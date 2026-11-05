"""The query engine: load one version of the artifact and answer counts, facets, pages.

Every query reduces to roaring-bitmap algebra over entity *positions* (0..n-1, sorted by
id). Nothing here touches the serving database; the API fetches display rows afterwards
with a primary-key lookup (``WHERE id = ANY(%s)``) or from ``values.parquet``.

No background threads, no locks: a FacetIndex is immutable once opened, so any number of
worker processes can hold one each. Lazy members (sort keys, numeric arrays, text values)
are loaded on first use from column-pruned Parquet reads and cached on the instance.
"""

from __future__ import annotations

import array
import base64
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyroaring import BitMap, FrozenBitMap

from . import artifact as A
from . import filters as F
from .build import _safe_name


class UnknownAttribute(KeyError):
    pass


class UnsupportedInIndex(Exception):
    """The filter needs data the index does not hold (e.g. contains() on long text).

    Fall back to SQL: ``filters.to_sql(f)`` renders the same filter for the serving table.
    """


@dataclass
class Page:
    total: int
    positions: np.ndarray
    ids: list
    sort_keys: list
    next_cursor: str | None


@dataclass
class FacetValue:
    value: str
    count: int


@dataclass
class Facet:
    attribute: str
    kind: str
    values: list[FacetValue] = field(default_factory=list)      # categorical/flag
    empty_count: int = 0
    filled_count: int = 0
    numeric: dict | None = None                                 # min/max/histogram for numerics


@dataclass
class ExplainRow:
    node: str
    cardinality: int
    note: str = ""


class FacetIndex:
    def __init__(self, path: str | Path, max_cached_attributes: int = 64):
        self.path = Path(path)
        self.max_cached_attributes = max_cached_attributes  # per lazy cache (keys, values, numeric)
        self.manifest = A.read_manifest(self.path)
        self.version = self.manifest.version
        self.n = self.manifest.row_count
        self.stats: dict[str, dict] = json.loads((self.path / A.STATS).read_text())
        self.kinds = self.manifest.kinds
        self._all = FrozenBitMap(range(self.n))
        self.ids: pa.Array = pq.read_table(self.path / A.IDS).column("id").combine_chunks()
        self._load_postings()
        self._numeric_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._key_cache: dict[str, np.ndarray] = {}
        self._values_cache: dict[str, pa.Array] = {}
        self._values_file = pq.ParquetFile(self.path / A.VALUES) if (self.path / A.VALUES).exists() else None
        self._numeric_file = (pq.ParquetFile(self.path / A.NUMERIC_SORTED)
                              if (self.path / A.NUMERIC_SORTED).exists() else None)

    # ------------------------------------------------------------------ loading
    @classmethod
    def open(cls, root_or_version_path: str | Path, version: str | None = None) -> FacetIndex:
        """Open ``<root>`` (CURRENT version), ``<root>`` + version, or a version directory directly."""
        p = Path(root_or_version_path)
        if version is not None:
            return cls(A.version_path(p, version))
        if (p / A.MANIFEST).exists():
            return cls(p)
        cur = A.current_version(p)
        if cur is None:
            raise FileNotFoundError(f"no CURRENT pointer under {p}")
        return cls(A.version_path(p, cur))

    def _load_postings(self) -> None:
        tbl = pq.read_table(self.path / A.POSTINGS)
        self.present: dict[str, FrozenBitMap] = {}
        self.flag_true: dict[str, FrozenBitMap] = {}
        self.flag_false: dict[str, FrozenBitMap] = {}
        self.values_postings: dict[str, dict[str, FrozenBitMap]] = {}
        for attr, kind, val, bm in zip(
            tbl.column("attribute").to_pylist(), tbl.column("kind").to_pylist(),
            tbl.column("value").to_pylist(), tbl.column("bitmap").to_pylist(),
        ):
            fbm = FrozenBitMap.deserialize(bm)
            if kind == "present":
                self.present[attr] = fbm
            elif kind == "true":
                self.flag_true[attr] = fbm
            elif kind == "false":
                self.flag_false[attr] = fbm
            else:
                self.values_postings.setdefault(attr, {})[val] = fbm

    def warm(self) -> None:
        """Load every lazy structure now (optional, e.g. right after a version switch).

        Lifts the per-attribute cache cap first: warming means you want everything resident.
        """
        self.max_cached_attributes = max(self.max_cached_attributes, len(self.kinds) + 1)
        for a in self.manifest.numeric_row_groups:
            self._numeric(a)
        for a in self.manifest.key_attrs:
            self._keys(a)
        for a in self.manifest.value_attrs:
            self._values(a)

    def _numeric(self, attr: str) -> tuple[np.ndarray, np.ndarray]:
        if attr not in self._numeric_cache:
            rg = self.manifest.numeric_row_groups.get(attr)
            if rg is None or self._numeric_file is None:
                self._numeric_cache[attr] = (np.zeros(0), np.zeros(0, dtype=np.uint32))
            else:
                t = self._numeric_file.read_row_group(rg)
                self._put(self._numeric_cache, attr, (
                    t.column("value").to_numpy(), t.column("position").to_numpy().astype(np.uint32, copy=False)))
        return self._numeric_cache[attr]

    def _keys(self, attr: str) -> np.ndarray:
        if attr not in self._key_cache:
            if attr not in self.manifest.key_attrs:
                raise UnsupportedInIndex(f"attribute {attr!r} is not sortable (kind={self.kinds.get(attr)})")
            t = pq.read_table(self.path / A.KEYS_DIR / f"{_safe_name(attr)}.parquet")
            self._put(self._key_cache, attr, t.column("key").to_numpy(zero_copy_only=False))  # NaN for NULL
        return self._key_cache[attr]

    def _put(self, cache: dict, key, value) -> None:
        if len(cache) >= self.max_cached_attributes:
            cache.pop(next(iter(cache)))  # drop the oldest inserted; good enough for an access-locality cache
        cache[key] = value

    def _values(self, attr: str) -> pa.Array:
        if attr not in self._values_cache:
            if attr not in self.manifest.value_attrs or self._values_file is None:
                raise UnsupportedInIndex(
                    f"attribute {attr!r} has no stored values (kind={self.kinds.get(attr)}); use filters.to_sql()")
            self._put(self._values_cache, attr, self._values_file.read(columns=[attr]).column(attr).combine_chunks())
        return self._values_cache[attr]

    # ------------------------------------------------------------------ evaluation
    def _check(self, attr: str, strict: bool) -> bool:
        if attr in self.kinds:
            return True
        if strict:
            raise UnknownAttribute(attr)
        return False

    def evaluate(self, f, segments: dict | None = None, strict: bool = False) -> BitMap:
        """Evaluate a filter to a bitmap of positions.

        ``segments`` maps segment name -> filter for ``Segment`` references.
        With ``strict=False`` an unknown attribute simply matches nothing (the schema is
        dynamic; a saved segment may outlive an attribute). ``validate`` reports those.
        """
        memo: dict[str, BitMap] = {}
        stack: list[str] = []

        def ev(n) -> BitMap:
            if isinstance(n, F.All):
                return BitMap(self._all)
            if isinstance(n, F.And):
                if not n.children:
                    return BitMap(self._all)
                parts = sorted((ev(c) for c in n.children), key=len)
                out = parts[0]
                for p in parts[1:]:
                    out &= p
                    if not out:
                        break
                return out
            if isinstance(n, F.Or):
                if not n.children:
                    return BitMap()
                return BitMap.union(*(ev(c) for c in n.children))
            if isinstance(n, F.Not):
                return self._all - ev(n.child)
            if isinstance(n, F.Segment):
                if segments is None or n.name not in segments:
                    if strict:
                        raise KeyError(f"unknown segment {n.name!r}")
                    return BitMap()
                if n.name in stack:
                    raise ValueError(f"segment cycle: {' -> '.join(stack + [n.name])}")
                if n.name not in memo:
                    stack.append(n.name)
                    try:
                        memo[n.name] = ev(segments[n.name])
                    finally:
                        stack.pop()
                return BitMap(memo[n.name])
            if not self._check(n.attr, strict):
                return BitMap()
            if isinstance(n, F.Flag):
                src = self.flag_true if n.value else self.flag_false
                return BitMap(src.get(n.attr, BitMap()))
            if isinstance(n, F.IsNotEmpty):
                return BitMap(self.present.get(n.attr, BitMap()))
            if isinstance(n, F.IsEmpty):
                return self._all - self.present.get(n.attr, BitMap())
            if isinstance(n, F.In):
                return self._eval_in(n)
            if isinstance(n, F.Between):
                return self._eval_between(n)
            if isinstance(n, F.Contains):
                return self._eval_contains(n)
            raise TypeError(f"not a filter node: {n!r}")

        return ev(f)

    def _eval_in(self, n: F.In) -> BitMap:
        wanted = list(n.values)
        post = self.values_postings.get(n.attr)
        if post is not None:
            hits = [post[v] for v in wanted if v in post]
            return BitMap.union(*hits) if hits else BitMap()
        if n.attr in self.manifest.value_attrs:
            arr = self._values(n.attr)
            mask = pc.fill_null(pc.is_in(arr, value_set=pa.array(wanted, pa.string())), False)
            return bitmap_from_positions(np.flatnonzero(mask.to_numpy(zero_copy_only=False)))
        if self.kinds.get(n.attr) == "empty":
            return BitMap()
        raise UnsupportedInIndex(f"'in' on long-text attribute {n.attr!r}; use filters.to_sql()")

    def _eval_between(self, n: F.Between) -> BitMap:
        values, positions = self._numeric(n.attr)
        lo = 0 if n.lo is None else int(np.searchsorted(values, float(n.lo), side="left"))
        hi = len(values) if n.hi is None else int(np.searchsorted(values, float(n.hi), side="right"))
        if hi <= lo:
            return BitMap()
        return bitmap_from_positions(positions[lo:hi])

    def _eval_contains(self, n: F.Contains) -> BitMap:
        if self.kinds.get(n.attr) == "empty":
            return BitMap()
        arr = self._values(n.attr)  # raises UnsupportedInIndex for long text
        if n.needle == "":
            return BitMap(self.present.get(n.attr, BitMap()))
        mask = pc.fill_null(pc.match_substring(arr, n.needle, ignore_case=True), False)
        return bitmap_from_positions(np.flatnonzero(mask.to_numpy(zero_copy_only=False)))

    # ------------------------------------------------------------------ public queries
    def count(self, f, segments: dict | None = None) -> int:
        return len(self.evaluate(f, segments))

    def validate(self, f, segments: dict | None = None) -> list[str]:
        """Human-readable problems: unknown attributes, unsupported ops, unknown segments."""
        problems = []
        for a in F.attributes(f):
            if a not in self.kinds:
                problems.append(f"unknown attribute {a!r}")
        for s in F.segments_referenced(f):
            if segments is None or s not in segments:
                problems.append(f"unknown segment {s!r}")

        def walk(n):
            if isinstance(n, (F.And, F.Or)):
                for c in n.children:
                    walk(c)
            elif isinstance(n, F.Not):
                walk(n.child)
            elif isinstance(n, (F.Contains, F.In)) and self.kinds.get(n.attr) == "long_text":
                problems.append(f"{type(n).__name__.lower()} on long-text attribute {n.attr!r} needs the SQL fallback")
            elif isinstance(n, F.Flag) and n.attr in self.kinds and self.kinds[n.attr] != "flag":
                problems.append(f"flag filter on non-flag attribute {n.attr!r} (kind={self.kinds[n.attr]})")

        walk(f)
        return problems

    def explain(self, f, segments: dict | None = None) -> list[ExplainRow]:
        """Per-node cardinalities, for debugging a slow or surprising segment."""
        rows: list[ExplainRow] = []

        def walk(n, depth):
            try:
                card = len(self.evaluate(n, segments))
                note = ""
            except UnsupportedInIndex as e:
                card, note = -1, str(e)
            rows.append(ExplainRow("  " * depth + _label(n), card, note))
            if isinstance(n, (F.And, F.Or)):
                for c in n.children:
                    walk(c, depth + 1)
            elif isinstance(n, F.Not):
                walk(n.child, depth + 1)

        walk(f, 0)
        return rows

    def ids_for(self, f, segments: dict | None = None, limit: int | None = None) -> list:
        bm = self.evaluate(f, segments)
        pos = _positions(bm)
        if limit is not None:
            pos = pos[:limit]
        return self.ids.take(pa.array(pos)).to_pylist()

    def facets(
        self,
        f,
        attributes: list[str] | None = None,
        top_n: int = 20,
        exclude_own: bool = True,
        segments: dict | None = None,
        numeric_bins: int = 10,
        include_zero: bool = False,
    ) -> dict[str, Facet]:
        """Facet counts under the current filter.

        For each facetable attribute, count the entities matching the filter per value.
        With ``exclude_own`` the attribute's own top-level clauses are dropped first, so a
        user who selected tier=A still sees how many tier=B rows there are (the standard
        multi-select behaviour). Numeric attributes get min/max and a fixed-bin histogram.
        """
        base = self.evaluate(f, segments)
        attrs = attributes or [a for a in self.manifest.columns
                               if self.stats[a]["facetable"] or self.kinds[a] == "numeric"]
        own_cache: dict[str, BitMap] = {}
        member_cache: dict[int, np.ndarray] = {}
        touched = F.attributes(f)
        out: dict[str, Facet] = {}
        for a in attrs:
            if a not in self.kinds:
                continue
            if exclude_own and a in touched:
                key = json.dumps(F.to_dict(F.without_attribute(f, a)), sort_keys=True)
                if key not in own_cache:
                    own_cache[key] = self.evaluate(F.without_attribute(f, a), segments)
                cur = own_cache[key]
            else:
                cur = base
            kind = self.kinds[a]
            present = self.present.get(a, BitMap())
            fc = Facet(attribute=a, kind=kind)
            fc.filled_count = cur.intersection_cardinality(present)
            fc.empty_count = len(cur) - fc.filled_count
            if kind == "flag":
                fc.values = [
                    FacetValue("true", cur.intersection_cardinality(self.flag_true.get(a, BitMap()))),
                    FacetValue("false", cur.intersection_cardinality(self.flag_false.get(a, BitMap()))),
                ]
            elif a in self.values_postings:
                vals = [(v, cur.intersection_cardinality(bm)) for v, bm in self.values_postings[a].items()]
                vals = [(v, c) for v, c in vals if include_zero or c > 0]
                vals.sort(key=lambda t: (-t[1], _numeric_or_str(t[0])))
                fc.values = [FacetValue(v, c) for v, c in vals[:top_n]]
            if kind == "numeric":
                mkey = id(cur)
                if mkey not in member_cache:
                    m = np.zeros(self.n, dtype=bool)
                    m[_positions(cur)] = True
                    member_cache[mkey] = m
                fc.numeric = self._numeric_facet(a, member_cache[mkey], numeric_bins)
            out[a] = fc
        return out

    def _numeric_facet(self, attr: str, member: np.ndarray, bins: int) -> dict:
        values, positions = self._numeric(attr)
        if len(values) == 0:
            return {"count": 0}
        sel = values[member[positions]]
        if sel.size == 0:
            return {"count": 0}
        lo, hi = float(sel[0]), float(sel[-1])
        if hi > lo and bins > 0:
            edges = np.linspace(lo, hi, bins + 1)
            counts, _ = np.histogram(sel, bins=edges)
            hist = [{"lo": float(edges[i]), "hi": float(edges[i + 1]), "count": int(counts[i])} for i in range(bins)]
        else:
            hist = [{"lo": lo, "hi": hi, "count": int(sel.size)}]
        return {"count": int(sel.size), "min": lo, "max": hi, "median": float(sel[sel.size // 2]), "histogram": hist}

    def page(
        self,
        f,
        sort_by: str | None = None,
        descending: bool = False,
        limit: int = 50,
        cursor: str | None = None,
        segments: dict | None = None,
    ) -> Page:
        """One page of matching entities with a stable order and keyset pagination.

        Order: ``sort_by`` key (numeric value, or dense rank of the lower-cased text),
        missing values last in both directions, ties broken by id ascending. The cursor
        encodes (last key, last position), so deep pages cost the same as the first.
        """
        bm = self.evaluate(f, segments)
        total = len(bm)
        pos = _positions(bm)
        if sort_by is None:
            keys = pos.astype(np.float64)  # id order == position order
            desc = False
        else:
            if sort_by not in self.kinds:
                raise UnknownAttribute(sort_by)
            keys = self._keys(sort_by)[pos]
            desc = descending
        if cursor:
            ck, cp = _decode_cursor(cursor)
            pos, keys = _after(pos, keys, ck, cp, desc)
        sel = _top_k(pos, keys, limit, desc)
        sel_pos = pos[sel]
        sel_keys = keys[sel]
        next_cursor = None
        if len(sel) == limit and len(sel) < len(pos):
            next_cursor = _encode_cursor(sel_keys[-1], int(sel_pos[-1]))
        return Page(
            total=total,
            positions=sel_pos,
            ids=self.ids.take(pa.array(sel_pos)).to_pylist(),
            sort_keys=[None if (isinstance(k, float) and math.isnan(k)) else float(k) for k in sel_keys],
            next_cursor=next_cursor,
        )

    def rows(self, positions, columns: list[str] | None = None) -> list[dict]:
        """Display rows from the artifact's narrow table (short attributes only).

        Long text is not in the artifact; fetch it from the serving table by id.
        """
        positions = np.asarray(positions, dtype=np.int64)
        cols = columns or self.manifest.value_attrs
        cols = [c for c in cols if c in self.manifest.value_attrs]
        idx = pa.array(positions)
        out = [{"id": v} for v in self.ids.take(idx).to_pylist()]
        for c in cols:
            vals = self._values(c).take(idx).to_pylist()
            for r, v in zip(out, vals):
                r[c] = v
        return out

    def catalog(self) -> list[dict]:
        """The attribute catalog (what the live profile used to compute), in source order."""
        return [self.stats[a] for a in self.manifest.columns]

    def segment_counts(self, segments: dict) -> dict[str, int]:
        return {name: len(self.evaluate(F.Segment(name), segments)) for name in segments}

    def __repr__(self) -> str:
        return f"FacetIndex(version={self.version!r}, rows={self.n}, attributes={len(self.kinds)})"


# ---------------------------------------------------------------------- helpers

def _positions(bm: BitMap) -> np.ndarray:
    return np.frombuffer(bm.to_array(), dtype=np.uint32) if len(bm) else np.zeros(0, dtype=np.uint32)


def bitmap_from_positions(positions: np.ndarray) -> BitMap:
    """Build a bitmap from a uint32 position array via pyroaring's bulk path (10x faster than
    handing it the numpy array directly, which iterates element by element)."""
    if len(positions) == 0:
        return BitMap()
    return BitMap(array.array("I", np.ascontiguousarray(positions, dtype=np.uint32).tobytes()))


def _label(n) -> str:
    if isinstance(n, (F.And, F.Or)):
        return type(n).__name__ + f"[{len(n.children)}]"
    return repr(n)


def _numeric_or_str(v: str):
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v.lower())


def _encode_cursor(key: float, pos: int) -> str:
    k = None if (isinstance(key, float) and math.isnan(key)) else float(key)
    return base64.urlsafe_b64encode(json.dumps([k, pos]).encode()).decode()


def _decode_cursor(cursor: str) -> tuple[float, int]:
    try:
        k, p = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
    except Exception as e:  # noqa: BLE001
        raise ValueError("malformed cursor") from e
    return (math.nan if k is None else float(k)), int(p)


def _after(pos: np.ndarray, keys: np.ndarray, ck: float, cp: int, desc: bool):
    """Keep rows strictly after cursor (ck, cp) in the page order; missing keys sort last."""
    isnan = np.isnan(keys)
    if math.isnan(ck):
        keep = isnan & (pos > cp)
    else:
        if desc:
            keep = (~isnan & (keys < ck)) | isnan
        else:
            keep = (~isnan & (keys > ck)) | isnan
        keep |= (keys == ck) & (pos > cp)
    return pos[keep], keys[keep]


def _top_k(pos: np.ndarray, keys: np.ndarray, k: int, desc: bool) -> np.ndarray:
    """Indices of the first k rows in page order. Exact, deterministic, O(n) for large n."""
    n = len(pos)
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, n)
    isnan = np.isnan(keys)
    primary = np.where(isnan, np.inf, -keys if desc else keys)  # NaN -> +inf: always last
    if n > 8 * k:
        # partition to find the k-th smallest primary key, keep everything <= it (ties included),
        # then do an exact lexsort on that much smaller candidate set.
        kth = np.partition(primary, k - 1)[k - 1]
        cand = np.flatnonzero(primary <= kth)
    else:
        cand = np.arange(n)
    order = np.lexsort((pos[cand], primary[cand]))
    return cand[order[:k]]
