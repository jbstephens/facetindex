"""Build a versioned facet index from a Parquet file. Runs inside the pipeline, after dedup.

    from facetindex import build_index
    build_index("load_0421.parquet", "/srv/facetindex", version="load-0421", id_column="id")

What it writes is described in ``artifact.py``. Cost model: a cheap length pass over every
column, one scan of the long-text columns for their fill bitmaps, then a single sweep over
the short columns in Arrow batches that produces statistics, postings, sort keys and the
numeric arrays together. Long text is never materialised.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyroaring import BitMap

from . import artifact as A
from .filters import FALSY, TRUTHY
from .profile import (
    AttributeProfile,
    Pass2,
    ProfileConfig,
    approx_distinct_long,
    is_long,
    make_profile,
    nonempty_expr,
    numeric_expr_normalised,
    pass1_lengths,
    qi,
)


@dataclass
class BuildResult:
    version: str
    path: Path
    row_count: int
    seconds: float
    bytes: int
    kinds: dict[str, str]
    published: bool


def source_fingerprint(path: str) -> str:
    """Cheap, stable identity for a Parquet file: size, mtime, row count and schema."""
    st = os.stat(path)
    md = pq.read_metadata(path)
    h = hashlib.sha1()
    h.update(f"{st.st_size}:{st.st_mtime_ns}:{md.num_rows}:{md.num_row_groups}:".encode())
    h.update(str(md.schema).encode())
    return h.hexdigest()[:12]


def bitmap_bytes(positions: np.ndarray) -> tuple[bytes, int]:
    """Serialised roaring bitmap + cardinality for a uint32 position array (bulk path)."""
    bm = BitMap(array.array("I", np.ascontiguousarray(positions, dtype=np.uint32).tobytes()))
    bm.run_optimize()
    bm.shrink_to_fit()
    return bm.serialize(), len(bm)


_TRUTHY = pa.array(TRUTHY, pa.string())
_FALSY = pa.array(FALSY, pa.string())


class _Postings:
    def __init__(self):
        self.attr, self.kind, self.val, self.card, self.bm = [], [], [], [], []

    def add(self, attr, kind, val, positions):
        b, card = bitmap_bytes(positions)
        self.attr.append(attr); self.kind.append(kind); self.val.append(val); self.card.append(card); self.bm.append(b)

    def table(self) -> pa.Table:
        return pa.table({
            "attribute": pa.array(self.attr, pa.string()), "kind": pa.array(self.kind, pa.string()),
            "value": pa.array(self.val, pa.string()), "cardinality": pa.array(self.card, pa.int64()),
            "bitmap": pa.array(self.bm, pa.binary()),
        })


def _column_stats(col: pa.Array, num: np.ndarray, top_n: int) -> tuple[Pass2, pa.Array, np.ndarray]:
    """Pass-2 statistics from the Arrow column (normalised text) and its numeric cast.

    Returns (stats, lower-cased column, dictionary-encoded indices of the raw column).
    """
    enc = pc.dictionary_encode(col)
    dictionary = enc.dictionary
    counts = pc.value_counts(enc.indices.filter(pc.is_valid(enc.indices)))  # struct<values, counts>
    idxs = counts.field("values").to_numpy(zero_copy_only=False)
    cnts = counts.field("counts").to_numpy(zero_copy_only=False)
    if len(cnts) > top_n:
        kth = np.partition(cnts, len(cnts) - top_n)[len(cnts) - top_n]
        cand = np.flatnonzero(cnts >= kth)          # everything that could be in the top n, ties included
    else:
        cand = np.arange(len(cnts))
    if len(cand):
        cand_vals = np.asarray(dictionary.take(pa.array(idxs[cand])).to_pylist(), dtype=object)
        order = cand[np.lexsort((cand_vals, -cnts[cand]))]
    else:
        order = np.zeros(0, int)
    top = [[dictionary[int(idxs[i])].as_py(), int(cnts[i])] for i in order[:top_n]]
    low = pc.utf8_lower(col)
    truthy = int(pc.sum(pc.is_in(low, value_set=_TRUTHY)).as_py() or 0)
    falsy = int(pc.sum(pc.is_in(low, value_set=_FALSY)).as_py() or 0)
    finite = num[~np.isnan(num)]
    p2 = Pass2(distinct=len(dictionary), numeric_count=int(finite.size),
               min=float(finite.min()) if finite.size else None, max=float(finite.max()) if finite.size else None,
               mean=float(finite.mean()) if finite.size else None, truthy=truthy, falsy=falsy, top_values=top)
    return p2, low, enc


def _text_rank_key(low: pa.Array) -> np.ndarray:
    """Dense rank (1-based) of the lower-cased value in byte order; NaN where missing."""
    enc = pc.dictionary_encode(low)
    d = enc.dictionary
    rank_of_dict = np.empty(len(d), dtype=np.float64)
    if len(d):
        order = pc.sort_indices(d).to_numpy(zero_copy_only=False)
        rank_of_dict[order] = np.arange(1, len(d) + 1, dtype=np.float64)
    valid = pc.is_valid(enc.indices).to_numpy(zero_copy_only=False)
    idx = pc.fill_null(enc.indices, 0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    out = np.full(len(low), np.nan)
    out[valid] = rank_of_dict[idx[valid]]
    return out


def build_index(
    source: str,
    root: str,
    version: str | None = None,
    id_column: str = "id",
    cfg: ProfileConfig | None = None,
    publish: bool = True,
    with_signatures: bool = False,
    overwrite: bool = False,
    retain: int | None = 3,
    threads: int | None = None,
    log=None,
) -> BuildResult:
    """Build ``versions/<version>`` under ``root`` and (optionally) point CURRENT at it.

    ``version`` should be the pipeline's own load identifier so the serving table and the
    index carry the same name; it defaults to a fingerprint of the source file.
    """
    t0 = time.time()
    cfg = cfg or ProfileConfig()
    _log = log or (lambda *_: None)

    def log(msg):
        _log(f"[{time.time() - t0:6.1f}s] {msg}")

    version = version or f"fp-{source_fingerprint(source)}"
    A.version_path(root, version)  # validates the name

    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={int(threads)}")
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet({_sql_str(source)})")
    columns = [r[0] for r in con.execute("DESCRIBE src").fetchall()]
    if id_column not in columns:
        raise ValueError(f"id column {id_column!r} not in {source}: {columns[:10]}...")
    n_rows, n_ids, n_null_ids = con.execute(
        f"SELECT count(*), count(DISTINCT {qi(id_column)}), count(*) - count({qi(id_column)}) FROM src"
    ).fetchone()
    if n_null_ids:
        raise ValueError(f"{n_null_ids} rows have a NULL {id_column}")
    if n_ids != n_rows:
        raise ValueError(f"{id_column} is not unique: {n_rows} rows, {n_ids} distinct ids (dedupe first)")
    attrs = [c for c in columns if c != id_column]

    # ---- pass 1: lengths / fill / sample for every column; decide long vs short -------------
    log("pass 1: lengths and fill")
    p1 = pass1_lengths(con, "src", attrs, cfg)
    long_attrs = [c for c in attrs if p1[c]["fill"] > 0 and is_long(cfg, c, p1[c])]
    empty_attrs = [c for c in attrs if p1[c]["fill"] == 0]
    short_attrs = [c for c in attrs if c not in long_attrs and c not in empty_attrs]
    p2: dict[str, Pass2] = approx_distinct_long(con, "src", long_attrs, cfg)

    # ---- canonical positions: rows sorted by id; narrow normalised table in DuckDB ------------
    log(f"materialising narrow table ({len(short_attrs)} short, {len(long_attrs)} long, {len(empty_attrs)} empty)")
    pos_expr = f"(row_number() OVER (ORDER BY CAST({qi(id_column)} AS VARCHAR)) - 1)::UINTEGER"
    sel = ", ".join(f"{nonempty_expr(f'CAST({qi(c)} AS VARCHAR)')} AS {qi(c)}" for c in short_attrs)
    con.execute(
        f"CREATE TABLE narrow AS SELECT CAST({qi(id_column)} AS VARCHAR) AS __id, {pos_expr} AS __pos"
        + (", " + sel if sel else "") + " FROM src"
    )

    build_dir = A.new_build_dir(root, version)
    try:
        ids_tbl = con.sql("SELECT __id AS id FROM narrow ORDER BY __pos").to_arrow_table()
        pq.write_table(ids_tbl, build_dir / A.IDS, compression="zstd")

        postings = _Postings()
        profiles: dict[str, AttributeProfile] = {}
        numeric_row_groups: dict[str, int] = {}
        key_attrs: list[str] = []
        (build_dir / A.KEYS_DIR).mkdir()
        ns_schema = pa.schema([("value", pa.float64()), ("position", pa.uint32())])
        ns_writer = pq.ParquetWriter(build_dir / A.NUMERIC_SORTED, ns_schema, compression="zstd")
        rg = 0

        # ---- long and empty columns: fill bitmaps only, one scan of the source -------------
        if long_attrs:
            log("long text: fill bitmaps")
            flags = ", ".join(f"{nonempty_expr(f'CAST({qi(c)} AS VARCHAR)')} IS NOT NULL AS {qi(c)}" for c in long_attrs)
            long_cols = ", ".join(qi(c) for c in long_attrs)
            lt = con.sql(f"SELECT {flags} FROM (SELECT {pos_expr} AS __pos, {long_cols} FROM src) ORDER BY __pos").to_arrow_table()
            for c in long_attrs:
                present = np.flatnonzero(lt.column(c).to_numpy(zero_copy_only=False))
                postings.add(c, "present", None, present)
                profiles[c] = make_profile(cfg, c, n_rows, p1[c], p2[c])
            del lt
        for c in empty_attrs:
            postings.add(c, "present", None, np.zeros(0, dtype=np.uint32))
            profiles[c] = make_profile(cfg, c, n_rows, p1[c], None)

        # ---- the sweep: short columns in Arrow batches ------------------------------------------
        log("sweep: statistics, postings, keys, numeric arrays")
        all_pos = np.arange(n_rows, dtype=np.uint32)
        for i in range(0, len(short_attrs), cfg.batch_size):
            batch = short_attrs[i : i + cfg.batch_size]
            cols = ", ".join(qi(c) for c in batch)
            nums = ", ".join(f"{numeric_expr_normalised(qi(c))} AS {qi('__n_' + c)}" for c in batch)
            tbl = con.sql(f"SELECT {cols}, {nums} FROM narrow ORDER BY __pos").to_arrow_table()
            for c in batch:
                col = tbl.column(c).combine_chunks()
                num = tbl.column("__n_" + c).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)  # NaN = NULL
                stats, low, enc = _column_stats(col, num, cfg.top_n)
                p2[c] = stats
                prof = make_profile(cfg, c, n_rows, p1[c], stats)
                profiles[c] = prof
                valid = pc.is_valid(col).to_numpy(zero_copy_only=False)
                present = all_pos[valid]
                postings.add(c, "present", None, present)
                if prof.kind == "flag":
                    postings.add(c, "true", None, all_pos[pc.fill_null(pc.is_in(low, value_set=_TRUTHY), False).to_numpy(zero_copy_only=False)])
                    postings.add(c, "false", None, all_pos[pc.fill_null(pc.is_in(low, value_set=_FALSY), False).to_numpy(zero_copy_only=False)])
                elif prof.facetable:
                    idx = pc.fill_null(enc.indices, 0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
                    vidx = idx[valid]
                    order = np.argsort(vidx, kind="stable")
                    bounds = np.searchsorted(vidx[order], np.arange(len(enc.dictionary) + 1))
                    for k, val in enumerate(enc.dictionary.to_pylist()):
                        postings.add(c, "value", val, present[order[bounds[k] : bounds[k + 1]]])
                # numeric sorted (value, position) for every column with any numeric values
                if stats.numeric_count > 0:
                    nz = np.flatnonzero(~np.isnan(num))
                    o = np.argsort(num[nz], kind="stable")  # stable => ties ordered by position
                    ns_writer.write_table(pa.table({"value": num[nz][o], "position": nz[o].astype(np.uint32)}, schema=ns_schema))
                    numeric_row_groups[c] = rg
                    rg += 1
                # dense sort key
                if prof.sortable:
                    if prof.kind == "numeric":
                        key = num
                    elif prof.kind == "flag":
                        key = np.full(n_rows, np.nan)
                        key[pc.fill_null(pc.is_in(low, value_set=_TRUTHY), False).to_numpy(zero_copy_only=False)] = 1.0
                        key[pc.fill_null(pc.is_in(low, value_set=_FALSY), False).to_numpy(zero_copy_only=False)] = 0.0
                    else:
                        key = _text_rank_key(low)
                    pq.write_table(pa.table({"key": pa.array(key, pa.float64(), from_pandas=True)}),
                                   build_dir / A.KEYS_DIR / f"{_safe_name(c)}.parquet", compression="zstd")
                    key_attrs.append(c)
            del tbl
        ns_writer.close()
        pq.write_table(postings.table(), build_dir / A.POSTINGS, compression="zstd")

        # ---- values: the narrow normalised table, one column per short attribute ------------------
        log("writing narrow values table")
        if short_attrs:
            con.execute(
                f"COPY (SELECT {', '.join(qi(c) for c in short_attrs)} FROM narrow ORDER BY __pos) "
                f"TO {_sql_str((build_dir / A.VALUES).as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
            )
        else:
            pq.write_table(pa.table({"__empty": pa.nulls(n_rows, pa.bool_())}), build_dir / A.VALUES)

        # ---- optional forward index (the "signals key") -----------------------------------------
        if with_signatures:
            log("writing signatures")
            _write_signatures(build_dir, postings.table(), n_rows, ids_tbl)

        # ---- stats + manifest -----------------------------------------------------------------
        kinds = {c: profiles[c].kind for c in attrs}
        stats_json = {c: profiles[c].to_dict() for c in attrs}
        (build_dir / A.STATS).write_text(json.dumps(stats_json, indent=1, sort_keys=True, default=_json_default))
        manifest = A.Manifest(
            version=version, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            format_version=A.FORMAT_VERSION, row_count=int(n_rows), id_column=id_column,
            source={"path": os.path.abspath(source), "fingerprint": source_fingerprint(source)},
            columns=attrs, kinds=kinds, numeric_row_groups=numeric_row_groups, key_attrs=key_attrs,
            value_attrs=short_attrs, files={}, build_seconds=0.0, has_signatures=with_signatures,
        )
        manifest.files = A.inventory(build_dir)
        manifest.build_seconds = round(time.time() - t0, 3)
        (build_dir / A.MANIFEST).write_text(manifest.to_json())
        final = A.finalize_version(build_dir, root, version, overwrite=overwrite)
        log("finalised " + ", ".join(f"{k}={sum(1 for v in kinds.values() if v == k)}"
                                     for k in ("flag", "categorical", "numeric", "short_text", "long_text", "empty")))
    except BaseException:
        shutil.rmtree(build_dir, ignore_errors=True)
        raise
    finally:
        con.close()

    if publish:
        A.set_current(root, version)
        if retain:
            A.retain(root, keep=retain)
    total = sum(manifest.files.values()) + (final / A.MANIFEST).stat().st_size
    return BuildResult(version, final, int(n_rows), round(time.time() - t0, 3), total, kinds, publish)


def _write_signatures(build_dir: Path, postings: pa.Table, n_rows: int, ids_tbl: pa.Table) -> None:
    """Forward index: one bitset per entity over the (attribute, value) dimensions.

    This is what a per-entity "signals key" is once you make it searchable: a fixed
    dictionary of dimensions and one bit per dimension. ``dimensions.json`` is the dictionary.
    It is NOT what the engine queries (the inverted postings are), but it is cheap to emit,
    and in Postgres a ``bit(n)`` column with ``(sig & mask) = mask`` gives a narrow, cacheable
    scan for ANDed flag/category filters without any other index.
    """
    dims = []
    for r in postings.to_pylist():
        if r["kind"] == "true":
            dims.append((r["attribute"], "=true", r["bitmap"]))
        elif r["kind"] == "value":
            dims.append((r["attribute"], r["value"], r["bitmap"]))
    nbits = len(dims)
    nbytes = (nbits + 7) // 8
    sig = np.zeros((n_rows, nbytes), dtype=np.uint8)
    for bit, (_, _, bm_bytes) in enumerate(dims):
        pos = np.frombuffer(BitMap.deserialize(bm_bytes).to_array(), dtype=np.uint32)
        sig[pos, bit // 8] |= np.uint8(1 << (bit % 8))
    tbl = pa.table({
        "id": ids_tbl.column("id"),
        "signature": pa.array([sig[i].tobytes() for i in range(n_rows)], pa.binary()),
    })
    pq.write_table(tbl, build_dir / A.SIGNATURES, compression="zstd")
    (build_dir / A.DIMENSIONS).write_text(json.dumps(
        {"bit_count": nbits, "byte_count": nbytes, "bit_order": "lsb-first within byte",
         "dimensions": [{"bit": i, "attribute": a, "value": v} for i, (a, v, _) in enumerate(dims)]},
        indent=1))


def _safe_name(attr: str) -> str:
    """File name for an attribute: readable, filesystem-safe and unique."""
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in attr)
    return f"{keep[:80]}-{hashlib.sha1(attr.encode()).hexdigest()[:8]}"


def _json_default(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"
