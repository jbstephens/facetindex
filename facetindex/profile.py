"""Attribute profiling and kind inference, done once per load inside the pipeline.

The output (one ``AttributeProfile`` per column) is written into the index artifact as
``stats.json`` and served from there, so the API never profiles the serving table again.

Two passes, mirroring the cost model from the problem statement:
  1. a cheap length/fill pass over every column (never hashes long values),
  2. full statistics only for columns whose values are all short.
Long columns get an approximate distinct count (HyperLogLog), fill count and a sample.

Two implementations share the same classification and output type:
  * ``profile_relation``: pure SQL, for the ``facetindex profile`` command and for anyone
    who wants the statistics without building an index;
  * the builder (``build.py``), which computes pass 2 from the Arrow columns it is already
    holding to build postings, in one sweep.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb

from .filters import FALSY, TRUTHY

KINDS = ("flag", "categorical", "numeric", "short_text", "long_text", "empty")


@dataclass
class ProfileConfig:
    max_categorical_distinct: int = 50     # <= this many distinct values => facetable
    long_text_max_len: int = 256           # any value longer than this => long_text
    long_text_avg_len: float = 96.0        # or an average longer than this
    numeric_ratio: float = 0.98            # share of filled values that parse as numbers
    top_n: int = 20
    batch_size: int = 24                   # columns per statement / per Arrow batch
    kind_overrides: dict[str, str] = field(default_factory=dict)  # admin metadata wins


@dataclass
class AttributeProfile:
    name: str
    kind: str
    fill_count: int
    empty_count: int
    distinct_count: int | None          # exact for short columns, approximate for long
    distinct_is_approx: bool
    true_count: int | None
    false_count: int | None
    numeric_count: int
    min: float | None
    max: float | None
    mean: float | None
    max_len: int
    avg_len: float | None
    top_values: list[list[Any]]         # [[value, count], ...]
    sample: str | None
    facetable: bool                     # has per-value postings in the index
    sortable: bool
    ops: list[str]                      # filter ops the index answers for this attribute

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pass2:
    """Full statistics of one short column, however they were computed."""
    distinct: int = 0
    numeric_count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    truthy: int = 0
    falsy: int = 0
    top_values: list[list[Any]] = field(default_factory=list)
    approx: bool = False


def nonempty_expr(col: str) -> str:
    return f"NULLIF(TRIM({col}), '')"


def numeric_expr(col: str) -> str:
    inner = f"TRY_CAST({nonempty_expr(col)} AS DOUBLE)"
    return f"(CASE WHEN isfinite({inner}) THEN {inner} END)"


def numeric_expr_normalised(col: str) -> str:
    """Same as numeric_expr for a column that is already trimmed and ''-> NULL."""
    inner = f"TRY_CAST({col} AS DOUBLE)"
    return f"(CASE WHEN isfinite({inner}) THEN {inner} END)"


def qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_list(vals) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in vals)


def classify(cfg: ProfileConfig, name: str, fill: int, distinct: int | None, max_len: int,
             avg_len: float | None, numeric_count: int, truthy: int, falsy: int) -> str:
    if name in cfg.kind_overrides:
        k = cfg.kind_overrides[name]
        if k not in KINDS:
            raise ValueError(f"override for {name!r} must be one of {KINDS}, got {k!r}")
        return k
    if fill == 0:
        return "empty"
    if max_len > cfg.long_text_max_len or (avg_len or 0) > cfg.long_text_avg_len:
        return "long_text"
    if truthy + falsy == fill and (distinct or 0) <= 2:
        return "flag"
    if numeric_count / fill >= cfg.numeric_ratio:
        return "numeric"
    if distinct is not None and distinct <= cfg.max_categorical_distinct:
        return "categorical"
    return "short_text"


def ops_for(kind: str, facetable: bool) -> list[str]:
    base = ["is_empty", "is_not_empty"]
    if kind == "flag":
        return ["flag"] + base
    if kind == "numeric":
        return (["in"] if facetable else []) + ["between"] + base
    if kind == "categorical":
        return ["in", "contains", "between"] + base
    if kind == "short_text":
        return ["in", "contains", "between"] + base   # between matches numeric-looking values
    return base                                        # long_text / empty: contains => SQL fallback


def is_long(cfg: ProfileConfig, name: str, p1: dict) -> bool:
    if cfg.kind_overrides.get(name) == "long_text":
        return True
    if name in cfg.kind_overrides:
        return False
    return p1["max_len"] > cfg.long_text_max_len or (p1["avg_len"] or 0) > cfg.long_text_avg_len


def pass1_lengths(con: duckdb.DuckDBPyConnection, relation: str, attrs: list[str],
                  cfg: ProfileConfig) -> dict[str, dict]:
    """Fill count, max/avg length and a sample for every column. Never hashes or sorts values."""
    out: dict[str, dict] = {}
    for i in range(0, len(attrs), cfg.batch_size):
        batch = attrs[i : i + cfg.batch_size]
        parts = []
        for c in batch:
            ne = nonempty_expr(f"CAST({qi(c)} AS VARCHAR)")
            parts.append(f"count({ne}), max(length({ne})), avg(length({ne})), any_value({ne})")
        row = con.execute(f"SELECT {', '.join(parts)} FROM {relation}").fetchone()
        for j, c in enumerate(batch):
            fill, max_len, avg_len, sample = row[4 * j : 4 * j + 4]
            out[c] = {"fill": int(fill or 0), "max_len": int(max_len or 0),
                      "avg_len": float(avg_len) if avg_len is not None else None, "sample": sample}
    return out


def pass2_sql(con: duckdb.DuckDBPyConnection, relation: str, short: list[str], cfg: ProfileConfig) -> dict[str, Pass2]:
    out: dict[str, Pass2] = {}
    for i in range(0, len(short), cfg.batch_size):
        batch = short[i : i + cfg.batch_size]
        parts = []
        for c in batch:
            col = f"CAST({qi(c)} AS VARCHAR)"
            ne = nonempty_expr(col)
            num = numeric_expr(col)
            low = f"LOWER({ne})"
            parts.append(
                f"count(DISTINCT {ne}), count({num}), min({num}), max({num}), avg({num}), "
                f"count(*) FILTER (WHERE {low} IN ({sql_list(TRUTHY)})), "
                f"count(*) FILTER (WHERE {low} IN ({sql_list(FALSY)}))")
        row = con.execute(f"SELECT {', '.join(parts)} FROM {relation}").fetchone()
        for j, c in enumerate(batch):
            d, nc, mn, mx, mean, tr, fa = row[7 * j : 7 * j + 7]
            out[c] = Pass2(int(d or 0), int(nc or 0), mn, mx, mean, int(tr or 0), int(fa or 0))
    for c in short:
        ne = nonempty_expr(f"CAST({qi(c)} AS VARCHAR)")
        rows = con.execute(
            f"SELECT {ne} AS v, count(*) AS n FROM {relation} WHERE {ne} IS NOT NULL "
            f"GROUP BY 1 ORDER BY n DESC, v LIMIT {int(cfg.top_n)}").fetchall()
        out[c].top_values = [[v, int(n)] for v, n in rows]
    return out


def approx_distinct_long(con: duckdb.DuckDBPyConnection, relation: str, long: list[str],
                         cfg: ProfileConfig) -> dict[str, Pass2]:
    """Long columns: one HyperLogLog pass, no sort, no exact distinct."""
    out: dict[str, Pass2] = {}
    for i in range(0, len(long), cfg.batch_size):
        batch = long[i : i + cfg.batch_size]
        parts = [f"approx_count_distinct({nonempty_expr(f'CAST({qi(c)} AS VARCHAR)')})" for c in batch]
        row = con.execute(f"SELECT {', '.join(parts)} FROM {relation}").fetchone()
        for j, c in enumerate(batch):
            out[c] = Pass2(distinct=int(row[j] or 0), approx=True)
    return out


def make_profile(cfg: ProfileConfig, name: str, n_rows: int, p1: dict, p2: Pass2 | None) -> AttributeProfile:
    p2 = p2 or Pass2()
    kind = classify(cfg, name, p1["fill"], p2.distinct, p1["max_len"], p1["avg_len"],
                    p2.numeric_count, p2.truthy, p2.falsy)
    facetable = kind in ("flag", "categorical") or (
        kind == "numeric" and p2.distinct <= cfg.max_categorical_distinct)
    sortable = kind in ("flag", "categorical", "numeric", "short_text")
    return AttributeProfile(
        name=name, kind=kind, fill_count=p1["fill"], empty_count=n_rows - p1["fill"],
        distinct_count=p2.distinct if p1["fill"] else 0, distinct_is_approx=p2.approx,
        true_count=p2.truthy if kind == "flag" else None, false_count=p2.falsy if kind == "flag" else None,
        numeric_count=p2.numeric_count, min=_f(p2.min), max=_f(p2.max), mean=_f(p2.mean),
        max_len=p1["max_len"], avg_len=p1["avg_len"], top_values=list(p2.top_values), sample=p1["sample"],
        facetable=facetable, sortable=sortable, ops=ops_for(kind, facetable))


def profile_relation(con: duckdb.DuckDBPyConnection, relation: str, id_column: str,
                     cfg: ProfileConfig | None = None, columns: list[str] | None = None) -> dict[str, AttributeProfile]:
    """Profile every non-id column of ``relation`` with SQL only. All columns are treated as text."""
    cfg = cfg or ProfileConfig()
    if columns is None:
        columns = [r[0] for r in con.execute(f"DESCRIBE {relation}").fetchall()]
    attrs = [c for c in columns if c != id_column]
    n_rows = con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0]
    p1 = pass1_lengths(con, relation, attrs, cfg)
    short = [c for c in attrs if p1[c]["fill"] > 0 and not is_long(cfg, c, p1[c])]
    long = [c for c in attrs if p1[c]["fill"] > 0 and c not in short]
    p2 = pass2_sql(con, relation, short, cfg)
    p2.update(approx_distinct_long(con, relation, long, cfg))
    return {c: make_profile(cfg, c, n_rows, p1[c], p2.get(c)) for c in attrs}


def profile_parquet(path: str, id_column: str = "id", cfg: ProfileConfig | None = None):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet({_sql_str(path)})")
    return profile_relation(con, "src", id_column, cfg)


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _sql_str(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"
