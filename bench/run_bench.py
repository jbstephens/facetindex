"""Benchmark: build cost, artifact size, and query latency of facetindex vs DuckDB-over-Parquet.

    python bench/run_bench.py --source data/org240k.parquet --root bench/out/idx --label 240k

Workloads mirror the problem statement: ANDed filter count, one sorted page of 50, facet
counts for every facetable attribute, 25 saved-segment counts at once, 25k-row export ids.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facetindex import filters as F  # noqa: E402
from facetindex.build import build_index  # noqa: E402
from facetindex.index import FacetIndex, UnsupportedInIndex  # noqa: E402
from facetindex.profile import numeric_expr  # noqa: E402
from tests.randfilters import Gen  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(name, ms):
    return {"workload": name, "n": len(ms), "p50_ms": round(statistics.median(ms), 2),
            "p95_ms": round(pct(ms, 95), 2), "max_ms": round(max(ms), 2), "mean_ms": round(statistics.mean(ms), 2)}


def typical_filters(stats, rng, n):
    """AND of 2-4 leaf clauses over flags/categoricals/numerics, like a sidebar selection."""
    g = Gen(stats, rng.randint(0, 10**9))
    out = []
    while len(out) < n:
        k = rng.randint(2, 4)
        kids = []
        for _ in range(k):
            r = rng.random()
            if r < 0.4:
                kids.append(F.Flag(g.pick(["flag"]), True))
            elif r < 0.75:
                a = g.pick(["categorical"])
                top = [v for v, _ in stats[a]["top_values"]]
                kids.append(F.In(a, rng.sample(top, min(len(top), rng.randint(1, 3)))))
            else:
                a = g.pick(["numeric"])
                lo, hi = stats[a]["min"], stats[a]["max"]
                x, y = sorted([rng.uniform(lo, hi), rng.uniform(lo, hi)])
                kids.append(F.Between(a, x, y))
        out.append(F.And(kids))
    return out


def key_sql(idx, attr):
    kind = idx.kinds[attr]
    col = f'"{attr}"'
    if kind == "numeric":
        return numeric_expr(col)
    if kind == "flag":
        return (f"CASE WHEN LOWER(TRIM({col})) IN ('true','t','1','yes','y') THEN 1.0 "
                f"WHEN LOWER(TRIM({col})) IN ('false','f','0','no','n') THEN 0.0 END")
    return f"LOWER(NULLIF(TRIM({col}), ''))"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--duckdb-threads", type=int, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    report = {"label": args.label, "source": args.source, "cpu_count": os.cpu_count()}

    # ---- build ----------------------------------------------------------------------
    if not args.skip_build:
        t0 = time.perf_counter()
        res = build_index(args.source, args.root, version=f"bench-{args.label}", overwrite=True,
                          with_signatures=False, log=lambda m: print(f"  [build] {m}", file=sys.stderr))
        report["build_seconds"] = round(time.perf_counter() - t0, 2)
        report["artifact_bytes"] = res.bytes
        report["rows"] = res.row_count
        report["kinds"] = {k: sum(1 for v in res.kinds.values() if v == k) for k in sorted(set(res.kinds.values()))}
    t0 = time.perf_counter()
    idx = FacetIndex.open(args.root)
    report["open_ms_cold"] = round((time.perf_counter() - t0) * 1000, 1)
    t0 = time.perf_counter()
    idx.warm()
    report["warm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    report["rows"] = idx.n
    report["attributes"] = len(idx.kinds)
    report["files"] = {k: v for k, v in idx.manifest.files.items() if not k.startswith("keys/")}
    report["files"]["keys/* (total)"] = sum(v for k, v in idx.manifest.files.items() if k.startswith("keys/"))
    report["postings_count"] = sum(len(v) for v in idx.values_postings.values()) + len(idx.present) + len(idx.flag_true) * 2

    filters = typical_filters(idx.stats, rng, args.queries)
    sortable = list(idx.manifest.key_attrs)
    facetable = [a for a in idx.manifest.columns if idx.stats[a]["facetable"]]
    results = []

    # ---- facetindex workloads -----------------------------------------------------------
    ms = []
    sel = []
    for f in filters:
        t0 = time.perf_counter(); c = idx.count(f); ms.append((time.perf_counter() - t0) * 1000); sel.append(c)
    results.append(summarize("facetindex count (AND of 2-4 clauses)", ms))
    report["selectivity_median_rows"] = int(statistics.median(sel))

    ms = []
    for f in filters:
        attr = rng.choice(sortable)
        t0 = time.perf_counter(); idx.page(f, sort_by=attr, descending=rng.random() < 0.5, limit=50)
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex page 50 sorted by random attribute (warm keys)", ms))

    ms = []
    for f in filters:
        t0 = time.perf_counter(); idx.page(f, limit=50); ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex page 50 by id", ms))

    ms = []
    for f in filters[:100]:
        t0 = time.perf_counter(); idx.facets(f, attributes=facetable, top_n=20)
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize(f"facetindex facets for all {len(facetable)} facetable attrs", ms))

    ms = []
    for f in filters[:100]:
        t0 = time.perf_counter(); idx.facets(f, top_n=20)   # includes numeric histograms
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex facets incl. numeric histograms", ms))

    segs = {f"seg_{i:02d}": f for i, f in enumerate(filters[:25])}
    segs["composed"] = F.Or(F.Segment("seg_00"), F.And(F.Segment("seg_01"), F.Not(F.Segment("seg_02"))))
    ms = []
    for _ in range(50):
        t0 = time.perf_counter(); idx.segment_counts(segs); ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex 26 saved-segment counts (one list render)", ms))

    ms = []
    for f in filters[:50]:
        t0 = time.perf_counter(); idx.ids_for(F.Or(f, F.Flag("has_api")), limit=25_000)
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex export: first 25k matching ids", ms))

    ms = []
    for f in filters[:100]:
        a = rng.choice([x for x in idx.manifest.value_attrs if idx.kinds[x] == "short_text"])
        t0 = time.perf_counter()
        try:
            idx.count(F.And(f, F.Contains(a, "an")))
        except UnsupportedInIndex:
            pass
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("facetindex count with text-contains on a short text attr", ms))

    # ---- DuckDB over the source Parquet (the 'serve straight from the columnar file' option) ----
    con = duckdb.connect()
    if args.duckdb_threads:
        con.execute(f"PRAGMA threads={args.duckdb_threads}")
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{args.source}')")
    con.execute("SELECT count(*) FROM src").fetchone()
    ms = []
    for f in filters[:100]:
        sql, params = F.to_sql(f, F.DUCKDB)
        t0 = time.perf_counter(); con.execute(f"SELECT count(*) FROM src WHERE {sql}", params).fetchone()
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("duckdb-on-parquet count (same filters)", ms))

    ms = []
    for f in filters[:50]:
        attr = rng.choice(sortable)
        sql, params = F.to_sql(f, F.DUCKDB)
        t0 = time.perf_counter()
        con.execute(f"SELECT id FROM src WHERE {sql} ORDER BY {key_sql(idx, attr)} NULLS LAST, id LIMIT 50",
                    params).fetchall()
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("duckdb-on-parquet page 50 sorted by random attribute", ms))

    ms = []
    facet_attrs = facetable[:20]
    for f in filters[:20]:
        sql, params = F.to_sql(f, F.DUCKDB)
        t0 = time.perf_counter()
        for a in facet_attrs:
            con.execute(f"SELECT NULLIF(TRIM(\"{a}\"), '') v, count(*) FROM src WHERE {sql} GROUP BY 1 "
                        f"ORDER BY 2 DESC LIMIT 20", params).fetchall()
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize(f"duckdb-on-parquet facets for {len(facet_attrs)} attrs (one GROUP BY each)", ms))

    # DuckDB over a narrow copy (no long text) - what a typed narrow serving table would see
    short_cols = [c for c in idx.manifest.value_attrs]
    con.execute("CREATE TABLE narrow AS SELECT id, " + ", ".join(f'"{c}"' for c in short_cols) + " FROM src")
    ms = []
    for f in filters[:100]:
        sql, params = F.to_sql(f, F.DUCKDB)
        t0 = time.perf_counter(); con.execute(f"SELECT count(*) FROM narrow WHERE {sql}", params).fetchone()
        ms.append((time.perf_counter() - t0) * 1000)
    results.append(summarize("duckdb in-memory narrow table count (same filters)", ms))

    report["results"] = results
    out = Path(__file__).parent / "out" / f"bench-{args.label}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1))
    print(f"\n{'workload':<70} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
    for r in results:
        print(f"{r['workload']:<70} {r['p50_ms']:>8} {r['p95_ms']:>8} {r['max_ms']:>8}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
