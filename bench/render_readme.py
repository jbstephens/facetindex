"""Render bench/out/*.json into the README between the BENCH markers."""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).parent / "out"
README = Path(__file__).parents[1] / "README.md"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def mb(b):
    return f"{b / 1e6:.0f} MB"


def main():
    b240, b1m, pg = load("bench-240k.json"), load("bench-1m.json"), load("pgbench-240k.json")
    lines = ["## Benchmarks", ""]
    lines.append("Synthetic table shaped like the problem (200 all-text columns: 62 flags, 44 categoricals, "
                 "49 numerics, 39 short text, 4 long text of 1-8 KB at 30% fill, 2 empty), 4-vCPU container, "
                 "single process. Filters are ANDs of 2-4 clauses over flags, categoricals and numeric ranges. "
                 "Reproduce with `bench/run_bench.py` and `bench/pg_bench.py`; raw results in `bench/out/`.")
    lines.append("")
    lines.append("### Build and footprint")
    lines.append("")
    lines.append("| | 240k rows | 1M rows |")
    lines.append("|---|---|---|")
    rows = [
        ("build time (pipeline side, DuckDB + Arrow)", lambda b: f"{b['build_seconds']:.0f} s"),
        ("artifact size, all files", lambda b: mb(b["artifact_bytes"])),
        ("of which postings (bitmaps)", lambda b: mb(b["files"]["postings.parquet"])),
        ("of which sort keys (lazy)", lambda b: mb(b["files"]["keys/* (total)"])),
        ("of which numeric arrays (lazy)", lambda b: mb(b["files"]["numeric_sorted.parquet"])),
        ("of which narrow text values (lazy)", lambda b: mb(b["files"].get("values/* (total)", b["files"].get("values.parquet", 0)))),
        ("open per worker (postings + ids)", lambda b: f"{b['open_ms_cold']:.0f} ms"),
        ("warm everything (optional)", lambda b: f"{b['warm_ms'] / 1000:.1f} s"),
        ("value bitmaps", lambda b: f"{b['postings_count']}"),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | {fn(b240) if b240 else '-'} | {fn(b1m) if b1m else '-'} |")
    lines.append("")
    lines.append("### Query latency (p50 / p95, milliseconds)")
    lines.append("")
    lines.append("| workload | 240k rows | 1M rows |")
    lines.append("|---|---|---|")
    def norm(w):  # "facets for all 110 facetable attrs" and "... 111 ..." are the same workload
        return re.sub(r"\ball \d+ facetable", "all facetable", w)

    names = [norm(r["workload"]) for r in b240["results"]] if b240 else []
    r240 = {norm(r["workload"]): r for r in (b240 or {"results": []})["results"]}
    r1m = {norm(r["workload"]): r for r in (b1m or {"results": []})["results"]}

    def cell(d, n):
        r = d.get(n)
        return f"{r['p50_ms']:.1f} / {r['p95_ms']:.1f}" if r else "-"

    for n in names:
        lines.append(f"| {n} | {cell(r240, n)} | {cell(r1m, n)} |")
    lines.append("")
    lines.append("The DuckDB rows query the same Parquet file directly with the SQL rendered by "
                 "`filters.to_sql(f, DUCKDB)`: that is the 'serve from the columnar file' alternative, "
                 "and it is a strong one for counts and pages. The facet sidebar is where the bitmaps earn "
                 "their keep: 110 attributes in ~10 ms versus one GROUP BY per attribute.")
    if pg:
        lines.append("")
        lines.append("### Postgres, same filters, 240k rows (local instance, table in page cache)")
        lines.append("")
        loads = []
        if pg.get("load_wide_seconds"):
            loads.append(f"wide table load {pg['load_wide_seconds']} s")
        if pg.get("load_jsonb_seconds"):
            loads.append(f"JSONB copy + GIN build {pg['load_jsonb_seconds']} s")
        lines.append(f"Wide table {pg['table_sizes'].get('entities', '?')}, JSONB copy {pg['table_sizes'].get('entities_doc', '?')}, "
                     f"narrow copy without long text {pg['table_sizes'].get('entities_narrow', '?')}; shared_buffers {pg['shared_buffers']}; "
                     + "; ".join(loads) + f"; one partial index per flag on the narrow table {pg.get('partial_index_build_seconds_per_flag', '?')} s to build.")
        lines.append("")
        lines.append("| workload | p50 ms | p95 ms | timeouts (30 s) |")
        lines.append("|---|---|---|---|")
        for r in pg["results"]:
            lines.append(f"| {r['workload']} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r.get('timeouts', 0)} |")
        lines.append("")
        lines.append(f"Caveat: {pg['caveat']}. The gap on real hardware is wider than shown here.")
        lines.append("")
        lines.append("Two things worth noticing. Once the table is cached, Postgres is CPU-bound on evaluating "
                     "TRIM/regex/CASE over text columns for every row, which is why the narrow copy (3x smaller) is "
                     "barely faster than the wide table: typed columns would help more than narrowing. And JSONB + GIN "
                     "is genuinely fast for the filters it can serve (flags and categories by containment), but it "
                     "cannot serve numeric ranges, sorting or facet counts, and its copy took two minutes to build, "
                     "which does not fit a one-load-per-minute cadence.")
    text = README.read_text()
    new = re.sub(r"<!-- BENCH:BEGIN -->.*<!-- BENCH:END -->",
                 "<!-- BENCH:BEGIN -->\n" + "\n".join(lines) + "\n<!-- BENCH:END -->", text, flags=re.S)
    README.write_text(new)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
