"""Postgres comparison: the serving table as it is today (wide, all text, PK only) vs
JSONB+GIN vs the facetindex numbers from run_bench.py.

Starts a throwaway Postgres cluster if --pgdata is given (needs initdb/pg_ctl on PATH and a
non-root user, e.g. `sudo -u postgres`), or connects to --dsn. Loads the Parquet via COPY.

    python bench/pg_bench.py --source data/org240k.parquet --dsn postgresql:///bench --label 240k

Caveat printed with the results: a local Postgres on NVMe with the table in page cache is a
*flattering* stand-in for a managed instance on network storage whose table does not fit
in its buffer cache; the measured 2-4 s scans from the problem statement were on the latter.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import duckdb
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_bench import summarize, typical_filters  # noqa: E402

from facetindex import filters as F  # noqa: E402
from facetindex.index import FacetIndex  # noqa: E402


def load_wide(con, source, cols, table="entities"):
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} (" + ", ".join(f'"{c}" text' for c in cols) + ")")
    d = duckdb.connect()
    csv = Path(source).with_suffix(".csv")
    d.execute(f"COPY (SELECT * FROM read_parquet('{source}')) TO '{csv}' (FORMAT CSV, HEADER, NULL '')")
    t0 = time.perf_counter()
    with con.cursor() as cur, open(csv, "rb") as fh, cur.copy(
        f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
    ) as cp:
        while chunk := fh.read(1 << 20):
            cp.write(chunk)
    con.execute(f"ALTER TABLE {table} ADD PRIMARY KEY (id)")
    con.execute(f"ANALYZE {table}")
    con.commit()
    csv.unlink()
    return time.perf_counter() - t0


def load_jsonb(con, idx, table="entities_doc"):
    """(id, doc jsonb) with normalised values: flags as booleans, numerics as numbers."""
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} (id text PRIMARY KEY, doc jsonb)")
    kinds = idx.kinds
    t0 = time.perf_counter()
    def pair(a):
        if kinds[a] == "flag":
            v = (f"(CASE WHEN LOWER(TRIM(\"{a}\")) IN ('true','t','1','yes','y') THEN TRUE "
                 f"WHEN LOWER(TRIM(\"{a}\")) IN ('false','f','0','no','n') THEN FALSE END)")
        elif kinds[a] == "numeric":
            v = f"(CASE WHEN \"{a}\" ~ '{F.NUMERIC_RE}' THEN \"{a}\"::numeric END)"
        else:
            v = f"NULLIF(TRIM(\"{a}\"), '')"
        return f"'{a}', {v}"
    attrs = list(idx.manifest.value_attrs)
    # jsonb_build_object takes at most 100 arguments: build in chunks and concatenate
    chunks = [attrs[i:i + 40] for i in range(0, len(attrs), 40)]
    objs = " || ".join("jsonb_build_object(" + ", ".join(pair(a) for a in ch) + ")" for ch in chunks)
    con.execute(f"INSERT INTO {table} SELECT id, jsonb_strip_nulls({objs}) FROM entities")
    con.execute(f"CREATE INDEX ON {table} USING gin (doc jsonb_path_ops)")
    con.execute(f"ANALYZE {table}")
    con.commit()
    return time.perf_counter() - t0


def jsonb_containment(f):
    """Render an AND of flag/in clauses as @> containment (what GIN jsonb_path_ops accelerates).
    Returns None when the filter has a clause GIN cannot serve (numeric range)."""
    doc = {}
    extra = []
    for c in f.children:
        if isinstance(c, F.Flag):
            doc[c.attr] = bool(c.value)
        elif isinstance(c, F.In) and len(c.values) == 1:
            doc[c.attr] = c.values[0]
        elif isinstance(c, F.In):
            extra.append("(" + " OR ".join("doc @> %s" for _ in c.values) + ")")
        elif isinstance(c, F.Between):
            return None, None
    where = ["doc @> %s"]
    params = [json.dumps(doc)]
    for c in f.children:
        if isinstance(c, F.In) and len(c.values) > 1:
            where.append("(" + " OR ".join("doc @> %s" for _ in c.values) + ")")
            params.extend(json.dumps({c.attr: v}) for v in c.values)
    return " AND ".join(where), params


def timed(con, sql, params=None, n=1):
    ms = []
    for _ in range(n):
        t0 = time.perf_counter()
        with con.cursor() as cur:
            cur.execute(sql, params)
            cur.fetchall()
        ms.append((time.perf_counter() - t0) * 1000)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--root", required=True, help="facetindex root (for kinds/stats)")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-load", action="store_true")
    ap.add_argument("--statement-timeout-ms", type=int, default=30_000)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    idx = FacetIndex.open(args.root)
    cols = [idx.manifest.id_column] + idx.manifest.columns
    con = psycopg.connect(args.dsn)
    report = {"label": args.label, "results": []}
    def exists(t):
        return con.execute("SELECT to_regclass(%s) IS NOT NULL", [t]).fetchone()[0]
    if not args.skip_load:
        if not exists("entities"):
            report["load_wide_seconds"] = round(load_wide(con, args.source, cols), 1)
            print(f"loaded wide table in {report['load_wide_seconds']}s", flush=True)
        if not exists("entities_doc"):
            report["load_jsonb_seconds"] = round(load_jsonb(con, idx), 1)
            print(f"loaded jsonb table + GIN in {report['load_jsonb_seconds']}s", flush=True)
    con.execute(f"SET statement_timeout = {args.statement_timeout_ms}")   # the app's per-statement limit
    sizes = con.execute("""
        SELECT relname, pg_size_pretty(pg_total_relation_size(oid)), pg_total_relation_size(oid)
        FROM pg_class WHERE relname IN ('entities','entities_doc') ORDER BY 1""").fetchall()
    report["table_sizes"] = {r[0]: r[1] for r in sizes}
    con.execute("SELECT pg_size_pretty(pg_relation_size('entities'))")
    mem = con.execute("SHOW shared_buffers").fetchone()[0]
    report["shared_buffers"] = mem

    filters = typical_filters(idx.stats, rng, args.queries)
    sortable = list(idx.manifest.key_attrs)

    def run(name, fn, n):
        ms = []
        timeouts = 0
        for f in filters[:n]:
            try:
                ms.extend(fn(f))
            except psycopg.errors.QueryCanceled:
                con.rollback()
                timeouts += 1
                ms.append(float(args.statement_timeout_ms))
        r = summarize(name, ms)
        r["timeouts"] = timeouts
        report["results"].append(r)
        print(f"{name:<70} p50={r['p50_ms']:>9} p95={r['p95_ms']:>9} timeouts={timeouts}", flush=True)

    # warm the cache once so numbers are 'table in page cache' (flattering) rather than cold
    con.execute("SELECT count(*) FROM entities WHERE name IS NOT NULL")

    def pg_count(f):
        sql, params = F.to_sql(f, F.POSTGRES)
        return timed(con, f"SELECT count(*) FROM entities WHERE {sql}", params)

    def pg_page(f):
        sql, params = F.to_sql(f, F.POSTGRES)
        attr = rng.choice(sortable)
        kind = idx.kinds[attr]
        col = f'"{attr}"'
        if kind == "numeric":
            key = f"(CASE WHEN {col} ~ '{F.NUMERIC_RE}' THEN {col}::double precision END)"
        elif kind == "flag":
            key = f"(CASE WHEN LOWER(TRIM({col})) IN ('true','t','1','yes','y') THEN 1 WHEN LOWER(TRIM({col})) IN ('false','f','0','no','n') THEN 0 END)"
        else:
            key = f"LOWER(NULLIF(TRIM({col}), ''))"
        return timed(con, f"SELECT id FROM entities WHERE {sql} ORDER BY {key} NULLS LAST, id LIMIT 50", params)

    facet_attrs = [a for a in idx.manifest.columns if idx.stats[a]["facetable"]][:20]

    def pg_facets(f):
        sql, params = F.to_sql(f, F.POSTGRES)
        t0 = time.perf_counter()
        for a in facet_attrs:
            with con.cursor() as cur:
                cur.execute(f"SELECT NULLIF(TRIM(\"{a}\"), ''), count(*) FROM entities WHERE {sql} "
                            f"GROUP BY 1 ORDER BY 2 DESC LIMIT 20", params)
                cur.fetchall()
        return [(time.perf_counter() - t0) * 1000]

    def pg_facets_onepass(f):
        """The stopgap's trick: unpivot with a lateral VALUES list and group once."""
        sql, params = F.to_sql(f, F.POSTGRES)
        vals = ", ".join(f"('{a}', NULLIF(TRIM(\"{a}\"), ''))" for a in facet_attrs)
        q = (f"SELECT attr, val, n FROM (SELECT attr, val, count(*) n, row_number() OVER (PARTITION BY attr ORDER BY count(*) DESC) rn "
             f"FROM entities e, LATERAL (VALUES {vals}) AS v(attr, val) WHERE {sql} GROUP BY attr, val) s WHERE rn <= 20")
        return timed(con, q, params)

    def pg_count_jsonb(f):
        where, params = jsonb_containment(f)
        if where is None:
            return []
        return timed(con, f"SELECT count(*) FROM entities_doc WHERE {where}", params)

    def pg_count_narrow(f):
        sql, params = F.to_sql(f, F.POSTGRES)
        return timed(con, f"SELECT count(*) FROM entities_narrow WHERE {sql}", params)

    run("postgres wide text table: count (same filters, PK only)", pg_count, args.queries)
    run("postgres wide text table: page 50 sorted by random attribute", pg_page, min(args.queries, 20))
    run(f"postgres wide text table: facets for {len(facet_attrs)} attrs (one GROUP BY each)", pg_facets, 10)
    run(f"postgres wide text table: facets for {len(facet_attrs)} attrs (unpivot, one pass)", pg_facets_onepass, 10)
    run("postgres jsonb + GIN (jsonb_path_ops): count, flag/category-only filters", pg_count_jsonb, args.queries)

    # a narrow copy without the long text columns: the cheapest schema-level change
    con.execute("DROP TABLE IF EXISTS entities_narrow")
    con.execute("CREATE TABLE entities_narrow AS SELECT id, " + ", ".join(f'"{c}"' for c in idx.manifest.value_attrs) + " FROM entities")
    con.execute("ALTER TABLE entities_narrow ADD PRIMARY KEY (id)")
    con.execute("ANALYZE entities_narrow")
    con.commit()
    narrow_size = con.execute("SELECT pg_size_pretty(pg_total_relation_size('entities_narrow'))").fetchone()[0]
    report["table_sizes"]["entities_narrow"] = narrow_size
    run("postgres narrow text table (no long text): count (same filters)", pg_count_narrow, args.queries)

    # partial indexes on flags: how long does the pipeline pay per load?
    flags = [a for a in idx.manifest.columns if idx.kinds[a] == "flag"][:10]
    t0 = time.perf_counter()
    for a in flags:
        con.execute(f"CREATE INDEX ON entities_narrow ((LOWER(TRIM(\"{a}\")))) WHERE LOWER(TRIM(\"{a}\")) IN ('true','t','1','yes','y')")
    con.commit()
    report["partial_index_build_seconds_per_flag"] = round((time.perf_counter() - t0) / len(flags), 2)

    report["caveat"] = ("local Postgres, NVMe, table in page cache; the production instance on network storage "
                        "with a table larger than shared_buffers measured 2-4 s per scan")
    out = Path(__file__).parent / "out" / f"pgbench-{args.label}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
