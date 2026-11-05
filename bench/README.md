# Benchmarks

```bash
facetindex synth data/org240k.parquet --rows 240000 --columns 200
python bench/run_bench.py --source data/org240k.parquet --root bench/out/idx240k --label 240k
```

`run_bench.py` builds the index (reporting build time and artifact size), then times the
workloads from the problem statement against facetindex and against DuckDB querying the
same Parquet file directly. Filters are ANDs of 2 to 4 clauses over flags, categoricals
and numeric ranges, like a sidebar selection; the median selectivity is printed.

```bash
# optional: Postgres comparison (wide text table with PK only, JSONB + GIN, narrow copy)
python bench/pg_bench.py --source data/org240k.parquet --root bench/out/idx240k \
    --dsn postgresql:///bench --label 240k
```

`pg_bench.py` loads the Parquet into Postgres as the serving table is today (every column
text, primary key only), runs the same filters as counts, sorted pages and facet GROUP BYs
under a 30 s statement timeout, then builds a JSONB + GIN copy and a narrow copy and
measures those too. Read the caveat it prints: a local Postgres with the table in page
cache flatters the database relative to a managed instance on network storage.

Results land in `bench/out/*.json`. The numbers in the top-level README came from a
4-vCPU container; expect variance.
