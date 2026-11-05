# facetindex

**A versioned facet index that your data pipeline builds from a Parquet file, so faceted
search, counts, saved segments and sort-by-anything over a wide, sparse, constantly
reloaded attribute table stop being full table scans on the serving database.**

```
pipeline (DuckDB, Parquet)  ──build──►  versions/<load_id>/  ──CURRENT──►  API workers (bitmaps in memory)
                                                                                 │
serving table (Postgres, 200 text columns) ◄──── WHERE id = ANY(...) ────────────┘  (display rows only)
```

* **No new infrastructure.** One Python package, a directory of Parquet files per load,
  a pointer file. Each API worker holds a read-only copy; no threads, no locks, no cache
  to invalidate.
* **Dynamic schema.** Attribute kinds (flag, categorical, numeric, short text, long text)
  are inferred per load; attributes may appear, disappear or change kind between loads.
  Admin overrides are honoured.
* **One filter language, two evaluators.** The same filter tree runs against the bitmaps
  in microseconds or renders to parameterised SQL for the serving table. The test suite
  proves they agree on random filters against DuckDB.
* **Atomic versions.** Build, then flip `CURRENT`. Readers never see a half-written index
  and finish a request on the version they started with.

The design notes in [`docs/DESIGN.md`](docs/DESIGN.md) answer the review questions from
the original problem statement, including why a per-entity "signals key" is the forward
index and the per-value bitmap is the one that makes multi-attribute search fast.
[`docs/INTEGRATION.md`](docs/INTEGRATION.md) is the hand-off for the pipeline and API teams.

## Quick start

```bash
pip install -e ".[dev]"

facetindex synth data/orgs.parquet --rows 240000 --columns 200       # a table shaped like the problem
facetindex build data/orgs.parquet ./index --version load-0001      # ~20 s, writes ./index/versions/load-0001
facetindex catalog ./index --brief                                   # the attribute profile, instantly
facetindex query ./index --sort revenue_usd --desc --limit 5 --facets region tier --rows --columns name region \
  --filter '{"op":"and","children":[{"op":"flag","attr":"has_api"},
                                   {"op":"in","attr":"region","values":["EMEA","APAC"]},
                                   {"op":"between","attr":"headcount","lo":50,"hi":5000}]}'
```

```python
from facetindex import build_index, Catalog, filters as F

# pipeline, once per load
build_index("load_0421.parquet", "/srv/facetindex", version="load-0421", id_column="id")

# API, every worker
catalog = Catalog("/srv/facetindex")
idx = catalog.get()                                   # pinned version for this request
f = F.And(F.Flag("has_api"), F.In("region", ["EMEA", "APAC"]), F.Between("headcount", 50, 5000))
idx.count(f)                                          # 316
page = idx.page(f, sort_by="revenue_usd", descending=True, limit=50)     # ids + next_cursor
facets = idx.facets(f)                                # counts per value for every facetable attribute
idx.segment_counts({"emea_api": F.And(F.Flag("has_api"), F.In("region", ["EMEA"])), "big": F.Between("headcount", 1000, None)})
idx.catalog()                                         # the per-attribute profile (stats.json)
F.to_sql(f)                                           # ('(LOWER(TRIM("has_api")) IN (%s, ...) AND ...)', [...])
```

## What it answers, and how

| Workload from the problem statement | Mechanism | Cost at 240k rows |
|---|---|---|
| Attribute catalog and profile | `stats.json`, computed in the pipeline | one file read per version |
| ANDed filter count | roaring bitmap AND + popcount | sub-millisecond |
| Facet counts under the current filter | filter bitmap ∧ each value bitmap | ~10 ms for 110 attributes |
| One sorted page of 50, any attribute | dense key array + partition + exact top-k, keyset cursor | ~1 ms |
| Numeric range | sorted (value, position) array, two binary searches | 1-10 ms |
| Text contains (short text) | vectorised substring over one Arrow column | ~25 ms |
| Saved segments, composed or nested | on-demand bitmap evaluation with memoisation and cycle check | ~1 ms each |
| Export of 25k ids | bitmap to array | ~2 ms |
| Single entity | unchanged: primary key on the serving table | |
| Text contains on long text | **not indexed**; `filters.to_sql()` for the serving table | SQL |

<!-- BENCH:BEGIN -->
## Benchmarks

Synthetic table shaped like the problem (200 all-text columns: 62 flags, 44 categoricals, 49 numerics, 39 short text, 4 long text of 1-8 KB at 30% fill, 2 empty), 4-vCPU container, single process. Filters are ANDs of 2-4 clauses over flags, categoricals and numeric ranges. Reproduce with `bench/run_bench.py` and `bench/pg_bench.py`; raw results in `bench/out/`.

### Build and footprint

| | 240k rows | 1M rows |
|---|---|---|
| build time (pipeline side, DuckDB + Arrow) | 20 s | 62 s |
| artifact size, all files | 134 MB | 481 MB |
| of which postings (bitmaps) | 16 MB | 66 MB |
| of which sort keys (lazy) | 33 MB | 127 MB |
| of which numeric arrays (lazy) | 49 MB | 142 MB |
| of which narrow text values (lazy) | 37 MB | 146 MB |
| open per worker (postings + ids) | 114 ms | 361 ms |
| warm everything (optional) | 6.3 s | 24.6 s |
| value bitmaps | 933 | 970 |

### Query latency (p50 / p95, milliseconds)

| workload | 240k rows | 1M rows |
|---|---|---|
| facetindex count (AND of 2-4 clauses) | 0.3 / 3.8 | 2.0 / 21.5 |
| facetindex page 50 sorted by random attribute (warm keys) | 0.7 / 4.5 | 2.2 / 20.6 |
| facetindex page 50 by id | 0.4 / 4.3 | 2.2 / 18.5 |
| facetindex facets for all facetable attrs | 9.2 / 18.6 | 34.2 / 76.4 |
| facetindex facets incl. numeric histograms | 40.0 / 63.6 | 166.3 / 241.3 |
| facetindex 26 saved-segment counts (one list render) | 40.3 / 47.6 | 171.9 / 206.1 |
| facetindex export: first 25k matching ids | 1.6 / 13.4 | 4.0 / 18.4 |
| facetindex count with text-contains on a short text attr | 21.0 / 34.1 | 82.1 / 106.2 |
| duckdb-on-parquet count (same filters) | 13.7 / 21.1 | 27.6 / 40.5 |
| duckdb-on-parquet page 50 sorted by random attribute | 18.7 / 28.5 | 45.7 / 62.8 |
| duckdb-on-parquet facets for 20 attrs (one GROUP BY each) | 292.2 / 400.8 | 633.0 / 837.5 |
| duckdb in-memory narrow table count (same filters) | 13.8 / 26.1 | 28.0 / 56.2 |

The DuckDB rows query the same Parquet file directly with the SQL rendered by `filters.to_sql(f, DUCKDB)`: that is the 'serve from the columnar file' alternative, and it is a strong one for counts and pages. The facet sidebar is where the bitmaps earn their keep: 110 attributes in ~10 ms versus one GROUP BY per attribute.

### Postgres, same filters, 240k rows (local instance, table in page cache)

Wide table 765 MB, JSONB copy 533 MB, narrow copy without long text 242 MB; shared_buffers 1GB; JSONB copy + GIN build 127.1 s; one partial index per flag on the narrow table 0.09 s to build.

| workload | p50 ms | p95 ms | timeouts (30 s) |
|---|---|---|---|
| postgres wide text table: count (same filters, PK only) | 395 | 579 | 0 |
| postgres wide text table: page 50 sorted by random attribute | 388 | 546 | 0 |
| postgres wide text table: facets for 20 attrs (one GROUP BY each) | 8243 | 12225 | 0 |
| postgres wide text table: facets for 20 attrs (unpivot, one pass) | 423 | 1126 | 0 |
| postgres jsonb + GIN (jsonb_path_ops): count, flag/category-only filters | 4 | 67 | 0 |
| postgres narrow text table (no long text): count (same filters) | 350 | 545 | 0 |

Caveat: local Postgres, NVMe, table in page cache; the production instance on network storage with a table larger than shared_buffers measured 2-4 s per scan. The gap on real hardware is wider than shown here.

Two things worth noticing. Once the table is cached, Postgres is CPU-bound on evaluating TRIM/regex/CASE over text columns for every row, which is why the narrow copy (3x smaller) is barely faster than the wide table: typed columns would help more than narrowing. And JSONB + GIN is genuinely fast for the filters it can serve (flags and categories by containment), but it cannot serve numeric ranges, sorting or facet counts, and its copy took two minutes to build, which does not fit a one-load-per-minute cadence.
<!-- BENCH:END -->

## Use it whole or in pieces

If you are already building something like this, the parts are separable:

* **Just the profile.** `facetindex.profile.profile_parquet()` gives the per-attribute
  statistics (fill, distinct, true count, min/max, top values, sample, inferred kind) from
  DuckDB in a few seconds. Write it wherever your catalog reads from and the live profile
  on the serving database is gone.
* **Just the filter language.** `facetindex.filters` is a small, serialisable AST with
  pinned semantics and `to_sql()` for Postgres or DuckDB. Saved segments become JSON;
  composition is a `segment` node. Nothing in it depends on the index.
* **Just the postings.** `postings.parquet` is `(attribute, kind, value, cardinality,
  roaring bitmap)`. Any language with a roaring library reads it; DuckDB reads it as a table.
  If you want the bitmaps in a different store, this is the file to load.
* **Just the versioning.** `facetindex.artifact` is the build-dir / rename / pointer-file
  pattern with `retain()`. It works for any per-load artefact, not only this one.
* **The forward signature.** `--signatures` emits the per-entity bitset with its bit
  dictionary, if a narrow `bit(n)` column in Postgres is the shape you want.

## The artifact

```
<root>/CURRENT                        {"version": "load-0421", "published_at": ...}
<root>/versions/load-0421/
  manifest.json                       rows, columns, kinds, file inventory, format version
  stats.json                          AttributeProfile per attribute: fill, distinct, true count, min/max, top values, sample, kind
  ids.parquet                         entity id per position (positions are sorted by id)
  postings.parquet                    (attribute, kind, value, cardinality, roaring bitmap): present / true / false / value
  numeric_sorted.parquet              one row group per numeric-capable attribute: (value, position) sorted
  keys/<attr>.parquet                 dense float64 sort key per position (NaN = missing)
  values/values-NNN.parquet           normalised text of the short attributes, ~24 per file (a narrow typed table in waiting)
  signatures.parquet, dimensions.json optional forward index: one bitset per entity (--signatures)
```

Everything is plain Parquet and JSON. DuckDB can read the postings table; the `values/`
files are literally the narrow serving table if you ever want one in Postgres.

## Filters

```json
{"op": "and", "children": [
  {"op": "flag", "attr": "has_api", "value": true},
  {"op": "in", "attr": "region", "values": ["EMEA", "APAC"]},
  {"op": "between", "attr": "headcount", "lo": 50, "hi": 5000},
  {"op": "contains", "attr": "name", "needle": "labs"},
  {"op": "is_not_empty", "attr": "funding_stage"},
  {"op": "not", "child": {"op": "segment", "name": "churn_risk"}}
]}
```

`all`, `and`, `or`, `not`, `flag`, `in`/`eq`, `between`, `contains`, `is_empty`,
`is_not_empty`, `segment`. Exact semantics (what counts as empty, which spellings are
true, trimming, null ordering) are pinned in `docs/DESIGN.md` section 5, and the
equivalence tests hold both evaluators to them.

## Kind inference

| Rule (first match wins) | Kind | Gets |
|---|---|---|
| admin override | as given | |
| no non-empty values | `empty` | present bitmap |
| a value longer than 256 chars or average over 96 | `long_text` | present bitmap, approx distinct, sample |
| all filled values are true/false spellings, ≤ 2 distinct | `flag` | true/false bitmaps, sort key |
| ≥ 98% of filled values parse as numbers | `numeric` | sorted array, sort key, value bitmaps if ≤ 50 distinct |
| ≤ 50 distinct values | `categorical` | value bitmaps, rank sort key |
| otherwise | `short_text` | rank sort key, stored values for contains |

Thresholds are in `ProfileConfig`. Long text is never read beyond its length.

## Where the "signals key" idea fits

A per-entity key that encodes its attribute values is a *forward* index. Hashed, it
answers only whole-combination equality. Encoded in a fixed order, it is a composite
B-tree that serves only leading-prefix queries. As a bitset, it gives a narrow cacheable
scan. None of those answer "how many entities match these three attributes, and how many
would match each other value" without scanning. The transpose, one bitmap per
(attribute, value), answers all of it with a handful of AND operations. That is what
`postings.parquet` holds. `--signatures` also emits the forward bitset with its bit
dictionary, because it is a legitimate cheap trick inside Postgres (`bit(n)` and `&`) and
because the idea deserved a precise, testable home. `docs/DESIGN.md` section 2 has the
full argument.

## Alternatives that also solve this

* **OpenSearch / Elasticsearch.** The industrial answer to faceted search: dynamic
  mapping, filters, aggregations under the current filter, doc values for sorting, index
  aliases for atomic swaps. Costs one new service.
* **DuckDB over the load's Parquet file, in the API.** The bench below shows it answers
  every workload here in tens of milliseconds with no index at all. Facet sidebars are the
  weak spot (one GROUP BY per attribute). If you can ship a Parquet file to the API hosts,
  this is the smallest possible change and facetindex's `to_sql(DUCKDB)` already speaks it.
* **Postgres JSONB + GIN.** Good for ANDed equality within SQL; no ranges, no sort, no
  facet counts without a scan.

facetindex is the option with no new service and no new query language that still makes
the facet sidebar and saved-segment counts constant-time.

## Repository layout

```
facetindex/        the package: filters, profile, build, artifact, index, segments, catalog, synth, cli
tests/             equivalence tests against DuckDB SQL on random filters, engine, versioning, CLI
bench/             run_bench.py (facetindex vs DuckDB-on-Parquet), pg_bench.py (Postgres wide / JSONB+GIN / narrow)
examples/          pipeline_hook.py, api.py (FastAPI, multi-worker)
docs/              DESIGN.md (the review questions answered), INTEGRATION.md (hand-off)
```

## Status

Reference implementation, MIT licensed. The format is versioned (`format_version` in the
manifest) and the semantics are tested, but the API surface is young. Things it does not
do: full-text relevance, indexing long text, sharing memory between worker processes.
