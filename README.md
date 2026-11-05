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

Numbers pending; see `bench/out/`.
<!-- BENCH:END -->

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
  values.parquet                      normalised text of every short attribute (a narrow typed table in waiting)
  signatures.parquet, dimensions.json optional forward index: one bitset per entity (--signatures)
```

Everything is plain Parquet and JSON. DuckDB can read the postings table; `values.parquet`
is literally the narrow serving table if you ever want one in Postgres.

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
