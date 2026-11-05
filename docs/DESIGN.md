# Design notes

This document answers the review questions from the original problem statement and
explains why facetindex is shaped the way it is. The short version: **the per-entity
"signals key" is a forward index; what makes multi-attribute filtering and counting fast
is the inverted index (one bitmap per attribute value), and the pipeline already has
everything it needs to emit one per load.**

## 1. The problem in one sentence

Faceted search over one wide (about 200 columns), sparse, all-text table of 240k rows
(growing to millions), where the column set changes without notice, the table is rebuilt
wholesale on every load (as often as once a minute), and the serving database is a
row-store on network storage whose sequential scans cost 2 to 4 s and whose random reads
cost 50 to 100 times more per page.

Everything interactive (catalog, counts, facet counts, a sorted page, saved-segment
counts) is a full scan today. That is the whole problem.

## 2. About the "signals key"

The idea on the table was: encode each entity's attribute values into one big key, index
it, and search the key. There are three distinct ideas tangled in that sentence.

**A hashed key.** A hash is one-way and whole-value. You can test "does this entity have
exactly this combination", nothing else. Ranges, substring matches, sorting and facet
counts are all impossible. Encryption adds a key-management problem and no speed; the
searchable-encryption literature exists precisely because it is hard, and even the best
schemes support equality only, slower than plaintext.

**An encoded composite key.** If the key is a fixed-order concatenation of values, a
B-tree on it is a composite index: fast for queries that constrain the *leading*
attributes in that order, useless for any other combination. Faceted search is the one
workload where the user picks arbitrary combinations. This is exactly why inverted and
bitmap indexes exist.

**A per-entity bitset ("signature").** Each entity gets one bit per (attribute, value)
dimension. An ANDed filter is `(sig & mask) = mask`, evaluated row by row. This works,
and it is fast only because the signature table is *narrow* (a few hundred bytes per row
instead of 2 KB) and therefore cacheable, so the scan is cheap. It still scans every row
for every query, and it still cannot do ranges, contains, sorting or counts-per-value
without a scan per value.

The kernel of truth is the narrowness. The structure that actually wins is the transpose:

**One bitmap per (attribute, value) listing the entities that have it.** A filter with k
clauses is k bitmap lookups and k-1 ANDs. A count is a popcount. The facet sidebar is the
filter bitmap ANDed with every value bitmap, which at 240k entities is a few thousand
microsecond-scale operations. This is what Lucene, Pinot, Druid and Postgres's own bitmap
heap scans do internally. Roaring bitmaps are the standard compressed representation.

facetindex emits both. The inverted postings are what the engine queries. The forward
signature (`--signatures`) is an optional export with a documented bit dictionary, because
it is a legitimate cheap trick inside Postgres for a narrow ANDed-flags scan when nothing
else is available, and because it gives the original idea a precise, testable home.

## 3. Answers to the review questions

### Q1. Precompute in the pipeline, serve from small tables plus indexes?

Yes. The pipeline already holds the full load in a columnar engine where every statistic
costs milliseconds, and it already rebuilds the table every load, so "rebuild the index
every load" is free in architectural terms. The alternative, making the serving database
compute on demand, has been measured and it is the thing that broke.

The split facetindex implements:

| Artefact | Built by | Read by | Replaces |
|---|---|---|---|
| `stats.json` (per-attribute profile) | pipeline, DuckDB | API, once per version | the live profile (15-20 s) |
| `postings.parquet` (roaring bitmaps) | pipeline | API, loaded per worker | every COUNT / facet scan |
| `numeric_sorted.parquet` + `keys/` | pipeline | API, lazily per attribute | numeric ranges and sort-by-anything scans |
| `values/*.parquet` (narrow normalised table) | pipeline | API, lazily per attribute | text-contains scans; optionally a narrow serving table |
| the wide serving table | pipeline (unchanged) | API, by primary key only | nothing, it stays |

If you would rather not run the engine in the API process, two off-the-shelf options do
the same job and should be on the table: **OpenSearch/Elasticsearch** (dynamic mapping,
filters, ranges, aggregations under the current filter, doc values for sorting, index
aliases for atomic swaps) and **DuckDB over the load's Parquet file** (the bench shows it
answers every workload here in milliseconds with no index at all). facetindex sits between
them: no new infrastructure, no new query language, one Parquet directory per load.

### Q2. Best representation for filtering and counting over 200 sparse attributes?

| Structure | ANDed equality and count | Numeric range | Text contains | Sort by any attribute | Dynamic schema | Size at 240k x 200 |
|---|---|---|---|---|---|---|
| JSONB + GIN (`jsonb_path_ops`) | good (containment) | poor (needs per-key expression index) | poor (trigram per key) | no | good | ~2x the data |
| EAV long table + (attr, value, id) index | ok, large | ok with a typed value column | poor | join | good | 20-50M rows |
| **Per-(attribute, value) roaring bitmaps** | excellent | via sorted (value, id) arrays | no | via dense key arrays | good | tens of MB |
| Wide table + curated indexes | good for the curated set | good for the curated set | trigram | curated only | poor | index per column per load |
| Per-entity signature / hash | narrow scan only | no | no | no | ok | small |

facetindex uses bitmaps for equality and counts, one sorted `(value, position)` array per
numeric-capable attribute for ranges (two binary searches and a slice), one dense float
key per sortable attribute for ordering, and the narrow normalised text table for
`contains`. Long text is deliberately not indexed; `contains` on it renders to SQL.

Tradeoffs worth knowing:

* **Counts and ANDed filters** are the bitmap sweet spot: microseconds, independent of
  how many columns the table has.
* **Numeric ranges** on a sorted array are O(log n + k). The result bitmap is built from
  the k matching positions; for very wide ranges (k close to n) this costs a few ms.
* **Text contains** is a vectorised substring scan over one Arrow column: a few ms at 240k
  for short text. It is the one op that does not get faster with the index, and it is
  fine, because the column is narrow and in memory.
* **Sorting** needs a key array per attribute (8 bytes per row per attribute). The page
  selects the top k with a partition and an exact lexsort on the candidates, with the
  position (= id order) as tiebreaker, so the order is total and the cursor is stable.

### Q3. Saved segments, including composed ones

Evaluate on demand against the bitmaps. A segment is a filter; a composed segment is a
filter with `Segment(name)` leaves. Evaluation memoises each referenced segment within a
query and refuses cycles. Twenty-six segment counts in one list render cost about
40 ms at 240k rows in the bench, most of it wide numeric ranges.

Do **not** materialise segment membership per load: segments are edited between loads,
so a materialised view is stale the moment an admin saves, and you would need an
invalidation path. The bitmap evaluation is already cheaper than reading a materialised
membership table would be.

If segments are edited between loads, nothing changes: the next request evaluates the
new definition against the current version. `validate()` reports segments that reference
attributes the latest load no longer has.

### Q4. Facet counts under the current filter

The standard approach at this scale is exactly bitmap intersection: for each facet
value, `popcount(filter_bitmap AND value_bitmap)`. Rollup tables only answer unfiltered
counts or a fixed set of filter combinations. facetindex also implements the
multi-select convention (counts for facet X ignore the user's own selection on X), which
is what makes the other values of X clickable, by re-evaluating the filter without X's
top-level clauses. Numeric attributes get min, max, median and a fixed-bin histogram so a
range slider can render.

### Q5. Sorting by an arbitrary attribute

Feasible without an index per column, provided the sort key is a narrow, in-memory array.
facetindex stores one dense float key per sortable attribute (numeric value, or dense rank
of the lower-cased text, or 1/0 for flags), loaded lazily per attribute. Top-50 of 240k
is a partition plus a small sort: a few milliseconds. Missing values sort last in both
directions, ties break on id, and pagination is keyset `(key, position)`, so page 400
costs the same as page 1. At low millions of rows this is still tens of milliseconds.

You can still curate which attributes the UI offers for sorting, but the engine does not
need you to.

### Q6. Database-level tactics before re-architecting

Worth doing regardless, in this order:

1. **Stop profiling on the serving database.** Ship `stats.json` from the pipeline.
   This alone removes the 15-20 s first-load cost and the thundering-herd risk.
2. **Check what "empty" is stored as.** A NULL costs no bytes in a Postgres row; an
   empty string costs at least one byte plus alignment, in every one of 200 columns.
   The main heap is about 2 KB per row, which is 200 text columns, not TOAST pointers
   (those are 18 bytes). If empties are `''`, converting them to NULL alone may cut the
   table size substantially.
3. **Typed columns in a narrow serving table.** Booleans as `boolean` (1 byte) and
   numbers as `numeric`/`double` shrink the heap further and make numeric sort correct.
   The `values/` files are exactly this table; `COPY` them if you want it in Postgres.
4. **Move long text to a side table** keyed by id. It helps less than people expect for
   scans (it is already out of line), but it stops anyone accidentally touching it in a
   wide `SELECT *` or a `count(DISTINCT)`.
5. **Instance memory.** A half-gigabyte table that does not fit the buffer cache implies a
   very small instance. More RAM may be the cheapest fix of all and buys time.
6. **Partial indexes on sparse flags** work but each one is a full scan to build per load.
   The bench measures the per-flag cost; at one load per minute, ten flags may be all you
   can afford. A single GIN on a `text[]` of true-flag names covers all flags with one
   build if you stay in SQL.
7. Cluster ordering does nothing for full scans. IOPS help random reads, which the
   right design avoids entirely.

### Q7. Freshness and atomic swaps

facetindex versions are immutable directories under `versions/<load_id>/` plus a tiny
`CURRENT` pointer file replaced by rename. Build, then flip. Readers pin the version they
started with for the duration of a request; the previous version stays loaded until two
newer ones exist. Nothing can observe a half-written version.

Apply the same pattern to the serving table: write `entities_<load_id>`, build its indexes,
then flip a one-row `current_load` table in a transaction and have the API read that row
once per request. This avoids the rename trap below. If you keep `ALTER TABLE ... RENAME`,
set a short `lock_timeout` on the pipeline's swap transaction.

To keep table and index in lockstep: `build_index(..., publish=False)`, swap the table,
`set_current(root, load_id)`. If the table swap fails, the index version is simply never
pointed at and is swept by `retain()`.

### Q8. Risks in the current stopgap

* **Rename swap behind a long reader.** `ALTER TABLE RENAME` needs an `AccessExclusiveLock`.
  It queues behind any running 20 s profile query, and every new `SELECT` then queues
  behind the `ALTER`. The whole API stalls for the duration of the slowest reader. Use a
  `lock_timeout` on the swap, or the pointer-table pattern above.
* **"One profile per version per process" times N workers** is N concurrent profiles,
  each about five sequential scans. Coordinate with `pg_try_advisory_lock(hash(version))`
  and store the result in a table so other workers read instead of recompute. Or, better,
  stop profiling in the API (Q6.1).
* **The five-minute cooldown after a failure** means five minutes of "catalog without
  statistics" during an active bulk push, when the catalog changes most. Acceptable only
  if the UI labels it.
* **Serving the last good result while the column set is unchanged** is sound. Serving
  stale counts after a load with the same columns but different rows is a product
  decision; label it with the version it came from.
* **The unpivot top-values query** materialises rows x columns tuples into a hash
  aggregate; watch `work_mem` spills. It is fine as a bridge.

## 4. What facetindex deliberately does not do

* It does not store or search long text. `contains` on long text renders to SQL
  (`filters.to_sql`) for the serving table. A trigram index there, or a search engine, is
  the right tool if that becomes a primary workload.
* It does not do relevance ranking or tokenised full-text search. It is a facet index.
* It does not replace the serving table. Display rows are fetched by primary key.
* It does not share state between processes. Each worker loads the artifact (tens of MB)
  on first use after a version change, lazily, in the request. There are no threads.

## 5. Semantics pinned down

So the engine and the SQL fallback always agree (the test suite checks this on random
filters against DuckDB):

* *Empty* means NULL or blank after trimming.
* `Flag(attr, True)` matches `true/t/1/yes/y`; `Flag(attr, False)` matches
  `false/f/0/no/n`; case-insensitive; empty matches neither. `Not(Flag(a, True))`
  includes empties.
* `In` compares the trimmed stored text with the trimmed query values, case-sensitively.
* `Between` is inclusive; values that do not parse as finite numbers never match.
  It works on any short attribute, so a text column with some numeric values behaves like
  SQL would.
* `Contains` is a case-insensitive substring match; `%`, `_` and `\` are escaped in SQL.
* Sorting: missing values last in both directions, ties by id ascending; text sorts by
  its lower-cased value in binary order.

## 6. Kind inference

Per attribute, from the two profiling passes:

| Rule (first match wins) | Kind |
|---|---|
| admin override | as given |
| no non-empty values | `empty` |
| any value longer than 256 chars, or average over 96 | `long_text` |
| every filled value is a truthy/falsy spelling and at most two distinct | `flag` |
| at least 98% of filled values parse as finite numbers | `numeric` |
| at most 50 distinct values | `categorical` |
| otherwise | `short_text` |

Numerics with at most 50 distinct values also get value postings, so a 1-5 score shows
up as facet values *and* supports ranges. Thresholds live in `ProfileConfig`.

## 7. Scaling notes

* Postings: 240k entities, about 1,000 bitmaps, 16 MB on disk (66 MB at 1M rows). Roaring keeps
  sparse flags (1% true) as tiny arrays and dense ones as bitsets or runs.
* Keys: 8 bytes x rows per sortable attribute, lazily loaded. At 1M rows and 150
  sortable attributes that is 1.2 GB if you warm everything, so do not; a worker loads
  the handful of sort keys users actually click.
* Build: dominated by the DuckDB column scans, roughly linear in rows x short columns.
  See the README for measured numbers at 240k and 1M rows.
* Beyond the low millions, or if the fleet of API workers grows large, moving the same
  artifact behind one shared process (or to OpenSearch) is the next step; the filter AST
  and the versioning do not change.
