# Integrating facetindex

Two touch points: the pipeline builds, the API reads. Nothing else changes. The wide
serving table stays exactly as it is and is only ever read by primary key.

```
 provider ──► pipeline (dedupe, DuckDB) ──┬──► serving table  entities_<load_id>   (Postgres)
                                          │
                                          └──► facetindex.build_index(load.parquet, root, version=load_id)
                                                     │
                                                     ▼
                                          <root>/versions/<load_id>/   +   <root>/CURRENT
                                                     ▲
                           API worker 1 ─ Catalog ───┤  (each worker: read CURRENT per request,
                           API worker 2 ─ Catalog ───┤   open the version lazily, pin it for the request)
                           API worker N ─ Catalog ───┘
```

## Pipeline

```python
from facetindex import build_index, set_current, ProfileConfig

cfg = ProfileConfig(kind_overrides=admin_overrides)   # optional: {"maturity_level": "categorical"}
build_index("load_0421.parquet", "/srv/facetindex", version="load-0421",
            id_column="id", cfg=cfg, publish=False)
swap_serving_table("load-0421")                       # your existing step
set_current("/srv/facetindex", "load-0421")           # pointer flip; readers switch on next request
```

* `version` should be the pipeline's load id. The index, the table and the logs then
  share one name.
* `publish=True` (the default) flips the pointer at the end of the build and prunes old
  versions (`retain=3`). Use `publish=False` + `set_current()` when the serving table swap
  must happen in between.
* The build is idempotent per version name; rebuilding an existing version needs
  `overwrite=True`.
* Input requirements: a Parquet file with a unique, non-null id column. All other columns
  are treated as text. Typed columns are fine; they are cast.
* The id column may be any type; it is stored as text in `ids.parquet`.

### Where the root lives

A directory any API worker can read: a shared volume, or an object-store prefix synced to
local disk. Each version is self-contained and immutable. `CURRENT` is a tiny JSON file.
On object storage, upload the version directory first, then overwrite the pointer object
(single-object PUT is atomic on S3 and GCS).

### Cost

See the README benchmark table. Build time is roughly linear in rows x short columns;
long text columns cost one column scan each for their fill bitmap. The build runs on the
pipeline host, not on the serving database.

## API

```python
from facetindex import Catalog, filters as F

catalog = Catalog("/srv/facetindex", keep=2)        # one per worker process

def handle(request):
    idx = catalog.get()                             # pinned version for this request
    segs = load_segments_from_admin_db()            # {name: filter}
    f = F.from_dict(request.json["filter"])
    page   = idx.page(f, sort_by="revenue_usd", descending=True, limit=50, cursor=request.json.get("cursor"), segments=segs)
    facets = idx.facets(f, segments=segs)           # all facetable attributes; pass attributes=[...] to narrow
    rows   = db.fetch("SELECT * FROM entities WHERE id = ANY(%s)", [page.ids])   # PK lookup, keep page order
    return {"version": idx.version, "total": page.total, "rows": order_like(rows, page.ids),
            "next_cursor": page.next_cursor, "facets": facets, "warnings": idx.validate(f, segs)}
```

* **Catalog screen**: `idx.catalog()` returns the per-attribute profile list. Merge your
  admin metadata (labels, hidden, category) by attribute name. No query to the database.
* **Faceted list**: `page()` + `facets()`. Display rows by `WHERE id = ANY(ids)`; preserve
  the order from `page.ids`. Short attributes are also available from `idx.rows(positions)`
  without touching the database.
* **Export**: `idx.ids_for(f, limit=25_000)` then stream `COPY (SELECT ... WHERE id = ANY(...))`.
* **Saved segments**: store filters as JSON (`filters.to_dict`). `idx.segment_counts(segs)`
  gives every live count for a list view in one call. `SegmentStore` is a reference
  implementation of validation (cycles, dangling references); keep your own storage.
* **Single entity**: unchanged, primary key on the serving table.

### Filters

JSON shape (see `filters.py`):

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

Ops: `all`, `and`, `or`, `not`, `flag`, `in` (or `eq`), `between`, `contains`, `is_empty`,
`is_not_empty`, `segment`. Semantics are pinned in `docs/DESIGN.md` section 5.

### The SQL fallback

`filters.to_sql(f, POSTGRES, segments)` renders the same filter as a parameterised WHERE
clause for the all-text serving table (regex-guarded numeric casts, trimmed comparisons,
escaped ILIKE). Use it for `contains` on long text, which the index refuses with
`UnsupportedInIndex`, or any time you want to cross-check the engine against the database.
Pass `SQLDialect(typed=True)` if your table has typed columns.

### Version switches

`Catalog.get()` stats the pointer file on every call (microseconds). When the version
changes, the next request opens the new artifact (about 150 ms at 240k rows for postings
and ids; sort keys and text values load lazily per attribute on first use, a few ms each).
The previous version stays loaded while `keep` is not exceeded, so in-flight requests
finish on the version they started with.

With N workers, every worker pays the open cost once per load. There is no shared cache
and nothing to coordinate. If you want zero first-request latency, call `idx.warm()` from
a post-publish job in your queue system, or warm only the attributes your UI sorts by.

### Memory per worker

Resident after open: postings (about 16 MB at 240k rows, 110 facetable attributes) and ids.
Lazy: 8 bytes x rows per sort key attribute touched; the text column per attribute used in
`contains` or `rows()`. Caches are capped at 64 attributes each by default
(`FacetIndex(path, max_cached_attributes=...)`).

## Admin metadata

Admin-side attribute metadata (label, hidden, category, staleness) is keyed by attribute
name and lives in your database. Two places it meets facetindex:

* `ProfileConfig.kind_overrides` lets admins pin a kind when inference gets it wrong.
* `idx.catalog()` gives the inferred profile; your admin screen joins the two.

## Operational checklist

* Alert if `CURRENT` is older than the newest serving table load.
* `retain` keeps 3 versions by default. Each version is a few hundred MB at 240k rows
  (most of it the lazily loaded keys and values). Lower `retain` or move old versions to
  cold storage if the cadence is one load per minute for hours.
* `facetindex versions <root>`, `facetindex catalog <root> --brief` and
  `facetindex query <root> --filter ... --explain` are the debugging tools.
* Everything in a version directory is immutable; it is safe to rsync.
