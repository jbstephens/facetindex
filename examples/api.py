"""API side: a minimal FastAPI service over a facetindex root. Run with several workers::

    pip install "facetindex[api]"
    FACETINDEX_ROOT=/srv/facetindex uvicorn examples.api:app --workers 4

Each worker holds its own Catalog (no shared state, no threads, no locks). A request pins
one version for its whole duration; a load published mid-request does not affect it.

Row display: the engine returns ids and positions. Fetch display rows from the serving
table by primary key (``WHERE id = ANY(%s)``), or from the artifact's narrow table for
short attributes. Long text always comes from the serving table (it is not in the index).
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from facetindex import Catalog, SegmentDef, SegmentStore, UnknownAttribute, UnsupportedInIndex
from facetindex import filters as F

ROOT = os.environ.get("FACETINDEX_ROOT", "./index")
SEGMENTS_FILE = os.environ.get("FACETINDEX_SEGMENTS", os.path.join(ROOT, "segments.json"))

app = FastAPI(title="facetindex example")
catalog = Catalog(ROOT, keep=2)


def segments() -> SegmentStore:
    # In production this is your admin DB; here it is a JSON file re-read per request.
    return SegmentStore.load(SEGMENTS_FILE)


class SearchRequest(BaseModel):
    filter: dict | None = None
    sort_by: str | None = None
    descending: bool = False
    limit: int = 50
    cursor: str | None = None
    facets: list[str] | None = None      # None = all facetable attributes
    columns: list[str] | None = None     # display columns from the narrow table


@app.get("/version")
def version():
    idx = catalog.get()
    return {"version": idx.version, "rows": idx.n, "attributes": len(idx.kinds), "created_at": idx.manifest.created_at}


@app.get("/catalog")
def attribute_catalog(hidden: bool = False):
    """The attribute profile that used to be computed live. Instant; comes from stats.json."""
    idx = catalog.get()
    return {"version": idx.version, "attributes": idx.catalog()}


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    idx = catalog.get()                      # pinned for this request
    segs = segments().filters()
    try:
        f = F.from_dict(req.filter) if req.filter else F.All()
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"bad filter: {e}") from e
    problems = idx.validate(f, segs)
    try:
        page = idx.page(f, sort_by=req.sort_by, descending=req.descending,
                        limit=min(req.limit, 500), cursor=req.cursor, segments=segs)
        facets = idx.facets(f, attributes=req.facets, segments=segs)
    except UnknownAttribute as e:
        raise HTTPException(400, f"unknown attribute {e}") from e
    except UnsupportedInIndex as e:
        # e.g. contains() on long text: hand the caller the SQL to run against the serving table
        sql, params = F.to_sql(f, F.POSTGRES, segs)
        raise HTTPException(422, {"error": str(e), "sql_where": sql, "params": params}) from e
    rows = idx.rows(page.positions, req.columns)   # or: SELECT ... FROM entities WHERE id = ANY(page.ids)
    return {
        "version": idx.version,
        "total": page.total,
        "ids": page.ids,
        "rows": rows,
        "next_cursor": page.next_cursor,
        "facets": {
            a: {"kind": fc.kind, "values": [{"value": v.value, "count": v.count} for v in fc.values],
                "empty": fc.empty_count, "filled": fc.filled_count, "numeric": fc.numeric}
            for a, fc in facets.items()
        },
        "warnings": problems,
    }


@app.get("/segments")
def list_segments():
    """Every saved segment with its live count. One list render = one call."""
    idx = catalog.get()
    st = segments()
    counts = idx.segment_counts(st.filters())
    return {"version": idx.version,
            "segments": [{**st.get(n).to_dict(), "count": counts[n]} for n in st.names()]}


class SegmentIn(BaseModel):
    name: str
    filter: dict
    label: str = ""
    description: str = ""


@app.put("/segments/{name}")
def put_segment(name: str, body: SegmentIn):
    st = segments()
    try:
        st.put(SegmentDef(name, F.from_dict(body.filter), body.label, body.description))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    st.save(SEGMENTS_FILE)
    idx = catalog.get()
    return {"name": name, "count": idx.count(F.Segment(name), st.filters()),
            "warnings": idx.validate(F.Segment(name), st.filters())}


@app.get("/export")
def export_ids(segment: str | None = None, limit: int = Query(25_000, le=100_000)):
    """Ids for an export; stream the CSV from the serving table with WHERE id = ANY(...)."""
    idx = catalog.get()
    st = segments()
    f = F.Segment(segment) if segment else F.All()
    return {"version": idx.version, "ids": idx.ids_for(f, st.filters(), limit=limit)}
