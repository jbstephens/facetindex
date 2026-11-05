"""facetindex: a versioned, pipeline-built facet index for wide, sparse attribute tables.

Pipeline side (once per load)::

    from facetindex import build_index
    build_index("load.parquet", "/srv/facetindex", version="load-0421", id_column="id")

API side (every worker)::

    from facetindex import Catalog, filters as F
    catalog = Catalog("/srv/facetindex")
    idx = catalog.get()                                 # pinned for this request
    f = F.And(F.Flag("has_api"), F.In("region", ["EMEA"]), F.Between("headcount", 50, 500))
    idx.count(f); idx.facets(f); idx.page(f, sort_by="revenue_usd", descending=True, limit=50)
"""

from . import filters
from .artifact import current_version, list_versions, retain, set_current
from .build import build_index
from .catalog import Catalog
from .index import FacetIndex, UnknownAttribute, UnsupportedInIndex
from .profile import AttributeProfile, ProfileConfig, profile_parquet
from .segments import SegmentDef, SegmentStore

__version__ = "0.1.0"
__all__ = [
    "filters", "build_index", "Catalog", "FacetIndex", "UnknownAttribute", "UnsupportedInIndex",
    "AttributeProfile", "ProfileConfig", "profile_parquet", "SegmentDef", "SegmentStore",
    "current_version", "list_versions", "retain", "set_current",
]
