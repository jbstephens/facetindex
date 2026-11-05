"""Pipeline side: call this right after the deduplicated load file is written and before
(or in the same step as) the serving table swap. One function call per load.

The version string should be the pipeline's own load id so the serving table, the index
and any logs all carry the same name. Publishing is a pointer flip, so if you want the
serving table and the index to switch together, build with publish=False, swap the table,
then call set_current().
"""

from __future__ import annotations

import logging
import sys

from facetindex import ProfileConfig, build_index, set_current

log = logging.getLogger("pipeline.facetindex")


def after_load(load_parquet: str, load_id: str, index_root: str, admin_kind_overrides: dict | None = None):
    # Admin metadata can pin an attribute's kind (e.g. force 'categorical' on a numeric code).
    cfg = ProfileConfig(kind_overrides=admin_kind_overrides or {})

    # 1. Build the artifact for this load. Nothing is visible to readers yet.
    result = build_index(
        load_parquet, index_root, version=load_id, id_column="id", cfg=cfg,
        publish=False, with_signatures=False, log=log.info,
    )
    log.info("facetindex %s built: %d rows, %.1fs, %.1f MB", load_id, result.row_count,
             result.seconds, result.bytes / 1e6)

    # 2. Swap the serving table here (your existing step). E.g. in one transaction:
    #      UPDATE current_load SET load_id = %s;     -- a pointer row the API reads per request
    #    or the rename dance, with a short lock_timeout so a long reader cannot queue the world.

    # 3. Flip the index pointer. Readers now see the new version on their next request.
    set_current(index_root, load_id)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    after_load(sys.argv[1], sys.argv[2], sys.argv[3])
