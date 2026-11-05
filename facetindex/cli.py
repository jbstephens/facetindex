"""Command line: synth / profile / build / publish / versions / query / catalog."""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import artifact as A
from . import filters as F


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


def cmd_synth(a):
    from . import synth
    t0 = time.time()
    t = synth.write(a.out, rows=a.rows, columns=a.columns, seed=a.seed,
                    long_text_columns=a.long_text_columns, long_text_fill=a.long_text_fill)
    print(f"wrote {t.num_rows} rows x {t.num_columns} columns to {a.out} in {time.time()-t0:.1f}s")


def cmd_profile(a):
    from .profile import ProfileConfig, profile_parquet
    cfg = ProfileConfig(kind_overrides=json.loads(a.overrides) if a.overrides else {})
    profs = profile_parquet(a.source, a.id_column, cfg)
    _print([p.to_dict() for p in profs.values()])


def cmd_build(a):
    from .build import build_index
    from .profile import ProfileConfig
    cfg = ProfileConfig(kind_overrides=json.loads(a.overrides) if a.overrides else {})
    r = build_index(a.source, a.root, version=a.version, id_column=a.id_column, cfg=cfg,
                    publish=not a.no_publish, with_signatures=a.signatures, overwrite=a.overwrite,
                    retain=a.retain, log=lambda m: print(f"[build] {m}", file=sys.stderr))
    _print({"version": r.version, "path": str(r.path), "rows": r.row_count, "seconds": r.seconds,
            "bytes": r.bytes, "published": r.published,
            "kinds": {k: sum(1 for v in r.kinds.values() if v == k) for k in set(r.kinds.values())}})


def cmd_publish(a):
    A.set_current(a.root, a.version)
    print(f"CURRENT -> {a.version}")


def cmd_versions(a):
    _print({"current": A.current_version(a.root), "versions": A.list_versions(a.root)})


def _load_segments(path):
    from .segments import SegmentStore
    return SegmentStore.load(path).filters() if path else {}


def cmd_query(a):
    from .index import FacetIndex
    idx = FacetIndex.open(a.root, a.version)
    f = F.from_dict(json.loads(a.filter)) if a.filter else F.All()
    segs = _load_segments(a.segments)
    t0 = time.perf_counter()
    out = {"version": idx.version, "problems": idx.validate(f, segs)}
    if a.explain:
        out["explain"] = [[r.node, r.cardinality, r.note] for r in idx.explain(f, segs)]
    pg = idx.page(f, sort_by=a.sort, descending=a.desc, limit=a.limit, cursor=a.cursor, segments=segs)
    out["total"] = pg.total
    out["ids"] = pg.ids
    out["next_cursor"] = pg.next_cursor
    if a.facets:
        fac = idx.facets(f, attributes=a.facets if a.facets != ["*"] else None, segments=segs)
        out["facets"] = {k: {"values": [[v.value, v.count] for v in fc.values], "empty": fc.empty_count,
                             "numeric": fc.numeric} for k, fc in fac.items()}
    if a.rows:
        out["rows"] = idx.rows(pg.positions, a.columns)
    out["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    sql, params = F.to_sql(f, F.POSTGRES, segs)
    out["sql_fallback"] = {"where": sql, "params": params}
    _print(out)


def cmd_catalog(a):
    from .index import FacetIndex
    idx = FacetIndex.open(a.root, a.version)
    rows = idx.catalog()
    if a.brief:
        for r in rows:
            print(f"{r['name']:<36} {r['kind']:<12} fill={r['fill_count']:<8} distinct={r['distinct_count']}"
                  f"{'~' if r['distinct_is_approx'] else ''}  top={r['top_values'][:3]}")
    else:
        _print(rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="facetindex", description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("synth", help="write a synthetic wide/sparse parquet file")
    s.add_argument("out"); s.add_argument("--rows", type=int, default=240_000)
    s.add_argument("--columns", type=int, default=200); s.add_argument("--seed", type=int, default=7)
    s.add_argument("--long-text-columns", type=int, default=4); s.add_argument("--long-text-fill", type=float, default=0.3)
    s.set_defaults(fn=cmd_synth)

    s = sp.add_parser("profile", help="profile a parquet file (what stats.json will contain)")
    s.add_argument("source"); s.add_argument("--id-column", default="id"); s.add_argument("--overrides")
    s.set_defaults(fn=cmd_profile)

    s = sp.add_parser("build", help="build a version of the index and publish it")
    s.add_argument("source"); s.add_argument("root"); s.add_argument("--version")
    s.add_argument("--id-column", default="id"); s.add_argument("--overrides", help='JSON {"attr": "kind"}')
    s.add_argument("--no-publish", action="store_true"); s.add_argument("--signatures", action="store_true")
    s.add_argument("--overwrite", action="store_true"); s.add_argument("--retain", type=int, default=3)
    s.set_defaults(fn=cmd_build)

    s = sp.add_parser("publish", help="point CURRENT at an existing version")
    s.add_argument("root"); s.add_argument("version"); s.set_defaults(fn=cmd_publish)

    s = sp.add_parser("versions", help="list versions and the current pointer")
    s.add_argument("root"); s.set_defaults(fn=cmd_versions)

    s = sp.add_parser("query", help="count / page / facets for a JSON filter")
    s.add_argument("root"); s.add_argument("--version"); s.add_argument("--filter", help="JSON filter")
    s.add_argument("--segments", help="segments JSON file"); s.add_argument("--sort"); s.add_argument("--desc", action="store_true")
    s.add_argument("--limit", type=int, default=50); s.add_argument("--cursor")
    s.add_argument("--facets", nargs="*", help="attributes to facet ('*' for all facetable)")
    s.add_argument("--rows", action="store_true"); s.add_argument("--columns", nargs="*")
    s.add_argument("--explain", action="store_true"); s.set_defaults(fn=cmd_query)

    s = sp.add_parser("catalog", help="print the attribute catalog of a version")
    s.add_argument("root"); s.add_argument("--version"); s.add_argument("--brief", action="store_true")
    s.set_defaults(fn=cmd_catalog)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
