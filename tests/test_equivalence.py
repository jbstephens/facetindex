"""The bitmap engine must agree with SQL over the source Parquet, filter for filter."""

import math

import pytest

from facetindex import filters as F
from facetindex.index import UnsupportedInIndex
from facetindex.profile import numeric_expr

from .randfilters import Gen

N_FILTERS = 150


def oracle_ids(duck, f, segments=None):
    sql, params = F.to_sql(f, F.DUCKDB, segments)
    return {r[0] for r in duck.execute(f"SELECT id FROM src WHERE {sql}", params).fetchall()}


@pytest.mark.parametrize("seed", range(N_FILTERS))
def test_random_filter_matches_sql(idx, duck, seed):
    g = Gen(idx.stats, seed)
    f = g.tree()
    # JSON round trip must be lossless
    assert F.from_dict(F.to_dict(f)) == f
    try:
        mine = set(idx.ids_for(f))
    except UnsupportedInIndex:
        # contains()/in() on long text is deliberately not indexed; SQL must still work
        assert oracle_ids(duck, f) is not None
        return
    assert mine == oracle_ids(duck, f), F.to_dict(f)


def test_segments_inline_like_sql(idx, duck):
    segs = {
        "emea_api": F.And(F.Flag("has_api"), F.In("region", ["EMEA"])),
        "mid_market": F.Between("headcount", 50, 1000),
        "combo": F.Or(F.Segment("emea_api"), F.And(F.Segment("mid_market"), F.Not(F.Flag("has_sso")))),
    }
    for name in segs:
        f = F.Segment(name)
        assert set(idx.ids_for(f, segs)) == oracle_ids(duck, f, segs)
    counts = idx.segment_counts(segs)
    assert counts["combo"] >= counts["emea_api"]


def _key_sql(idx, attr):
    kind = idx.kinds[attr]
    col = f'"{attr}"'
    if kind == "numeric":
        return numeric_expr(col)
    if kind == "flag":
        return (f"CASE WHEN LOWER(TRIM({col})) IN ('true','t','1','yes','y') THEN 1.0 "
                f"WHEN LOWER(TRIM({col})) IN ('false','f','0','no','n') THEN 0.0 END")
    return f"LOWER(NULLIF(TRIM({col}), ''))"


@pytest.mark.parametrize("seed", range(12))
def test_sorted_pagination_matches_sql(idx, duck, seed):
    g = Gen(idx.stats, 1000 + seed)
    f = g.tree()
    sortable = [a for a in idx.manifest.key_attrs]
    attr = g.rng.choice(sortable)
    desc = g.rng.random() < 0.5
    try:
        idx.evaluate(f)
    except UnsupportedInIndex:
        pytest.skip("filter needs SQL fallback")
    sql, params = F.to_sql(f, F.DUCKDB)
    order = f"{_key_sql(idx, attr)} {'DESC' if desc else 'ASC'} NULLS LAST, id ASC"
    expected = [r[0] for r in duck.execute(f"SELECT id FROM src WHERE {sql} ORDER BY {order}", params).fetchall()]
    got = []
    cursor = None
    limit = 37
    pages = 0
    while True:
        pg = idx.page(f, sort_by=attr, descending=desc, limit=limit, cursor=cursor)
        assert pg.total == len(expected)
        got.extend(pg.ids)
        pages += 1
        if not pg.next_cursor:
            break
        cursor = pg.next_cursor
        assert pages < 10_000
    assert got == expected, (attr, desc, F.to_dict(f))
    assert len(got) == pg.total


def test_facet_counts_match_group_by(idx, duck):
    f = F.And(F.In("region", ["NA", "EMEA"]), F.Flag("has_api"))
    fac = idx.facets(f, attributes=["tier", "region", "has_sso", "headcount"], exclude_own=True)
    # tier counts under the full filter
    sql, params = F.to_sql(f, F.DUCKDB)
    rows = duck.execute(
        f"SELECT NULLIF(TRIM(tier), ''), count(*) FROM src WHERE {sql} GROUP BY 1", params).fetchall()
    exp = {v: n for v, n in rows if v is not None}
    assert {fv.value: fv.count for fv in fac["tier"].values} == exp
    assert fac["tier"].empty_count == dict(rows).get(None, 0)
    # region counts ignore the user's own region selection (multi-select behaviour)
    f2 = F.Flag("has_api")
    sql2, params2 = F.to_sql(f2, F.DUCKDB)
    rows2 = duck.execute(
        f"SELECT NULLIF(TRIM(region), ''), count(*) FROM src WHERE {sql2} GROUP BY 1", params2).fetchall()
    assert {fv.value: fv.count for fv in fac["region"].values} == {v: n for v, n in rows2 if v is not None}
    # numeric facet: min/max/count agree
    num = fac["headcount"].numeric
    mn, mx, cnt = duck.execute(
        f"SELECT min(v), max(v), count(v) FROM (SELECT {numeric_expr('headcount')} v FROM src WHERE {sql})",
        params).fetchone()
    assert (num["min"], num["max"], num["count"]) == (mn, mx, cnt)
    assert sum(b["count"] for b in num["histogram"]) == cnt


def test_catalog_matches_sql_profile(idx, duck):
    for attr in ["region", "has_api", "headcount", "name", "deprecated_attribute_1"]:
        s = idx.stats[attr]
        fill, distinct = duck.execute(
            f"SELECT count(NULLIF(TRIM(\"{attr}\"), '')), count(DISTINCT NULLIF(TRIM(\"{attr}\"), '')) FROM src"
        ).fetchone()
        assert s["fill_count"] == fill
        assert s["distinct_count"] == distinct
        assert s["empty_count"] == idx.n - fill
    assert idx.stats["has_api"]["kind"] == "flag"
    assert idx.stats["region"]["kind"] == "categorical"
    assert idx.stats["headcount"]["kind"] == "numeric"
    assert idx.stats["alexa_rank_legacy"]["kind"] == "numeric"  # 1% junk tolerated
    assert idx.stats["name"]["kind"] == "short_text"
    assert idx.stats["ai_summary_1"]["kind"] == "long_text"
    assert idx.stats["deprecated_attribute_1"]["kind"] == "empty"
    assert idx.stats["deprecated_attribute_2"]["kind"] == "empty"
    assert not math.isnan(idx.stats["headcount"]["mean"])
