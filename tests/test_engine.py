import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from pyroaring import BitMap

from facetindex import artifact as A
from facetindex import filters as F
from facetindex.catalog import Catalog
from facetindex.index import FacetIndex, UnknownAttribute, UnsupportedInIndex
from facetindex.segments import SegmentDef, SegmentStore


def test_all_and_empty_logic(idx):
    assert idx.count(F.All()) == idx.n
    assert idx.count(F.And()) == idx.n
    assert idx.count(F.Or()) == 0
    assert idx.count(F.Not(F.All())) == 0
    assert idx.count(F.In("region", [])) == 0
    assert idx.count(F.Or(F.IsEmpty("region"), F.IsNotEmpty("region"))) == idx.n


def test_unknown_attribute_policy(idx):
    f = F.And(F.Flag("has_api"), F.In("vanished_attribute", ["x"]))
    assert idx.count(f) == 0
    with pytest.raises(UnknownAttribute):
        idx.evaluate(f, strict=True)
    assert any("vanished_attribute" in p for p in idx.validate(f))


def test_long_text_contains_needs_sql(idx):
    with pytest.raises(UnsupportedInIndex):
        idx.count(F.Contains("ai_summary_1", "alpha"))
    sql, params = F.to_sql(F.Contains("ai_summary_1", "al%pha"))
    assert "ILIKE" in sql and params == ["%al\\%pha%"]
    assert idx.count(F.IsNotEmpty("ai_summary_1")) == idx.stats["ai_summary_1"]["fill_count"]


def test_flag_semantics(idx):
    t = idx.count(F.Flag("has_api", True))
    f = idx.count(F.Flag("has_api", False))
    e = idx.count(F.IsEmpty("has_api"))
    assert t + f + e == idx.n
    assert t == idx.stats["has_api"]["true_count"]


def test_page_without_sort_is_id_order(idx):
    pg = idx.page(F.All(), limit=10)
    assert pg.ids == sorted(pg.ids)
    assert pg.total == idx.n
    pg2 = idx.page(F.All(), limit=10, cursor=pg.next_cursor)
    assert pg2.ids[0] > pg.ids[-1]


def test_rows_and_explain(idx):
    f = F.And(F.Flag("has_api"), F.In("region", ["EMEA"]))
    pg = idx.page(f, sort_by="name", limit=5)
    rows = idx.rows(pg.positions, ["name", "region", "has_api"])
    assert [r["id"] for r in rows] == pg.ids
    assert all(r["region"] == "EMEA" for r in rows)
    names = [r["name"].lower() for r in rows]
    assert names == sorted(names)
    ex = idx.explain(f)
    assert ex[0].cardinality == pg.total and len(ex) == 3


def test_facets_zero_filter_sum(idx):
    fac = idx.facets(F.All(), attributes=["tier"])
    assert sum(v.count for v in fac["tier"].values) + fac["tier"].empty_count == idx.n


def test_segment_store_cycles_and_deps(tmp_path):
    st = SegmentStore()
    st.put(SegmentDef("a", F.Flag("has_api")))
    st.put(SegmentDef("b", F.And(F.Segment("a"), F.In("region", ["NA"]))))
    with pytest.raises(ValueError):
        st.put(SegmentDef("a", F.Segment("b")))  # a -> b -> a
    with pytest.raises(ValueError):
        st.remove("a")  # b depends on it
    assert st.dependencies("b") == ["a"]
    st.save(tmp_path / "segs.json")
    again = SegmentStore.load(tmp_path / "segs.json")
    assert again.filters()["b"] == st.filters()["b"]


def test_segment_cycle_detected_at_eval(idx):
    segs = {"x": F.Segment("y"), "y": F.Segment("x")}
    with pytest.raises(ValueError):
        idx.count(F.Segment("x"), segs)


def test_signatures_match_postings(dataset, idx):
    vp = dataset["result"].path
    dims = json.loads((vp / A.DIMENSIONS).read_text())
    sig = pq.read_table(vp / A.SIGNATURES)
    assert sig.column("id").to_pylist() == idx.ids.to_pylist()
    blobs = sig.column("signature").to_pylist()
    arr = np.frombuffer(b"".join(blobs), dtype=np.uint8).reshape(idx.n, dims["byte_count"])
    for d in dims["dimensions"][:40]:
        bit = d["bit"]
        rows = np.flatnonzero(arr[:, bit // 8] & (1 << (bit % 8)))
        if d["value"] == "=true":
            expect = idx.flag_true[d["attribute"]]
        else:
            expect = idx.values_postings[d["attribute"]][d["value"]]
        assert BitMap(rows.astype(np.uint32)) == expect


def test_versioning_and_catalog(dataset, tmp_path):
    root = tmp_path / "root"
    # publish v1 then v2 from the same source; CURRENT flips atomically; catalog follows
    from facetindex import build_index
    r1 = build_index(dataset["src"], str(root), version="v1")
    cat = Catalog(root, keep=2)
    i1 = cat.get()
    assert i1.version == "v1" and A.current_version(root) == "v1"
    r2 = build_index(dataset["src"], str(root), version="v2", publish=False)
    assert A.current_version(root) == "v1"           # built but not yet published
    assert cat.get().version == "v1"
    A.set_current(root, "v2")
    i2 = cat.get()
    assert i2.version == "v2" and i2 is not i1
    assert cat.get("v1") is i1                        # old version still pinned for in-flight work
    build_index(dataset["src"], str(root), version="v3", retain=2)
    assert set(A.list_versions(root)) == {"v2", "v3"}
    with pytest.raises(FileExistsError):
        build_index(dataset["src"], str(root), version="v3", publish=False)
    assert r1.row_count == r2.row_count
    manifest = A.read_manifest(r2.path)
    assert manifest.format_version == A.FORMAT_VERSION and manifest.row_count == r2.row_count


def test_kind_override(dataset, tmp_path):
    from facetindex import ProfileConfig, build_index
    cfg = ProfileConfig(kind_overrides={"maturity_level": "categorical", "name": "long_text"})
    r = build_index(dataset["src"], str(tmp_path / "o"), version="ov", cfg=cfg)
    i = FacetIndex(r.path)
    assert i.kinds["maturity_level"] == "categorical" and "maturity_level" in i.values_postings
    assert i.kinds["name"] == "long_text"
    with pytest.raises(UnsupportedInIndex):
        i.count(F.Contains("name", "a"))


def test_duplicate_ids_rejected(tmp_path):
    import pyarrow as pa
    p = tmp_path / "dup.parquet"
    pq.write_table(pa.table({"id": ["a", "a", "b"], "x": ["1", "2", "3"]}), p)
    from facetindex import build_index
    with pytest.raises(ValueError, match="not unique"):
        build_index(str(p), str(tmp_path / "r"), version="d")


def test_sql_emitter_postgres_shapes():
    f = F.And(F.Flag("has_api"), F.Between("headcount", 10, None), F.Not(F.IsEmpty("region")),
              F.Contains("name", "a_b"))
    sql, params = F.to_sql(f)
    assert sql.count("%s") == len(params)
    assert '"has_api"' in sql and "::double precision" in sql and "NOT COALESCE" in sql
    assert params[-1] == "%a\\_b%"
    sqlt, _ = F.to_sql(f, F.SQLDialect("postgres", "%s", typed=True))
    assert "TRIM" not in sqlt and "::double precision" not in sqlt


def test_without_attribute():
    f = F.And(F.Flag("a"), F.In("b", ["x"]), F.Or(F.Flag("b"), F.Flag("c")))
    g = F.without_attribute(f, "b")
    assert g == F.And(F.Flag("a"), F.Or(F.Flag("b"), F.Flag("c")))  # nested OR touching b/c is kept
    assert F.without_attribute(F.Flag("a"), "a") == F.All()
