"""Smoke test for examples/api.py; skipped when FastAPI is not installed."""

import importlib
import os

import pytest

pytest.importorskip("fastapi")


def test_api_example_roundtrip(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("FACETINDEX_ROOT", str(dataset["root"]))
    monkeypatch.setenv("FACETINDEX_SEGMENTS", str(tmp_path / "segments.json"))
    import examples.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    c = TestClient(api.app)
    assert c.get("/version").json()["version"] == "t1"
    seg = {"op": "and", "children": [{"op": "flag", "attr": "has_api"}, {"op": "in", "attr": "region", "values": ["EMEA"]}]}
    assert c.put("/segments/emea_api", json={"name": "emea_api", "filter": seg}).status_code == 200
    r = c.post("/search", json={"filter": {"op": "segment", "name": "emea_api"}, "sort_by": "headcount",
                                "descending": True, "limit": 5, "facets": ["tier"], "columns": ["name"]})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == c.get("/segments").json()["segments"][0]["count"]
    assert len(body["ids"]) == min(5, body["total"]) and "tier" in body["facets"]
    r = c.post("/search", json={"filter": {"op": "contains", "attr": "ai_summary_1", "needle": "x"}})
    assert r.status_code == 422 and "sql_where" in r.json()["detail"]
    assert os.path.exists(tmp_path / "segments.json")
