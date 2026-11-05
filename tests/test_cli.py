import json
import subprocess
import sys


def run(*args):
    out = subprocess.run([sys.executable, "-m", "facetindex.cli", *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_cli_roundtrip(tmp_path):
    src = tmp_path / "s.parquet"
    root = tmp_path / "root"
    run("synth", str(src), "--rows", "2000", "--columns", "30", "--long-text-columns", "1")
    built = json.loads(run("build", str(src), str(root), "--version", "v1", "--signatures"))
    assert built["rows"] == 2000 and built["published"]
    versions = json.loads(run("versions", str(root)))
    assert versions["current"] == "v1"
    q = json.loads(run("query", str(root), "--filter", json.dumps(
        {"op": "and", "children": [{"op": "flag", "attr": "has_api"}, {"op": "in", "attr": "region", "values": ["NA"]}]}),
        "--sort", "headcount", "--desc", "--limit", "5", "--facets", "tier", "--rows", "--columns", "name", "--explain"))
    assert q["total"] >= len(q["ids"]) and q["problems"] == []
    assert "tier" in q["facets"] and "sql_fallback" in q
    brief = run("catalog", str(root), "--brief")
    assert "has_api" in brief and "flag" in brief
    prof = json.loads(run("profile", str(src)))
    assert any(p["name"] == "region" and p["kind"] == "categorical" for p in prof)
