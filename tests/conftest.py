import pathlib

import pytest

from facetindex import build_index, synth
from facetindex.index import FacetIndex

ROWS = 6000
COLUMNS = 48


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("data")
    src = d / "src.parquet"
    synth.write(str(src), rows=ROWS, columns=COLUMNS, seed=11, long_text_columns=2, long_text_fill=0.4,
                long_text_kb=(1, 2))
    root = d / "idx"
    res = build_index(str(src), str(root), version="t1", id_column="id", with_signatures=True)
    return {"src": str(src), "root": root, "result": res}


@pytest.fixture(scope="session")
def idx(dataset) -> FacetIndex:
    return FacetIndex.open(dataset["root"])


@pytest.fixture(scope="session")
def duck(dataset):
    import duckdb
    con = duckdb.connect()
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{dataset['src']}')")
    return con


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slower tests")


__all__ = ["pathlib"]
