"""Integration test: build a real Lucene index on the synthetic fixture corpus."""

import json

import pytest

pytest.importorskip("pyserini")

from src.bm25 import build_index, prepare_data  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402


@pytest.fixture
def prepared_dir(raw_dir, tmp_path):
    out = tmp_path / "processed"
    prepare_data.prepare(raw_dir, out)
    return out


def run_build(prepared_dir, tmp_path, rebuild=False, k1=0.9, b=0.4):
    return build_index.build(
        prepared_dir,
        tmp_path / "index",
        tmp_path / "results" / "index_build.json",
        k1=k1,
        b=b,
        threads=1,
        smoke_query="vitamin D cancer",
        smoke_k=3,
        rebuild=rebuild,
    )


def test_build_index_and_record(prepared_dir, tmp_path):
    record = run_build(prepared_dir, tmp_path, k1=1.2, b=0.75)
    assert record["num_docs_indexed"] == len(FIXTURE_DOCS)
    assert record["all_doc_ids_found"]
    assert record["smoke_test"]["top_hits"][0]["docid"] == "MED-14"
    assert record["bm25"]["k1"] == 1.2 and record["bm25"]["b"] == 0.75
    assert record["index_size_bytes"] > 0
    assert record["build_duration_seconds"] > 0
    java_major = int(record["versions"]["java"].split(".")[0])
    assert java_major >= 21, f"Expected Java >= 21, got {record['versions']['java']}"

    saved = json.loads((tmp_path / "results" / "index_build.json").read_text())
    assert saved["num_docs_indexed"] == len(FIXTURE_DOCS)
    assert json.loads((tmp_path / "index" / "bm25_params.json").read_text()) == {"k1": 1.2, "b": 0.75}


def test_build_refuses_overwrite_without_rebuild(prepared_dir, tmp_path):
    run_build(prepared_dir, tmp_path)
    with pytest.raises(SystemExit):
        run_build(prepared_dir, tmp_path)
    record = run_build(prepared_dir, tmp_path, rebuild=True)
    assert record["num_docs_indexed"] == len(FIXTURE_DOCS)
