import json

import pytest

from src.bm25 import prepare_data
from tests.conftest import FIXTURE_DOCS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_to_pyserini_doc_keeps_id_and_joins_title_text():
    doc = prepare_data.to_pyserini_doc({"_id": "MED-10", "title": " A ", "text": "b c "})
    assert doc == {"id": "MED-10", "contents": "A b c"}


def test_to_pyserini_doc_empty_title():
    assert prepare_data.to_pyserini_doc({"_id": "x", "title": "", "text": "t"})["contents"] == "t"


def test_prepare_preserves_ids_and_order(raw_dir, tmp_path):
    out = tmp_path / "processed"
    manifest = prepare_data.prepare(raw_dir, out)
    docs = read_jsonl(out / "corpus" / "docs.jsonl")
    assert [d["id"] for d in docs] == [d["_id"] for d in FIXTURE_DOCS]
    assert all(set(d) == {"id", "contents"} for d in docs)
    assert manifest["num_docs"] == len(FIXTURE_DOCS)
    # Only the corpus file may be in the indexer's input directory.
    assert [p.name for p in (out / "corpus").iterdir()] == ["docs.jsonl"]
    assert (out / "manifest.json").exists()


def test_prepare_does_not_modify_raw_data(raw_dir, tmp_path):
    before = prepare_data.file_hash(raw_dir / "corpus.jsonl")
    manifest = prepare_data.prepare(raw_dir, tmp_path / "processed")
    assert prepare_data.file_hash(raw_dir / "corpus.jsonl") == before == manifest["source_corpus_sha256"]


def test_prepare_refuses_overwrite_without_rebuild(raw_dir, tmp_path):
    out = tmp_path / "processed"
    prepare_data.prepare(raw_dir, out)
    sentinel = out / "corpus" / "docs.jsonl"
    original = sentinel.read_text()
    with pytest.raises(SystemExit):
        prepare_data.prepare(raw_dir, out)
    assert sentinel.read_text() == original
    prepare_data.prepare(raw_dir, out, rebuild=True)
    assert sentinel.read_text() == original


def test_duplicate_ids_rejected(raw_dir, tmp_path):
    with open(raw_dir / "corpus.jsonl", "a") as f:
        f.write(json.dumps(FIXTURE_DOCS[0]) + "\n")
    out = tmp_path / "processed"
    with pytest.raises(ValueError, match="duplicate"):
        prepare_data.prepare(raw_dir, out)
    assert not out.exists()


def test_missing_corpus_exits(tmp_path):
    with pytest.raises(SystemExit):
        prepare_data.prepare(tmp_path / "nope", tmp_path / "processed")
