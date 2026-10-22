"""Tests for BM25 evaluation pipeline (src.bm25.evaluate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bm25.evaluate import (
    DEFAULT_INDEX_DIR,
    DEFAULT_QRELS_FILE,
    DEFAULT_QUERIES_FILE,
    TrecEvaluator,
    load_qrels,
    load_queries,
    run_retrieval,
    write_trec_run,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not DEFAULT_INDEX_DIR.exists() or not DEFAULT_QRELS_FILE.exists(),
    reason="Index or qrels not found, skipping evaluation test.",
)


def test_load_qrels():
    qrels = load_qrels(DEFAULT_QRELS_FILE)
    assert len(qrels) == 323
    assert "PLAIN-2" in qrels
    assert "MED-2427" in qrels["PLAIN-2"]
    assert qrels["PLAIN-2"]["MED-2427"] == 2


def test_load_queries():
    qrels = load_qrels(DEFAULT_QRELS_FILE)
    test_qids = set(qrels.keys())
    all_queries = load_queries(DEFAULT_QUERIES_FILE)
    test_queries = {q["_id"]: q["text"] for q in all_queries if q.get("_id") in test_qids}
    assert len(test_queries) == 323
    assert set(test_queries.keys()) == test_qids
    assert len(test_queries["PLAIN-2"].strip()) > 0


def test_write_trec_run(tmp_path: Path):
    class DummyHit:
        def __init__(self, docid: str, score: float):
            self.docid = docid
            self.score = score

    batch_hits = {
        "q1": [DummyHit("MED-1", 1.5), DummyHit("MED-2", 0.9)],
        "q2": [DummyHit("MED-3", 2.1)],
    }
    out_file = tmp_path / "test.run"
    count = write_trec_run(batch_hits, out_file, tag="bm25_test")
    assert count == 3
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == "q1 Q0 MED-1 1 1.500000 bm25_test"
    assert lines[1] == "q1 Q0 MED-2 2 0.900000 bm25_test"
    assert lines[2] == "q2 Q0 MED-3 1 2.100000 bm25_test"


def test_trec_evaluator_on_test_subset():
    qrels = load_qrels(DEFAULT_QRELS_FILE)
    test_qids = set(qrels.keys())
    all_queries = load_queries(DEFAULT_QUERIES_FILE)
    test_queries = {q["_id"]: q["text"] for q in all_queries if q.get("_id") in test_qids}

    # Run retrieval on a small subset (5 queries)
    sample_queries = {qid: test_queries[qid] for qid in list(test_qids)[:5]}
    batch_hits, elapsed = run_retrieval(
        index_dir=DEFAULT_INDEX_DIR,
        queries=sample_queries,
        k=50,
        k1=0.9,
        b=0.4,
        threads=2,
    )
    assert len(batch_hits) == 5

    sample_qrels = {qid: qrels[qid] for qid in sample_queries}
    evaluator = TrecEvaluator(sample_qrels)
    try:
        metrics = evaluator.evaluate(batch_hits)
    finally:
        evaluator.close()

    assert "ndcg@10" in metrics
    assert "map@10" in metrics
    assert "recall@100" in metrics
    assert "mrr@10" in metrics
    assert metrics["ndcg@10"] > 0.0
