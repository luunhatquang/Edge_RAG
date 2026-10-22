"""Integration test: kiểm tra BM25 index với dữ liệu NFCorpus thực.

Chỉ chạy nếu index đã được build (`indexes/bm25/nfcorpus-flat`).
Skip nếu index hoặc pyserini không có sẵn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "indexes" / "bm25" / "nfcorpus-flat"
QUERIES_FILE = REPO_ROOT / "data" / "nfcorpus" / "queries.jsonl"
QRELS_FILE = REPO_ROOT / "data" / "nfcorpus" / "qrels" / "test.tsv"
CORPUS_FILE = REPO_ROOT / "data" / "nfcorpus" / "corpus.jsonl"

pyserini = pytest.importorskip("pyserini")
pytestmark = pytest.mark.skipif(
    not INDEX_DIR.exists(),
    reason=f"BM25 index not built yet: {INDEX_DIR}",
)

NUM_DOCS_EXPECTED = 3633


@pytest.fixture(scope="module")
def searcher():
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(str(INDEX_DIR))
    s.set_bm25(k1=0.9, b=0.4)
    yield s
    s.close()


@pytest.fixture(scope="module")
def qrels() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with open(QRELS_FILE, encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            result.setdefault(qid, {})[did] = score
    return result


@pytest.fixture(scope="module")
def test_queries() -> list[dict]:
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_index_doc_count(searcher):
    """Index phải có đúng số documents."""
    assert searcher.num_docs == NUM_DOCS_EXPECTED, (
        f"Expected {NUM_DOCS_EXPECTED} docs, got {searcher.num_docs}"
    )


def test_known_doc_retrievable(searcher):
    """Document đầu tiên trong corpus phải truy xuất được bằng ID."""
    doc = searcher.doc("MED-10")
    assert doc is not None, "MED-10 not found in index"
    raw = json.loads(doc.raw())
    assert "id" in raw
    assert "contents" in raw
    assert "Statin" in raw["contents"]


def test_smoke_query_returns_hits(searcher):
    """Query đơn giản phải trả về kết quả."""
    hits = searcher.search("vitamin D cancer risk", k=10)
    assert len(hits) > 0
    docids = [h.docid for h in hits]
    print(f"\n  Top hits for 'vitamin D cancer risk': {docids[:5]}")


def test_bm25_scoring_decreasing(searcher):
    """Scores phải giảm dần theo rank."""
    hits = searcher.search("breast cancer statin cholesterol", k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "Scores are not in descending order"


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg_at_k(hits, qrel: dict[str, int], k: int) -> float:
    if not qrel:
        return 0.0
    retrieved_rels = [qrel.get(h.docid, 0) for h in hits[:k]]
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    ideal = _dcg(ideal_rels)
    return _dcg(retrieved_rels) / ideal if ideal > 0 else 0.0


def test_ndcg_at_10_above_baseline(searcher, test_queries, qrels):
    """
    nDCG@10 trên 50 test queries phải >= 0.28
    (baseline Pyserini BEIR BM25 cho NFCorpus là ~0.32)
    """
    test_qids = set(qrels.keys())
    sample = [q for q in test_queries if q["_id"] in test_qids][:50]
    assert sample, "No test queries with qrels found"

    scores = []
    for q in sample:
        hits = searcher.search(q["text"], k=10)
        scores.append(_ndcg_at_k(hits, qrels.get(q["_id"], {}), 10))

    mean_ndcg = sum(scores) / len(scores)
    print(f"\n  Mean nDCG@10 over {len(sample)} queries: {mean_ndcg:.4f}")
    assert mean_ndcg >= 0.28, f"nDCG@10 too low: {mean_ndcg:.4f} < 0.28"


def test_all_corpus_docs_indexed(searcher):
    """Một sample 20 doc IDs từ corpus phải tất cả có trong index."""
    import itertools
    doc_ids = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in itertools.islice(f, 20):
            if line.strip():
                doc_ids.append(json.loads(line)["_id"])

    missing = [d for d in doc_ids if searcher.doc(d) is None]
    assert not missing, f"These doc IDs not found in index: {missing}"
