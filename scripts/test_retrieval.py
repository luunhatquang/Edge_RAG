"""
test_retrieval.py — Kiểm tra BM25 index với dữ liệu NFCorpus thực.

Chạy một số queries từ queries.jsonl, in top-k hits, và
tính nDCG@10 trên test split của qrels (subset 50 queries).

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --queries 20 --k 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INDEX_DIR = REPO_ROOT / "indexes" / "bm25" / "nfcorpus-flat"
QUERIES_FILE = REPO_ROOT / "data" / "nfcorpus" / "queries.jsonl"
QRELS_FILE = REPO_ROOT / "data" / "nfcorpus" / "qrels" / "test.tsv"


def load_queries(path: Path, limit: int | None = None) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries[:limit] if limit else queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """qrels[query_id][doc_id] = relevance_score"""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[did] = score
    return qrels


def dcg(relevances: list[int]) -> float:
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(hits: list, qrel: dict[str, int], k: int) -> float:
    if not qrel:
        return 0.0
    retrieved_rels = [qrel.get(h.docid, 0) for h in hits[:k]]
    ideal_rels = sorted(qrel.values(), reverse=True)[:k]
    ideal = dcg(ideal_rels)
    if ideal == 0:
        return 0.0
    return dcg(retrieved_rels) / ideal


def run(num_queries: int, k: int, k1: float, b: float, verbose: bool) -> None:
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError:
        sys.exit("pyserini not found. Run: pip install pyserini==2.4.0")

    if not INDEX_DIR.exists():
        sys.exit(
            f"Index not found at {INDEX_DIR}.\n"
            "Run: python -m src.bm25.prepare_data --raw-dir data/nfcorpus\n"
            "     python -m src.bm25.build_index --prepared-dir data/processed/nfcorpus"
        )

    print(f"[retrieval] Loading index from {INDEX_DIR}")
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1, b)
    print(f"[retrieval] Index has {searcher.num_docs} documents, BM25(k1={k1}, b={b})")

    # Load queries filtered to those that have qrels (test split)
    all_queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)
    test_qids = set(qrels.keys())
    test_queries = [q for q in all_queries if q["_id"] in test_qids]

    if not test_queries:
        sys.exit("No test queries found that have qrels.")

    sample = test_queries[:num_queries]
    print(f"[retrieval] Evaluating {len(sample)}/{len(test_queries)} test queries at k={k}\n")

    ndcg_scores = []
    for q in sample:
        qid = q["_id"]
        text = q["text"]
        hits = searcher.search(text, k=k)
        qrel = qrels.get(qid, {})
        score = ndcg_at_k(hits, qrel, k)
        ndcg_scores.append(score)

        if verbose:
            print(f"  Query [{qid}]: {text[:70]}")
            print(f"    nDCG@{k} = {score:.4f} | top hits:")
            for i, h in enumerate(hits[:3]):
                rel = qrel.get(h.docid, 0)
                print(f"      {i+1}. {h.docid:10s}  score={h.score:.3f}  rel={rel}")
            print()

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    print("=" * 50)
    print(f"  Mean nDCG@{k} over {len(sample)} queries: {mean_ndcg:.4f}")
    print("=" * 50)

    searcher.close()
    return mean_ndcg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=int, default=50, help="Number of test queries to evaluate (default: 50)")
    p.add_argument("--k", type=int, default=10, help="Cutoff for nDCG (default: 10)")
    p.add_argument("--k1", type=float, default=0.9, help="BM25 k1")
    p.add_argument("--b", type=float, default=0.4, help="BM25 b")
    p.add_argument("--verbose", "-v", action="store_true", help="Print per-query hits")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.queries, args.k, args.k1, args.b, args.verbose)
