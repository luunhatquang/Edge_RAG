"""Shared retrieval evaluation helpers for NFCorpus (BEIR format).

- ``load_queries`` / ``load_qrels`` read the BEIR ``queries.jsonl`` and
  ``qrels/<split>.tsv`` files (used by ``scripts/test_retrieval.py`` and
  ``src.bm25.tune``).
- ``TrecEvaluator`` scores a run with the official trec_eval (9.0.4) bundled in
  Pyserini's Anserini jar, called in-process through
  ``io.anserini.eval.TrecEval`` so no extra JVM is started per evaluation.

Metric definitions follow trec_eval with ``-c`` (queries in the qrels with no
retrieved documents count as 0):

    ndcg@10     ndcg_cut.10   (linear gain, as in BEIR / Pyserini)
    map@10      map_cut.10
    recall@100  recall.100
    recall@1000 recall.1000
    mrr@10      recip_rank with -M 10 (ranks beyond 10 count as 0)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

METRICS = ("ndcg@10", "map@10", "recall@100", "recall@1000", "mrr@10")
_TREC_NAMES = {
    "ndcg_cut_10": "ndcg@10",
    "map_cut_10": "map@10",
    "recall_100": "recall@100",
    "recall_1000": "recall@1000",
    "recip_rank": "mrr@10",
}


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


def write_trec_qrels(qrels: dict[str, dict[str, int]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in qrels.items():
            for did, rel in docs.items():
                f.write(f"{qid} 0 {did} {rel}\n")


def write_trec_run(results: dict[str, list], path: Path, tag: str = "bm25") -> None:
    """Write ``{qid: [hit, ...]}`` (hits with ``.docid``/``.score``) as a TREC run file."""
    with open(path, "w", encoding="utf-8") as f:
        for qid, hits in results.items():
            for rank, h in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {h.docid} {rank} {h.score:.6f} {tag}\n")


class TrecEvaluator:
    """Evaluate runs against one fixed set of qrels with trec_eval.

    Requires the Pyserini JVM to be running (import ``pyserini.search.lucene``
    first). The qrels are converted to TREC format once, in a temporary
    directory that is removed by ``close()``.
    """

    def __init__(self, qrels: dict[str, dict[str, int]]):
        from jnius import autoclass

        self._trec_eval = autoclass("io.anserini.eval.TrecEval")
        self._tmp = tempfile.TemporaryDirectory(prefix="trec_eval-")
        self.qrels_path = Path(self._tmp.name) / "qrels.txt"
        write_trec_qrels(qrels, self.qrels_path)

    def _run(self, args: list[str]) -> dict[str, float]:
        te = self._trec_eval()
        rows = te.runAndGetOutput(args)
        if te.getLastExitCode() != 0:
            raise RuntimeError(f"trec_eval exited with code {te.getLastExitCode()}: {args}")
        out = {}
        for row in rows:
            name, qid, value = list(row)
            if qid == "all" and name in _TREC_NAMES:
                out[_TREC_NAMES[name]] = float(value)
        return out

    def evaluate(self, results: dict[str, list]) -> dict[str, float]:
        fd, run_path = tempfile.mkstemp(suffix=".run", dir=self._tmp.name)
        os.close(fd)
        try:
            write_trec_run(results, Path(run_path))
            qrels = str(self.qrels_path)
            metrics = self._run(
                ["-c", "-m", "ndcg_cut.10", "-m", "map_cut.10", "-m", "recall.100,1000", qrels, run_path]
            )
            metrics.update(self._run(["-c", "-M", "10", "-m", "recip_rank", qrels, run_path]))
        finally:
            os.remove(run_path)
        missing = set(METRICS) - set(metrics)
        if missing:
            raise RuntimeError(f"trec_eval did not report {sorted(missing)}")
        return {m: metrics[m] for m in METRICS}

    def close(self) -> None:
        self._tmp.cleanup()
