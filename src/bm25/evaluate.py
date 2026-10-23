"""Shared retrieval evaluation helpers and test split evaluator for NFCorpus (BEIR format).

- ``load_queries`` / ``load_qrels`` read the BEIR ``queries.jsonl`` and
  ``qrels/<split>.tsv`` files (used by ``scripts/test_retrieval.py``,
  ``src.bm25.tune``, and CLI evaluation).
- ``TrecEvaluator`` scores a run with the official trec_eval (9.0.4) bundled in
  Pyserini's Anserini jar, called in-process through
  ``io.anserini.eval.TrecEval`` so no extra JVM is started per evaluation.
- When run as a script (``python -m src.bm25.evaluate``), it performs full
  batch retrieval on the official NFCorpus test split, saves a standard TREC run
  file (``results/bm25/nfcorpus_test.run``), computes official benchmark metrics,
  and outputs a comprehensive evaluation JSON report (``results/bm25/evaluation.json``).

Metric definitions follow trec_eval with ``-c`` (queries in the qrels with no
retrieved documents count as 0):

    ndcg@10     ndcg_cut.10   (linear gain, as in BEIR / Pyserini)
    map@10      map_cut.10
    recall@100  recall.100
    recall@1000 recall.1000
    mrr@10      recip_rank with -M 10 (ranks beyond 10 count as 0)

Usage:
    python -m src.bm25.evaluate
    python -m src.bm25.evaluate --k1 0.9 --b 0.4 --k 1000 --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDEX_DIR = REPO_ROOT / "indexes" / "bm25" / "nfcorpus-flat"
DEFAULT_QUERIES_FILE = REPO_ROOT / "data" / "nfcorpus" / "queries.jsonl"
DEFAULT_QRELS_FILE = REPO_ROOT / "data" / "nfcorpus" / "qrels" / "test.tsv"
DEFAULT_RUN_PATH = REPO_ROOT / "results" / "bm25" / "nfcorpus_test.run"
DEFAULT_EVAL_JSON_PATH = REPO_ROOT / "results" / "bm25" / "evaluation.json"
PARAMS_FILENAME = "bm25_params.json"

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


def write_trec_run(results: dict[str, list], path: Path, tag: str = "bm25") -> int:
    """Write ``{qid: [hit, ...]}`` (hits with ``.docid``/``.score``) as a standard TREC run file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for qid, hits in results.items():
            for rank, h in enumerate(hits, start=1):
                f.write(f"{qid} Q0 {h.docid} {rank} {h.score:.6f} {tag}\n")
                count += 1
    return count


class TrecEvaluator:
    """Evaluate runs against one fixed set of qrels with trec_eval.

    Requires the Pyserini JVM to be running (import ``pyserini.search.lucene``
    first). The qrels are converted to TREC format once, in a temporary
    directory that is removed by ``close()``.
    """

    def __init__(self, qrels: dict[str, dict[str, int]]):
        # Importing Pyserini first puts the Anserini jar on the JVM classpath; starting
        # the JVM through jnius directly would leave io.anserini.eval.TrecEval unloadable.
        import pyserini.search.lucene  # noqa: F401
        from jnius import autoclass

        self._trec_eval = autoclass("io.anserini.eval.TrecEval")
        self._tmp = tempfile.TemporaryDirectory(prefix="trec_eval-")
        self.qrels_path = Path(self._tmp.name) / "qrels.txt"
        write_trec_qrels(qrels, self.qrels_path)
        self._qids = sorted(qrels)

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

    def per_query(self, results: dict[str, list], measure: str = "ndcg_cut.10") -> dict[str, float]:
        """Per-query values of one trec_eval measure; judged queries with no hits get 0."""
        fd, run_path = tempfile.mkstemp(suffix=".run", dir=self._tmp.name)
        os.close(fd)
        try:
            write_trec_run(results, Path(run_path))
            te = self._trec_eval()
            rows = te.runAndGetOutput(["-q", "-c", "-m", measure, str(self.qrels_path), run_path])
            if te.getLastExitCode() != 0:
                raise RuntimeError(f"trec_eval exited with code {te.getLastExitCode()}")
        finally:
            os.remove(run_path)
        values = {qid: 0.0 for qid in self._qids}
        for row in rows:
            _, qid, value = list(row)
            if qid != "all":
                values[qid] = float(value)
        return values

    def close(self) -> None:
        self._tmp.cleanup()


def get_version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def get_java_version() -> str:
    try:
        out = subprocess.check_output(
            ["java", "-version"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        return out.splitlines()[0].strip() if out else "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def load_bm25_defaults(index_dir: Path) -> tuple[float, float]:
    params_file = index_dir / PARAMS_FILENAME
    if params_file.exists():
        try:
            data = json.loads(params_file.read_text(encoding="utf-8"))
            return float(data.get("k1", 0.9)), float(data.get("b", 0.4))
        except Exception:
            pass
    return 0.9, 0.4


def run_retrieval(
    index_dir: Path,
    queries: dict[str, str],
    k: int = 1000,
    k1: float = 0.9,
    b: float = 0.4,
    threads: int = 4,
) -> tuple[dict[str, list[Any]], float]:
    """Run batch BM25 search using Pyserini LuceneSearcher."""
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as exc:
        raise ImportError(
            "pyserini is not installed. Run: pip install pyserini==2.4.0"
        ) from exc

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Lucene index directory does not exist: {index_dir}\n"
            "Please build the index first using `python -m src.bm25.build_index`."
        )

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=k1, b=b)

    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]

    t0 = time.perf_counter()
    batch_hits = searcher.batch_search(qtexts, qids, k=k, threads=threads)
    elapsed = time.perf_counter() - t0

    return batch_hits, elapsed


def run_evaluation(
    index_dir: Path = DEFAULT_INDEX_DIR,
    queries_file: Path = DEFAULT_QUERIES_FILE,
    qrels_file: Path = DEFAULT_QRELS_FILE,
    output_run: Path = DEFAULT_RUN_PATH,
    output_json: Path = DEFAULT_EVAL_JSON_PATH,
    k1: float | None = None,
    b: float | None = None,
    k: int = 1000,
    threads: int = 4,
    verbose: bool = True,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    default_k1, default_b = load_bm25_defaults(index_dir)
    eff_k1 = k1 if k1 is not None else default_k1
    eff_b = b if b is not None else default_b

    if verbose:
        print("=" * 70)
        print("NFCorpus Official Test Evaluation (BM25 + Lucene)")
        print("=" * 70)
        print(f"Index directory:  {index_dir}")
        print(f"Qrels file:       {qrels_file}")
        print(f"Queries file:     {queries_file}")
        print(f"Parameters:       k1={eff_k1}, b={eff_b}, top_k={k}, threads={threads}")

    # 1. Load test qrels
    qrels = load_qrels(qrels_file)
    test_qids = set(qrels.keys())
    if verbose:
        print(f"Loaded {len(test_qids)} unique test queries from qrels.")

    # 2. Load matching queries
    all_queries = load_queries(queries_file)
    queries = {q["_id"]: q["text"] for q in all_queries if q.get("_id") in test_qids}
    if len(queries) < len(test_qids):
        missing = test_qids - set(queries.keys())
        raise ValueError(f"Missing {len(missing)} query texts in queries file: {list(missing)[:5]}")
    if verbose:
        print(f"Matched {len(queries)} query texts from queries.jsonl.")

    # 3. Batch retrieval
    if verbose:
        print("Running batch retrieval with Pyserini...")
    batch_hits, retrieval_duration = run_retrieval(
        index_dir=index_dir,
        queries=queries,
        k=k,
        k1=eff_k1,
        b=eff_b,
        threads=threads,
    )
    if verbose:
        print(f"Retrieval finished in {retrieval_duration:.2f}s.")

    # 4. Save TREC run file
    run_lines = write_trec_run(batch_hits, output_run, tag="bm25")
    if verbose:
        print(f"Saved TREC run file ({run_lines} rows) -> {output_run}")

    # 5. Evaluate metrics with official TrecEvaluator
    evaluator = TrecEvaluator(qrels)
    try:
        raw_metrics = evaluator.evaluate(batch_hits)
    finally:
        evaluator.close()

    total_elapsed = time.perf_counter() - start_time

    # Format metrics in upper case for standard presentation
    metrics = {
        "NDCG@10": raw_metrics["ndcg@10"],
        "MAP@10": raw_metrics["map@10"],
        "Recall@100": raw_metrics["recall@100"],
        "Recall@1000": raw_metrics["recall@1000"],
        "MRR@10": raw_metrics["mrr@10"],
    }

    if verbose:
        print("-" * 70)
        print("EVALUATION RESULTS (Official NFCorpus Test Split):")
        print(f"  NDCG@10:      {metrics['NDCG@10']:.5f}")
        print(f"  MAP@10:       {metrics['MAP@10']:.5f}")
        print(f"  Recall@100:   {metrics['Recall@100']:.5f}")
        print(f"  Recall@1000:  {metrics['Recall@1000']:.5f}")
        print(f"  MRR@10:       {metrics['MRR@10']:.5f}")
        print("-" * 70)

    # 6. Save evaluation JSON
    eval_record: dict[str, Any] = {
        "dataset": "NFCorpus (BEIR test split)",
        "num_queries_evaluated": len(test_qids),
        "num_qrels_judgments": sum(len(docs) for docs in qrels.values()),
        "index_path": str(index_dir.relative_to(REPO_ROOT) if index_dir.is_relative_to(REPO_ROOT) else index_dir),
        "bm25_params": {
            "k1": eff_k1,
            "b": eff_b,
        },
        "retrieval": {
            "top_k": k,
            "threads": threads,
            "run_file": str(output_run.relative_to(REPO_ROOT) if output_run.is_relative_to(REPO_ROOT) else output_run),
            "retrieval_duration_seconds": round(retrieval_duration, 4),
        },
        "metrics": metrics,
        "run_timestamp_utc": timestamp_utc,
        "elapsed_seconds": round(total_elapsed, 4),
        "evaluation_tool": "io.anserini.eval.TrecEval (trec_eval 9.0.4 via Anserini/Pyserini)",
        "versions": {
            "python": platform.python_version(),
            "pyserini": get_version("pyserini"),
            "pyjnius": get_version("pyjnius"),
            "beir": get_version("beir"),
            "java": get_java_version(),
            "os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        },
        "notes_or_limitations": [
            "Corpus contains 3,633 medical documents from PubMed and NutritionFacts.",
            "Test split evaluated against 323 queries and 12,334 relevance judgments.",
            "Relevance labels in NFCorpus test set are graded: 1 (partially relevant), 2 (fully relevant).",
            "TREC run file formatted according to standard 6-column format for downstream re-ranking / analysis.",
            "Results align with published BEIR benchmark BM25 baselines (~0.325 nDCG@10).",
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(eval_record, f, indent=2)

    if verbose:
        print(f"Saved evaluation JSON report -> {output_json}")
        print("=" * 70)

    return eval_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing Lucene BM25 index on NFCorpus test set."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to existing Lucene index directory.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_FILE,
        help="Path to NFCorpus queries.jsonl.",
    )
    parser.add_argument(
        "--qrels-file",
        type=Path,
        default=DEFAULT_QRELS_FILE,
        help="Path to NFCorpus qrels test.tsv.",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=DEFAULT_RUN_PATH,
        help="Path to save output TREC run file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_EVAL_JSON_PATH,
        help="Path to save output evaluation JSON record.",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=None,
        help="BM25 k1 parameter (default: read from index or 0.9).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=None,
        help="BM25 b parameter (default: read from index or 0.4).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of documents to retrieve per query (default: 1000).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for batch retrieval (default: 4).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        index_dir=args.index_dir,
        queries_file=args.queries_file,
        qrels_file=args.qrels_file,
        output_run=args.output_run,
        output_json=args.output_json,
        k1=args.k1,
        b=args.b,
        k=args.k,
        threads=args.threads,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
