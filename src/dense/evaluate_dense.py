"""Evaluate dense retrieval (default: bekko-embedding-v1-a8m) against BM25 on the NFCorpus dev split.

Reuses the existing pieces:
  - ``src.bm25.tune.load_split``          dev/train queries + qrels; the test split
                                          is not selectable and dev/test overlap aborts;
  - ``src.bm25.evaluate.TrecEvaluator``   trec_eval 9.0.4 metrics and per-query NDCG@10;
  - ``src.bm25.evaluate_rm3.paired_test`` paired t-test / Wilcoxon / wins-losses-ties;
  - the dense index from ``src.dense.build_index`` and the BM25 Lucene index.

Queries are processed one at a time on ``--threads`` CPU threads (default 1,
like a small edge device): each query is encoded once, then scored against
the index at every requested Matryoshka dimension (``--dims``; truncated
vectors are re-normalised). Query encoding and search are timed separately.
BM25 (k1=1.7, b=0.4 by default) runs on the same queries for comparison.

Outputs (in --out-dir, default results/dense; refused if present unless --overwrite):
  dev_results.json        metrics, deltas and significance vs BM25, latency, memory, versions
  dense_dev_<dim>.run     TREC run for each evaluated dimension

Usage:
    python -m src.dense.evaluate_dense
    python -m src.dense.evaluate_dense --dims 384,256,128,64
    python -m src.dense.evaluate_dense --model models/bekko-embedding-v1-a8m --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

from src.bm25.build_index import DEFAULT_INDEX_DIR as BM25_INDEX_DIR
from src.bm25.build_index import git_commit
from src.bm25.evaluate import METRICS, TrecEvaluator, write_trec_run
from src.bm25.evaluate_rm3 import paired_test, percentile, rss_mb
from src.bm25.tune import DEFAULT_DATA_DIR, file_sha256, load_split
from src.dense.build_index import DEFAULT_INDEX_DIR, load_model, model_info, versions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "dense"

Hit = namedtuple("Hit", "docid score")


class DenseIndex:
    """L2-normalised document embeddings with exact cosine top-k search."""

    def __init__(self, index_dir: Path, dims: list[int]):
        import numpy as np

        self.np = np
        for name in ("embeddings.npy", "docids.json", "manifest.json"):
            if not (index_dir / name).exists():
                sys.exit(f"[dense] {index_dir / name} not found. Run `python -m src.dense.build_index` first.")
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        self.docids = json.loads((index_dir / "docids.json").read_text())
        full = np.load(index_dir / "embeddings.npy", mmap_mode="r")
        if full.shape[0] != len(self.docids):
            sys.exit(f"[dense] {index_dir}: {full.shape[0]} embeddings but {len(self.docids)} document IDs")
        self.full_dim = full.shape[1]
        bad = [d for d in dims if not 1 <= d <= self.full_dim]
        if bad:
            sys.exit(f"[dense] --dims {bad} outside 1..{self.full_dim}")
        # One contiguous, re-normalised matrix per dimension (Matryoshka truncation).
        self.matrices = {d: self._normalise(np.ascontiguousarray(full[:, :d], dtype=np.float32)) for d in dims}

    def _normalise(self, m):
        norms = self.np.linalg.norm(m, axis=-1, keepdims=True)
        return m / self.np.maximum(norms, 1e-12)

    def search(self, query_vec, dim: int, k: int) -> list[Hit]:
        np = self.np
        q = self._normalise(np.asarray(query_vec[:dim], dtype=np.float32))
        scores = self.matrices[dim] @ q
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [Hit(self.docids[i], float(scores[i])) for i in top]


def latency_summary(values: list[float]) -> dict:
    return {"mean_ms": round(statistics.fmean(values), 3), "p50_ms": round(percentile(values, 50), 3),
            "p95_ms": round(percentile(values, 95), 3)}


def run_bm25(index_dir: Path, qids: list[str], texts: list[str], k1: float, b: float, hits: int, warmup: int):
    from pyserini.search.lucene import LuceneSearcher

    t0 = time.perf_counter()
    searcher = LuceneSearcher(str(index_dir))
    init_s = time.perf_counter() - t0
    searcher.set_bm25(k1, b)
    for text in texts[:warmup]:
        searcher.search(text, k=hits)
    results, lat = {}, []
    for qid, text in zip(qids, texts):
        t = time.perf_counter()
        results[qid] = searcher.search(text, k=hits)
        lat.append((time.perf_counter() - t) * 1000)
    searcher.close()
    return results, lat, init_s


def evaluate(args: argparse.Namespace) -> dict:
    out = args.out_dir
    outputs = [out / "dev_results.json", *(out / f"dense_dev_{d}.run" for d in args.dims)]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        sys.exit(f"[dense] {existing[0]} already exists. Pass --overwrite, or choose another --out-dir.")

    qids, texts, qrels, qrels_path = load_split(args.data_dir, args.split)
    index = DenseIndex(args.index_dir, args.dims)
    built_with = index.manifest["model"]["name"]
    model_name = args.model or built_with
    warnings = []
    if model_name != built_with:
        warnings.append(f"Query model {model_name!r} differs from the model the index was built with "
                        f"({built_with!r}); this is only valid if they are the same weights.")
    model, load_s = load_model(model_name, args.revision or index.manifest["model"].get("revision"),
                               index.manifest["model"]["max_seq_length"], args.threads)
    if model.get_embedding_dimension() != index.full_dim:
        sys.exit(f"[dense] Model dimension {model.get_embedding_dimension()} != index {index.full_dim}")
    print(f"[dense] {len(qids)} {args.split} queries; index {args.index_dir} ({len(index.docids)} docs, "
          f"dim {index.full_dim}); model loaded in {load_s:.2f}s; dims {args.dims}; {args.threads} thread(s)")

    def encode(text):
        return model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]

    for text in texts[:args.warmup_queries]:
        encode(text)
    encode_lat, search_lat = [], {d: [] for d in args.dims}
    dense_results = {d: {} for d in args.dims}
    t_all = time.perf_counter()
    for qid, text in zip(qids, texts):
        t = time.perf_counter()
        vec = encode(text)
        encode_lat.append((time.perf_counter() - t) * 1000)
        for d in args.dims:
            t = time.perf_counter()
            dense_results[d][qid] = index.search(vec, d, args.hits)
            search_lat[d].append((time.perf_counter() - t) * 1000)
    dense_total_s = time.perf_counter() - t_all
    print(f"[dense] Encoded and searched {len(qids)} queries in {dense_total_s:.1f}s "
          f"(encode {statistics.fmean(encode_lat):.2f} ms/query)")

    evaluator = TrecEvaluator(qrels)
    try:
        methods, per_query = {}, {}
        if not args.no_bm25:
            bm25_results, bm25_lat, bm25_init = run_bm25(args.bm25_index_dir, qids, texts, args.k1, args.b,
                                                         args.hits, args.warmup_queries)
            methods["bm25"] = {"k1": args.k1, "b": args.b, "metrics": evaluator.evaluate(bm25_results),
                               "latency": latency_summary(bm25_lat), "searcher_init_seconds": round(bm25_init, 4)}
            per_query["bm25"] = evaluator.per_query(bm25_results)
        for d in args.dims:
            role = f"dense_{d}"
            total = [e + s for e, s in zip(encode_lat, search_lat[d])]
            methods[role] = {
                "dim": d,
                "metrics": evaluator.evaluate(dense_results[d]),
                "latency": {"total": latency_summary(total), "query_encode": latency_summary(encode_lat),
                            "search": latency_summary(search_lat[d])},
                "index_matrix_bytes": index.matrices[d].nbytes,
            }
            per_query[role] = evaluator.per_query(dense_results[d])
            write_trec_run(dense_results[d], out / f"dense_dev_{d}.run", tag=f"bekko_a8m_dim{d}")
    finally:
        evaluator.close()

    deltas, significance = {}, {}
    if "bm25" in methods:
        base = methods["bm25"]["metrics"]
        for d in args.dims:
            role = f"dense_{d}"
            deltas[role] = {m: {"absolute": methods[role]["metrics"][m] - base[m],
                                "relative_pct": 100 * (methods[role]["metrics"][m] - base[m]) / base[m]
                                if base[m] else None} for m in METRICS}
            deltas[role]["latency_mean_ratio"] = (methods[role]["latency"]["total"]["mean_ms"]
                                                  / methods["bm25"]["latency"]["mean_ms"])
            significance[role] = paired_test(per_query["bm25"], per_query[role])

    record = {
        "split": args.split,
        "methods": methods,
        "delta_vs_bm25": deltas,
        "significance_vs_bm25": significance,
        "model": {**model_info(model, model_name, args.revision), "load_seconds": round(load_s, 3),
                  "index_built_with": index.manifest["model"]},
        "dense_index": {"path": os.path.relpath(args.index_dir, REPO_ROOT), "num_docs": len(index.docids),
                        "full_dim": index.full_dim, "corpus_sha256": index.manifest.get("corpus_sha256"),
                        "encode": index.manifest.get("encode"),
                        "documents_truncated": index.manifest.get("documents_truncated")},
        "data": {"qrels": os.path.relpath(qrels_path, REPO_ROOT), "qrels_sha256": file_sha256(qrels_path),
                 "num_queries": len(qids), "hits_per_query": args.hits},
        "cost": {
            "threads": args.threads,
            "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "rss_mb_at_end": round(rss_mb(), 1),
            "dense_total_seconds": round(dense_total_s, 3),
            "timing": "per query, sequential, after warm-up; dense total = query encoding + exact cosine "
                      "search; BM25 = LuceneSearcher.search; trec_eval excluded",
            "memory_note": "RSS covers the whole process (Python, PyTorch, model, embeddings, and the JVM when "
                           "BM25 is evaluated).",
        },
        "versions": versions(),
        "git_commit": git_commit(),
        "warnings": warnings,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if "bm25" in methods:
        from src.bm25.build_index import software_versions

        record["versions"].update({k: v for k, v in software_versions().items() if k not in record["versions"]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_results.json").write_text(json.dumps(record, indent=2) + "\n")
    print_summary(record)
    return record


def print_summary(record: dict) -> None:
    print(f"\n[dense] Dev results ({record['split']}, {record['data']['num_queries']} queries)")
    print(f"  {'method':<12}" + "".join(f"{m:>12}" for m in METRICS) + f"{'ms/query':>10}")
    for role, m in record["methods"].items():
        ms = m["latency"]["mean_ms"] if role == "bm25" else m["latency"]["total"]["mean_ms"]
        print(f"  {role:<12}" + "".join(f"{m['metrics'][k]:>12.4f}" for k in METRICS) + f"{ms:>10.2f}")
    for role, s in record["significance_vs_bm25"].items():
        p = s["paired_t_p_value"]
        print(f"  {role} vs bm25: ΔNDCG@10={record['delta_vs_bm25'][role]['ndcg@10']['absolute']:+.4f}, "
              f"wins/losses/ties={s['wins']}/{s['losses']}/{s['ties']}, paired t p="
              + (f"{p:.4f}" if p is not None else "n/a"))
    for w in record["warnings"]:
        print(f"[dense] Warning: {w}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Model ID or local dir (default: the one the index was built with)")
    p.add_argument("--revision", default=None)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Dense index directory")
    p.add_argument("--dims", default="384", help="Comma-separated Matryoshka dimensions, e.g. 384,256,128,64")
    p.add_argument("--split", choices=["dev", "train"], default="dev", help="Evaluation split (test is not used)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--bm25-index-dir", type=Path, default=BM25_INDEX_DIR)
    p.add_argument("--k1", type=float, default=1.7)
    p.add_argument("--b", type=float, default=0.4)
    p.add_argument("--no-bm25", action="store_true", help="Skip the BM25 comparison")
    p.add_argument("--hits", type=int, default=1000)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads for query encoding (default 1)")
    p.add_argument("--warmup-queries", type=int, default=20)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="Replace existing results in --out-dir")
    a = p.parse_args(argv)
    try:
        a.dims = sorted({int(x) for x in a.dims.split(",") if x.strip()}, reverse=True)
    except ValueError:
        p.error("--dims must be comma-separated integers")
    if not a.dims or a.hits < 1 or a.threads < 1 or a.warmup_queries < 0:
        p.error("need at least one dim, --hits >= 1, --threads >= 1, --warmup-queries >= 0")
    if not a.no_bm25 and not a.bm25_index_dir.exists():
        p.error(f"BM25 index {a.bm25_index_dir} not found (or pass --no-bm25)")
    return a


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
