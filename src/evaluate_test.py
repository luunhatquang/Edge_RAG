"""One-shot evaluation of BM25, BM25+RM3, dense and hybrid RRF on the NFCorpus test split.

Every configuration is read from the dev-split results; this script has no
retrieval hyperparameter options, so nothing can be tuned on test:

    BM25    k1, b                 results/rm3/best_config.json  ("bm25_baseline")
    RM3     fb_docs, fb_terms,    results/rm3/best_config.json  ("best")
            original_query_weight
    Dense   model, dense index    results/dense/dev_results.json
            dimension             results/hybrid/dev_comparison.json (dense run used for fusion)
    Hybrid  combination, k,       results/hybrid/best_config.json ("best")
            weights, depth

The BM25 k1/b in the dense results must match the RM3 file, and every source
must say it was selected on the dev split; otherwise the run stops.

All four methods run on the same 323 test queries, one query at a time on a
single thread (after a warm-up on dev queries), so latency is measured on this
machine: BM25 and RM3 with LuceneSearcher, dense = query encoding + exact
cosine search, hybrid = its retrievers + RRF fusion, run one after the other.
Metrics come from the same trec_eval (TrecEvaluator); per-query NDCG@10 is
compared with paired t-tests and Wilcoxon tests.

Outputs (in --out-dir, default results/test; refused if present unless --overwrite):
  test_results.json            metrics, significance, latency, memory, locked configs, checksums
  {bm25,rm3,dense,hybrid}_test.run

Usage:
    python -m src.evaluate_test
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.bm25.build_index import DEFAULT_INDEX_DIR as BM25_INDEX_DIR
from src.bm25.build_index import git_commit, software_versions
from src.bm25.evaluate import METRICS, TrecEvaluator, load_qrels, load_queries, write_trec_run
from src.bm25.evaluate_rm3 import paired_test, percentile, rss_mb
from src.bm25.tune import DEFAULT_DATA_DIR, file_sha256, load_split
from src.dense.build_index import load_model, versions
from src.dense.evaluate_dense import DenseIndex, Hit
from src.hybrid.rrf import rrf, weights_for

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DEFAULT_RM3_CONFIG = RESULTS / "rm3" / "best_config.json"
DEFAULT_DENSE_RESULTS = RESULTS / "dense" / "dev_results.json"
DEFAULT_HYBRID_CONFIG = RESULTS / "hybrid" / "best_config.json"
DEFAULT_HYBRID_COMPARISON = RESULTS / "hybrid" / "dev_comparison.json"
DEFAULT_OUT_DIR = RESULTS / "test"
METHODS = ("bm25", "rm3", "dense", "hybrid")


def read_json(path: Path, what: str) -> dict:
    if not path.exists():
        sys.exit(f"[test] {what} not found: {path}. Run the dev stage that writes it first.")
    return json.loads(path.read_text())


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def lock_configs(a: argparse.Namespace) -> dict:
    """Read the dev-selected configurations; stop on anything inconsistent."""
    rm3_file = read_json(a.rm3_config, "RM3 best config")
    dense_file = read_json(a.dense_results, "Dense dev results")
    hybrid_file = read_json(a.hybrid_config, "Hybrid best config")
    for name, rec in (("RM3", rm3_file), ("dense", dense_file), ("hybrid", hybrid_file)):
        if rec.get("split") != "dev":
            sys.exit(f"[test] The {name} configuration was selected on split {rec.get('split')!r}, not 'dev'.")

    base, best = rm3_file["bm25_baseline"], rm3_file["best"]
    dense_bm25 = dense_file["methods"].get("bm25")
    if dense_bm25 and (dense_bm25["k1"], dense_bm25["b"]) != (base["k1"], base["b"]):
        sys.exit(f"[test] BM25 k1/b differ between {a.rm3_config.name} ({base['k1']}, {base['b']}) and "
                 f"{a.dense_results.name} ({dense_bm25['k1']}, {dense_bm25['b']}).")

    dim = None
    if a.hybrid_comparison.exists():
        dense_input = json.loads(a.hybrid_comparison.read_text()).get("inputs", {}).get("dense", {})
        m = re.search(r"dense_dev_(\d+)\.run$", dense_input.get("path", ""))
        dim = int(m.group(1)) if m else None
    if dim is None:
        dim = dense_file["dense_index"]["full_dim"]
    if f"dense_{dim}" not in dense_file["methods"]:
        sys.exit(f"[test] Dense dimension {dim} was not evaluated on dev ({a.dense_results}).")

    h = hybrid_file["best"]
    return {
        "bm25": {"k1": base["k1"], "b": base["b"]},
        "rm3": {"k1": best["k1"], "b": best["b"], "fb_docs": best["fb_docs"], "fb_terms": best["fb_terms"],
                "original_query_weight": best["original_query_weight"], "config_id": best["config_id"]},
        "dense": {"model": dense_file["model"]["name"], "revision": dense_file["model"].get("revision"),
                  "max_seq_length": dense_file["model"]["max_seq_length"],
                  "index_dir": dense_file["dense_index"]["path"], "dim": dim},
        "hybrid": {"combo": h["combo"], "k": h["k"], "weights": h["weights"], "depth": h["depth"],
                   "config_id": h["config_id"]},
        "sources": {os.path.relpath(p, REPO_ROOT): file_sha256(p)
                    for p in (a.rm3_config, a.dense_results, a.hybrid_config, a.hybrid_comparison) if p.exists()},
    }


def load_test(data_dir: Path) -> tuple[list[str], list[str], dict, Path]:
    qrels_path = data_dir / "qrels" / "test.tsv"
    if not qrels_path.exists():
        sys.exit(f"[test] {qrels_path} not found.")
    qrels = load_qrels(qrels_path)
    texts = {q["_id"]: q["text"] for q in load_queries(data_dir / "queries.jsonl")}
    missing = [q for q in qrels if q not in texts]
    if missing:
        sys.exit(f"[test] {len(missing)} test query IDs have no text, e.g. {missing[:5]}")
    qids = sorted(qrels)
    return qids, [texts[q] for q in qids], qrels, qrels_path


def summary(values: list[float]) -> dict:
    return {"mean_ms": round(statistics.fmean(values), 3), "p50_ms": round(percentile(values, 50), 3),
            "p95_ms": round(percentile(values, 95), 3)}


def evaluate(a: argparse.Namespace) -> dict:
    out = a.out_dir
    outputs = [out / "test_results.json", *(out / f"{m}_test.run" for m in METHODS)]
    existing = [p for p in outputs if p.exists()]
    if existing and not a.overwrite:
        sys.exit(f"[test] {existing[0]} already exists; the test split is meant to be evaluated once. "
                 "Pass --overwrite to replace it, or choose another --out-dir.")

    cfg = lock_configs(a)
    combo = tuple(cfg["hybrid"]["combo"])
    unknown = set(combo) - {"bm25", "rm3", "dense"}
    if unknown:
        sys.exit(f"[test] Hybrid combination uses runs this script cannot reproduce: {sorted(unknown)}")
    qids, texts, qrels, qrels_path = load_test(a.data_dir)
    dev_qids, dev_texts, _, _ = load_split(a.data_dir, "dev")  # also checks dev/test separation
    print(f"[test] {len(qids)} test queries; locked configs: BM25 k1={cfg['bm25']['k1']} b={cfg['bm25']['b']} | "
          f"RM3 {cfg['rm3']['config_id']} | dense {cfg['dense']['model']} dim {cfg['dense']['dim']} | "
          f"hybrid {cfg['hybrid']['config_id']}")

    from pyserini.search.lucene import LuceneSearcher

    hits = max(a.hits, cfg["hybrid"]["depth"])
    t0 = time.perf_counter()
    searcher = LuceneSearcher(str(a.bm25_index_dir))
    lucene_init = time.perf_counter() - t0
    dense_index = DenseIndex(resolve(cfg["dense"]["index_dir"]), [cfg["dense"]["dim"]])
    model, model_load = load_model(cfg["dense"]["model"], cfg["dense"]["revision"],
                                   cfg["dense"]["max_seq_length"], a.threads)

    def lexical(name, text):
        c = cfg[name]
        searcher.set_bm25(c["k1"], c["b"])
        if name == "rm3":
            searcher.set_rm3(fb_terms=c["fb_terms"], fb_docs=c["fb_docs"],
                             original_query_weight=c["original_query_weight"])
        else:
            searcher.unset_rm3()
        return [Hit(h.docid, float(h.score)) for h in searcher.search(text, k=hits)]

    def dense(text):
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        return dense_index.search(vec, cfg["dense"]["dim"], hits)

    retrieve = {"bm25": lambda t: lexical("bm25", t), "rm3": lambda t: lexical("rm3", t), "dense": dense}
    for text in dev_texts[:a.warmup_queries]:  # warm-up on dev queries, never test
        for fn in retrieve.values():
            fn(text)

    results = {m: {} for m in METHODS}
    latency = {m: [] for m in METHODS}
    fusion_ms = []
    weights = cfg["hybrid"]["weights"]
    for qid, text in zip(qids, texts):
        for name in ("bm25", "rm3", "dense"):
            t = time.perf_counter()
            results[name][qid] = retrieve[name](text)
            latency[name].append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        runs = {n: {qid: [h.docid for h in results[n][qid]]} for n in combo}
        results["hybrid"][qid] = rrf(runs, weights, cfg["hybrid"]["k"], cfg["hybrid"]["depth"], [qid])[qid]
        fusion_ms.append((time.perf_counter() - t) * 1000)
        latency["hybrid"].append(sum(latency[n][-1] for n in combo) + fusion_ms[-1])
    searcher.close()

    evaluator = TrecEvaluator(qrels)
    try:
        metrics = {m: evaluator.evaluate(results[m]) for m in METHODS}
        per_query = {m: evaluator.per_query(results[m]) for m in METHODS}
    finally:
        evaluator.close()
    for m in METHODS:
        write_trec_run(results[m], out / f"{m}_test.run", tag=f"{m}_test")

    significance = {f"hybrid_vs_{m}": paired_test(per_query[m], per_query["hybrid"]) for m in ("bm25", "rm3", "dense")}
    for x, y in itertools.combinations(("bm25", "rm3", "dense"), 2):
        significance[f"{y}_vs_{x}"] = paired_test(per_query[x], per_query[y])

    record = {
        "split": "test",
        "data": {"qrels": os.path.relpath(qrels_path, REPO_ROOT), "qrels_sha256": file_sha256(qrels_path),
                 "num_queries": len(qids), "dev_queries_disjoint": not (set(qids) & set(dev_qids))},
        "locked_configs": cfg,
        "metrics": metrics,
        "significance": significance,
        "latency": {**{m: summary(latency[m]) for m in METHODS}, "hybrid_fusion": summary(fusion_ms),
                    "note": "single thread, sequential per query after warm-up on dev queries; hybrid = sum of its "
                            "retrievers' latency for that query + fusion (retrievers not run in parallel)"},
        "cost": {"lucene_searcher_init_seconds": round(lucene_init, 4), "dense_model_load_seconds": round(model_load, 3),
                 "dense_matrix_bytes": dense_index.matrices[cfg["dense"]["dim"]].nbytes,
                 "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                 "rss_mb_at_end": round(rss_mb(), 1), "torch_threads": a.threads,
                 "memory_note": "RSS covers the whole process: JVM + Lucene, PyTorch + embedding model, embeddings."},
        "versions": {**versions(), **software_versions()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overwrote_previous": bool(existing),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_results.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"\n[test] Test results ({len(qids)} queries)")
    print(f"  {'method':<8}" + "".join(f"{m:>12}" for m in METRICS) + f"{'ms/query':>10}")
    for m in METHODS:
        print(f"  {m:<8}" + "".join(f"{metrics[m][k]:>12.4f}" for k in METRICS)
              + f"{record['latency'][m]['mean_ms']:>10.2f}")
    for key, s in significance.items():
        p = s["paired_t_p_value"]
        print(f"  {key}: ΔNDCG@10={s['mean_difference']:+.4f}, wins/losses/ties={s['wins']}/{s['losses']}/"
              f"{s['ties']}, paired t p=" + (f"{p:.4g}" if p is not None else "n/a"))
    print(f"[test] Outputs in {out}")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--bm25-index-dir", type=Path, default=BM25_INDEX_DIR)
    p.add_argument("--rm3-config", type=Path, default=DEFAULT_RM3_CONFIG)
    p.add_argument("--dense-results", type=Path, default=DEFAULT_DENSE_RESULTS)
    p.add_argument("--hybrid-config", type=Path, default=DEFAULT_HYBRID_CONFIG)
    p.add_argument("--hybrid-comparison", type=Path, default=DEFAULT_HYBRID_COMPARISON)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hits", type=int, default=1000, help="Hits per query for every method (>= hybrid depth)")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads for query encoding")
    p.add_argument("--warmup-queries", type=int, default=20, help="Dev queries used to warm up (not scored)")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing test evaluation")
    a = p.parse_args(argv)
    if a.hits < 1 or a.threads < 1 or a.warmup_queries < 0:
        p.error("need --hits >= 1, --threads >= 1, --warmup-queries >= 0")
    if not a.bm25_index_dir.exists():
        p.error(f"BM25 index {a.bm25_index_dir} not found")
    return a


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
