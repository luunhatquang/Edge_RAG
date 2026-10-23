"""Hybrid retrieval by weighted Reciprocal Rank Fusion (RRF), tuned on the NFCorpus dev split.

Fuses existing dev TREC runs instead of re-running retrieval:

    bm25   results/rm3/bm25_dev.run        (src.bm25.evaluate_rm3)
    rm3    results/rm3/rm3_dev.run         (src.bm25.evaluate_rm3, best RM3 on dev)
    dense  results/dense/dense_dev_384.run (src.dense.evaluate_dense)

Weighted RRF score of document d for a query:

    score(d) = sum_i  w_i / (k + rank_i(d))

where rank_i(d) is d's 1-based rank in run i (only the top --depth hits of each
run count; a document missing from a run gets nothing from it). For a
combination "a+b", w_a = w and w_b = 1 - w; with more runs the first gets w
and the rest share 1 - w equally.

Grid (defaults): combinations rm3+dense and bm25+dense, k in {10,20,40,60,100},
w in {0.3,0.4,0.5,0.6,0.7}. The best fusion is the one with the highest dev
NDCG@10. Each run file is checked to contain only dev queries (a run with test
query IDs aborts), and test qrels are never used for scoring.

Reuses ``tune.load_split`` (dev qrels, test overlap check), ``TrecEvaluator``
(trec_eval metrics, per-query NDCG@10), ``TrialLog`` (CSV checkpoint/resume)
and ``evaluate_rm3.paired_test``.

Outputs (in --out-dir, default results/hybrid):
  tuning_results.csv   one row per fusion configuration (--resume continues it)
  dev_comparison.json  inputs vs best hybrid: metrics, deltas, significance, latency estimate
  best_config.json     best fusion configuration on dev
  hybrid_dev.run       fused run of the best configuration

Usage:
    python -m src.hybrid.rrf
    python -m src.hybrid.rrf --combos rm3+dense --k 60 --weights 0.5
    python -m src.hybrid.rrf --run dense=results/dense/dense_dev_256.run --resume
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from collections import defaultdict, namedtuple
from datetime import datetime, timezone
from pathlib import Path

from src.bm25.build_index import git_commit
from src.bm25.evaluate import METRICS, TrecEvaluator, write_trec_run
from src.bm25.evaluate_rm3 import paired_test, percentile
from src.bm25.tune import DEFAULT_DATA_DIR, TrialLog, file_sha256, load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "hybrid"
DEFAULT_RUNS = {
    "bm25": REPO_ROOT / "results" / "rm3" / "bm25_dev.run",
    "rm3": REPO_ROOT / "results" / "rm3" / "rm3_dev.run",
    "dense": REPO_ROOT / "results" / "dense" / "dense_dev_384.run",
}
# Optional sources of per-query latency for the end-to-end estimate.
LATENCY_SOURCES = {
    "bm25": (REPO_ROOT / "results" / "rm3" / "dev_comparison.json", ("methods", "bm25", "latency", "latency_mean_ms")),
    "rm3": (REPO_ROOT / "results" / "rm3" / "dev_comparison.json", ("methods", "rm3_best", "latency", "latency_mean_ms")),
    "dense": (REPO_ROOT / "results" / "dense" / "dev_results.json", ("methods", "dense_384", "latency", "total", "mean_ms")),
}
OBJECTIVE = "ndcg@10"
FIELDS = ["trial", "config_id", "combo", "k", "weight_first", *METRICS, "num_queries", "fuse_ms_per_query",
          "status", "error", "split", "timestamp_utc"]

Hit = namedtuple("Hit", "docid score")


def read_run(path: Path) -> dict[str, list[str]]:
    """TREC run -> {qid: [docid, ...]} ordered by rank (score desc, as written)."""
    rows = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{path}:{n}: expected 6 columns, got {len(parts)}")
            qid, _, docid, rank, score, _ = parts
            rows[qid].append((int(rank), -float(score), docid))
    return {q: [d for _, _, d in sorted(r)] for q, r in rows.items()}


def weights_for(combo: tuple[str, ...], w: float) -> dict[str, float]:
    rest = (1 - w) / (len(combo) - 1)
    return {name: (w if i == 0 else rest) for i, name in enumerate(combo)}


def rrf(runs: dict[str, dict[str, list[str]]], weights: dict[str, float], k: int, depth: int,
        qids: list[str]) -> dict[str, list[Hit]]:
    fused = {}
    for qid in qids:
        scores: dict[str, float] = defaultdict(float)
        for name, w in weights.items():
            for rank, docid in enumerate(runs[name].get(qid, [])[:depth], start=1):
                scores[docid] += w / (k + rank)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:depth]
        fused[qid] = [Hit(d, s) for d, s in ranked]
    return fused


def config_id(combo: tuple[str, ...], k: int, w: float) -> str:
    return f"{'+'.join(combo)}:k={k}:w={w:.2f}"


def lookup(path: Path, keys: tuple[str, ...]):
    try:
        value = json.loads(path.read_text())
        for key in keys:
            value = value[key]
        return float(value)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class HybridTuning:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        a = args
        csv_path = a.out_dir / "tuning_results.csv"
        derived = [a.out_dir / n for n in ("dev_comparison.json", "best_config.json", "hybrid_dev.run")]
        if not csv_path.exists() and any(p.exists() for p in derived):
            sys.exit(f"[hybrid] {a.out_dir} has outputs but no tuning_results.csv; refusing to overwrite them.")

        self.qids, _, qrels, self.qrels_path = load_split(a.data_dir, a.split)
        test_path = a.data_dir / "qrels" / "test.tsv"
        test_qids = set()
        if test_path.exists():
            from src.bm25.evaluate import load_qrels

            test_qids = set(load_qrels(test_path))

        needed = sorted({name for combo in a.combos for name in combo})
        missing = [n for n in needed if n not in a.runs]
        if missing:
            sys.exit(f"[hybrid] No run file given for {missing}; pass --run NAME=PATH.")
        self.runs, self.run_info = {}, {}
        dev = set(self.qids)
        for name in needed:
            path = a.runs[name]
            if not path.exists():
                sys.exit(f"[hybrid] Run file for {name!r} not found: {path}")
            run = read_run(path)
            extra = set(run) - dev
            if extra & test_qids:
                sys.exit(f"[hybrid] {path} contains {len(extra & test_qids)} test query IDs; "
                         "only dev runs may be fused for tuning.")
            if extra:
                sys.exit(f"[hybrid] {path} contains {len(extra)} query IDs that are not in {a.split}.tsv, "
                         f"e.g. {sorted(extra)[:5]}")
            self.runs[name] = run
            self.run_info[name] = {"path": os.path.relpath(path, REPO_ROOT), "sha256": file_sha256(path),
                                   "queries": len(run), "missing_queries": len(dev - set(run)),
                                   "max_depth": max((len(v) for v in run.values()), default=0)}
            print(f"[hybrid] {name}: {path} ({len(run)}/{len(dev)} {a.split} queries, depth "
                  f"{self.run_info[name]['max_depth']})")

        self.evaluator = TrecEvaluator(qrels)
        self.log = TrialLog(csv_path, a.resume, fields=FIELDS, key_fn=lambda r: r["config_id"])
        self.warnings: list[str] = []
        for name, info in self.run_info.items():
            if info["missing_queries"]:
                self.warn(f"{name} run has no results for {info['missing_queries']} {a.split} queries.")

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
            print(f"[hybrid] Warning: {msg}")

    def plan(self) -> list[tuple[tuple[str, ...], int, float]]:
        a = self.args
        return [(c, k, w) for c in a.combos for k in a.k for w in a.weights]

    def run_trial(self, combo, k, w, label: str) -> None:
        a = self.args
        row = {"config_id": config_id(combo, k, w), "combo": "+".join(combo), "k": k, "weight_first": w,
               "num_queries": len(self.qids), "split": a.split, **{m: "" for m in METRICS}}
        try:
            t0 = time.perf_counter()
            fused = rrf(self.runs, weights_for(combo, w), k, a.depth, self.qids)
            fuse_ms = (time.perf_counter() - t0) * 1000 / len(self.qids)
            metrics = self.evaluator.evaluate(fused)
            row.update({m: f"{v:.6f}" for m, v in metrics.items()}, fuse_ms_per_query=f"{fuse_ms:.4f}",
                       status="ok", error="")
        except Exception as e:  # recorded, never silently dropped
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:500], fuse_ms_per_query="")
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.log.append(row)
        best = self.log.best()
        body = (" ".join(f"{m}={float(row[m]):.4f}" for m in METRICS) if row["status"] == "ok"
                else f"ERROR {row['error']}")
        tail = f"best {best['config_id']} {OBJECTIVE}={float(best[OBJECTIVE]):.4f}" if best else "best -"
        print(f"[{label}] {row['config_id']} {body} | {tail}", flush=True)

    def run(self) -> None:
        plan = self.plan()
        done = self.log.completed
        todo = [p for p in plan if config_id(*p) not in done]
        print(f"[hybrid] Plan: {len(plan)} configurations, {len(plan) - len(todo)} already completed, "
              f"{len(todo)} to run")
        try:
            for i, (combo, k, w) in enumerate(todo, start=1):
                self.run_trial(combo, k, w, f"{len(plan) - len(todo) + i}/{len(plan)}")
            self.finalize(plan)
        finally:
            self.evaluator.close()

    def finalize(self, plan) -> None:
        a = self.args
        done = self.log.completed
        rows = [done[config_id(*p)] for p in plan if config_id(*p) in done]
        best = self.log.best(rows)
        if best is None:
            sys.exit(f"[hybrid] No fusion configuration succeeded; see {self.log.path}.")
        if len(rows) < len(plan):
            self.warn(f"Only {len(rows)} of {len(plan)} planned configurations completed.")
        combo = tuple(best["combo"].split("+"))
        k, w = int(best["k"]), float(best["weight_first"])

        # Re-fuse the best configuration: run file, reproducibility, per-query scores, fusion latency.
        lat = []
        fused = {}
        for qid in self.qids:
            t0 = time.perf_counter()
            fused.update(rrf(self.runs, weights_for(combo, w), k, a.depth, [qid]))
            lat.append((time.perf_counter() - t0) * 1000)
        metrics = self.evaluator.evaluate(fused)
        reproduced = all(abs(metrics[m] - float(best[m])) < 1e-6 for m in METRICS)
        if not reproduced:
            self.warn(f"Best configuration did not reproduce: {metrics} vs CSV")
        write_trec_run(fused, a.out_dir / "hybrid_dev.run", tag=best["config_id"].replace(":", "_"))

        per_query = {"hybrid": self.evaluator.per_query(fused)}
        inputs = {}
        for name, run in self.runs.items():
            hits = {q: [Hit(d, -i) for i, d in enumerate(run.get(q, [])[:a.depth])] for q in self.qids}
            inputs[name] = {"metrics": self.evaluator.evaluate(hits), **self.run_info[name]}
            per_query[name] = self.evaluator.per_query(hits)

        significance = {f"hybrid_vs_{n}": paired_test(per_query[n], per_query["hybrid"]) for n in self.runs}
        for x, y in itertools.combinations(sorted(self.runs), 2):
            significance[f"{y}_vs_{x}"] = paired_test(per_query[x], per_query[y])
        deltas = {n: {m: metrics[m] - inputs[n]["metrics"][m] for m in METRICS} for n in self.runs}

        component_ms = {n: lookup(*LATENCY_SOURCES[n]) if n in LATENCY_SOURCES else None for n in combo}
        latency = {
            "fusion": {"mean_ms": round(statistics.fmean(lat), 4), "p50_ms": round(percentile(lat, 50), 4),
                       "p95_ms": round(percentile(lat, 95), 4)},
            "component_mean_ms": component_ms,
            "estimated_sequential_mean_ms": (round(sum(component_ms.values()) + statistics.fmean(lat), 3)
                                             if all(v is not None for v in component_ms.values()) else None),
            "note": "Fusion time is measured here (pure Python over the stored runs). Component latencies are "
                    "read from their own dev result files, measured in separate processes; the end-to-end figure "
                    "is an estimate assuming the retrievers run one after the other on the device.",
        }
        if latency["estimated_sequential_mean_ms"] is None:
            self.warn("Component latency files not found; end-to-end latency not estimated.")

        best_record = {"config_id": best["config_id"], "combo": list(combo), "k": k,
                       "weights": weights_for(combo, w), "depth": a.depth, "metrics": metrics}
        context = {
            "split": a.split,
            "data": {"qrels": os.path.relpath(self.qrels_path, REPO_ROOT),
                     "qrels_sha256": file_sha256(self.qrels_path), "num_queries": len(self.qids)},
            "selection": {
                "rule": f"highest dev {OBJECTIVE}; earliest trial on ties",
                "grid": {"combos": ["+".join(c) for c in a.combos], "k": a.k, "weight_first": a.weights},
                "configs_evaluated": len(rows),
                "caveat": "The best fusion is selected and measured on the same dev queries, so its dev scores "
                          "are optimistic; confirm once on the test split.",
            },
            "git_commit": git_commit(),
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        comparison = {**context, "best_hybrid": best_record, "inputs": inputs, "delta_vs_inputs": deltas,
                      "significance": significance, "latency": latency, "reproduced_on_rerun": reproduced,
                      "warnings": self.warnings}
        for name, record in (("dev_comparison.json", comparison),
                             ("best_config.json", {**context, "best": best_record, "latency": latency})):
            path = a.out_dir / name
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(path)

        print(f"\n[hybrid] Dev results ({a.split}, {len(self.qids)} queries)")
        print(f"  {'run':<34}" + "".join(f"{m:>12}" for m in METRICS))
        for name, info in inputs.items():
            print(f"  {name:<34}" + "".join(f"{info['metrics'][m]:>12.4f}" for m in METRICS))
        print(f"  {'hybrid ' + best['config_id']:<34}" + "".join(f"{metrics[m]:>12.4f}" for m in METRICS))
        for key, s in significance.items():
            p = s["paired_t_p_value"]
            print(f"  {key}: ΔNDCG@10={s['mean_difference']:+.4f}, wins/losses/ties={s['wins']}/{s['losses']}/"
                  f"{s['ties']}, paired t p=" + (f"{p:.4f}" if p is not None else "n/a"))
        if latency["estimated_sequential_mean_ms"] is not None:
            print(f"  estimated latency: {latency['estimated_sequential_mean_ms']} ms/query "
                  f"(fusion {latency['fusion']['mean_ms']} ms)")
        print(f"[hybrid] Outputs in {a.out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="NAME=PATH",
                   help="Add or override an input run (defaults: bm25, rm3, dense)")
    p.add_argument("--combos", default="rm3+dense,bm25+dense", help="Comma-separated run combinations to fuse")
    p.add_argument("--k", default="10,20,40,60,100", help="RRF k values")
    p.add_argument("--weights", default="0.3,0.4,0.5,0.6,0.7", help="Weight of the first run in each combination")
    p.add_argument("--depth", type=int, default=1000, help="Hits used from each run and kept after fusion")
    p.add_argument("--split", choices=["dev", "train"], default="dev", help="Tuning split (test is never used)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--resume", action="store_true", help="Continue an existing tuning_results.csv")
    a = p.parse_args(argv)
    a.runs = dict(DEFAULT_RUNS)
    for spec in a.run:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            p.error(f"--run expects NAME=PATH, got {spec!r}")
        a.runs[name] = Path(path)
    try:
        a.combos = [tuple(c.split("+")) for c in a.combos.split(",") if c.strip()]
        a.k = sorted({int(x) for x in a.k.split(",") if x.strip()})
        a.weights = sorted({round(float(x), 4) for x in a.weights.split(",") if x.strip()})
    except ValueError:
        p.error("--k must be integers and --weights numbers")
    if not a.combos or any(len(c) < 2 or len(set(c)) != len(c) for c in a.combos):
        p.error("each combination needs at least two distinct runs, e.g. rm3+dense")
    if not a.k or min(a.k) < 0 or not a.weights or not all(0 < w < 1 for w in a.weights) or a.depth < 1:
        p.error("need k >= 0, 0 < weight < 1 and --depth >= 1")
    return a


def main(argv: list[str] | None = None) -> None:
    HybridTuning(parse_args(argv)).run()


if __name__ == "__main__":
    main()
