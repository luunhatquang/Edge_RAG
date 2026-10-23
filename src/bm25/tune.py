"""Automatic BM25 (k1, b) tuning on the NFCorpus dev split.

Opens the existing Lucene index once with Pyserini's ``LuceneSearcher`` and,
for every trial, only changes k1/b with ``set_bm25`` - the index is never
rebuilt. Each trial retrieves the top ``--hits`` documents for every query in
the selected split and scores the run with trec_eval (``src.bm25.evaluate``).
NDCG@10 is the objective; MAP@10, Recall@100, Recall@1000 and MRR@10 are
recorded alongside.

Search methods:
  grid  every (k1, b) on a grid with --k1-step / --b-step, in a seeded
        shuffled order so that a --max-trials budget covers the space evenly.
  tpe   Optuna's TPE sampler (optional dependency: requirements-tuning.txt),
        with k1 and b on a 0.01 grid.

The Pyserini default (k1=0.9, b=0.4) is evaluated first as a reference
(when it lies inside the search ranges, even if it is not on the grid).

Only the dev (default) or train split can be used for selection. The test
split is never read for tuning; the run aborts if the tuning queries overlap
the test queries.

Each trial is appended to the CSV (and fsync'ed) as soon as it finishes;
``--resume`` skips configurations already completed successfully and retries
failed ones. The best configuration so far is rewritten after every trial.

Usage:
    python -m src.bm25.tune --method grid --max-trials 25
    python -m src.bm25.tune --method grid --k1-step 0.1 --b-step 0.1 --resume
    python -m src.bm25.tune --method tpe --max-trials 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.bm25.build_index import DEFAULT_INDEX_DIR, git_commit, software_versions
from src.bm25.evaluate import METRICS, TrecEvaluator, load_qrels, load_queries

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "nfcorpus"
DEFAULT_RESULTS_CSV = REPO_ROOT / "results" / "bm25" / "tuning_results.csv"
DEFAULT_BEST_CONFIG = REPO_ROOT / "results" / "bm25" / "best_config.json"

OBJECTIVE = "ndcg@10"
DEFAULT_CONFIG = (0.9, 0.4)
TPE_STEP = 0.01
CSV_FIELDS = [
    "trial", "method", "k1", "b", *METRICS,
    "num_queries", "duration_s", "status", "error", "split", "seed", "timestamp_utc",
]


def key(k1: float, b: float) -> tuple[float, float]:
    return round(k1, 4), round(b, 4)


def axis(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive grid from ``lo`` to ``hi`` with spacing ``step``."""
    n = int(math.floor((hi - lo) / step + 1e-9))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def grid_points(args: argparse.Namespace) -> list[tuple[float, float]]:
    points = [(k1, b) for k1 in axis(args.k1_min, args.k1_max, args.k1_step)
              for b in axis(args.b_min, args.b_max, args.b_step)]
    random.Random(args.seed).shuffle(points)
    # The Pyserini default goes first as a reference, even if it is off-grid.
    if DEFAULT_CONFIG in points:
        points.remove(DEFAULT_CONFIG)
    if args.k1_min <= DEFAULT_CONFIG[0] <= args.k1_max and args.b_min <= DEFAULT_CONFIG[1] <= args.b_max:
        points.insert(0, DEFAULT_CONFIG)
    return points


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Pre-flight checks
# --------------------------------------------------------------------------

def check_python() -> None:
    if sys.version_info < (3, 12):
        sys.exit(f"[tune] Python >= 3.12 is required by Pyserini 2.x, found {sys.version.split()[0]}")


def check_java(versions: dict) -> None:
    major = int(versions["java"].split(".")[0])
    if major < 21:
        sys.exit(f"[tune] Java >= 21 is required by Pyserini 2.x / Anserini, found {versions['java']}")


def load_split(data_dir: Path, split: str) -> tuple[list[str], list[str], dict, Path]:
    """Return (qids, query texts, qrels, qrels path) for the tuning split."""
    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    if not queries_path.exists():
        sys.exit(f"[tune] {queries_path} not found. Place the BEIR NFCorpus files in {data_dir}.")
    if not qrels_path.exists():
        available = sorted(p.name for p in (data_dir / "qrels").glob("*.tsv")) if (data_dir / "qrels").exists() else []
        sys.exit(
            f"[tune] {qrels_path} not found (available qrels: {available or 'none'}).\n"
            "  The test split is never used for tuning. Options:\n"
            "    - restore the BEIR NFCorpus qrels/dev.tsv (it ships with the official zip), or\n"
            "    - tune on the training split instead: --split train\n"
            "  Report final numbers on test.tsv once, with the selected configuration."
        )
    qrels = load_qrels(qrels_path)
    if not qrels:
        sys.exit(f"[tune] {qrels_path} contains no judgments.")

    test_path = data_dir / "qrels" / "test.tsv"
    if test_path.exists():
        overlap = set(qrels) & set(load_qrels(test_path))
        if overlap:
            sys.exit(
                f"[tune] {len(overlap)} {split} query IDs also appear in test.tsv (e.g. {sorted(overlap)[:5]}). "
                "Refusing to tune on queries that are used for testing."
            )

    texts = {q["_id"]: q["text"] for q in load_queries(queries_path)}
    missing = [qid for qid in qrels if qid not in texts]
    if missing:
        sys.exit(f"[tune] {len(missing)} {split} query IDs have no text in queries.jsonl, e.g. {missing[:5]}")
    qids = sorted(qrels)
    return qids, [texts[q] for q in qids], qrels, qrels_path


# --------------------------------------------------------------------------
# Results bookkeeping
# --------------------------------------------------------------------------

def bm25_key(row: dict) -> tuple[float, float]:
    return key(float(row["k1"]), float(row["b"]))


class TrialLog:
    """Append-only CSV of trials, plus the in-memory view used for resume.

    ``fields`` are the CSV columns and ``key_fn`` maps a row to the
    configuration it evaluated; the defaults are the k1/b tuning ones.
    """

    def __init__(self, path: Path, resume: bool, fields: list[str] = CSV_FIELDS, key_fn=bm25_key):
        self.path = path
        self.fields = fields
        self.key_fn = key_fn
        self.rows: list[dict] = []
        if path.exists():
            if not resume:
                sys.exit(f"[tune] {path} already exists. Pass --resume to continue it, or choose another output path.")
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != fields:
                    sys.exit(f"[tune] {path} has unexpected columns {reader.fieldnames}; not resuming.")
                self.rows = list(reader)
            print(f"[tune] Resuming from {path}: {len(self.completed)} completed trials, "
                  f"{sum(r['status'] != 'ok' for r in self.rows)} failed rows")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    @property
    def completed(self) -> dict[tuple, dict]:
        return {self.key_fn(r): r for r in self.rows if r["status"] == "ok"}

    def append(self, row: dict) -> None:
        row = {**row, "trial": len(self.rows) + 1}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self.rows.append({k: str(v) for k, v in row.items()})

    def best(self, rows: list[dict] | None = None) -> dict | None:
        ok = [r for r in (self.rows if rows is None else rows) if r["status"] == "ok"]
        # Highest objective; earliest trial wins ties.
        return max(ok, key=lambda r: (float(r[OBJECTIVE]), -int(r["trial"]))) if ok else None


def metrics_of(row: dict) -> dict[str, float]:
    return {m: float(row[m]) for m in METRICS}


def write_best_config(path: Path, log: TrialLog, context: dict) -> None:
    best = log.best()
    if best is None:
        return
    default = log.completed.get(key(*DEFAULT_CONFIG))
    record = {
        "objective": OBJECTIVE,
        "split": context["split"],
        "best": {
            "k1": float(best["k1"]),
            "b": float(best["b"]),
            "metrics": metrics_of(best),
            "trial": int(best["trial"]),
            "method": best["method"],
        },
        "default_config": {
            "k1": DEFAULT_CONFIG[0],
            "b": DEFAULT_CONFIG[1],
            "metrics": metrics_of(default) if default else None,
        },
        "trials": {
            "ok": len(log.completed),
            "failed_rows": sum(r["status"] != "ok" for r in log.rows),
            "results_csv": os.path.relpath(log.path, REPO_ROOT),
        },
        **{k: v for k, v in context.items() if k != "split"},
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Tuning loop
# --------------------------------------------------------------------------

class Tuner:
    def __init__(self, args: argparse.Namespace):
        check_python()
        self.args = args
        if not args.index_dir.exists():
            sys.exit(f"[tune] Index not found: {args.index_dir}. Run `python -m src.bm25.build_index` first.")
        if args.best_config.exists() and not args.resume:
            sys.exit(f"[tune] {args.best_config} already exists. Pass --resume, or choose another --best-config.")

        self.qids, self.texts, qrels, qrels_path = load_split(args.data_dir, args.split)

        from pyserini.search.lucene import LuceneSearcher  # starts the JVM

        self.versions = software_versions()
        check_java(self.versions)
        self.searcher = LuceneSearcher(str(args.index_dir))
        self.evaluator = TrecEvaluator(qrels)

        judged_docs = {d for docs in qrels.values() for d in docs}
        unknown = sum(self.searcher.doc(d) is None for d in judged_docs)
        if unknown:
            print(f"[tune] Warning: {unknown}/{len(judged_docs)} judged documents are not in the index")

        print(f"[tune] Python {self.versions['python']} · Java {self.versions['java']} · "
              f"Pyserini {self.versions['pyserini']} · Lucene {self.versions['lucene']}")
        print(f"[tune] Index {args.index_dir} ({self.searcher.num_docs} docs); "
              f"{len(self.qids)} {args.split} queries from {qrels_path}")

        self.log = TrialLog(args.results_csv, args.resume)
        self.context = {
            "split": args.split,
            "search": {
                "method": args.method,
                "k1_range": [args.k1_min, args.k1_max],
                "b_range": [args.b_min, args.b_max],
                "k1_step": args.k1_step if args.method == "grid" else TPE_STEP,
                "b_step": args.b_step if args.method == "grid" else TPE_STEP,
                "max_trials": args.max_trials,
                "seed": args.seed,
                "hits": args.hits,
            },
            "data": {
                "queries": os.path.relpath(args.data_dir / "queries.jsonl", REPO_ROOT),
                "qrels": os.path.relpath(qrels_path, REPO_ROOT),
                "qrels_sha256": file_sha256(qrels_path),
                "num_queries": len(self.qids),
            },
            "index": {"path": os.path.relpath(args.index_dir, REPO_ROOT), "num_docs": self.searcher.num_docs},
            "environment": self.versions,
            "git_commit": git_commit(),
        }
        self.start = time.perf_counter()
        self.new_trials = 0
        self.consecutive_errors = 0

    def budget_left(self, budget: int) -> bool:
        return len(self.log.completed) < budget

    def evaluate(self, k1: float, b: float) -> dict | None:
        """Run one trial, log it, and return its metrics (None on failure)."""
        a = self.args
        t0 = time.perf_counter()
        row = {"method": a.method, "k1": k1, "b": b, "num_queries": len(self.qids), "split": a.split,
               "seed": a.seed, **{m: "" for m in METRICS}}
        try:
            self.searcher.set_bm25(k1, b)
            results = self.searcher.batch_search(self.texts, self.qids, k=a.hits, threads=a.threads)
            metrics = self.evaluator.evaluate(results)
            row.update({m: f"{v:.6f}" for m, v in metrics.items()}, status="ok", error="")
            self.consecutive_errors = 0
        except Exception as e:  # recorded in the CSV, never silently dropped
            metrics = None
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
            self.consecutive_errors += 1
        row.update(duration_s=f"{time.perf_counter() - t0:.3f}",
                   timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.log.append(row)
        self.new_trials += 1
        write_best_config(a.best_config, self.log, self.context)
        self.report(row)
        if self.consecutive_errors >= a.max_consecutive_errors:
            sys.exit(f"[tune] {self.consecutive_errors} consecutive failed trials; stopping. See {a.results_csv}.")
        return metrics

    def report(self, row: dict) -> None:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - self.start))
        head = f"[trial {len(self.log.completed)}/{self.budget}] k1={float(row['k1']):.2f} b={float(row['b']):.2f}"
        if row["status"] == "ok":
            body = " ".join(f"{m}={float(row[m]):.4f}" for m in METRICS) + f" ({float(row['duration_s']):.1f}s)"
        else:
            body = f"ERROR {row['error']}"
        best = self.log.best()
        tail = (f"best k1={float(best['k1']):.2f} b={float(best['b']):.2f} {OBJECTIVE}={float(best[OBJECTIVE]):.4f}"
                if best else "best -")
        print(f"{head} {body} | {tail} | elapsed {elapsed}", flush=True)

    def run_grid(self) -> None:
        points = grid_points(self.args)
        self.budget = min(self.args.max_trials or len(points), len(points))
        done = self.log.completed
        print(f"[tune] Grid: {len(points)} configurations, budget {self.budget}, {len(done)} already completed")
        for k1, b in points:
            if not self.budget_left(self.budget):
                break
            if key(k1, b) not in done:
                self.evaluate(k1, b)

    def run_tpe(self) -> None:
        try:
            import optuna
        except ImportError:
            sys.exit("[tune] --method tpe needs Optuna, which is not installed. "
                     "Install it explicitly with: pip install -r requirements-tuning.txt")
        from optuna.distributions import FloatDistribution

        a = self.args
        self.versions["optuna"] = metadata.version("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        dists = {"k1": FloatDistribution(a.k1_min, a.k1_max, step=TPE_STEP),
                 "b": FloatDistribution(a.b_min, a.b_max, step=TPE_STEP)}
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=a.seed))
        # Seed the sampler with every completed trial in range, from any method.
        for (k1, b), r in self.log.completed.items():
            if a.k1_min <= k1 <= a.k1_max and a.b_min <= b <= a.b_max:
                study.add_trial(optuna.trial.create_trial(
                    params={"k1": k1, "b": b}, distributions=dists, value=float(r[OBJECTIVE])))
        if key(*DEFAULT_CONFIG) not in self.log.completed:
            study.enqueue_trial({"k1": DEFAULT_CONFIG[0], "b": DEFAULT_CONFIG[1]})

        self.budget = a.max_trials or 50
        print(f"[tune] TPE (Optuna {self.versions['optuna']}, seed {a.seed}): budget {self.budget}, "
              f"{len(self.log.completed)} already completed")
        attempts = 0
        while self.budget_left(self.budget) and attempts < 10 * self.budget:
            attempts += 1
            trial = study.ask(dists)
            k1, b = key(trial.params["k1"], trial.params["b"])
            cached = self.log.completed.get((k1, b))
            if cached:  # TPE proposed a configuration that was already evaluated
                study.tell(trial, float(cached[OBJECTIVE]))
                continue
            metrics = self.evaluate(k1, b)
            if metrics is None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, metrics[OBJECTIVE])

    def run(self) -> None:
        try:
            self.run_grid() if self.args.method == "grid" else self.run_tpe()
        finally:
            self.evaluator.close()
            self.searcher.close()
        best = self.log.best()
        total = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - self.start))
        print(f"[tune] Done: {self.new_trials} new trials this run, {len(self.log.completed)} completed in total, "
              f"elapsed {total}")
        if best:
            print(f"[tune] Best on {self.args.split}: k1={float(best['k1'])} b={float(best['b'])} "
                  + " ".join(f"{m}={float(best[m]):.4f}" for m in METRICS))
            print(f"[tune] Results: {self.args.results_csv}  Best config: {self.args.best_config}")
        else:
            print("[tune] No successful trials.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["grid", "tpe"], default="grid")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Total successful trials to reach, counting resumed ones (grid: full grid; tpe: 50)")
    p.add_argument("--resume", action="store_true", help="Continue an existing results CSV, skipping completed configs")
    p.add_argument("--seed", type=int, default=42, help="Seed for grid order and the TPE sampler")
    p.add_argument("--k1-min", type=float, default=0.4)
    p.add_argument("--k1-max", type=float, default=2.0)
    p.add_argument("--b-min", type=float, default=0.2)
    p.add_argument("--b-max", type=float, default=1.0)
    p.add_argument("--k1-step", type=float, default=0.1, help="Grid spacing for k1")
    p.add_argument("--b-step", type=float, default=0.1, help="Grid spacing for b")
    p.add_argument("--split", choices=["dev", "train"], default="dev",
                   help="Split used to select parameters (test is never allowed)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="BEIR NFCorpus directory")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--hits", type=int, default=1000, help="Documents retrieved per query (>= 1000 for Recall@1000)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    p.add_argument("--best-config", type=Path, default=DEFAULT_BEST_CONFIG)
    p.add_argument("--max-consecutive-errors", type=int, default=5)
    a = p.parse_args(argv)
    if not (0 <= a.k1_min <= a.k1_max):
        p.error("need 0 <= --k1-min <= --k1-max")
    if not (0 <= a.b_min <= a.b_max <= 1):
        p.error("need 0 <= --b-min <= --b-max <= 1")
    if a.k1_step <= 0 or a.b_step <= 0:
        p.error("--k1-step and --b-step must be > 0")
    if a.max_trials is not None and a.max_trials < 1:
        p.error("--max-trials must be >= 1")
    if a.hits < 1 or a.threads < 1 or a.max_consecutive_errors < 1:
        p.error("--hits, --threads and --max-consecutive-errors must be >= 1")
    if a.hits < 1000:
        print(f"[tune] Note: --hits {a.hits} < 1000, so Recall@1000 is computed on {a.hits} hits")
    return a


def main(argv: list[str] | None = None) -> None:
    Tuner(parse_args(argv)).run()


if __name__ == "__main__":
    main()
