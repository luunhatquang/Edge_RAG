"""BM25 vs BM25+RM3 on the NFCorpus dev split, with optional RM3 tuning.

Reuses the existing pieces instead of a second pipeline:
  - ``src.bm25.tune.load_split``  dev/train queries + qrels, refuses the test
    split, missing dev qrels and dev/test query overlap;
  - ``src.bm25.evaluate.TrecEvaluator``  official trec_eval 9.0.4 metrics
    (NDCG@10, MAP@10, Recall@100, Recall@1000, MRR@10) and per-query NDCG@10;
  - ``src.bm25.tune.TrialLog``  append-only CSV checkpoint with resume;
  - the Lucene index built by ``src.bm25.build_index`` (it stores document
    vectors, which RM3 needs). The index is opened read-only, never rebuilt.

Every configuration is run query by query with ``LuceneSearcher.search`` on a
single thread, so the same pass yields the metrics and the per-query latency
an edge device would see. BM25 uses k1=1.7, b=0.4 for both methods; RM3 is
switched on with ``set_rm3`` and off with ``unset_rm3`` between trials.

Trials:
  bm25          BM25 only (baseline)
  rm3 default   fb_docs=10, fb_terms=10, original_query_weight=0.5
  --tune        RM3 grid fb_docs {5,10,20} x fb_terms {10,20,30} x
                original_query_weight {0.3,0.5,0.7}: all 27 (--grid full)
                or the 9-run L9 orthogonal array (--grid l9), which covers
                every pair of levels of any two parameters exactly once.

The best RM3 configuration is the one with the highest dev NDCG@10. At the end,
the baseline, the default RM3 and the best RM3 are run again to write the TREC
run files, check that the metrics reproduce, and compare per-query NDCG@10
with a paired t-test. Test qrels are never read.

Outputs (in --out-dir, default results/rm3):
  tuning_results.csv   one row per trial (checkpoint; --resume continues it)
  dev_comparison.json  BM25 vs RM3 default vs RM3 best: metrics, deltas,
                       significance, latency/memory, warnings
  best_config.json     best RM3 configuration on dev
  rm3_dev.run          run of the best RM3 configuration on dev
  bm25_dev.run         run of the BM25 baseline on dev

Usage:
    python -m src.bm25.evaluate_rm3                    # BM25 vs RM3 default
    python -m src.bm25.evaluate_rm3 --tune             # + 27-config RM3 grid
    python -m src.bm25.evaluate_rm3 --tune --grid l9   # + 9-config L9 design
    python -m src.bm25.evaluate_rm3 --tune --resume    # continue after an interruption
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.bm25.build_index import DEFAULT_INDEX_DIR, dir_size_bytes, git_commit, software_versions
from src.bm25.evaluate import METRICS, TrecEvaluator, write_trec_run
from src.bm25.tune import DEFAULT_DATA_DIR, TrialLog, check_java, check_python, file_sha256, load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "rm3"
BM25_TUNING_CSV = REPO_ROOT / "results" / "bm25" / "tuning_results.csv"
BM25_BEST_CONFIG = REPO_ROOT / "results" / "bm25" / "best_config.json"

OBJECTIVE = "ndcg@10"
FB_DOCS = (5, 10, 20)
FB_TERMS = (10, 20, 30)
ORIG_WEIGHTS = (0.3, 0.5, 0.7)

FIELDS = [
    "trial", "config_id", "method", "k1", "b", "fb_docs", "fb_terms", "original_query_weight", *METRICS,
    "num_queries", "eval_seconds", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
    "jvm_heap_used_mb", "rss_mb", "status", "error", "split", "timestamp_utc",
]


def config_id(cfg: tuple | None) -> str:
    """``None`` is plain BM25; otherwise (fb_docs, fb_terms, original_query_weight)."""
    if cfg is None:
        return "bm25"
    return f"rm3:fb_docs={cfg[0]}:fb_terms={cfg[1]}:oqw={cfg[2]:.2f}"


def rm3_grid(mode: str) -> list[tuple[int, int, float]]:
    if mode == "full":
        return list(itertools.product(FB_DOCS, FB_TERMS, ORIG_WEIGHTS))
    # L9(3^3) orthogonal array: fb_docs level i, fb_terms level j, weight level (i + j) mod 3.
    return [(FB_DOCS[i], FB_TERMS[j], ORIG_WEIGHTS[(i + j) % 3]) for i in range(3) for j in range(3)]


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q / 100 * (len(s) - 1))))
    return s[idx]


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def metrics_of(row: dict) -> dict[str, float]:
    return {m: float(row[m]) for m in METRICS}


def latency_of(row: dict) -> dict[str, float]:
    return {k: float(row[k]) for k in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "eval_seconds")}


def paired_test(a: dict[str, float], b: dict[str, float]) -> dict:
    """Paired comparison of per-query values, b minus a."""
    from scipy import stats

    qids = sorted(a)
    diffs = [b[q] - a[q] for q in qids]
    nonzero = [d for d in diffs if d != 0]
    result = {
        "measure": "ndcg@10 per query",
        "num_queries": len(qids),
        "mean_difference": statistics.fmean(diffs),
        "wins": sum(d > 0 for d in diffs),
        "losses": sum(d < 0 for d in diffs),
        "ties": sum(d == 0 for d in diffs),
        "paired_t_p_value": None,
        "wilcoxon_p_value": None,
    }
    if len(nonzero) >= 2:
        t_p = float(stats.ttest_rel([b[q] for q in qids], [a[q] for q in qids]).pvalue)
        w_p = float(stats.wilcoxon(nonzero).pvalue)
        # NaN (e.g. all differences identical) is not valid JSON; report it as undefined.
        result["paired_t_p_value"] = None if t_p != t_p else t_p
        result["wilcoxon_p_value"] = None if w_p != w_p else w_p
    return result


class RM3Evaluation:
    def __init__(self, args: argparse.Namespace):
        check_python()
        self.args = args
        self.warnings: list[str] = []
        a = args
        if not a.index_dir.exists():
            sys.exit(f"[rm3] Index not found: {a.index_dir}. Build it with `python -m src.bm25.build_index` "
                     "(it stores the document vectors RM3 needs).")

        csv_path = a.out_dir / "tuning_results.csv"
        derived = [a.out_dir / n for n in ("dev_comparison.json", "best_config.json", "rm3_dev.run", "bm25_dev.run")]
        if not csv_path.exists() and any(p.exists() for p in derived):
            sys.exit(f"[rm3] {a.out_dir} has RM3 outputs but no tuning_results.csv; refusing to overwrite them.")

        self.qids, self.texts, qrels, self.qrels_path = load_split(a.data_dir, a.split)

        t0 = time.perf_counter()
        from pyserini.search.lucene import LuceneSearcher  # starts the JVM with Anserini on the classpath

        from jnius import autoclass  # only after Pyserini has configured and started the JVM

        self.jvm_start_s = time.perf_counter() - t0
        self.versions = software_versions()
        check_java(self.versions)
        self.runtime = autoclass("java.lang.Runtime").getRuntime()

        t0 = time.perf_counter()
        self.searcher = LuceneSearcher(str(a.index_dir))
        self.searcher_init_s = time.perf_counter() - t0
        self._check_rm3_support()

        self.evaluator = TrecEvaluator(qrels)
        self.log = TrialLog(csv_path, a.resume, fields=FIELDS, key_fn=lambda r: r["config_id"])
        print(f"[rm3] Python {self.versions['python']} · Java {self.versions['java']} · "
              f"Pyserini {self.versions['pyserini']} · Lucene {self.versions['lucene']}")
        print(f"[rm3] Index {a.index_dir} ({self.searcher.num_docs} docs), JVM start {self.jvm_start_s:.2f}s, "
              f"searcher init {self.searcher_init_s:.3f}s")
        print(f"[rm3] {len(self.qids)} {a.split} queries from {self.qrels_path}; BM25 k1={a.k1} b={a.b}")
        self._check_bm25_params()
        self.start = time.perf_counter()
        self.consecutive_errors = 0

    # ---------------------------------------------------------------- checks

    def _check_rm3_support(self) -> None:
        try:
            self.searcher.set_rm3(fb_terms=10, fb_docs=10, original_query_weight=0.5)
        except TypeError as e:
            sys.exit(f"[rm3] {e}\n  The index at {self.args.index_dir} has no document vectors. Rebuild it with "
                     "`python -m src.bm25.build_index --rebuild` (stores docvectors); this script never rebuilds it.")
        self.searcher.unset_rm3()

    def _check_bm25_params(self) -> None:
        if BM25_BEST_CONFIG.exists():
            best = json.loads(BM25_BEST_CONFIG.read_text()).get("best", {})
            if (best.get("k1"), best.get("b")) != (self.args.k1, self.args.b):
                self.warn(f"BM25 k1={self.args.k1}, b={self.args.b} differ from {BM25_BEST_CONFIG.name} "
                          f"(k1={best.get('k1')}, b={best.get('b')}).")
        else:
            self.warn(f"{os.path.relpath(BM25_BEST_CONFIG, REPO_ROOT)} not found; BM25 k1={self.args.k1}, "
                      f"b={self.args.b} are taken from the command line, not verified against BM25 tuning.")

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
            print(f"[rm3] Warning: {msg}")

    # -------------------------------------------------------------- running

    def configure(self, cfg: tuple | None) -> None:
        self.searcher.set_bm25(self.args.k1, self.args.b)
        if cfg is None:
            self.searcher.unset_rm3()
        else:
            self.searcher.set_rm3(fb_terms=cfg[1], fb_docs=cfg[0], original_query_weight=cfg[2])

    def retrieve(self, cfg: tuple | None, limit: int | None = None) -> tuple[dict, list[float]]:
        """Search every query sequentially; return results and per-query latency (ms)."""
        self.configure(cfg)
        results, latencies = {}, []
        for qid, text in list(zip(self.qids, self.texts))[:limit]:
            t0 = time.perf_counter()
            results[qid] = self.searcher.search(text, k=self.args.hits)
            latencies.append((time.perf_counter() - t0) * 1000)
        return results, latencies

    def warmup(self) -> None:
        n = min(self.args.warmup_queries, len(self.qids))
        if n:
            t0 = time.perf_counter()
            for cfg in (None, (10, 10, 0.5)):
                self.retrieve(cfg, limit=n)
            print(f"[rm3] JVM warm-up: {n} queries x (BM25, RM3) in {time.perf_counter() - t0:.2f}s (not recorded)")

    def run_trial(self, cfg: tuple | None, label: str) -> None:
        a = self.args
        row = {
            "config_id": config_id(cfg), "method": "bm25" if cfg is None else "rm3", "k1": a.k1, "b": a.b,
            "fb_docs": "" if cfg is None else cfg[0], "fb_terms": "" if cfg is None else cfg[1],
            "original_query_weight": "" if cfg is None else cfg[2], "num_queries": len(self.qids), "split": a.split,
            **{m: "" for m in METRICS},
            **{k: "" for k in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms")},
        }
        t0 = time.perf_counter()
        try:
            results, lat = self.retrieve(cfg)
            metrics = self.evaluator.evaluate(results)
            row.update({m: f"{v:.6f}" for m, v in metrics.items()}, status="ok", error="",
                       latency_mean_ms=f"{statistics.fmean(lat):.3f}", latency_p50_ms=f"{percentile(lat, 50):.3f}",
                       latency_p95_ms=f"{percentile(lat, 95):.3f}")
            self.consecutive_errors = 0
        except Exception as e:  # recorded in the CSV, never silently dropped
            row.update(status="error", error=f"{type(e).__name__}: {e}"[:500])
            self.consecutive_errors += 1
        heap = (self.runtime.totalMemory() - self.runtime.freeMemory()) / 2**20
        row.update(eval_seconds=f"{time.perf_counter() - t0:.3f}", jvm_heap_used_mb=f"{heap:.1f}",
                   rss_mb=f"{rss_mb():.1f}", timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.log.append(row)
        self.report(row, label)
        if self.consecutive_errors >= a.max_consecutive_errors:
            sys.exit(f"[rm3] {self.consecutive_errors} consecutive failed trials; stopping. See {self.log.path}.")

    def report(self, row: dict, label: str) -> None:
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - self.start))
        if row["status"] == "ok":
            body = (" ".join(f"{m}={float(row[m]):.4f}" for m in METRICS)
                    + f" | {float(row['latency_mean_ms']):.1f} ms/query, {float(row['eval_seconds']):.1f}s")
        else:
            body = f"ERROR {row['error']}"
        best = self.best_rm3()
        tail = f"best RM3 {best['config_id']} {OBJECTIVE}={float(best[OBJECTIVE]):.4f}" if best else "best RM3 -"
        print(f"[{label}] {row['config_id']} {body} | {tail} | elapsed {elapsed}", flush=True)

    def best_rm3(self) -> dict | None:
        return self.log.best([r for r in self.log.rows if r["method"] == "rm3"])

    def run(self) -> None:
        a = self.args
        default = (a.fb_docs, a.fb_terms, a.original_query_weight)
        plan = [None, default]
        if a.tune:
            plan += [c for c in rm3_grid(a.grid) if c != default]
        todo = [c for c in plan if config_id(c) not in self.log.completed]
        print(f"[rm3] Plan: {len(plan)} configurations ({'BM25 + RM3 default' if not a.tune else 'BM25 + RM3 default + '
              + a.grid + ' grid'}), {len(plan) - len(todo)} already completed, {len(todo)} to run")
        try:
            if todo:
                self.warmup()
            for i, cfg in enumerate(todo, start=1):
                self.run_trial(cfg, f"{len(plan) - len(todo) + i}/{len(plan)}")
            self.finalize(plan, default)
        finally:
            self.evaluator.close()
            self.searcher.close()

    # ------------------------------------------------------------ reporting

    def finalize(self, plan: list, default: tuple) -> None:
        a = self.args
        done = self.log.completed
        base_row, default_row = done.get("bm25"), done.get(config_id(default))
        rm3_rows = [done[config_id(c)] for c in plan if c is not None and config_id(c) in done]
        best_row = self.log.best(rm3_rows)
        if base_row is None or best_row is None:
            sys.exit("[rm3] BM25 baseline or every RM3 configuration failed; no comparison written. "
                     f"See {self.log.path}.")
        if len(rm3_rows) < sum(c is not None for c in plan):
            self.warn(f"Only {len(rm3_rows)} of {sum(c is not None for c in plan)} planned RM3 configurations "
                      "completed; the best is chosen among those.")

        def cfg_of(row):
            return None if row["method"] == "bm25" else (
                int(row["fb_docs"]), int(row["fb_terms"]), float(row["original_query_weight"]))

        # Re-run the key configurations: run files, reproducibility check, per-query scores.
        per_query, reproduced = {}, {}
        roles = {"bm25": base_row, "rm3_best": best_row}
        if default_row:
            roles["rm3_default"] = default_row
        for role, row in roles.items():
            if role == "rm3_default" and row is best_row:
                per_query[role] = per_query["rm3_best"]
                reproduced[role] = reproduced["rm3_best"]
                continue
            results, _ = self.retrieve(cfg_of(row))
            again = self.evaluator.evaluate(results)
            reproduced[role] = all(abs(again[m] - float(row[m])) < 1e-6 for m in METRICS)
            if not reproduced[role]:
                self.warn(f"{role} metrics did not reproduce on re-run: {again} vs CSV {metrics_of(row)}")
            per_query[role] = self.evaluator.per_query(results)
            if role == "bm25":
                write_trec_run(results, a.out_dir / "bm25_dev.run", tag=f"bm25_k1_{a.k1}_b_{a.b}")
            elif role == "rm3_best":
                write_trec_run(results, a.out_dir / "rm3_dev.run", tag=row["config_id"].replace(":", "_"))

        def summary(row: dict) -> dict:
            out = {"config_id": row["config_id"], "method": row["method"], "k1": float(row["k1"]),
                   "b": float(row["b"])}
            if row["method"] == "rm3":
                out.update(fb_docs=int(row["fb_docs"]), fb_terms=int(row["fb_terms"]),
                           original_query_weight=float(row["original_query_weight"]))
            return {**out, "metrics": metrics_of(row), "latency": latency_of(row),
                    "jvm_heap_used_mb": float(row["jvm_heap_used_mb"]), "rss_mb": float(row["rss_mb"])}

        base = summary(base_row)
        methods = {"bm25": base, "rm3_best": summary(best_row)}
        if default_row:
            methods["rm3_default"] = summary(default_row)
        deltas, significance = {}, {}
        for role in ("rm3_default", "rm3_best"):
            if role not in methods:
                continue
            m = methods[role]
            deltas[role] = {
                k: {"absolute": m["metrics"][k] - base["metrics"][k],
                    "relative_pct": 100 * (m["metrics"][k] - base["metrics"][k]) / base["metrics"][k]
                    if base["metrics"][k] else None}
                for k in METRICS
            }
            deltas[role]["latency_mean_ratio"] = m["latency"]["latency_mean_ms"] / base["latency"]["latency_mean_ms"]
            significance[role] = paired_test(per_query["bm25"], per_query[role])

        context = self.context()
        comparison = {
            "split": a.split,
            "objective": OBJECTIVE,
            "methods": methods,
            "delta_vs_bm25": deltas,
            "significance_vs_bm25": significance,
            "reproduced_on_rerun": reproduced,
            "selection": {
                "rule": f"highest dev {OBJECTIVE} among completed RM3 configurations; earliest trial on ties",
                "grid": a.grid if a.tune else "none (default RM3 only)",
                "rm3_configs_evaluated": len(rm3_rows),
                "caveat": "rm3_best is selected and measured on the same dev queries, so its dev scores are "
                          "optimistic; confirm once on the test split before claiming an improvement.",
            },
            "cost": self.cost(),
            **context,
            "warnings": self.warnings,
        }
        best_config = {
            "objective": OBJECTIVE,
            "split": a.split,
            "best": methods["rm3_best"],
            "bm25_baseline": base,
            "delta_vs_bm25": deltas["rm3_best"],
            "significance_vs_bm25": significance["rm3_best"],
            "selection": comparison["selection"],
            "run_file": os.path.relpath(a.out_dir / "rm3_dev.run", REPO_ROOT),
            **context,
        }
        for name, record in (("dev_comparison.json", comparison), ("best_config.json", best_config)):
            path = a.out_dir / name
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(path)
        self.print_summary(methods, deltas, significance)

    def context(self) -> dict:
        a = self.args
        consistency = None
        if BM25_TUNING_CSV.exists():
            import csv

            with open(BM25_TUNING_CSV, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("split") == a.split
                        and abs(float(r["k1"]) - a.k1) < 1e-9 and abs(float(r["b"]) - a.b) < 1e-9]
            if rows:
                base = metrics_of(self.log.completed["bm25"])
                diff = max(abs(float(rows[-1][m]) - base[m]) for m in METRICS)
                consistency = {"source": os.path.relpath(BM25_TUNING_CSV, REPO_ROOT),
                               "max_abs_metric_difference": diff, "match": diff < 1e-4}
                if diff >= 1e-4:
                    self.warn(f"BM25 baseline differs from {BM25_TUNING_CSV.name} by up to {diff:.6f}.")
        return {
            "data": {"qrels": os.path.relpath(self.qrels_path, REPO_ROOT), "qrels_sha256": file_sha256(self.qrels_path),
                     "num_queries": len(self.qids), "hits_per_query": a.hits},
            "index": {"path": os.path.relpath(a.index_dir, REPO_ROOT), "num_docs": self.searcher.num_docs},
            "bm25_baseline_matches_bm25_tuning": consistency,
            "environment": self.versions,
            "git_commit": git_commit(),
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def cost(self) -> dict:
        return {
            "jvm_start_seconds": round(self.jvm_start_s, 3),
            "searcher_init_seconds": round(self.searcher_init_s, 4),
            "index_size_bytes": dir_size_bytes(self.args.index_dir),
            "process_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "timing": "single-threaded LuceneSearcher.search per query, after a JVM warm-up; "
                      "latency excludes trec_eval",
            "memory_note": "RSS covers the whole process (Python + JVM + trec_eval buffers); JVM heap is "
                           "sampled after each trial and depends on GC timing.",
        }

    def print_summary(self, methods: dict, deltas: dict, significance: dict) -> None:
        print("\n[rm3] Dev results (" + self.args.split + f", {len(self.qids)} queries)")
        header = f"  {'method':<45}" + "".join(f"{m:>12}" for m in METRICS) + f"{'ms/query':>10}"
        print(header)
        for role in ("bm25", "rm3_default", "rm3_best"):
            if role in methods:
                m = methods[role]
                print(f"  {role + ' ' + m['config_id']:<45}" + "".join(f"{m['metrics'][k]:>12.4f}" for k in METRICS)
                      + f"{m['latency']['latency_mean_ms']:>10.2f}")
        for role, s in significance.items():
            p = s["paired_t_p_value"]
            print(f"  {role} vs bm25: ΔNDCG@10={deltas[role]['ndcg@10']['absolute']:+.4f}, "
                  f"wins/losses/ties={s['wins']}/{s['losses']}/{s['ties']}, paired t p="
                  + (f"{p:.4f}" if p is not None else "n/a")
                  + f", latency x{deltas[role]['latency_mean_ratio']:.2f}")
        print(f"[rm3] Outputs in {self.args.out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k1", type=float, default=1.7, help="BM25 k1 for both methods (default: 1.7)")
    p.add_argument("--b", type=float, default=0.4, help="BM25 b for both methods (default: 0.4)")
    p.add_argument("--fb-docs", type=int, default=10, help="Default RM3 feedback documents")
    p.add_argument("--fb-terms", type=int, default=10, help="Default RM3 expansion terms")
    p.add_argument("--original-query-weight", type=float, default=0.5, help="Default RM3 original query weight")
    p.add_argument("--tune", action="store_true", help="Also run the RM3 grid on the dev split")
    p.add_argument("--grid", choices=["full", "l9"], default="full", help="27 configurations or the 9-run L9 design")
    p.add_argument("--resume", action="store_true", help="Continue an existing tuning_results.csv")
    p.add_argument("--split", choices=["dev", "train"], default="dev", help="Selection split (test is never used)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hits", type=int, default=1000, help="Documents retrieved per query")
    p.add_argument("--warmup-queries", type=int, default=50, help="Queries run before timing (0 disables)")
    p.add_argument("--max-consecutive-errors", type=int, default=5)
    a = p.parse_args(argv)
    if a.k1 < 0 or not 0 <= a.b <= 1:
        p.error("need k1 >= 0 and 0 <= b <= 1")
    if a.fb_docs < 1 or a.fb_terms < 1 or not 0 <= a.original_query_weight <= 1:
        p.error("need fb_docs >= 1, fb_terms >= 1 and 0 <= original_query_weight <= 1")
    if a.hits < 1 or a.warmup_queries < 0 or a.max_consecutive_errors < 1:
        p.error("need --hits >= 1, --warmup-queries >= 0 and --max-consecutive-errors >= 1")
    return a


def main(argv: list[str] | None = None) -> None:
    RM3Evaluation(parse_args(argv)).run()


if __name__ == "__main__":
    main()
