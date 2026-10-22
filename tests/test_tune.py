"""Tests for BM25 tuning, against a real index built from the synthetic fixture corpus.

The metrics here come from a 4-document toy corpus and say nothing about NFCorpus.
"""

import csv
import json

import pytest

pytest.importorskip("pyserini")

from src.bm25 import build_index, prepare_data, tune  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402

QUERIES = [
    {"_id": "d1", "text": "green tea"},
    {"_id": "d2", "text": "vitamin D cancer"},
    {"_id": "d3", "text": "fiber colorectal"},
    {"_id": "t1", "text": "statin"},
    {"_id": "s1", "text": "breast cancer statin"},
]
QRELS = {
    "dev": [("d1", "MED-301", 2), ("d2", "MED-14", 2), ("d2", "MED-10", 1), ("d3", "MED-118", 1)],
    "train": [("t1", "MED-10", 1)],
    "test": [("s1", "MED-10", 2)],
}


def write_qrels(path, rows):
    path.write_text("query-id\tcorpus-id\tscore\n" + "".join(f"{q}\t{d}\t{s}\n" for q, d, s in rows))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tune")
    data = tmp / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in FIXTURE_DOCS))
    (data / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    for split, rows in QRELS.items():
        write_qrels(data / "qrels" / f"{split}.tsv", rows)
    prepare_data.prepare(data, tmp / "processed")
    build_index.build(tmp / "processed", tmp / "index", tmp / "rec.json",
                      k1=0.9, b=0.4, threads=1, smoke_query="tea", smoke_k=1)
    return tmp


def run(env, out, *extra):
    argv = ["--data-dir", str(env / "nfcorpus"), "--index-dir", str(env / "index"),
            "--results-csv", str(out / "tuning_results.csv"), "--best-config", str(out / "best_config.json"),
            "--threads", "1", "--k1-step", "0.4", "--b-step", "0.4", *extra]
    tune.main(argv)
    with open(out / "tuning_results.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_axis_is_inclusive():
    assert tune.axis(0.4, 2.0, 0.4) == [0.4, 0.8, 1.2, 1.6, 2.0]
    assert tune.axis(0.2, 1.0, 0.1) == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_grid_puts_default_first_and_is_seeded():
    args = tune.parse_args(["--seed", "7"])
    points = tune.grid_points(args)
    assert points[0] == tune.DEFAULT_CONFIG
    assert len(points) == len(set(points)) == 17 * 9
    assert points == tune.grid_points(tune.parse_args(["--seed", "7"]))
    assert all(0.4 <= k1 <= 2.0 and 0.2 <= b <= 1.0 for k1, b in points)


def test_default_added_when_off_grid():
    points = tune.grid_points(tune.parse_args(["--k1-step", "0.4", "--b-step", "0.4"]))
    assert points[0] == tune.DEFAULT_CONFIG and len(points) == 5 * 3 + 1
    assert tune.DEFAULT_CONFIG not in tune.grid_points(tune.parse_args(["--k1-min", "1.0"]))


def test_trec_evaluator_against_hand_computation(env):
    from collections import namedtuple

    from pyserini.search.lucene import LuceneSearcher  # noqa: F401  (starts the JVM)

    Hit = namedtuple("Hit", "docid score")
    ev = tune.TrecEvaluator({"q1": {"A": 2, "B": 1}, "q2": {"B": 1}})
    try:
        m = ev.evaluate({"q1": [Hit("B", 5.0), Hit("A", 4.0), Hit("C", 3.0)], "q2": [Hit("C", 3.0)]})
    finally:
        ev.close()
    # q1: DCG = 1/log2(2) + 2/log2(3), IDCG = 2/log2(2) + 1/log2(3); q2 retrieves nothing relevant.
    ndcg_q1 = (1 + 2 / 1.5849625) / (2 + 1 / 1.5849625)
    assert m["ndcg@10"] == pytest.approx(ndcg_q1 / 2, abs=1e-4)
    assert m["map@10"] == pytest.approx(0.5, abs=1e-4)  # q1 AP = 1, q2 AP = 0
    assert m["mrr@10"] == pytest.approx(0.5, abs=1e-4)
    assert m["recall@100"] == pytest.approx(0.5, abs=1e-4)


def test_test_split_is_not_selectable():
    with pytest.raises(SystemExit):
        tune.parse_args(["--split", "test"])


def test_grid_run_writes_results_and_best(env, tmp_path):
    rows = run(env, tmp_path, "--method", "grid", "--max-trials", "4")
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    assert (float(rows[0]["k1"]), float(rows[0]["b"])) == tune.DEFAULT_CONFIG
    assert all(r["split"] == "dev" and r["num_queries"] == "3" for r in rows)
    for m in tune.METRICS:
        assert all(0.0 <= float(r[m]) <= 1.0 for r in rows)

    best = json.loads((tmp_path / "best_config.json").read_text())
    top = max(rows, key=lambda r: float(r["ndcg@10"]))
    assert best["best"]["metrics"]["ndcg@10"] == pytest.approx(float(top["ndcg@10"]))
    assert best["default_config"]["metrics"] is not None
    assert best["data"]["qrels"].endswith("dev.tsv")
    assert best["environment"]["pyserini"] and best["environment"]["java"]
    assert best["trials"]["ok"] == 4


def test_metrics_match_hand_computation(env, tmp_path):
    # With every query's relevant documents ranked first, all metrics are 1.
    rows = run(env, tmp_path, "--max-trials", "1")
    assert {m: float(rows[0][m]) for m in tune.METRICS} == {m: 1.0 for m in tune.METRICS}


def test_resume_skips_completed_and_refuses_overwrite(env, tmp_path):
    first = run(env, tmp_path, "--max-trials", "3")
    with pytest.raises(SystemExit):
        run(env, tmp_path, "--max-trials", "5")
    rows = run(env, tmp_path, "--max-trials", "5", "--resume")
    assert len(rows) == 5
    assert rows[:3] == first
    configs = [(r["k1"], r["b"]) for r in rows]
    assert len(configs) == len(set(configs))


def test_failed_trial_is_recorded(env, tmp_path, monkeypatch):
    original = tune.TrecEvaluator.evaluate
    calls = {"n": 0}

    def flaky(self, results):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return original(self, results)

    monkeypatch.setattr(tune.TrecEvaluator, "evaluate", flaky)
    rows = run(env, tmp_path, "--max-trials", "3")
    assert [r["status"] for r in rows] == ["ok", "error", "ok", "ok"]
    assert rows[1]["error"] == "RuntimeError: boom"

    # A resumed run retries nothing that succeeded and keeps the error row.
    monkeypatch.setattr(tune.TrecEvaluator, "evaluate", original)
    rows = run(env, tmp_path, "--max-trials", "4", "--resume")
    assert len(rows) == 5 and rows[1]["status"] == "error"


def test_missing_dev_split_exits_with_advice(env, tmp_path):
    data = tmp_path / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "queries.jsonl").write_text((env / "nfcorpus" / "queries.jsonl").read_text())
    write_qrels(data / "qrels" / "test.tsv", QRELS["test"])
    with pytest.raises(SystemExit, match="--split train"):
        tune.load_split(data, "dev")


def test_dev_overlapping_test_is_rejected(env, tmp_path):
    data = tmp_path / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "queries.jsonl").write_text((env / "nfcorpus" / "queries.jsonl").read_text())
    write_qrels(data / "qrels" / "dev.tsv", QRELS["dev"] + QRELS["test"])
    write_qrels(data / "qrels" / "test.tsv", QRELS["test"])
    with pytest.raises(SystemExit, match="test"):
        tune.load_split(data, "dev")


def test_tpe_run(env, tmp_path):
    pytest.importorskip("optuna")
    rows = run(env, tmp_path, "--method", "tpe", "--max-trials", "5", "--seed", "1")
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 5 and (float(ok[0]["k1"]), float(ok[0]["b"])) == tune.DEFAULT_CONFIG
    assert all(0.4 <= float(r["k1"]) <= 2.0 and 0.2 <= float(r["b"]) <= 1.0 for r in ok)
    assert json.loads((tmp_path / "best_config.json").read_text())["environment"]["optuna"]
