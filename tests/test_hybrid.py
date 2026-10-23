"""Tests for weighted RRF fusion (src/hybrid/rrf.py) on hand-made dev runs.

The fixture runs are synthetic; the metrics they produce say nothing about NFCorpus.
"""

import csv
import json
import subprocess
import sys

import pytest

pytest.importorskip("pyserini")

from src.hybrid import rrf  # noqa: E402
from tests.test_tune import QRELS, QUERIES, write_qrels  # noqa: E402

# dev qrels: d1 -> MED-301 (2); d2 -> MED-14 (2), MED-10 (1); d3 -> MED-118 (1)
RUNS = {
    "rm3": {"d1": ["MED-301", "MED-10"], "d2": ["MED-10", "MED-118", "MED-14"], "d3": ["MED-14", "MED-118"]},
    "dense": {"d1": ["MED-10", "MED-301"], "d2": ["MED-14", "MED-10"], "d3": ["MED-118"]},
    "bm25": {"d1": ["MED-10"], "d2": ["MED-118"], "d3": ["MED-14"]},
}


def write_run(path, run):
    path.write_text("".join(f"{q} Q0 {d} {i} {10 - i:.3f} t\n"
                            for q, docs in run.items() for i, d in enumerate(docs, start=1)))


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    for split, rows in QRELS.items():
        write_qrels(data / "qrels" / f"{split}.tsv", rows)
    for name, run in RUNS.items():
        write_run(tmp_path / f"{name}.run", run)
    return tmp_path


def args(env, *extra):
    return ["--data-dir", str(env / "nfcorpus"), "--out-dir", str(env / "out"),
            *(f"--run={n}={env / (n + '.run')}" for n in RUNS), *extra]


def test_rrf_matches_hand_computation():
    runs = {"a": {"q": ["x", "y"]}, "b": {"q": ["y", "z"]}}
    fused = rrf.rrf(runs, {"a": 0.6, "b": 0.4}, k=1, depth=10, qids=["q"])["q"]
    expected = {"x": 0.6 / 2, "y": 0.6 / 3 + 0.4 / 2, "z": 0.4 / 3}
    assert [h.docid for h in fused] == ["y", "x", "z"]
    assert {h.docid: h.score for h in fused} == pytest.approx(expected)
    # depth limits both the inputs and the output
    assert [h.docid for h in rrf.rrf(runs, {"a": 0.5, "b": 0.5}, 1, 1, ["q"])["q"]] == ["x"]


def test_weights_for_three_runs():
    assert rrf.weights_for(("a", "b", "c"), 0.5) == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_read_run_orders_by_rank(tmp_path):
    (tmp_path / "r.run").write_text("q Q0 b 2 1.0 t\nq Q0 a 1 2.0 t\n")
    assert rrf.read_run(tmp_path / "r.run") == {"q": ["a", "b"]}


def test_tuning_outputs(env):
    rrf.main(args(env, "--combos", "rm3+dense,bm25+dense", "--k", "1,60", "--weights", "0.4,0.5"))
    out = env / "out"
    with open(out / "tuning_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * 2 * 2 and all(r["status"] == "ok" and r["split"] == "dev" for r in rows)
    comp = json.loads((out / "dev_comparison.json").read_text())
    best = max(rows, key=lambda r: float(r["ndcg@10"]))
    assert comp["best_hybrid"]["metrics"]["ndcg@10"] == pytest.approx(float(best["ndcg@10"]))
    assert comp["reproduced_on_rerun"] is True
    assert set(comp["inputs"]) == {"rm3", "dense", "bm25"}
    # fusing complementary runs beats each of them on this fixture
    for name in ("rm3", "dense"):
        assert comp["best_hybrid"]["metrics"]["ndcg@10"] > comp["inputs"][name]["metrics"]["ndcg@10"]
    assert {"hybrid_vs_rm3", "hybrid_vs_dense", "rm3_vs_dense"} <= set(comp["significance"])
    assert comp["data"]["qrels"].endswith("dev.tsv")
    run = (out / "hybrid_dev.run").read_text().splitlines()
    assert {line.split()[0] for line in run} == {"d1", "d2", "d3"}
    assert json.loads((out / "best_config.json").read_text())["best"]["config_id"] == comp["best_hybrid"]["config_id"]


def test_resume_and_overwrite_guard(env):
    rrf.main(args(env, "--combos", "rm3+dense", "--k", "60", "--weights", "0.5"))
    with pytest.raises(SystemExit):
        rrf.main(args(env, "--combos", "rm3+dense", "--k", "60", "--weights", "0.5,0.6"))
    rrf.main(args(env, "--combos", "rm3+dense", "--k", "60", "--weights", "0.5,0.6", "--resume"))
    with open(env / "out" / "tuning_results.csv", newline="") as f:
        assert [r["config_id"] for r in csv.DictReader(f)] == ["rm3+dense:k=60:w=0.50", "rm3+dense:k=60:w=0.60"]


def test_run_with_test_queries_is_rejected(env):
    write_run(env / "dense.run", {**RUNS["dense"], "s1": ["MED-10"]})
    with pytest.raises(SystemExit, match="test query IDs"):
        rrf.main(args(env))
    assert not (env / "out").exists()


def test_run_with_unknown_queries_is_rejected(env):
    write_run(env / "dense.run", {**RUNS["dense"], "zz": ["MED-10"]})
    with pytest.raises(SystemExit, match="not in dev.tsv"):
        rrf.main(args(env))


def test_bad_arguments():
    for bad in (["--combos", "rm3"], ["--weights", "1.0"], ["--split", "test"], ["--run", "nopath"]):
        with pytest.raises(SystemExit):
            rrf.parse_args(bad)


def test_cli_fresh_process(env):
    out = subprocess.run([sys.executable, "-m", "src.hybrid.rrf", *args(env, "--k", "60", "--weights", "0.5")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Dev results" in out.stdout
