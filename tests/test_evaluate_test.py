"""End-to-end test of the one-shot test-split evaluation on the synthetic fixture.

Runs the real dev chain (RM3, dense with a tiny offline model, hybrid RRF) and then
src.evaluate_test. Metrics from this toy corpus say nothing about NFCorpus.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("pyserini")
pytest.importorskip("sentence_transformers")

from src import evaluate_test  # noqa: E402
from src.bm25 import build_index as bm25_build  # noqa: E402
from src.bm25 import evaluate_rm3, prepare_data  # noqa: E402
from src.dense import build_index as dense_build  # noqa: E402
from src.dense import evaluate_dense  # noqa: E402
from src.hybrid import rrf  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402
from tests.test_dense import make_tiny_model  # noqa: E402
from tests.test_tune import QRELS, QUERIES, write_qrels  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("testsplit")
    data = tmp / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "corpus.jsonl").write_text("".join(json.dumps(d) + "\n" for d in FIXTURE_DOCS))
    (data / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in QUERIES))
    for split, rows in QRELS.items():
        write_qrels(data / "qrels" / f"{split}.tsv", rows)
    prepare_data.prepare(data, tmp / "processed")
    bm25_build.build(tmp / "processed", tmp / "bm25", tmp / "rec.json",
                     k1=0.9, b=0.4, threads=1, smoke_query="tea", smoke_k=1)
    model = make_tiny_model(tmp / "model")
    dense_build.main(["--model", str(model), "--prepared-dir", str(tmp / "processed"),
                      "--index-dir", str(tmp / "dense_idx"), "--threads", "1", "--max-length", "64"])
    evaluate_rm3.main(["--data-dir", str(data), "--index-dir", str(tmp / "bm25"), "--out-dir", str(tmp / "rm3"),
                       "--warmup-queries", "1"])
    evaluate_dense.main(["--index-dir", str(tmp / "dense_idx"), "--data-dir", str(data),
                         "--bm25-index-dir", str(tmp / "bm25"), "--out-dir", str(tmp / "dense"),
                         "--dims", "32,16", "--warmup-queries", "1"])
    rrf.main(["--data-dir", str(data), "--out-dir", str(tmp / "hybrid"), "--combos", "rm3+dense",
              "--k", "60", "--weights", "0.5", f"--run=bm25={tmp / 'rm3' / 'bm25_dev.run'}",
              f"--run=rm3={tmp / 'rm3' / 'rm3_dev.run'}", f"--run=dense={tmp / 'dense' / 'dense_dev_16.run'}"])
    return tmp


def args(env, out, **override):
    paths = {"--data-dir": env / "nfcorpus", "--bm25-index-dir": env / "bm25",
             "--rm3-config": env / "rm3" / "best_config.json", "--dense-results": env / "dense" / "dev_results.json",
             "--hybrid-config": env / "hybrid" / "best_config.json",
             "--hybrid-comparison": env / "hybrid" / "dev_comparison.json", "--out-dir": out}
    paths.update(override)
    return [x for k, v in paths.items() for x in (k, str(v))] + ["--warmup-queries", "1"]


def test_one_shot_test_evaluation(env, tmp_path):
    record = evaluate_test.evaluate(evaluate_test.parse_args(args(env, tmp_path)))
    assert record["split"] == "test" and record["data"]["qrels"].endswith("test.tsv")
    assert record["data"]["num_queries"] == 1 and record["data"]["dev_queries_disjoint"]
    cfg = record["locked_configs"]
    assert cfg["bm25"] == {"k1": 1.7, "b": 0.4}
    assert cfg["rm3"]["config_id"] == "rm3:fb_docs=10:fb_terms=10:oqw=0.50"
    assert cfg["dense"]["dim"] == 16  # the dense run the hybrid was tuned with
    assert cfg["hybrid"]["combo"] == ["rm3", "dense"] and cfg["hybrid"]["k"] == 60
    for m in evaluate_test.METHODS:
        assert set(record["metrics"][m]) == set(evaluate_test.METRICS)
        assert (tmp_path / f"{m}_test.run").exists()
        assert record["latency"][m]["mean_ms"] > 0
    assert {"hybrid_vs_bm25", "hybrid_vs_rm3", "hybrid_vs_dense", "rm3_vs_bm25", "dense_vs_rm3"} <= set(
        record["significance"])
    # The hybrid test run is exactly the RRF of the component test runs it reports.
    runs = {n: rrf.read_run(tmp_path / f"{n}_test.run") for n in ("rm3", "dense")}
    fused = rrf.rrf(runs, cfg["hybrid"]["weights"], 60, 1000, ["s1"])["s1"]
    assert [h.docid for h in fused] == rrf.read_run(tmp_path / "hybrid_test.run")["s1"]
    with pytest.raises(SystemExit, match="evaluated once"):
        evaluate_test.evaluate(evaluate_test.parse_args(args(env, tmp_path)))


def test_no_retrieval_hyperparameter_flags():
    for flag in (["--k1", "1.2"], ["--fb-docs", "5"], ["--dims", "64"], ["--k", "10"], ["--split", "dev"]):
        with pytest.raises(SystemExit):
            evaluate_test.parse_args(flag)


def test_inconsistent_bm25_params_stop(env, tmp_path):
    dense = json.loads((env / "dense" / "dev_results.json").read_text())
    dense["methods"]["bm25"]["k1"] = 0.9
    (tmp_path / "dense.json").write_text(json.dumps(dense))
    with pytest.raises(SystemExit, match="k1/b differ"):
        evaluate_test.evaluate(evaluate_test.parse_args(args(env, tmp_path / "out",
                                                             **{"--dense-results": tmp_path / "dense.json"})))
    assert not (tmp_path / "out").exists()


def test_config_not_selected_on_dev_stops(env, tmp_path):
    hybrid = json.loads((env / "hybrid" / "best_config.json").read_text())
    hybrid["split"] = "test"
    (tmp_path / "h.json").write_text(json.dumps(hybrid))
    with pytest.raises(SystemExit, match="not 'dev'"):
        evaluate_test.evaluate(evaluate_test.parse_args(args(env, tmp_path / "out",
                                                             **{"--hybrid-config": tmp_path / "h.json"})))


def test_cli_fresh_process(env, tmp_path):
    out = subprocess.run([sys.executable, "-m", "src.evaluate_test", *args(env, tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Test results" in out.stdout
