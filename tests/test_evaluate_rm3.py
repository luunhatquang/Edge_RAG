"""Tests for BM25 vs BM25+RM3 evaluation, on a real index built from the synthetic fixture.

The metrics here come from a 4-document toy corpus and say nothing about NFCorpus.
"""

import csv
import itertools
import json
import subprocess
import sys

import pytest

pytest.importorskip("pyserini")

from src.bm25 import build_index, evaluate_rm3, prepare_data, tune  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402
from tests.test_tune import QRELS, QUERIES, write_qrels  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rm3")
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


def run(env, out, *extra, index=None):
    evaluate_rm3.main(["--data-dir", str(env / "nfcorpus"), "--index-dir", str(index or env / "index"),
                       "--out-dir", str(out), "--warmup-queries", "2", *extra])
    with open(out / "tuning_results.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_l9_is_an_orthogonal_array():
    design = evaluate_rm3.rm3_grid("l9")
    assert len(design) == len(set(design)) == 9
    assert (10, 10, 0.5) in design
    for i, j in itertools.combinations(range(3), 2):
        assert len({(c[i], c[j]) for c in design}) == 9  # every pair of levels exactly once
    assert len(evaluate_rm3.rm3_grid("full")) == 27


def test_default_comparison(env, tmp_path):
    rows = run(env, tmp_path)
    assert [r["config_id"] for r in rows] == ["bm25", "rm3:fb_docs=10:fb_terms=10:oqw=0.50"]
    assert all(r["status"] == "ok" and r["k1"] == "1.7" and r["b"] == "0.4" for r in rows)
    assert all(float(r["latency_mean_ms"]) > 0 and r["num_queries"] == "3" for r in rows)

    comp = json.loads((tmp_path / "dev_comparison.json").read_text())
    assert set(comp["methods"]) == {"bm25", "rm3_default", "rm3_best"}
    assert comp["methods"]["rm3_best"]["config_id"] == comp["methods"]["rm3_default"]["config_id"]
    assert all(comp["reproduced_on_rerun"].values())
    assert comp["data"]["qrels"].endswith("dev.tsv") and comp["split"] == "dev"
    assert comp["cost"]["searcher_init_seconds"] > 0 and comp["cost"]["process_peak_rss_mb"] > 0
    sig = comp["significance_vs_bm25"]["rm3_default"]
    assert sig["wins"] + sig["losses"] + sig["ties"] == 3
    for name in ("rm3_dev.run", "bm25_dev.run", "best_config.json"):
        assert (tmp_path / name).exists()
    # Run file keeps the original query and document IDs.
    qids = {line.split()[0] for line in (tmp_path / "rm3_dev.run").read_text().splitlines()}
    docids = {line.split()[2] for line in (tmp_path / "rm3_dev.run").read_text().splitlines()}
    assert qids <= {"d1", "d2", "d3"} and docids <= {d["_id"] for d in FIXTURE_DOCS}


def test_tune_l9_resume_and_overwrite_guard(env, tmp_path):
    first = run(env, tmp_path)
    with pytest.raises(SystemExit):
        run(env, tmp_path, "--tune", "--grid", "l9")
    rows = run(env, tmp_path, "--tune", "--grid", "l9", "--resume")
    assert rows[:2] == first
    assert len(rows) == 1 + 9 and len({r["config_id"] for r in rows}) == 10
    best = json.loads((tmp_path / "best_config.json").read_text())
    rm3 = [r for r in rows if r["method"] == "rm3"]
    assert best["best"]["metrics"]["ndcg@10"] == pytest.approx(max(float(r["ndcg@10"]) for r in rm3))
    assert best["selection"]["rm3_configs_evaluated"] == 9


def test_baseline_matches_bm25_tuning_pipeline(env, tmp_path):
    rows = run(env, tmp_path / "rm3")
    tune.main(["--data-dir", str(env / "nfcorpus"), "--index-dir", str(env / "index"),
               "--results-csv", str(tmp_path / "t.csv"), "--best-config", str(tmp_path / "t.json"),
               "--k1-min", "1.7", "--k1-max", "1.7", "--b-min", "0.4", "--b-max", "0.4", "--threads", "1"])
    with open(tmp_path / "t.csv", newline="") as f:
        tuned = next(csv.DictReader(f))
    for m in evaluate_rm3.METRICS:
        assert float(rows[0][m]) == pytest.approx(float(tuned[m]), abs=1e-6)


def test_only_dev_qrels_reach_the_evaluator(env, tmp_path, monkeypatch):
    seen = []
    original = evaluate_rm3.TrecEvaluator.__init__

    def spy(self, qrels):
        seen.append(set(qrels))
        original(self, qrels)

    monkeypatch.setattr(evaluate_rm3.TrecEvaluator, "__init__", spy)
    run(env, tmp_path)
    assert seen == [{"d1", "d2", "d3"}]


def test_index_without_docvectors_is_rejected(env, tmp_path):
    novec = tmp_path / "novec"
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", str(env / "processed" / "corpus"), "--index", str(novec),
                    "--generator", "DefaultLuceneDocumentGenerator", "--threads", "1", "--storeRaw"],
                   check=True, capture_output=True)
    with pytest.raises(SystemExit, match="document vectors"):
        run(env, tmp_path / "out", index=novec)
    assert not (tmp_path / "out").exists()


def test_missing_dev_split_stops(env, tmp_path):
    data = tmp_path / "nfcorpus"
    (data / "qrels").mkdir(parents=True)
    (data / "queries.jsonl").write_text((env / "nfcorpus" / "queries.jsonl").read_text())
    write_qrels(data / "qrels" / "test.tsv", QRELS["test"])
    with pytest.raises(SystemExit, match="--split train"):
        evaluate_rm3.main(["--data-dir", str(data), "--index-dir", str(env / "index"),
                           "--out-dir", str(tmp_path / "out")])


def test_test_split_not_selectable():
    with pytest.raises(SystemExit):
        evaluate_rm3.parse_args(["--split", "test"])


def test_paired_test_handles_identical_runs():
    a = {"q1": 0.5, "q2": 0.2}
    res = evaluate_rm3.paired_test(a, dict(a))
    assert res["ties"] == 2 and res["paired_t_p_value"] is None and res["mean_difference"] == 0


def test_cli_in_fresh_process(env, tmp_path):
    # A new interpreter catches JVM start-up/classpath problems that in-process tests hide.
    out = subprocess.run(
        [sys.executable, "-m", "src.bm25.evaluate_rm3", "--data-dir", str(env / "nfcorpus"),
         "--index-dir", str(env / "index"), "--out-dir", str(tmp_path), "--warmup-queries", "1"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "Dev results" in out.stdout
    assert json.loads((tmp_path / "dev_comparison.json").read_text())["methods"]["bm25"]["metrics"]
