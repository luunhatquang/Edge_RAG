"""Tests for dense retrieval (src/dense) using a tiny randomly initialised ModernBERT model.

The real model (hotchpotch/bekko-embedding-v1-a8m) is not downloaded here; the tiny model
has the same architecture family and the same Sentence Transformers interface, so these
tests exercise the pipeline, not retrieval quality.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("pyserini")
pytest.importorskip("sentence_transformers")

import numpy as np  # noqa: E402

from src.bm25 import build_index as bm25_build  # noqa: E402
from src.bm25 import prepare_data  # noqa: E402
from src.dense import build_index, evaluate_dense  # noqa: E402
from tests.conftest import FIXTURE_DOCS  # noqa: E402
from tests.test_tune import QRELS, QUERIES, write_qrels  # noqa: E402

WORDS = ("green tea vitamin d cancer risk statin statins breast survival fiber colorectal health catechins "
         "cardiovascular adults dietary intake status use and in").split()


def make_tiny_model(path):
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules as stm
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    torch.manual_seed(0)
    cfg = ModernBertConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=2, max_position_embeddings=128, pad_token_id=0,
                           cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3)
    ModernBertModel(cfg).save_pretrained(path / "hf")
    fast.save_pretrained(path / "hf")
    t = stm.Transformer(str(path / "hf"), max_seq_length=64)
    SentenceTransformer(modules=[t, stm.Pooling(t.get_embedding_dimension(), "mean")],
                        device="cpu").save(str(path / "st"))
    return path / "st"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dense")
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
    build_index.main(["--model", str(model), "--prepared-dir", str(tmp / "processed"),
                      "--index-dir", str(tmp / "dense"), "--threads", "1", "--max-length", "64"])
    return tmp


def eval_args(env, out, *extra):
    return ["--index-dir", str(env / "dense"), "--data-dir", str(env / "nfcorpus"),
            "--bm25-index-dir", str(env / "bm25"), "--out-dir", str(out), "--warmup-queries", "1", *extra]


def test_dense_index_files(env):
    emb = np.load(env / "dense" / "embeddings.npy")
    docids = json.loads((env / "dense" / "docids.json").read_text())
    manifest = json.loads((env / "dense" / "manifest.json").read_text())
    assert docids == [d["_id"] for d in FIXTURE_DOCS]
    assert emb.shape == (4, 32) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert manifest["num_docs"] == 4 and manifest["prefixes"] == {"query": "", "document": ""}
    assert manifest["model"]["dimension"] == 32 and manifest["model"]["max_seq_length"] == 64
    assert manifest["encode"]["docs_per_second"] > 0


def test_build_refuses_overwrite(env):
    before = (env / "dense" / "embeddings.npy").read_bytes()
    with pytest.raises(SystemExit):
        build_index.main(["--model", str(env / "model" / "st"), "--prepared-dir", str(env / "processed"),
                          "--index-dir", str(env / "dense"), "--threads", "1"])
    assert (env / "dense" / "embeddings.npy").read_bytes() == before


def test_search_matches_brute_force_and_truncation(env):
    index = evaluate_dense.DenseIndex(env / "dense", [32, 16])
    emb = np.load(env / "dense" / "embeddings.npy")
    q = emb[2] + 0.1 * emb[0]
    for d in (32, 16):
        m = emb[:, :d] / np.linalg.norm(emb[:, :d], axis=1, keepdims=True)
        expected = m @ (q[:d] / np.linalg.norm(q[:d]))
        hits = index.search(q, d, 10)
        assert [h.docid for h in hits] == [index.docids[i] for i in np.argsort(-expected, kind="stable")]
        assert [h.score for h in hits] == pytest.approx(sorted(expected, reverse=True), abs=1e-5)
    assert len(index.search(q, 32, 2)) == 2


def test_evaluate_dense_vs_bm25(env, tmp_path):
    record = evaluate_dense.evaluate(evaluate_dense.parse_args(eval_args(env, tmp_path, "--dims", "32,16")))
    assert set(record["methods"]) == {"bm25", "dense_32", "dense_16"}
    assert record["split"] == "dev" and record["data"]["num_queries"] == 3
    assert record["data"]["qrels"].endswith("dev.tsv")
    for role in ("dense_32", "dense_16"):
        m = record["methods"][role]
        assert all(0.0 <= v <= 1.0 for v in m["metrics"].values())
        assert m["latency"]["total"]["mean_ms"] >= m["latency"]["query_encode"]["mean_ms"] > 0
        s = record["significance_vs_bm25"][role]
        assert s["wins"] + s["losses"] + s["ties"] == 3
    assert record["methods"]["dense_16"]["index_matrix_bytes"] == 4 * 16 * 4
    run = (tmp_path / "dense_dev_32.run").read_text().splitlines()
    assert {line.split()[0] for line in run} == {"d1", "d2", "d3"}
    assert {line.split()[2] for line in run} == {d["_id"] for d in FIXTURE_DOCS}
    with pytest.raises(SystemExit):
        evaluate_dense.evaluate(evaluate_dense.parse_args(eval_args(env, tmp_path, "--dims", "32")))


def test_missing_model_gives_actionable_error(env, tmp_path):
    with pytest.raises(SystemExit, match="--model"):
        build_index.main(["--model", str(tmp_path / "no-such-model"), "--prepared-dir", str(env / "processed"),
                          "--index-dir", str(tmp_path / "idx"), "--threads", "1"])
    assert not (tmp_path / "idx").exists()


def test_test_split_not_selectable():
    with pytest.raises(SystemExit):
        evaluate_dense.parse_args(["--split", "test", "--no-bm25"])


def test_cli_fresh_process_without_bm25(env, tmp_path):
    # A new interpreter catches JVM classpath problems (trec_eval without Pyserini loaded first).
    out = subprocess.run([sys.executable, "-m", "src.dense.evaluate_dense",
                          *eval_args(env, tmp_path, "--no-bm25", "--dims", "32")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    record = json.loads((tmp_path / "dev_results.json").read_text())
    assert set(record["methods"]) == {"dense_32"} and record["significance_vs_bm25"] == {}
