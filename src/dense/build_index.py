"""Encode the prepared NFCorpus corpus into a dense index with a Sentence Transformers model.

Default model: hotchpotch/bekko-embedding-v1-a8m (MIT; mmBERT-based, ~7.7M active
parameters, 384-dim embeddings truncatable to 256/128/64, mean pooling, cosine).
Its model card uses no query/document prefixes, so none are added here.

Input is the same ``data/processed/nfcorpus/corpus/docs.jsonl`` the BM25 index
was built from (``contents = title + " " + text``), so document IDs match the
original NFCorpus IDs. The index is a plain directory:

    embeddings.npy   float32 [num_docs, dim], L2-normalised (full model dimension)
    docids.json      document IDs in row order
    manifest.json    model, revision, max length, corpus checksum, timings, versions

Exact (brute-force) cosine search over 3,633 x 384 floats is ~5.6 MB and needs
no ANN library, which suits an edge device. The index is never overwritten
without ``--rebuild``; a new one is written to a temporary directory first.

Usage:
    python -m src.dense.build_index
    python -m src.dense.build_index --model /path/to/local/bekko-embedding-v1-a8m
    python -m src.dense.build_index --max-length 512 --batch-size 32 --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src.bm25.build_index import dir_size_bytes, git_commit
from src.bm25.prepare_data import file_hash

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "hotchpotch/bekko-embedding-v1-a8m"
DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "processed" / "nfcorpus"
DEFAULT_INDEX_DIR = REPO_ROOT / "indexes" / "dense" / "bekko-embedding-v1-a8m"


def versions() -> dict:
    import torch

    out = {"python": platform.python_version(), "torch": torch.__version__, "os": platform.platform()}
    for dist in ("sentence-transformers", "transformers", "tokenizers", "numpy"):
        out[dist] = metadata.version(dist)
    return out


def load_model(name: str, revision: str | None, max_length: int, threads: int):
    """Load a Sentence Transformers model on CPU, with a clear error when it can't be fetched."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(name, revision=revision, device="cpu")
    except Exception as e:  # network, missing files, incompatible versions
        sys.exit(
            f"[dense] Could not load model {name!r}: {type(e).__name__}: {e}\n"
            "  If huggingface.co is unreachable, download the model elsewhere "
            "(e.g. `hf download hotchpotch/bekko-embedding-v1-a8m --local-dir models/bekko-embedding-v1-a8m`) "
            "and pass --model models/bekko-embedding-v1-a8m."
        )
    model.max_seq_length = max_length
    return model, time.perf_counter() - t0


def model_info(model, name: str, revision: str | None) -> dict:
    info = {"name": name, "revision": revision, "max_seq_length": model.max_seq_length,
            "dimension": model.get_embedding_dimension()}
    if Path(name).exists():
        info["local_path_size_bytes"] = dir_size_bytes(Path(name))
    info["num_parameters"] = sum(p.numel() for p in model.parameters())
    return info


def read_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                ids.append(doc["id"])
                texts.append(doc["contents"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate document IDs")
    return ids, texts


def build(args: argparse.Namespace) -> dict:
    import numpy as np

    corpus = args.prepared_dir / "corpus" / "docs.jsonl"
    if not corpus.exists():
        sys.exit(f"[dense] {corpus} not found. Run `python -m src.bm25.prepare_data` first.")
    if args.index_dir.exists() and not args.rebuild:
        sys.exit(f"[dense] {args.index_dir} already exists. Pass --rebuild to overwrite it.")

    docids, texts = read_corpus(corpus)
    model, load_s = load_model(args.model, args.revision, args.max_length, args.threads)
    info = model_info(model, args.model, args.revision)
    print(f"[dense] Model {args.model} loaded in {load_s:.2f}s: dim {info['dimension']}, "
          f"{info['num_parameters']:,} parameters, max_seq_length {model.max_seq_length}")
    print(f"[dense] Encoding {len(texts)} documents (batch {args.batch_size}, {args.threads} threads)")

    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=args.progress)
    encode_s = time.perf_counter() - t0
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape != (len(docids), info["dimension"]) or not np.isfinite(emb).all():
        raise RuntimeError(f"Unexpected embeddings: shape {emb.shape}, finite={np.isfinite(emb).all()}")

    # Truncated inputs: how many documents exceed max_seq_length tokens.
    lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=True)["input_ids"]]
    truncated = sum(n > model.max_seq_length for n in lengths)

    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=args.index_dir.name + ".tmp-", dir=args.index_dir.parent))
    try:
        np.save(tmp / "embeddings.npy", emb)
        (tmp / "docids.json").write_text(json.dumps(docids) + "\n")
        manifest = {
            "model": info,
            "corpus": os.path.relpath(corpus, REPO_ROOT),
            "corpus_sha256": file_hash(corpus),
            "num_docs": len(docids),
            "normalized": True,
            "similarity": "cosine (dot product of L2-normalised vectors)",
            "prefixes": {"query": "", "document": ""},
            "documents_truncated": truncated,
            "doc_tokens": {"mean": round(sum(lengths) / len(lengths), 1), "max": max(lengths)},
            "encode": {"seconds": round(encode_s, 3), "docs_per_second": round(len(docids) / encode_s, 1),
                       "batch_size": args.batch_size, "threads": args.threads, "device": "cpu",
                       "model_load_seconds": round(load_s, 3)},
            "versions": versions(),
            "git_commit": git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.index_dir.exists():
            print(f"[dense] --rebuild: replacing {args.index_dir}")
            shutil.rmtree(args.index_dir)
        tmp.rename(args.index_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)

    manifest["index_size_bytes"] = dir_size_bytes(args.index_dir)
    print(f"[dense] {len(docids)} docs encoded in {encode_s:.1f}s ({manifest['encode']['docs_per_second']} docs/s); "
          f"{truncated} truncated at {model.max_seq_length} tokens; index {manifest['index_size_bytes']} bytes "
          f"at {args.index_dir}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local directory")
    p.add_argument("--revision", default=None, help="Model revision (commit hash) to pin")
    p.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR, help="Output dir of prepare_data")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--max-length", type=int, default=512, help="Max tokens per document (model supports 8192)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="torch CPU threads")
    p.add_argument("--progress", action="store_true", help="Show an encoding progress bar")
    p.add_argument("--rebuild", action="store_true", help="Overwrite an existing dense index")
    a = p.parse_args(argv)
    if a.max_length < 8 or a.batch_size < 1 or a.threads < 1:
        p.error("need --max-length >= 8, --batch-size >= 1 and --threads >= 1")
    return a


def main(argv: list[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
