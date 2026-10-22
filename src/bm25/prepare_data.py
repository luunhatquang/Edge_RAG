"""Prepare the NFCorpus (BEIR) corpus for Pyserini Lucene indexing.

Reads the original BEIR ``corpus.jsonl`` (fields ``_id``, ``title``, ``text``)
and writes a Pyserini ``JsonCollection`` file with one ``{"id", "contents"}``
object per line. Document IDs are copied verbatim from ``_id``; ``contents`` is
``title + " " + text`` (the same "flat" layout used by Pyserini's
``beir-v1.0.0-*.flat`` indexes).

The original dataset is only ever read. Existing outputs are never overwritten
unless ``--rebuild`` is given.

Usage:
    python -m src.bm25.prepare_data --download   # fetch NFCorpus if missing
    python -m src.bm25.prepare_data              # convert corpus
    python -m src.bm25.prepare_data --rebuild    # regenerate outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
# MD5 published in the BEIR README dataset table.
NFCORPUS_MD5 = "a89dba18a62ef92f7d323ec890a0d38d"

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "nfcorpus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "nfcorpus"
# Pyserini indexes every .json/.jsonl file in its input directory, so the
# corpus lives in its own subdirectory, separate from the manifest.
CORPUS_SUBDIR = "corpus"
OUTPUT_FILENAME = "docs.jsonl"
MANIFEST_FILENAME = "manifest.json"


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_read_only(root: Path) -> None:
    """Strip write permission from every file under ``root``."""
    for path in root.rglob("*"):
        if path.is_file():
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def download_nfcorpus(raw_dir: Path, url: str = NFCORPUS_URL, expected_md5: str | None = NFCORPUS_MD5) -> None:
    """Download and extract the BEIR NFCorpus archive into ``raw_dir``.

    Does nothing if ``raw_dir/corpus.jsonl`` already exists.
    """
    if (raw_dir / "corpus.jsonl").exists():
        print(f"[download] NFCorpus already present at {raw_dir}; skipping download.")
        return
    if raw_dir.exists() and any(raw_dir.iterdir()):
        sys.exit(f"[download] {raw_dir} exists but has no corpus.jsonl; refusing to touch it. Remove it manually.")
    if raw_dir.exists():
        raw_dir.rmdir()  # empty; shutil.move would otherwise nest the data inside it

    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=raw_dir.parent) as tmp:
        zip_path = Path(tmp) / "nfcorpus.zip"
        print(f"[download] Fetching {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except urllib.error.URLError as e:
            sys.exit(f"[download] Failed to download {url}: {e}")
        md5 = file_hash(zip_path, "md5")
        print(f"[download] MD5 {md5}")
        if expected_md5 and md5 != expected_md5:
            sys.exit(f"[download] MD5 mismatch: expected {expected_md5}, got {md5}. Aborting.")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # The archive contains a single top-level "nfcorpus/" directory.
        shutil.move(str(Path(tmp) / "nfcorpus"), str(raw_dir))
    make_read_only(raw_dir)
    print(f"[download] Extracted to {raw_dir} (files marked read-only).")


def to_pyserini_doc(beir_doc: dict) -> dict:
    """Map a BEIR corpus record to a Pyserini JsonCollection record."""
    doc_id = beir_doc["_id"]
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError(f"Invalid document id: {doc_id!r}")
    title = (beir_doc.get("title") or "").strip()
    text = (beir_doc.get("text") or "").strip()
    contents = f"{title} {text}".strip()
    return {"id": doc_id, "contents": contents}


def convert_corpus(corpus_path: Path, output_path: Path) -> dict:
    """Convert ``corpus_path`` to ``output_path``; return statistics.

    Raises ``ValueError`` on duplicate IDs or malformed records.
    """
    seen: set[str] = set()
    num_docs = 0
    num_empty = 0
    with open(corpus_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                doc = to_pyserini_doc(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise ValueError(f"{corpus_path}:{line_no}: {e}") from e
            if doc["id"] in seen:
                raise ValueError(f"{corpus_path}:{line_no}: duplicate document id {doc['id']!r}")
            seen.add(doc["id"])
            if not doc["contents"]:
                num_empty += 1
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            num_docs += 1
    if num_docs == 0:
        raise ValueError(f"{corpus_path} contains no documents")
    return {"num_docs": num_docs, "num_empty_contents": num_empty}


def prepare(raw_dir: Path, output_dir: Path, rebuild: bool = False) -> dict:
    corpus_path = raw_dir / "corpus.jsonl"
    if not corpus_path.exists():
        sys.exit(
            f"[prepare] {corpus_path} not found. Run `python -m src.bm25.prepare_data --download` "
            "or place the BEIR NFCorpus files there."
        )

    output_path = output_dir / CORPUS_SUBDIR / OUTPUT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if output_dir.exists() and any(output_dir.iterdir()):
        if not rebuild:
            sys.exit(f"[prepare] {output_dir} already exists. Pass --rebuild to regenerate it.")
        print(f"[prepare] --rebuild: replacing {output_dir}")

    input_sha_before = file_hash(corpus_path)

    # Write into a temporary sibling directory and swap in only on success,
    # so a failed run never leaves a half-written corpus behind.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    tmp_corpus = tmp_dir / CORPUS_SUBDIR / OUTPUT_FILENAME
    try:
        tmp_corpus.parent.mkdir()
        stats = convert_corpus(corpus_path, tmp_corpus)
        if file_hash(corpus_path) != input_sha_before:
            raise RuntimeError(f"{corpus_path} changed during preparation")
        manifest = {
            "dataset": "nfcorpus (BEIR)",
            "source_corpus": os.path.relpath(corpus_path, REPO_ROOT),
            "source_corpus_sha256": input_sha_before,
            "output_corpus": os.path.relpath(output_path, REPO_ROOT),
            "output_corpus_sha256": file_hash(tmp_corpus),
            "format": "Pyserini JsonCollection: {id, contents}; contents = title + ' ' + text",
            **stats,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (tmp_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    print(f"[prepare] Wrote {stats['num_docs']} documents to {output_path}")
    if stats["num_empty_contents"]:
        print(f"[prepare] Warning: {stats['num_empty_contents']} documents have empty contents")
    print(f"[prepare] Manifest: {manifest_path}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="BEIR NFCorpus directory (read-only input)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory (corpus/docs.jsonl + manifest.json)")
    p.add_argument("--download", action="store_true", help="Download NFCorpus into --raw-dir if it is missing")
    p.add_argument("--download-only", action="store_true", help="Download NFCorpus and exit without converting")
    p.add_argument("--rebuild", action="store_true", help="Overwrite an existing prepared corpus")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.download or args.download_only:
        download_nfcorpus(args.raw_dir)
        if args.download_only:
            return
    prepare(args.raw_dir, args.output_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
