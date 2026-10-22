"""Build a Lucene BM25 index over the prepared NFCorpus corpus with Pyserini.

Runs ``pyserini.index.lucene`` on the JsonCollection produced by
``src.bm25.prepare_data``, then opens the index, checks that every prepared
document ID is present, runs a BM25 smoke query, and records document count,
build duration, index size and software versions in
``results/bm25/index_build.json``.

Note on k1 / b: Lucene stores raw term statistics and document lengths, so BM25
k1 and b are applied at search time, not baked into the index. The values given
here are used for the smoke query and recorded in the build record (and in
``bm25_params.json`` inside the index) as the defaults for later retrieval.
The defaults (k1=0.9, b=0.4) match Pyserini's BEIR BM25 baselines.

Usage:
    python -m src.bm25.build_index
    python -m src.bm25.build_index --k1 1.2 --b 0.75 --rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PREPARED_DIR = REPO_ROOT / "data" / "processed" / "nfcorpus"
DEFAULT_INDEX_DIR = REPO_ROOT / "indexes" / "bm25" / "nfcorpus-flat"
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "bm25" / "index_build.json"
DEFAULT_SMOKE_QUERY = "vitamin D cancer risk"
PARAMS_FILENAME = "bm25_params.json"


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def read_doc_ids(corpus_file: Path) -> list[str]:
    with open(corpus_file, encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def run_indexer(corpus_dir: Path, index_dir: Path, threads: int) -> float:
    """Invoke the Pyserini Lucene indexer; return wall-clock seconds."""
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    print("[index] " + " ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    return time.perf_counter() - start


def software_versions() -> dict:
    """Versions of the stack actually used (call after the JVM has started)."""
    from jnius import autoclass

    import pyserini

    system = autoclass("java.lang.System")
    lucene_version = autoclass("org.apache.lucene.util.Version")
    jars = sorted((Path(pyserini.__file__).parent / "resources" / "jars").glob("anserini-*-fatjar.jar"))
    return {
        "python": platform.python_version(),
        "pyserini": metadata.version("pyserini"),
        "pyjnius": metadata.version("pyjnius"),
        "anserini_jar": jars[0].name if jars else None,
        "lucene": lucene_version.LATEST.toString(),
        "java": system.getProperty("java.version"),
        "java_vm": system.getProperty("java.vm.name"),
        "os": platform.platform(),
    }


def verify_index(index_dir: Path, doc_ids: list[str], k1: float, b: float, query: str, k: int) -> dict:
    """Open the index, check document IDs, and run a BM25 smoke query."""
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    num_indexed = searcher.num_docs
    if num_indexed != len(doc_ids):
        raise RuntimeError(f"Index has {num_indexed} documents, prepared corpus has {len(doc_ids)}")

    missing = [d for d in doc_ids if searcher.doc(d) is None]
    if missing:
        raise RuntimeError(f"{len(missing)} prepared IDs not found in index, e.g. {missing[:5]}")

    searcher.set_bm25(k1, b)
    hits = searcher.search(query, k=k)
    if not hits:
        raise RuntimeError(f"Smoke query {query!r} returned no hits")
    id_set = set(doc_ids)
    unknown = [h.docid for h in hits if h.docid not in id_set]
    if unknown:
        raise RuntimeError(f"Smoke query returned IDs not in the prepared corpus: {unknown}")

    searcher.close()
    return {
        "num_docs_indexed": num_indexed,
        "all_doc_ids_found": True,
        "smoke_test": {
            "query": query,
            "k": k,
            "k1": k1,
            "b": b,
            "num_hits": len(hits),
            "top_hits": [{"docid": h.docid, "score": round(float(h.score), 4)} for h in hits],
        },
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build(
    prepared_dir: Path,
    index_dir: Path,
    results_path: Path,
    k1: float,
    b: float,
    threads: int,
    smoke_query: str,
    smoke_k: int,
    rebuild: bool = False,
) -> dict:
    corpus_dir = prepared_dir / "corpus"
    corpus_file = corpus_dir / "docs.jsonl"
    if not corpus_file.exists():
        sys.exit(f"[index] {corpus_file} not found. Run `python -m src.bm25.prepare_data` first.")

    for target in (index_dir, results_path):
        if target.exists() and not rebuild:
            sys.exit(f"[index] {target} already exists. Pass --rebuild to overwrite it.")

    doc_ids = read_doc_ids(corpus_file)

    # Build into a temporary sibling directory and swap in only after the
    # index has been verified, so a failed rebuild keeps the old index.
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_index = Path(tempfile.mkdtemp(prefix=index_dir.name + ".tmp-", dir=index_dir.parent))
    try:
        duration = run_indexer(corpus_dir, tmp_index, threads)
        check = verify_index(tmp_index, doc_ids, k1, b, smoke_query, smoke_k)
        (tmp_index / PARAMS_FILENAME).write_text(json.dumps({"k1": k1, "b": b}, indent=2) + "\n")
        if index_dir.exists():
            print(f"[index] --rebuild: replacing {index_dir}")
            shutil.rmtree(index_dir)
        tmp_index.rename(index_dir)
    finally:
        if tmp_index.exists():
            shutil.rmtree(tmp_index)

    manifest_path = prepared_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    record = {
        "dataset": "nfcorpus (BEIR)",
        "corpus": os.path.relpath(corpus_file, REPO_ROOT),
        "corpus_sha256": manifest.get("output_corpus_sha256"),
        "index_path": os.path.relpath(index_dir, REPO_ROOT),
        "num_docs_prepared": len(doc_ids),
        **check,
        "build_duration_seconds": round(duration, 3),
        "index_size_bytes": dir_size_bytes(index_dir),
        "bm25": {"k1": k1, "b": b, "note": "applied at search time; stored in index/" + PARAMS_FILENAME},
        "indexer": {
            "collection": "JsonCollection",
            "generator": "DefaultLuceneDocumentGenerator",
            "threads": threads,
            "store": ["positions", "docvectors", "raw"],
        },
        "versions": software_versions(),
        "machine": {"cpu_count": os.cpu_count()},
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(record, indent=2) + "\n")

    print(
        f"[index] {record['num_docs_indexed']} docs indexed in {record['build_duration_seconds']}s, "
        f"{record['index_size_bytes']} bytes on disk"
    )
    print(f"[index] Smoke query {smoke_query!r} top hit: {check['smoke_test']['top_hits'][0]}")
    print(f"[index] Build record: {results_path}")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR, help="Output dir of prepare_data")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH, help="Build record JSON path")
    p.add_argument("--k1", type=float, default=0.9, help="BM25 k1 (default: 0.9)")
    p.add_argument("--b", type=float, default=0.4, help="BM25 b (default: 0.4)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    p.add_argument("--smoke-query", default=DEFAULT_SMOKE_QUERY)
    p.add_argument("--smoke-k", type=int, default=10)
    p.add_argument("--rebuild", action="store_true", help="Overwrite an existing index and build record")
    args = p.parse_args(argv)
    if args.k1 < 0:
        p.error("--k1 must be >= 0")
    if not 0 <= args.b <= 1:
        p.error("--b must be in [0, 1]")
    if args.threads < 1:
        p.error("--threads must be >= 1")
    if args.smoke_k < 1:
        p.error("--smoke-k must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    build(a.prepared_dir, a.index_dir, a.results, a.k1, a.b, a.threads, a.smoke_query, a.smoke_k, a.rebuild)


if __name__ == "__main__":
    main()
