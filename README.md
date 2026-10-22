# Edge RAG — NFCorpus

Research code for retrieval-augmented generation on edge devices, using the
[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) dataset in
[BEIR](https://github.com/beir-cellar/beir) format.

**Current status — stage 1 only:** NFCorpus data preparation and a BM25 Lucene
index built with [Pyserini](https://github.com/castorini/pyserini). Dense
retrieval, hybrid fusion (RRF), evaluation and LLM generation are not implemented yet.

## Layout

```
src/bm25/prepare_data.py      BEIR corpus.jsonl -> Pyserini JsonCollection ({id, contents})
src/bm25/build_index.py       Lucene index build + smoke test + build record
src/bm25/search_app.py        Local web UI / JSON API for trying BM25 queries
scripts/run_pipeline.sh       One-shot script: prepare → build → test
scripts/test_retrieval.py     Chạy BM25 queries + tính nDCG@10 trên NFCorpus test split
tests/test_prepare_data.py    Unit tests: corpus conversion, download, overwrite guard
tests/test_build_index.py     Integration test: build index trên 4-doc fixture
tests/test_real_index.py      Integration test: kiểm tra index với dữ liệu NFCorpus thực
results/bm25/index_build.json Build record (git-ignored)
data/nfcorpus/                NFCorpus gốc (corpus.jsonl, queries.jsonl, qrels/)
data/processed/nfcorpus/      corpus/docs.jsonl + manifest.json (git-ignored)
indexes/bm25/nfcorpus-flat/   Lucene index (git-ignored)
```

## Requirements

- Python ≥ 3.12 (tested with 3.13.16)
- Java 21 JDK, with `JAVA_HOME` set (tested with OpenJDK 21.0.12). Pyserini 2.x bundles
  Anserini/Lucene, which needs Java 21.
- Pyserini 2.4.0 (pinned in `requirements.txt`). It pulls in PyTorch and
  transformers. BM25 doesn't use them, but Pyserini lists them as hard dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
java -version          # must report 21.x
```

## Quickstart (dữ liệu NFCorpus đã có sẵn)

Nếu `data/nfcorpus/` đã có sẵn (corpus.jsonl, queries.jsonl, qrels/), chạy toàn bộ pipeline:

```bash
bash scripts/run_pipeline.sh
# hoặc build lại từ đầu:
bash scripts/run_pipeline.sh --rebuild
```

Hoặc từng bước:

```bash
# Bước 1: Chuyển corpus sang Pyserini format
python -m src.bm25.prepare_data --raw-dir data/nfcorpus

# Bước 2: Build Lucene BM25 index
python -m src.bm25.build_index --prepared-dir data/processed/nfcorpus

# Bước 3: Test retrieval (nDCG@10 trên 50 test queries)
python scripts/test_retrieval.py --queries 50 --verbose
```

## Stage 1: data preparation and BM25 index

### 1. Get NFCorpus

NFCorpus đã có sẵn tại `data/nfcorpus/`. Nếu cần tải lại:

```bash
python -m src.bm25.prepare_data --download-only
```

This downloads
`https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip`,
checks it against the MD5 from the BEIR README (`a89dba18a62ef92f7d323ec890a0d38d`),
extracts it to `data/raw/nfcorpus/` and marks the files read-only. If the corpus is
already there, the download is skipped. The equivalent manual command is:

```bash
mkdir -p data/raw && cd data/raw \
  && wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip \
  && md5sum nfcorpus.zip && unzip nfcorpus.zip && rm nfcorpus.zip
```

Expected contents: `corpus.jsonl`, `queries.jsonl`, `qrels/{train,dev,test}.tsv`.

### 2. Prepare the Pyserini corpus

```bash
# Nếu dùng data/nfcorpus/ (đã có sẵn):
python -m src.bm25.prepare_data --raw-dir data/nfcorpus

# Nếu đã download về data/raw/nfcorpus/ (mặc định):
python -m src.bm25.prepare_data            # add --rebuild to regenerate
```

- Reads `data/raw/nfcorpus/corpus.jsonl` and never writes to it. The SHA-256 is
  checked before and after the run.
- Writes `data/processed/nfcorpus/corpus/docs.jsonl`, one
  `{"id": <original _id>, "contents": title + " " + text}` per line. This is the
  layout of Pyserini's `beir-v1.0.0-*.flat` indexes. IDs such as `MED-10` are kept
  as they are, and duplicate IDs abort the run.
- Writes `data/processed/nfcorpus/manifest.json` with document counts and checksums.
- Refuses to overwrite existing output unless `--rebuild` is given. Output is
  written to a temporary directory and swapped in only on success.

### 3. Build the BM25 index

```bash
python -m src.bm25.build_index                       # k1=0.9, b=0.4
python -m src.bm25.build_index --k1 1.2 --b 0.75 --rebuild
```

Options: `--threads N`, `--index-dir`, `--results`, `--smoke-query "..."`, `--smoke-k 10`.

The script runs `python -m pyserini.index.lucene` (JsonCollection,
DefaultLuceneDocumentGenerator, with positions, docvectors and raw text stored).
It then runs a smoke test:

- opens the index with `LuceneSearcher`
- checks that the indexed document count equals the prepared count
- checks that every original document ID can be fetched
- runs one BM25 query with the given `k1`/`b`

The old index is replaced only if the new one passes. Without `--rebuild`, the
script won't overwrite an existing index or build record.

**k1 and b:** Lucene stores raw term statistics, so BM25 `k1`/`b` are applied at
search time and are not baked into the index. The chosen values are used for the
smoke query, recorded in the build record, and saved as
`indexes/bm25/nfcorpus-flat/bm25_params.json` as defaults for later stages.

The build record `results/bm25/index_build.json` contains:

- document counts
- build duration in seconds (wall clock of the indexer)
- index size on disk in bytes
- BM25 parameters
- smoke-test hits
- software versions: Python, Pyserini, pyjnius, Anserini jar, Lucene, Java, OS
- git commit and timestamp

### 4. Try queries in the browser

```bash
python -m src.bm25.search_app                      # http://127.0.0.1:8000
python -m src.bm25.search_app --index-dir indexes/bm25/nfcorpus-flat --port 8080
```

This is a small Flask app that opens the index read-only. Each query lets you set
`top-k`, `k1` and `b` (defaults come from `bm25_params.json`). The page shows:

- the analyzed query terms (after stopword removal and Porter stemming) with
  their `df` and IDF
- the ranked hits with BM25 score and document length
- each term's share of the score; these add up to the Lucene score
- the document text with matched terms highlighted

The same data is available as JSON:

```bash
curl "http://127.0.0.1:8000/api/search?q=vitamin+d+cancer&k=10&k1=0.9&b=0.4"
```

The latency shown is the wall-clock time of the Lucene search call on that machine.
It is meant for interactive use, not as a benchmark.

## Tests

```bash
# Unit tests + integration (synthetic fixture)
python -m pytest -q

# Termasuk test với dữ liệu NFCorpus thực (cần index đã build)
python -m pytest -q tests/test_real_index.py -v

# Chạy hết tất cả:
python -m pytest -v
```

| Test file | Mô tả |
|-----------|-------|
| `test_prepare_data.py` | Unit tests: ID preservation, hash check, overwrite guard, download |
| `test_build_index.py` | Integration: build Lucene index trên 4-doc fixture (cần pyserini) |
| `test_real_index.py` | Integration: kiểm tra index thực với NFCorpus (skip nếu chưa build) |

`test_real_index.py` kiểm tra:
- Số document khớp với corpus gốc (3633)
- Doc IDs truy xuất được
- BM25 scores giảm dần theo rank
- Mean nDCG@10 ≥ 0.28 trên 50 test queries
