# Edge RAG — NFCorpus

Research code for retrieval-augmented generation on edge devices, using the
[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) dataset in
[BEIR](https://github.com/beir-cellar/beir) format.

**Current status:** NFCorpus data preparation, a BM25 Lucene index built with
[Pyserini](https://github.com/castorini/pyserini) (stage 1), and automatic BM25
k1/b tuning on the dev split (stage 2). RM3, dense retrieval, hybrid fusion (RRF)
and LLM generation are not implemented yet.

## Layout

```
src/bm25/prepare_data.py      BEIR corpus.jsonl -> Pyserini JsonCollection ({id, contents})
src/bm25/build_index.py       Lucene index build + smoke test + build record
src/bm25/search_app.py        Local web UI / JSON API for trying BM25 queries
src/bm25/evaluate.py          Shared query/qrels loaders + trec_eval metrics (NDCG@10, MAP@10, R@100/1000, MRR@10)
src/bm25/tune.py              Automatic BM25 k1/b tuning on the dev split (grid or Optuna TPE)
scripts/run_pipeline.sh       One-shot script: prepare → build → test
scripts/test_retrieval.py     Chạy BM25 queries + tính nDCG@10 trên NFCorpus test split
tests/test_prepare_data.py    Unit tests: corpus conversion, download, overwrite guard
tests/test_build_index.py     Integration test: build index trên 4-doc fixture
tests/test_real_index.py      Integration test: kiểm tra index với dữ liệu NFCorpus thực
tests/test_tune.py            Tuning tests trên fixture (grid, resume, TPE, lỗi trial, chặn test split)
results/bm25/index_build.json Build record (git-ignored)
results/bm25/tuning_results.csv  One row per tuning trial (git-ignored)
results/bm25/best_config.json    Best k1/b on dev + metrics + environment (git-ignored)
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

## Stage 2: automatic BM25 tuning (k1, b)

`src/bm25/tune.py` searches k1 ∈ [0.4, 2.0] and b ∈ [0.2, 1.0] on the **dev split**
and keeps the configuration with the best NDCG@10.

- The existing index is opened once with `LuceneSearcher`. Each trial only calls
  `set_bm25(k1, b)` and then `batch_search` over all dev queries (top 1000 hits),
  so the index is never rebuilt.
- Each run is scored with the official **trec_eval 9.0.4** bundled in Pyserini's
  Anserini jar. It's called in-process through `io.anserini.eval.TrecEval`, with
  `-c`, so a query with no hits counts as 0. The objective is NDCG@10 (`ndcg_cut.10`).
  MAP@10 (`map_cut.10`), Recall@100/1000 (`recall.100,1000`) and MRR@10
  (`recip_rank` with `-M 10`) are also recorded.
- The Pyserini default **k1=0.9, b=0.4** is always evaluated first as a reference.

```bash
# Grid search (step 0.1 → 17 × 9 = 153 configs), visited in a seeded shuffled order
python -m src.bm25.tune --method grid --max-trials 25
python -m src.bm25.tune --method grid --max-trials 153 --resume      # continue to the full grid
python -m src.bm25.tune --method grid --k1-step 0.2 --b-step 0.2      # coarser grid

# Optuna TPE (optional dependency, install it explicitly first)
pip install -r requirements-tuning.txt                                # optuna==5.0.0
python -m src.bm25.tune --method tpe --max-trials 50 --seed 42
```

Main options:
`--method {grid,tpe}`, `--max-trials N`, `--resume`, `--seed`,
`--k1-min/--k1-max/--b-min/--b-max`, `--k1-step/--b-step` (grid; TPE uses 0.01),
`--split {dev,train}`, `--data-dir data/nfcorpus`, `--index-dir`, `--hits 1000`, `--threads`,
`--results-csv`, `--best-config`.

**Dev split.** The tuner reads `data/nfcorpus/queries.jsonl` and
`data/nfcorpus/qrels/dev.tsv`, which come with the BEIR NFCorpus zip. Before it
runs, it checks that:

- `dev.tsv` exists and has judgments
- every dev query ID has text in `queries.jsonl`
- no dev query ID appears in `test.tsv`

`--split` only accepts `dev` or `train`, so the **test split can't be used for
tuning**. If `dev.tsv` is missing, the run stops and suggests restoring the
official file or tuning on `--split train`. Report test numbers once, with the
selected configuration.

**Outputs.**

- **`results/bm25/tuning_results.csv`:** one row per trial, appended and fsync'ed
  as soon as the trial ends. Columns: `trial, method, k1, b, ndcg@10, map@10,
  recall@100, recall@1000, mrr@10, num_queries, duration_s, status, error, split,
  seed, timestamp_utc`.
  - A failed trial is written with `status=error` and its error message, never
    dropped silently. The run stops after 5 consecutive failures.
- **`results/bm25/best_config.json`:** rewritten after every trial. It holds:
  - the best k1/b with all metrics, plus the default k1=0.9/b=0.4 metrics for comparison
  - the search settings
  - the qrels file and its SHA-256
  - the index, environment versions (Python, Java, Pyserini, Lucene, Optuna) and git commit

**Resume and overwrite protection.** Without `--resume`, the tuner refuses to
touch an existing CSV or best-config file. With `--resume`, completed
configurations are skipped and failed ones are retried. `--max-trials` is the
total number of successful trials to reach, counting resumed ones. For TPE,
earlier trials are loaded into the Optuna study first.

Tuning never writes to `data/` or the index, and never touches
`results/bm25/index_build.json`.

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
| `test_search_app.py` | Web UI / JSON API trên index fixture |
| `test_tune.py` | Tuning: grid, resume, TPE (skip nếu chưa cài Optuna), lỗi trial, metric so với tính tay |

`test_real_index.py` kiểm tra:
- Số document khớp với corpus gốc (3633)
- Doc IDs truy xuất được
- BM25 scores giảm dần theo rank
- Mean nDCG@10 ≥ 0.28 trên 50 test queries
