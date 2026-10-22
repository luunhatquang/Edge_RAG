#!/usr/bin/env bash
# run_pipeline.sh — chạy toàn bộ pipeline BM25 với dữ liệu NFCorpus có sẵn.
#
# Usage:
#   bash scripts/run_pipeline.sh               # lần đầu
#   bash scripts/run_pipeline.sh --rebuild      # build lại từ đầu
#
# Yêu cầu: Python >= 3.12, Java 21+, pyserini đã cài trong .venv

set -euo pipefail
cd "$(dirname "$0")/.."

REBUILD=""
if [[ "${1:-}" == "--rebuild" ]]; then
  REBUILD="--rebuild"
fi

RAW_DIR="data/nfcorpus"
PROCESSED_DIR="data/processed/nfcorpus"
INDEX_DIR="indexes/bm25/nfcorpus-flat"

echo "============================================"
echo "  Edge RAG — BM25 Pipeline"
echo "============================================"

# Activate venv nếu tồn tại
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

# Bước 1: Chuẩn bị corpus
echo ""
echo "[1/3] Chuẩn bị corpus..."
python -m src.bm25.prepare_data \
  --raw-dir "$RAW_DIR" \
  --output-dir "$PROCESSED_DIR" \
  $REBUILD

# Bước 2: Build Lucene index
echo ""
echo "[2/3] Build BM25 index..."
python -m src.bm25.build_index \
  --prepared-dir "$PROCESSED_DIR" \
  --index-dir "$INDEX_DIR" \
  --k1 0.9 --b 0.4 \
  $REBUILD

# Bước 3: Quick retrieval test
echo ""
echo "[3/3] Quick retrieval test..."
python scripts/test_retrieval.py

echo ""
echo "============================================"
echo "  Pipeline hoàn thành!"
echo "============================================"
