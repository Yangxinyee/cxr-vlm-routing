#!/bin/bash
# Download google/medgemma-27b-it to local disk (HF format for vLLM).
# Requires: huggingface-cli login (MedGemma is gated - accept terms on HF first)
#
# Usage:
#   bash scripts/download_medgemma27b.sh
#   HF_HUB_ENABLE_HF_TRANSFER=1 bash scripts/download_medgemma27b.sh  # faster

set -e
REPO="google/medgemma-27b-it"
LOCAL_DIR="${LOCAL_DIR:-models/medgemma-27b-it}"

echo "Downloading $REPO to $LOCAL_DIR"
echo "Ensure you have: huggingface-cli login (and accepted MedGemma terms on HF)"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
huggingface-cli download "$REPO" \
  --local-dir "$LOCAL_DIR" \
  --local-dir-use-symlinks False

echo "Done. Model at: $LOCAL_DIR"
echo ""
echo "Start vLLM server:"
echo "  python -m vllm.entrypoints.openai.api_server \\"
echo "      --model $LOCAL_DIR \\"
echo "      --host 0.0.0.0 --port 8000 \\"
echo "      --tensor-parallel-size 2 --gpu-memory-utilization 0.85 \\"
echo "      --max-model-len 2048 --max-num-seqs 32 --trust-remote-code"
echo ""
echo "Run toy experiment:"
echo "  python scripts/run_toy_experiment.py --vllm-url http://localhost:8000"
