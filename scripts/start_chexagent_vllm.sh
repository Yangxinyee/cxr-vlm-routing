#!/bin/bash
# Start CheXagent-8b via vLLM (OpenAI-compatible API).
# CheXagent uses custom architecture - vLLM may require trust-remote-code.
# If vLLM fails to load CheXagent, use local inference: python scripts/run_toy_experiment.py --model-path models/CheXagent-8b
#
# Usage:
#   bash scripts/start_chexagent_vllm.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_chexagent_vllm.sh  # 2 GPUs

set -e
MODEL_PATH="${MODEL_PATH:-models/CheXagent-8b}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-1}"  # CheXagent-8B fits on 1 GPU

echo "Starting CheXagent vLLM server: $MODEL_PATH"
echo "Tensor parallel size: $TP_SIZE"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --trust-remote-code
