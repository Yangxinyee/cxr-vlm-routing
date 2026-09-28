#!/usr/bin/env bash
# Run every prompt style for one model on one dataset.
# Usage: MODEL=google/medgemma-4b-it VLLM_URL=http://127.0.0.1:8000 TAG=medgemma4b_vllm \
#        bash scripts/run_all.sh data/mimic_formal_50
set -euo pipefail
DATASET="${1:-data/mimic_formal_50}"
MODEL="${MODEL:?set MODEL to a model id or local path}"
ARGS=(--data-dir "$DATASET" --model-path "$MODEL")
[[ -n "${VLLM_URL:-}" ]] && ARGS+=(--vllm-url "$VLLM_URL")
[[ -n "${TAG:-}" ]] && ARGS+=(--experiment-tag "$TAG")
for STYLE in structured concise labels_only; do
  echo "=== $(basename "$DATASET") / $MODEL / $STYLE ==="
  python scripts/run_toy_experiment.py "${ARGS[@]}" --prompt-style "$STYLE"
done
