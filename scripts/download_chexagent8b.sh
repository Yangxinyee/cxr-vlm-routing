#!/bin/bash
# Download StanfordAIMI/CheXagent-8b to local disk (same path as MedGemma).
# CheXagent is chest X-ray specific, no gating - no login required.
#
# Usage:
#   bash scripts/download_chexagent8b.sh
#   HF_HUB_ENABLE_HF_TRANSFER=1 bash scripts/download_chexagent8b.sh  # faster
#   MAX_WORKERS=32 bash scripts/download_chexagent8b.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export LOCAL_DIR="${LOCAL_DIR:-models/CheXagent-8b}"
export MAX_WORKERS="${MAX_WORKERS:-16}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

cd "$PROJECT_ROOT"
python scripts/download_chexagent8b.py
