#!/usr/bin/env bash
# Continue from a live model downloader without tying installation to SSH.
set -euo pipefail
ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
MODEL_STAGING_PID=${MODEL_STAGING_PID:?Required live model staging PID}
SRC="$ROOT/source/llm-quant-bench"
while [[ ! -f "$ROOT/models/qwen38-27b-bf16/.READY" || ! -f "$ROOT/models/qwen38-27b-fp8/.READY" ]]; do
  if ! kill -0 "$MODEL_STAGING_PID" 2>/dev/null; then
    echo 'Model staging stopped before both receipts completed' >&2
    exit 1
  fi
  sleep 20
done
echo 'Both model receipts completed; preparing runtime'
export PIP_NO_CACHE_DIR=1
export QWEN38_ROOT="$ROOT"
bash "$SRC/scripts/prepare_qwen38_runtime.sh"
echo 'RUNTIME_PREPARATION_COMPLETE'
