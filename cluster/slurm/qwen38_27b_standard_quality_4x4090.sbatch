#!/usr/bin/env bash
#SBATCH --job-name=q38-quality
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00

set -euo pipefail

ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
ARM=${ARM:?Set ARM=baseline or ARM=candidate}
MODE=${MODE:-smoke}
SRC="$ROOT/source/llm-quant-bench"
LMEVAL="$ROOT/source/lm-evaluation-harness"
PY="$ROOT/venv/bin/python"
PROTOCOL=${QWEN38_PROTOCOL:-$SRC/protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v2.json}

case "$ARM" in
  baseline)
    MODEL="$ROOT/models/qwen38-27b-bf16"
    ;;
  candidate)
    MODEL="$ROOT/models/qwen38-27b-fp8"
    ;;
  *)
    echo "ARM must be baseline or candidate" >&2
    exit 2
    ;;
esac

LIMIT_ARGS=()
case "$MODE" in
  smoke)
    LIMIT_ARGS=(--limit 1)
    ;;
  pilot)
    LIMIT_ARGS=(--limit 0.1)
    ;;
  full)
    ;;
  *)
    echo "MODE must be smoke, pilot, or full" >&2
    exit 2
    ;;
esac

[[ -x "$PY" && -d "$SRC" && -d "$LMEVAL" && -d "$MODEL" ]]
RUN="$ROOT/runs/standard-quality/$ARM/$MODE/job-$SLURM_JOB_ID"
mkdir -p "$RUN"

export HF_HOME="$ROOT/hf-cache"
export HF_DATASETS_CACHE="$ROOT/hf-datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export TMPDIR="/tmp/qwen38-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

"$PY" "$SRC/scripts/validate_qwen38_pareto_protocol.py" \
  --protocol "$PROTOCOL" \
  --lm-eval-root "$LMEVAL" \
  --runtime-root "$ROOT" \
  --check-gpus \
  --verify-model-content \
  --out "$RUN/preflight.json"

nvidia-smi \
  --query-gpu=index,name,uuid,compute_cap,memory.total,driver_version \
  --format=csv,noheader > "$RUN/gpus-start.csv"
cp "$PROTOCOL" "$RUN/protocol.json"
cp "$MODEL/SNAPSHOT_MANIFEST.json" "$RUN/model_snapshot_manifest.json"
cp "$ROOT/data/qwen38-standard-datasets-receipt.json" "$RUN/dataset_receipt.json"
cp "$ROOT/data/qwen38-standard-datasets.READY" "$RUN/dataset_receipt.READY"
git -C "$SRC" rev-parse HEAD > "$RUN/source_revision.txt"

"$PY" "$SRC/scripts/run_qwen38_standard_eval.py" \
  --protocol "$PROTOCOL" \
  --lm-eval-root "$LMEVAL" \
  --arm "$ARM" \
  --model-path "$MODEL" \
  --gpqa-csv "$ROOT/data/gpqa_diamond.csv" \
  --out "$RUN/eval" \
  "${LIMIT_ARGS[@]}"

if grep -Eiq \
  'CUDA[^[:alnum:]]+out of memory|Traceback \(most recent call last\)|Killed process|segmentation fault|fatal error' \
  "$RUN/eval/lm_eval.log"; then
  echo "fatal signature found in lm-eval log" >&2
  exit 1
fi

nvidia-smi \
  --query-gpu=index,name,uuid,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader > "$RUN/gpus-final.csv"
(
  cd "$RUN"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "$RUN/SHA256SUMS"
