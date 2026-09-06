#!/usr/bin/env bash
# Stage licensed existing data and submit smoke jobs after runtime preparation.
set -euo pipefail
ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
RUNTIME_STAGING_PID=${RUNTIME_STAGING_PID:?Required live runtime staging PID}
SRC="$ROOT/source/llm-quant-bench"
PY="$ROOT/venv/bin/python"
PROTOCOL="$SRC/protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v4.json"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/smoke-dispatch.lock"
flock -n 9 || exit 1
[[ ! -e "$ROOT/logs/smoke-submissions.txt" ]]
while [[ ! -f "$ROOT/RUNTIME_MANIFEST.json" ]]; do
  if ! kill -0 "$RUNTIME_STAGING_PID" 2>/dev/null; then
    echo 'Runtime preparation stopped without a manifest' >&2
    exit 1
  fi
  sleep 20
done
export HF_HOME="$ROOT/hf-cache" HF_DATASETS_CACHE="$ROOT/hf-datasets"
export HF_ENDPOINT=https://hf-mirror.com
"$PY" "$SRC/scripts/stage_qwen38_datasets.py" \
  --protocol "$PROTOCOL" --runtime-root "$ROOT" \
  --gpqa-csv /ssd/scxi253/gpu-evidence-20260905/data/gpqa/gpqa_diamond.csv
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" "$SRC/scripts/validate_qwen38_pareto_protocol.py" \
  --protocol "$PROTOCOL" --runtime-root "$ROOT" \
  --lm-eval-root "$ROOT/source/lm-evaluation-harness" --verify-model-content \
  --out "$ROOT/logs/smoke-preflight.json"
cd "$SRC"
baseline=$(sbatch --parsable --job-name=q38-bf16-smoke \
  --output="$ROOT/logs/smoke-%j.out" --error="$ROOT/logs/smoke-%j.err" \
  --export=ALL,ARM=baseline,MODE=smoke \
  cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch)
[[ "$baseline" =~ ^[0-9]+$ ]]
printf 'baseline %s\n' "$baseline" >> "$ROOT/logs/smoke-submissions.txt"
candidate=$(sbatch --parsable --job-name=q38-fp8-smoke --dependency="afterok:$baseline" \
  --output="$ROOT/logs/smoke-%j.out" --error="$ROOT/logs/smoke-%j.err" \
  --export=ALL,ARM=candidate,MODE=smoke \
  cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch)
[[ "$candidate" =~ ^[0-9]+$ ]]
printf 'candidate %s\n' "$candidate" >> "$ROOT/logs/smoke-submissions.txt"
printf 'SMOKE_SUBMITTED baseline=%s candidate=%s\n' "$baseline" "$candidate"
