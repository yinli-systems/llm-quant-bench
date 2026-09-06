#!/usr/bin/env bash
#SBATCH --job-name=q38-cuda-smoke
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
set -euo pipefail
ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
SRC="$ROOT/source/llm-quant-bench"
PY="$ROOT/venv/bin/python"
export TMPDIR="/tmp/qwen38-cuda-$SLURM_JOB_ID"
source "$SRC/scripts/qwen38_job_env.sh"
export MAX_JOBS=4
"$PY" "$SRC/scripts/qwen38_cuda_smoke.py"
