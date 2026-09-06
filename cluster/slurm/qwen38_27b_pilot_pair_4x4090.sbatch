#!/usr/bin/env bash
#SBATCH --job-name=q38-pilot-pair
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
set -euo pipefail
ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
SRC="$ROOT/source/llm-quant-bench"
export MODE=pilot
for ARM in baseline candidate; do
  export ARM
  # Execute within this allocation: both arms inherit exactly the same GPUs.
  bash "$SRC/cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch"
done
cmp "$ROOT/runs/standard-quality/baseline/pilot/job-$SLURM_JOB_ID/gpus-start.csv" \
    "$ROOT/runs/standard-quality/candidate/pilot/job-$SLURM_JOB_ID/gpus-start.csv"
echo "PILOT_PAIR_COMPLETED job=$SLURM_JOB_ID"
