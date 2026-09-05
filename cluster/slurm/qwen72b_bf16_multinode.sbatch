#!/usr/bin/env bash
#SBATCH --partition=gpu_4090
#SBATCH --qos=gpugpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
# Two GPUs per node are capped at 12 CPU cores on gpu_4090. Twelve cores also
# remain valid for the four-GPU-per-node topology.
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00

set -euo pipefail
ROOT=/ssd/scxi253/gpu-evidence-20260905
source "$ROOT/source/orchestration/runtime_env.sh"
SRC="$ROOT/source/llm-quant-bench"
PY="$ROOT/venv/bin/python"
RAY="$ROOT/venv/bin/ray"
MODEL="$ROOT/models/qwen25-72b-bf16"
SERVED=qwen25-72b-bf16
RUN="$ROOT/runs/qwen25-72b-retention/bf16"
ITEMS="$ROOT/data/qwen25-72b-quality-items-v1.jsonl"
SMOKE_ITEMS="$ROOT/data/qwen25-72b-quality-smoke-v1.jsonl"
API_PORT=8001
RAY_PORT=6379
EXPECTED_NODES=${EXPECTED_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-2}
[[ $((EXPECTED_NODES * GPUS_PER_NODE)) == $((TP_SIZE * PP_SIZE)) ]]
"$PY" -c 'import pkg_resources'

mkdir -p "$RUN"
[[ -f "$MODEL/.READY" && -f "$ITEMS" && -f "$SMOKE_ITEMS" && -x "$PY" && -x "$RAY" ]]
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export TMPDIR="$ROOT/tmp"
# Ray uses TMPDIR to construct AF_UNIX socket paths. The shared experiment path
# is long enough to exceed Linux's 107-byte socket limit on worker nodes.
export RAY_TMPDIR=/tmp
# Keep compiler caches node-local. Pipeline stages compile the same bootstrap
# kernels concurrently, and a shared /ssd cache makes their atomic directory
# replacement race across hosts. A common /tmp path is isolated per node.
export TRITON_CACHE_DIR="/tmp/triton-cache-$SLURM_JOB_ID"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-cache-$SLURM_JOB_ID"
export XDG_CONFIG_HOME="$ROOT/tmp/xdg-config-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/tmp/xdg-cache-$SLURM_JOB_ID"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Several compute nodes also expose a link-local 169.254/16 USB interface.
# NCCL's automatic interface selection is nondeterministic across those nodes;
# force all distributed control/data traffic onto the routable cluster bond.
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#NODES[@]}" == "$EXPECTED_NODES" ]]
HEAD_NODE=${NODES[0]}
WORKER_NODES=("${NODES[@]:1}")
# Avoid link-local addresses, which are visible in `hostname -I` but are not
# routable from Ray workers on the other Slurm nodes.
HEAD_IP=$(srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" hostname -I | \
  awk '{for (i = 1; i <= NF; i++) if ($i !~ /^127\./ && $i !~ /^169\.254\./ && $i !~ /:/) {print $i; exit}}')
[[ -n "$HEAD_IP" ]]

srun --nodes="$EXPECTED_NODES" --ntasks="$EXPECTED_NODES" --ntasks-per-node=1 \
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "$RUN/gpus-${SLURM_JOB_ID}.csv"
cp "$MODEL/SNAPSHOT_MANIFEST.json" "$RUN/model_snapshot_manifest.json"
cp "$ROOT/data/qwen25-72b-quality-items-v1.sha256" "$RUN/frozen_items.sha256"

SERVER_PID=
WORKER_STEP_PIDS=()
CLUSTER_STOPPED=0
cleanup() {
  if [[ "$CLUSTER_STOPPED" == 1 ]]; then
    return
  fi
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  # The blocking worker `srun` steps own the worker nodes. Terminate and reap
  # those exact steps before asking Slurm for an all-node telemetry step.
  # Launching another non-overlapping `srun` while they are alive deadlocks.
  for worker_step_pid in "${WORKER_STEP_PIDS[@]}"; do
    kill "$worker_step_pid" >/dev/null 2>&1 || true
  done
  for worker_step_pid in "${WORKER_STEP_PIDS[@]}"; do
    wait "$worker_step_pid" >/dev/null 2>&1 || true
  done
  "$RAY" stop --force >/dev/null 2>&1 || true
  CLUSTER_STOPPED=1
}
trap cleanup EXIT

"$RAY" start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" \
  --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" \
  --include-dashboard=false --disable-usage-stats
for worker_node in "${WORKER_NODES[@]}"; do
  srun --nodes=1 --ntasks=1 --nodelist="$worker_node" \
    "$RAY" start --address="$HEAD_IP:$RAY_PORT" \
    --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" \
    --disable-usage-stats --block &
  WORKER_STEP_PIDS+=("$!")
done

RAY_READY=0
for _ in $(seq 1 60); do
  if "$PY" - "$HEAD_IP:$RAY_PORT" "$EXPECTED_NODES" \
    "$((EXPECTED_NODES * GPUS_PER_NODE))" <<'PY'
import ray
import sys

ray.init(address=sys.argv[1], logging_level="ERROR")
alive = [node for node in ray.nodes() if node["Alive"]]
gpus = ray.cluster_resources().get("GPU", 0)
ray.shutdown()
expected_nodes = int(sys.argv[2])
expected_gpus = int(sys.argv[3])
raise SystemExit(0 if len(alive) == expected_nodes and gpus == expected_gpus else 1)
PY
  then
    RAY_READY=1
    break
  fi
  sleep 2
done
[[ "$RAY_READY" == 1 ]]
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"

"$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --tensor-parallel-size "$TP_SIZE" \
  --pipeline-parallel-size "$PP_SIZE" \
  --distributed-executor-backend ray \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.96 \
  --compilation-config 0 \
  --disable-custom-all-reduce \
  --disable-log-requests \
  --enforce-eager \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  > "$RUN/server.log" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 300 "$RUN/server.log"
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$API_PORT/v1/models" > "$RUN/models.json"; then
    READY=1
    break
  fi
  sleep 15
done
[[ "$READY" == 1 ]]

"$PY" "$SRC/scripts/run_quality_eval.py" \
  --out "$RUN/smoke" \
  --base-url "http://127.0.0.1:$API_PORT/v1" \
  --model "$SERVED" \
  --benchmarks mmlu cmmlu gsm8k \
  --items-file "$SMOKE_ITEMS" \
  --concurrency 3 \
  --temperature 0 \
  --timeout-s 600

"$PY" - "$RUN/smoke/summary.json" "$SMOKE_ITEMS" <<'PY'
import hashlib
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
item_bytes = open(sys.argv[2], "rb").read()
assert summary["items_sha256"] == hashlib.sha256(item_bytes).hexdigest()
assert summary["total_items"] == 3
assert summary["ok_items"] == 3
assert summary["failed_items"] == 0
assert set(summary["by_benchmark"]) == {"mmlu", "cmmlu", "gsm8k"}
PY

"$PY" "$SRC/scripts/run_quality_eval.py" \
  --out "$RUN/eval" \
  --base-url "http://127.0.0.1:$API_PORT/v1" \
  --model "$SERVED" \
  --benchmarks mmlu cmmlu gsm8k \
  --items-file "$ITEMS" \
  --concurrency 8 \
  --temperature 0 \
  --timeout-s 600 \
  --sample-log-every 100

"$PY" - "$RUN/eval/summary.json" "$ITEMS" <<'PY'
import hashlib
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
item_bytes = open(sys.argv[2], "rb").read()
expected_items = sum(bool(line.strip()) for line in item_bytes.splitlines())
assert summary["items_sha256"] == hashlib.sha256(item_bytes).hexdigest()
assert summary["total_items"] == expected_items
assert summary["ok_items"] == expected_items
assert summary["failed_items"] == 0
assert set(summary["by_benchmark"]) == {"mmlu", "cmmlu", "gsm8k"}
PY

if grep -Eiq 'CUDA[^[:alnum:]]+out of memory|Traceback \(most recent call last\)|Killed process|segmentation fault|fatal error' "$RUN/server.log"; then
  exit 1
fi
# Release the long-lived server and Ray worker steps before the final all-node
# snapshot. The EXIT trap remains as an idempotent fallback for earlier errors.
cleanup
srun --nodes="$EXPECTED_NODES" --ntasks="$EXPECTED_NODES" --ntasks-per-node=1 \
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader > "$RUN/gpus-final-${SLURM_JOB_ID}.csv"
sha256sum "$RUN/eval/summary.json" "$RUN/eval/samples.jsonl" > "$RUN/SHA256SUMS"
