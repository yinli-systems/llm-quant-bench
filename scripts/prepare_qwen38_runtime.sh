#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v1}
BASE_PYTHON=${BASE_PYTHON:-/ssd/scxi253/biomm/miniconda3/bin/python}
LMEVAL_COMMIT=b954108c9baaaa934b4ad842033b31a97ee30816
VLLM_VERSION=0.23.0
MINIMUM_FREE_GIB=160

[[ -x "$BASE_PYTHON" ]]
disk_anchor="$ROOT"
while [[ ! -e "$disk_anchor" && "$disk_anchor" != / ]]; do
  disk_anchor=$(dirname "$disk_anchor")
done
free_gib=$("$BASE_PYTHON" - "$disk_anchor" <<'PY'
import os
import sys

stats = os.statvfs(sys.argv[1])
print((stats.f_bavail * stats.f_frsize) // (1024**3))
PY
)
if (( free_gib < MINIMUM_FREE_GIB )); then
  echo "insufficient disk before runtime/model staging: ${free_gib} GiB free; need ${MINIMUM_FREE_GIB} GiB" >&2
  exit 1
fi

mkdir -p "$ROOT/source"
LMEVAL="$ROOT/source/lm-evaluation-harness"
if [[ ! -d "$LMEVAL/.git" ]]; then
  git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$LMEVAL"
fi
[[ -z "$(git -C "$LMEVAL" status --porcelain)" ]]
git -C "$LMEVAL" fetch --depth=1 origin "$LMEVAL_COMMIT"
git -C "$LMEVAL" checkout --detach "$LMEVAL_COMMIT"
[[ "$(git -C "$LMEVAL" rev-parse HEAD)" == "$LMEVAL_COMMIT" ]]

PY="$ROOT/venv/bin/python"
if [[ ! -x "$PY" ]]; then
  "$BASE_PYTHON" -m venv "$ROOT/venv"
fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "vllm==$VLLM_VERSION"
"$PY" -m pip install -e "${LMEVAL}[vllm]"
"$PY" -m pip check
"$PY" -c 'import datasets, lm_eval, ray, transformers, vllm'
"$PY" - "$ROOT" "$LMEVAL_COMMIT" "$VLLM_VERSION" <<'PY'
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
lm_eval_commit = sys.argv[2]
expected_vllm = sys.argv[3]
actual_vllm = importlib.metadata.version("vllm")
if actual_vllm != expected_vllm:
    raise SystemExit(f"vLLM version mismatch: {actual_vllm}")
freeze = subprocess.run(
    [sys.executable, "-m", "pip", "freeze", "--all"],
    text=True,
    capture_output=True,
    check=True,
).stdout
freeze_path = root / "RUNTIME_FREEZE.txt"
freeze_path.write_text(freeze, encoding="utf-8")
manifest = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "vllm": actual_vllm,
    "lm_eval_commit": lm_eval_commit,
    "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
}
(root / "RUNTIME_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2))
PY
