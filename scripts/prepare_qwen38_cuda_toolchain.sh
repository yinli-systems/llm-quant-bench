#!/usr/bin/env bash
# Isolated compiler pins matching torch's CUDA 13.0 runtime. Never upgrades the venv.
set -euo pipefail
ROOT=${QWEN38_ROOT:-/ssd/scxi253/qwen38-27b-pareto-v2}
PY="$ROOT/venv/bin/python"
TARGET="$ROOT/cuda-toolchain-13.0"
[[ ! -e "$TARGET" ]] || { echo "Refusing to overwrite existing toolchain: $TARGET" >&2; exit 1; }
export PIP_CACHE_DIR="$ROOT/pip-cache"
export TMPDIR="$ROOT/tmp-cuda-toolchain"
mkdir -p "$TMPDIR"
"$PY" -m pip install --ignore-installed --no-deps --target "$TARGET" \
  --report "$ROOT/logs/cuda-toolchain-install-report.json" \
  nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 \
  nvidia-nvvm==13.0.88 nvidia-cuda-runtime==13.0.96
"$PY" - "$TARGET" <<'PY'
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

target = Path(sys.argv[1])
cuda = target / "nvidia/cu13"
nvcc = subprocess.check_output([str(cuda / "bin/nvcc"), "--version"], text=True)
assert "release 13.0," in nvcc, nvcc
header = cuda / "include/cuda_runtime_api.h"
assert "#define CUDART_VERSION  13000" in header.read_text()
files = {}
for path in sorted(target.rglob("*")):
    if path.is_file():
        files[str(path.relative_to(target))] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "status": "prepared_not_gpu_validated",
    "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(target)])},
    "nvcc": nvcc, "header_cuda_version": 13000, "files_sha256": files,
}
(target / "TOOLCHAIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: v for k, v in manifest.items() if k != "files_sha256"}, indent=2))
PY
