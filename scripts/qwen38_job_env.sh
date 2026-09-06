#!/usr/bin/env bash
# Source before importing CUDA clients. Does not alter Python package versions.
: "${PY:?Set the pinned runtime Python}"
: "${TMPDIR:?Set a job-specific scratch directory}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export CUDA_CACHE_PATH="$TMPDIR/cuda"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
export CUDA_HOME
CUDA_HOME=$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')
[[ -x "$CUDA_HOME/bin/nvcc" && -f "$CUDA_HOME/include/cuda_runtime.h" ]] || {
  echo "Pinned CUDA compiler or headers are missing: $CUDA_HOME" >&2
  return 1
}
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
command -v ninja >/dev/null || {
  echo "ninja is missing from the pinned runtime" >&2
  return 1
}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE"
