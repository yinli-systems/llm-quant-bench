#!/usr/bin/env bash
# Source before importing CUDA clients. Does not alter Python package versions.
: "${PY:?Set the pinned runtime Python}"
: "${ROOT:?Set the project runtime root}"
: "${TMPDIR:?Set a job-specific scratch directory}"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
export CUDA_CACHE_PATH="$TMPDIR/cuda"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export FLASHINFER_WORKSPACE_BASE="$TMPDIR/flashinfer"
export CUDA_HOME
CUDA_HOME="$ROOT/cuda-toolchain-13.0-r2/nvidia/cu13"
[[ -f "$ROOT/cuda-toolchain-13.0-r2/TOOLCHAIN_MANIFEST.json" ]] || {
  echo "Run prepare_qwen38_cuda_toolchain.sh before GPU evaluation" >&2
  return 1
}
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
# NVIDIA wheels provide lib/libcudart.so.13 without the development linker
# name expected by FlashInfer. Keep the compatibility link on job scratch.
[[ -f "$CUDA_HOME/lib/libcudart.so.13" ]] || return 1
mkdir -p "$TMPDIR/cuda-link"
if [[ ! -e "$TMPDIR/cuda-link/libcudart.so" && ! -L "$TMPDIR/cuda-link/libcudart.so" ]]; then
  ln -s "$CUDA_HOME/lib/libcudart.so.13" "$TMPDIR/cuda-link/libcudart.so"
fi
[[ "$(readlink "$TMPDIR/cuda-link/libcudart.so")" == "$CUDA_HOME/lib/libcudart.so.13" ]] || return 1
export LIBRARY_PATH="$TMPDIR/cuda-link${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
