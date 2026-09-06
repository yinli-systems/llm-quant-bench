# Qwen3.8 GPU smoke-test recovery — 2026-09-06

No scored Qwen3.8 samples were produced by the attempts below. This report
records runtime failures, not BF16/FP8 quality or throughput measurements.

The active scoring protocol is v4 (text `vllm` backend, unchanged pinned
model/dataset revisions, thinking parameters and 32,768-token generation
budget). The earlier `VALIDATION.md` records the historical v3 CPU preflight.

| Job | Purpose | Observed outcome |
|---|---|---|
| 1560916 | BF16, four RTX 4090s, 15-sample smoke | Failed before generation, after 13m08s; FlashInfer could not discover `nvcc`. |
| 1560917 | FP8 after successful BF16 | Never started; dependency became unsatisfiable. Canceled only this queued job. |
| 1560956 | Isolated one-GPU FlashInfer sampler | Failed after 19s; `ninja` was installed but missing from PATH. |
| 1560962 | Repeat sampler with corrected PATH | Reached compilation; CUDA compiler and toolkit header versions disagreed. |
| 1560981 | Isolated matching CUDA 13.0 compiler | Version mismatch resolved; failed after 36s because the minimal toolchain lacked cuRAND headers. |
| 1560994 | Complete compiler and headers | All three CUDA translation units compiled; failed after 2m01s linking `-lcudart`, because NVIDIA wheels omit the unversioned development linker name. |

BF16 job 1560916 did verify all 18 checkpoint shards could be loaded by
vLLM 0.23.0 on TP4. Rank 0 logged 13.11 GiB model-loading memory and 56.06s
weight loading. These exclude cache reservation and steady-state generation
and must not be reported as peak serving footprint or inference throughput.

## Root causes and scoped repairs

- Added the pinned virtualenv executable directory and CUDA compiler directory
  to the job PATH. Added fail-fast checks for compiler, headers and `ninja`.
- Runtime packaging had installed nvcc 13.2.86, CRT 13.3.73 and NVVM 13.2.86
  alongside CUDA runtime 13.0.96. `pip check` did not detect this because the
  compiler's dependencies have no version constraints. The actual FlashInfer
  JIT rejected the compiler/header mismatch.
- Added an isolated `cuda-toolchain-13.0-r2` installation with nvcc/CRT/NVVM
  13.0.88, runtime 13.0.96, cuRAND 10.4.0.35 and CCCL 13.0.85.
  The original minimal toolchain is retained. These component pins match the existing NVIDIA
  CUDA Toolkit 13.0.2 package metadata. The original virtualenv is untouched.
  Preparation alone does not prove the repaired GPU path works.
- Redirected vLLM, Triton, TorchInductor, CUDA and FlashInfer caches to
  job-specific scratch. The original vLLM default used the account's 1 GiB
  home filesystem; this was a separate risk, not the observed failure cause.
- Added a job-local `libcudart.so` link to the exact toolchain's
  `lib/libcudart.so.13`, with explicit compiler and runtime library search paths.
  This does not modify the installed toolkit or its content receipt.
- All failed logs and run receipts are retained on the cluster. No checkpoint,
  dataset, existing result, or unrelated running job was removed by recovery.

The isolated sampler must compile and return the expected two token IDs on
an RTX 4090 before restarting the full BF16/FP8 smoke pair. Formal quality
and performance conclusions remain gated on subsequent complete runs.
