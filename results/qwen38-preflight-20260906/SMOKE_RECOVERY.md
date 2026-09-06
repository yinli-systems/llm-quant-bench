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
| 1561008 | Complete toolchain and job-local linker compatibility | **Passed** on RTX 4090 after 2m03s, exit 0. Real FlashInfer JIT sampling returned expected token IDs `[7, 42]`. |

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

The isolated sampler gate passed under source `62ff725`, torch `2.11.0+cu130`,
FlashInfer `0.6.12`, and nvcc `13.0.88`. Raw output is retained at
`/ssd/scxi253/qwen38-27b-pareto-v2/logs/cuda-smoke-1561008.out`.
The raw JSON is also archived alongside this report as
`cuda-smoke-1561008.json`, SHA-256
`5feae4272519359130d6516bb7830f9fb5690b7cfeb6ad8f2875cc7cec2b5d28`.
The four-GPU model smoke pair was restarted under source `62ff725`:
BF16 `1561019`, FP8 `1561020` with an `afterok` dependency on BF16.
Both model jobs completed with exit 0: BF16 in 17m44s, FP8 in 17m51s.
Each produced 15 scored samples, with zero length-limited completions and
zero unclosed thinking sections. Both run SHA256SUMS archives verified.
BF16 generated 9,228 tokens in 186.65s of instrumented generation calls;
FP8 generated 10,172 tokens in 225.67s. These include different output
lengths and cold JIT effects and are **not a matched-shape speed benchmark**.

All paired document, target and prompt hashes match. MMLU-Pro scored 11/14
in each arm, with identical parsed answer letters. The single GPQA response
was `[invalid]` under the frozen upstream parser in both arms: each used a
LaTeX boxed answer, which the upstream extraction rule did not recognize.
The original GPQA score remains 0/1 in each arm; no post-hoc score correction
was applied. These 15 development samples establish pipeline execution,
not general quality retention or formal GPQA accuracy.

The two jobs used the same node but only three of four GPU UUIDs matched.
Added a sequential, single-allocation 10% pilot wrapper so both future arms
use the same four devices. The pilot covers 1,229 samples per arm (20 GPQA
and 1,209 MMLU-Pro) and retains the original v4 prompts and scoring rules.
Parsing failure rates must be reported separately; pilot/full runs are not
to be presented as an official Qwen score reproduction.

Pilot pair job `1561180` was submitted from source `2c0f118` and observed
RUNNING on `wqd10nba06g8` with four RTX 4090s. Its paired outputs are under
`runs/standard-quality/{baseline,candidate}/pilot/job-1561180` at the same
remote root. A thread heartbeat checks every 15 minutes, with notifications
reserved for meaningful changes, completion, failures or required decisions.
The first 15 samples imply roughly 9–12 hours for both pilot arms together;
this is a rough extrapolation, not a guaranteed completion time.
