# Qwen3.8-27B pilot: BF16 completeness failure, FP8 not run

Status: **FAILED_BASELINE_COMPLETENESS_CANDIDATE_NOT_RUN**.
This is a completed audit of a failed pilot, not a passing quality comparison.
Original generation settings, samples, filters, scores and v4 gates are unchanged.

## Execution and coverage

- Slurm job `1561180`, node `wqd10nba06g8`, four RTX 4090 GPUs.
- UTC start `2026-09-06T12:28:33`; end `2026-09-07T01:10:57`.
- Allocation elapsed `12:42:24` (50.83 GPU-hours); Slurm `FAILED`, exit `1:0`.
- All **1,229** requests completed and all primary-filter samples were saved:
  GPQA 20, MMLU-Pro 1,209 across 14 subjects. Zero duplicate task/doc IDs,
  missing provenance hashes, or empty saved responses in the audited samples.
- GPQA has 40 JSONL records because both strict and flexible filters are logged;
  this is 20 unique items, not 40 generation requests.
- The pinned task-integrity receipt passed. Per-task coverage equals the ceiling
  of 10% of each pinned split; scores recomputed from the existing binary
  `exact_match` fields agree with the harness aggregate JSON.
- The wrapper stopped on the explicit generation-completeness gate. No traceback,
  CUDA OOM, Ninja failure or engine-init failure signature was found in the eval log.
- FP8's pilot directory is absent. No restart, duplicate submission, extra model
  deletion, full evaluation or performance benchmark was performed.

## Original scores: diagnostic only

| Task | Correct / items | Original score | Primary-filter invalid |
|---|---:|---:|---:|
| GPQA Diamond | 12 / 20 | 60.00% | 4 / 20 (20.00%) |
| MMLU-Pro | 1,008 / 1,209 | 83.3747% | 16 / 1,209 (1.3234%) |

GPQA uses the preregistered `flexible-extract` primary filter, not strict-match
(whose recorded score is 0/20). All four flexible-filter invalid saved responses
contain LaTeX boxed text. This is a format diagnostic, not proof that those answers
are correct. They remain invalid and retain their original zero scores.
No alternative answer extraction or post-hoc score correction was performed.

These pilot fractions are not official scores or estimates with full-benchmark
coverage. FP8 retention, paired confidence intervals and answer disagreements are
**not assessable**, since no FP8 pilot samples exist.

## Completeness failure and observability limits

- 9 / 1,229 requests (0.7323%) ended at the 32,768-token limit.
- 26 / 1,229 requests (2.1155%) lacked `</think>` in the original completion
  telemetry. This includes all 9 length-limited requests; 17 others stopped
  without the marker. Do not add 9 and 26 as independent failures.
- Total generated tokens: 2,397,502. Sum of instrumented generation time:
  44,889.352596 seconds. These are run-accounting observations, not a matched
  serving benchmark.
- The telemetry wrapper records IDs, lengths, finish reasons and marker presence,
  but not original generated text or a prompt hash. IDs are generation order,
  not dataset doc IDs. No unsupported per-item telemetry-to-sample mapping is made.
- Verified pinned harness source: `lm_eval/models/vllm_causallms.py:646-653`
  passes generated text through `postprocess_generated_text`; `models/utils.py:960-968`
  strips the thinking prefix and applies task stop strings. All saved sample
  responses lack the end marker after that processing. The full original
  reasoning traces cannot be reconstructed from those saved responses.
- Therefore marker omission versus genuinely unfinished reasoning cannot be
  conclusively diagnosed for every affected request. The observed length limits
  and frozen gate failure remain valid; the gate was not relaxed.

## Provenance and archive verification

- Protocol v4 SHA-256:
  `7a11c55ec0def9194473ccec67019eec561c024b59d47666ad078b9e9c9c06fd`.
- Running evaluation source: clean `2c0f118a203fc99f53c62bdc8451f4792b8758ba`.
- Harness: `b954108c9baaaa934b4ad842033b31a97ee30816`.
- All four start GPU UUIDs, model/dataset locks, per-task counts, anomalies,
  sample-provenance digest and **48 file hashes** are in `AUDIT.json`.
- A second independent read of those 48 remote artifacts matched every archived
  SHA-256. Raw licensed questions, prompts and complete answers are not committed.
- The original wrapper never reached its success-only `gpus-final.csv` or
  `SHA256SUMS` steps. Both are absent. This post-run hash inventory is explicitly
  an audit receipt, not a retroactively fabricated successful-run receipt.
- No paired GPU-UUID or paired sample-hash check can pass without the candidate.
  Start UUIDs alone do not establish a completed matched-topology comparison.

## Timing and next decision

A naive sample-count extrapolation of the observed 12.7067-hour BF16 allocation
to 12,230 full items is about **126.4 hours (5.3 days) for BF16 alone**. If FP8 took
the same time, the sequential pair would be about **252.9 hours (10.5 days)** on
four GPUs. This is planning arithmetic only: task mix, variable reasoning lengths,
startup costs and unknown FP8 speed make it uncertain. It exceeds a 24-hour job
limit, so a future full run would need an explicitly designed resumable workflow.
No full run is authorized or submitted by this audit.

Recommended next step, requiring a new protocol decision before new inference:
use CPU-only synthetic format tests to diagnose parser behavior, and preregister
request-to-prompt linkage plus raw-completion preservation and an explicit policy
for missing thinking markers/length limits. Any changed scoring or generation
budget must be a new version with this negative v4 result retained. Do not bypass
the BF16 gate simply to run FP8.

Reproduce the read-only aggregate audit with `scripts/audit_qwen38_pilot.py RUN_DIR`
using the retained remote run directory. No model inference is involved. Helper
tests and the existing protocol test module passed: 20 tests. A Unicode JSONL
line-separator regression test protects against incorrectly splitting licensed
text embedded inside JSON strings.
