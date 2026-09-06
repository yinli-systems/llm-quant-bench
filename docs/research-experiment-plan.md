# Research Experiment Plan

This document tracks the experiments needed for a credible technical report on single-L20 70B-class quantized serving.

## Current 24h Soak

Status: completed.

- Run directory: `/home/USER/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z`
- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`
- Runtime: vLLM `0.8.5.post1`
- GPU: 1x NVIDIA L20
- Serving config: `awq_marlin`, `max_model_len=1024`, `max_num_seqs=48`, `max_num_batched_tokens=4096`
- Workload: fixed-shape ~512 input / 256 output
- Concurrency: 10
- Duration: 86,400 seconds
- Load output: `load/load_summary.json`
- Power log: `gpu_power.csv`
- Energy output: `energy_summary.json`

Energy was computed with:

```bash
python3 scripts/summarize_energy.py \
  --summary /home/USER/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/load/load_summary.json \
  --power-log /home/USER/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/gpu_power.csv \
  --out /home/USER/llm-quant-bench/runs/qwen72b-awq-l20/soak-24h-fixed512x256-c10-20260519T034856Z/energy_summary.json
```

Result:

| Metric | Value |
|---|---:|
| Duration | 86,412.60s |
| Total requests | 36,740 |
| Successful requests | 36,740 |
| Failed requests | 0 |
| Success rate | 100% |
| Output tokens/s | 108.84 |
| Request throughput | 0.425 req/s |
| p95 TTFT | 6.61s |
| p95 latency | 23.54s |
| Prompt tokens | 19,361,980 |
| Output tokens | 9,405,440 |
| Total tokens | 28,767,420 |
| Average GPU power | 330.15 W |
| GPU energy | 7.92 kWh |
| Output tokens/J | 0.330 |
| Total tokens/J | 1.008 |
| CUDA OOM / error signatures | 0 |

## Quality Retention

Status: completed for the frozen MMLU, CMMLU, and GSM8K bundle. See
[the committed evidence](../results/qwen25-72b-retention-v1/EVIDENCE.md).

Goal: compare a high-quality baseline against the AWQ candidate.

Required baseline:

- FP16/BF16 Qwen2.5-72B-Instruct endpoint, or a trusted hosted Qwen2.5-72B-Instruct endpoint.
- The current L20 cannot host FP16/BF16 72B locally, and the current machine has only 69GB free disk, so the baseline must be external or multi-GPU.

Important caveat:

The earlier single-L20 AWQ benchmark alone was not a retention measurement. The
new ParaCloud run supplies the matched FP16/BF16 baseline with the same frozen
items, prompts, decoding parameters, scoring code, and answer extraction. The
correct wording is now:

```text
On the exact 26,943-item frozen MMLU, CMMLU, and GSM8K bundle, AWQ retained
99.451%, 99.359%, and 98.122% of BF16 score respectively, with zero request
failures in either arm.
```

Do not use wording such as:

```text
AWQ is lossless versus BF16/FP16.
```

Benchmarks:

| Benchmark | Purpose | Candidate Command Shape |
|---|---|---|
| MMLU / MMLU-Pro | general knowledge and reasoning | `lm-evaluation-harness` local OpenAI-compatible endpoint |
| CMMLU / C-Eval | Chinese knowledge and reasoning | OpenCompass preferred |
| GSM8K / MATH | math reasoning | `lm-evaluation-harness` or OpenCompass |
| MT-Bench / Arena-Hard | instruction following and preference | judge with answer-order swap |
| LongBench / LongBench v2 | long-context behavior | run separate 8K/16K service config |

Acceptance metrics:

- quality retention = candidate score / baseline score
- pass if task-level retention is at least 0.98 for business-critical categories
- report all drops over 2 percentage points individually
- keep judge inconsistency under 5% for preference benchmarks

Current execution status:

- AWQ candidate endpoint: tested on the L20 through vLLM and AWQ Marlin.
- MMLU, CMMLU, and GSM8K: completed with `scripts/run_quality_eval.py`; see [docs/l20-qwen72b-awq-quality-results.md](l20-qwen72b-awq-quality-results.md).
- MT-Bench: 80/80 answer generations completed; official scoring still requires an external judge endpoint/key and answer-order-swapped judging.
- LongBench: 8K subset completed with 60/60 successful requests using `max_model_len=8192`; the reported score is lightweight max token-F1, not an official leaderboard score.
- FP16/BF16 baseline: completed on 8x RTX 4090 across four ParaCloud nodes;
  model revision, GPU inventory, summaries, paired deltas, and checksums are
  committed under `results/qwen25-72b-retention-v1/`.
- Strict follow-up tooling: added readiness checks, LongBench v1 task-metric postprocessing, repeated load runs, and confidence-interval summaries. See [strict-experiment-suite.md](strict-experiment-suite.md).
- Repeated fixed-shape CI: completed c1/c4/c8/c16 with 3 repeats each and 100% success. c16 averaged 127.22 +/- 12.68 output tok/s and matched the earlier 127.70 output tok/s screening result.
- MT-Bench judge preflight remains blocked because a judge endpoint/model/key is
  not configured. This does not block the completed objective MMLU/CMMLU/GSM8K
  retention run.

Current AWQ candidate quality snapshot:

| Evaluation | Items | OK | Failed | Score | p95 Latency |
|---|---:|---:|---:|---:|---:|
| MMLU | 14,042 | 14,039 | 3 | 0.8130 | 3.19s |
| CMMLU | 11,582 | 11,582 | 0 | 0.8309 | 1.49s |
| GSM8K | 1,319 | 1,319 | 0 | 0.8082 | 16.65s |
| MT-Bench generation | 80 | 80 | 0 | pending judge | 31.13s |
| LongBench 8K subset | 60 | 60 | 0 | 0.2038 | 16.03s |
| LongBench official-style metrics | 60 | 60 | 0 | 16.16% | n/a |

Current repeated fixed-shape serving snapshot:

| Concurrency | Runs | Success Rate | Output tok/s mean +/- 95% CI | p95 TTFT mean +/- 95% CI | p95 Latency mean +/- 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 100% | 16.57 +/- 1.21 | 0.54s +/- 1.90s | 15.66s +/- 1.94s |
| 4 | 3 | 100% | 55.75 +/- 3.54 | 3.01s +/- 0.17s | 18.70s +/- 0.16s |
| 8 | 3 | 100% | 93.26 +/- 0.23 | 6.05s +/- 0.06s | 22.16s +/- 0.05s |
| 16 | 3 | 100% | 127.22 +/- 12.68 | 11.91s +/- 0.15s | 34.95s +/- 1.00s |

Baseline runbook for later:

```bash
python3 scripts/run_quality_eval.py \
  --out runs/qwen72b-bf16-quality \
  --base-url http://BASELINE_HOST:BASELINE_PORT/v1 \
  --model qwen72b-bf16 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /path/to/cmmlu/test \
  --concurrency 8

python3 scripts/summarize_quality_retention.py \
  --baseline-summary runs/qwen72b-bf16-quality/summary.json \
  --candidate-summary runs/qwen72b-awq-quality/summary.json \
  --out runs/qwen72b-awq-quality/quality_retention.json
```

Retention formula:

```text
quality_retention = candidate_score / baseline_score
delta_percentage_points = 100 * (candidate_score - baseline_score)
```

Implemented support code:

| Missing evidence | Implemented entry point | Notes |
|---|---|---|
| BF16/FP16 baseline retention | `scripts/run_quality_retention.py` | Runs the same benchmark set against baseline and candidate endpoints, then calls `scripts/summarize_quality_retention.py`. |
| MT-Bench judge score | `scripts/score_mt_bench.py` | Supports single-answer scoring and pairwise baseline-vs-candidate judging with answer-order swapping. Requires an external judge endpoint/key. |
| LongBench 8K quality | `scripts/run_longbench_8k.py` | Runs a documented LongBench subset against an 8K service; can optionally start a provided vLLM command. |
| LongBench official-style metrics | `scripts/score_longbench_official.py` | Applies LongBench v1 task-specific metric mapping to generated samples. Use the exact official repo revision for leaderboard claims. |
| Runtime ablation | `scripts/run_ablation_matrix.py`, `examples/runtime_ablation.example.json` | Keeps workload constant while swapping vLLM, SGLang, and llama.cpp/GGUF endpoints. |
| Quantization ablation | `scripts/run_ablation_matrix.py`, `examples/quant_ablation.example.json` | Keeps runtime/workload constant while swapping AWQ, GPTQ, and FP8 endpoints where available. |
| Repeated runs and confidence intervals | `scripts/run_repeated_load.py`, `scripts/summarize_repeats_ci.py` | Runs repeated fixed-shape conditions and reports run-level mean, standard deviation, and two-sided 95% CI. |
| Matrix readiness | `scripts/check_experiment_readiness.py`, `examples/full_research_matrix.example.json` | Checks missing runtimes, model paths, and required endpoint environment variables before launching expensive jobs. |

## Energy

Primary formula:

```text
energy_j = integral(power_watts dt)
output_tokens_per_joule = output_tokens / energy_j
total_tokens_per_joule = (prompt_tokens + output_tokens) / energy_j
joules_per_output_token = energy_j / output_tokens
```

For the current setup, `gpu_power.csv` is sampled every 10 seconds using `nvidia-smi`. This measures GPU board power, not full-system wall power.

## More Models

Current disk status leaves about 69GB free. A single 70B AWQ checkpoint usually needs about 35-45GB, so only one additional model can be staged safely unless older artifacts are removed.

Candidate order:

| Priority | Model | Quant | Why |
|---:|---|---|---|
| 1 | `casperhansen/llama-3.3-70b-instruct-awq` or another Llama 3.3 70B AWQ | AWQ | strong multilingual 70B baseline, available as AWQ |
| 2 | `Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ` | AWQ | reasoning-oriented 70B distilled model, available as AWQ |
| 3 | Qwen3 70B-class model | AWQ/GPTQ/FP8 if available | Qwen3 AWQ docs exist, but an official dense 70B-class AWQ checkpoint must be verified before download |

Do not download all candidates at once on the current disk.

## More Runtimes

| Runtime | Status | Notes |
|---|---|---|
| vLLM | complete for Qwen2.5-72B AWQ | current best result uses AWQ Marlin |
| SGLang | not installed | install in a separate env and test AWQ support |
| llama.cpp / GGUF | not installed | requires a GGUF 70B Q4 checkpoint; results are not directly comparable to AWQ Marlin |

Runtime comparisons should use the same fixed-shape workload:

- ~512 input / 256 output
- concurrency 1, 4, 8, 10, 16
- streaming or equivalent token timing
- GPU power monitor

## Ablations

| Ablation | Current Status | Notes |
|---|---|---|
| AWQ Marlin | complete | best fixed-shape point: c16, 127.70 output tok/s |
| AWQ non-Marlin | partially tested | slower than AWQ Marlin in earlier runs |
| GPTQ | not tested | requires compatible 70B GPTQ checkpoint |
| FP8 | not tested | NVIDIA NIM L20 profiles for Qwen2.5 72B use 4 or 8 L20 GPUs, not single L20 |
| max concurrency | tested | c24 fixed-shape saturated and regressed versus c16 |
| max context | tested at c1 | 4096 and 8192 context rows completed without OOM |

## Paper Positioning

This should be written as a systems benchmark or empirical technical report, not as a new quantization algorithm paper.

Claim that is supported:

```text
Single-L20 serving of Qwen2.5-72B AWQ is feasible, measurable, and stable under fixed-shape load, with c10 sustaining 108.84 output tok/s for 24 hours and c16 reaching 127.70 output tok/s in the short fixed-shape sweep for ~512 input / 256 output.
```

Claims that are not yet supported:

- lossless quality
- superiority over A100/H100
- generalization to all 70B models
- production SLA beyond the tested fixed-shape workload
