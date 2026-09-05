# Strict Experiment Suite

> Historical readiness snapshot: the BF16 endpoint listed as blocked below was
> later supplied by an eight-GPU ParaCloud run. See
> [the completed retention evidence](../results/qwen25-72b-retention-v1/EVIDENCE.md).

This document tracks the evidence still needed before making stronger research
claims about single-L20 70B serving.

## Current Blockers

The current L20 host has only the AWQ checkpoint staged:

- `/home/USER/models/Qwen2.5-72B-Instruct-AWQ`

No BF16/FP16 baseline endpoint, judge endpoint, SGLang runtime, llama.cpp
runtime, GGUF checkpoint, GPTQ checkpoint, or FP8 checkpoint is currently
available on that host. Those missing prerequisites block the corresponding
claims; they should not be inferred from the AWQ run.

## BF16 Baseline And MT-Bench Judge Preflight

Status: blocked as of `2026-05-20T12:39:37Z`.

Remote preflight artifact:

```text
/home/USER/llm-quant-bench/runs/qwen72b-awq-l20/bf16-mtbench-preflight-20260520T123916Z
```

The host was checked for BF16/FP16 baseline and MT-Bench judge prerequisites:

| Item | Status | Missing |
|---|---|---|
| BF16/FP16 baseline endpoint | blocked | `BASELINE_BASE_URL`, `BASELINE_MODEL`, external or multi-GPU Qwen2.5-72B BF16/FP16 endpoint |
| MT-Bench judge endpoint | blocked | `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY` |

Only AWQ checkpoint directories are currently staged under `/home/USER/models`.
A local single-L20 BF16/FP16 72B baseline is not feasible because 72B BF16/FP16
weights require roughly 144GB before KV cache and runtime overhead, while the
host exposes about 46GB of L20 VRAM and about 63GB free disk. Running the AWQ
candidate as its own judge is intentionally not reported, because that would not
be an MT-Bench judge result and would contaminate the claim.

## Evidence Checklist

| Evidence | Status | Entry Point | Blocking Condition |
|---|---|---|---|
| BF16/FP16 baseline vs AWQ retention | ready to run once baseline exists | `scripts/run_quality_retention.py` | needs matched BF16/FP16 endpoint |
| MT-Bench judge score | ready to run once judge exists | `scripts/score_mt_bench.py` | needs judge endpoint/key, preferably GPT-4-class |
| LongBench official-metric score | completed on the current 60-sample subset: 16.16% macro/micro | `scripts/score_longbench_official.py` | leaderboard claims still require exact official repo revision |
| vLLM repeated load with CI | completed: c1/c4/c8/c16, 3 repeats each, 100% success | `scripts/run_repeated_load.py` | none for current AWQ/vLLM setup |
| vLLM vs SGLang vs llama.cpp | scaffolded | `scripts/run_ablation_matrix.py` | needs SGLang, llama.cpp, and GGUF checkpoint |
| AWQ vs GPTQ vs FP8 | scaffolded | `scripts/run_ablation_matrix.py` | needs GPTQ and FP8 checkpoints/endpoints |
| Multi-run confidence intervals | runnable for any repeated summaries | `scripts/summarize_repeats_ci.py` | needs repeated runs per condition |

## Commands

Readiness check:

```bash
python3 scripts/check_experiment_readiness.py \
  --manifest examples/full_research_matrix.example.json \
  --out runs/research-matrix-readiness/readiness.json
```

LongBench official-metric postprocess from the existing 8K samples:

```bash
python3 scripts/score_longbench_official.py \
  --samples runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/eval/samples.jsonl \
  --out runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/official-metrics
```

Repeated fixed-shape load with confidence intervals:

```bash
python3 scripts/run_repeated_load.py \
  --config runs/qwen72b-awq-l20/config.fixed512x256.json \
  --dataset runs/qwen72b-awq-l20/fixed-512in-256out.jsonl \
  --out runs/qwen72b-awq-l20/repeated-fixed512x256-vllm-awq \
  --concurrencies 1 4 8 16 \
  --repeats 3 \
  --requests 80 \
  --stream-usage
```

Current repeated-load result:

| Concurrency | Runs | Success Rate | Output tok/s | p95 TTFT | p95 Latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 100% +/- 0.00% | 16.57 +/- 1.21 | 0.54s +/- 1.90s | 15.66s +/- 1.94s |
| 4 | 3 | 100% +/- 0.00% | 55.75 +/- 3.54 | 3.01s +/- 0.17s | 18.70s +/- 0.16s |
| 8 | 3 | 100% +/- 0.00% | 93.26 +/- 0.23 | 6.05s +/- 0.06s | 22.16s +/- 0.05s |
| 16 | 3 | 100% +/- 0.00% | 127.22 +/- 12.68 | 11.91s +/- 0.15s | 34.95s +/- 1.00s |

BF16/FP16 retention once a baseline endpoint is available:

```bash
python3 scripts/run_quality_retention.py \
  --out runs/qwen72b-bf16-vs-awq-retention \
  --baseline-base-url "$BASELINE_BASE_URL" \
  --baseline-model "$BASELINE_MODEL" \
  --candidate-base-url http://127.0.0.1:8001/v1 \
  --candidate-model qwen72b-awq-l20 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /home/USER/quality-eval-data/cmmlu/test \
  --concurrency 8
```

MT-Bench judge once a judge endpoint is available:

```bash
python3 scripts/score_mt_bench.py \
  --out runs/qwen72b-awq-l20/mt-bench-judge \
  --samples runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-mt-bench-generation/samples.jsonl \
  --judge-base-url "$JUDGE_BASE_URL" \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-key "$JUDGE_API_KEY" \
  --concurrency 4
```

## Reporting Rules

- Report AWQ absolute quality separately from BF16/FP16 retention.
- Report MT-Bench generation success separately from judge score.
- Report LongBench dependency-light official-metric scores separately from
  official leaderboard scores.
- For repeated load results, report mean, standard deviation, and two-sided 95%
  CI over run-level summaries.
- For runtime and quantization ablations, hold model family, prompt shape,
  decoding settings, concurrency, and measurement window constant.
