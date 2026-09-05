# L20 Qwen2.5-72B AWQ Quality Evaluation Notes

> Historical snapshot: this document records the earlier AWQ-only L20 run.
> The subsequently completed matched BF16-versus-AWQ result is available in
> [the ParaCloud evidence bundle](../results/qwen25-72b-retention-v1/EVIDENCE.md).

This note records the first quality-evaluation pass for `Qwen/Qwen2.5-72B-Instruct-AWQ` served by vLLM on a single NVIDIA L20.

## Scope

- GPU: 1x NVIDIA L20, 46 GB visible VRAM
- Runtime: vLLM `0.8.5.post1`
- Model: `Qwen/Qwen2.5-72B-Instruct-AWQ`
- Quantization: AWQ Marlin
- Endpoint: OpenAI-compatible `/v1/chat/completions`
- Short-context service: `max_model_len=1024`, `max_num_seqs=48`, `max_num_batched_tokens=4096`
- LongBench service: `max_model_len=8192`, `max_num_seqs=1`, `max_num_batched_tokens=2048`

These results are AWQ candidate absolute scores. They are not BF16/FP16-vs-AWQ retention results because a matched BF16/FP16 Qwen2.5-72B baseline endpoint was not available.

## Short-Context Quality

Run directory:

```text
/home/USER/llm-quant-bench/runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-awq-short-quality
```

Command shape:

```bash
python scripts/run_quality_eval.py \
  --out "$OUT" \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen72b-awq-l20 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /home/USER/hf-cache-l20-eval/cmmlu_data2/test \
  --concurrency 8
```

Results:

| Benchmark | Items | OK | Failed | Score | p50 Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| MMLU | 14,042 | 14,039 | 3 | 0.8130 | 0.98s | 3.19s |
| CMMLU | 11,582 | 11,582 | 0 | 0.8309 | 0.88s | 1.49s |
| GSM8K | 1,319 | 1,319 | 0 | 0.8082 | 12.21s | 16.65s |
| Total | 26,943 | 26,940 | 3 | n/a | n/a | n/a |

The three failures were MMLU `college_medicine` prompts that exceeded the 1024-token service context limit. They were HTTP 400 validation failures, not CUDA OOMs. No OOM or killed-process signatures were observed.

Method notes:

- MMLU and CMMLU use zero-shot direct-answer multiple-choice prompting and exact A/B/C/D extraction.
- GSM8K uses direct generation with exact final-number extraction.
- These scores should be compared only against a baseline produced by the same runner, dataset snapshot, prompt template, decoding parameters, and answer extraction logic.

## MT-Bench Generation

Run directory:

```text
/home/USER/llm-quant-bench/runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-mt-bench-generation
```

Results:

| Benchmark | Items | OK | Failed | Judge Score | p50 Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| MT-Bench generation | 80 | 80 | 0 | pending judge | 29.75s | 31.13s |

This run only generated answers. Official MT-Bench scoring requires a judge endpoint and answer-order-swapped judging. The repo includes `scripts/score_mt_bench.py` for that follow-up, but no judge endpoint/key was available during this run.

## LongBench 8K Subset

Run directory:

```text
/home/USER/llm-quant-bench/runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k
```

Command shape:

```bash
python scripts/run_longbench_8k.py \
  --out "$OUT" \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen72b-awq-l20 \
  --longbench-dir /home/USER/hf-cache-l20-eval/longbench_data/data \
  --tasks multifieldqa_en hotpotqa passage_count lcc gov_report multi_news \
  --max-per-task 10 \
  --concurrency 1 \
  --longbench-max-context-chars 24000 \
  --longbench-max-tokens 128
```

Results:

| Task | Items | OK | Failed | Token-F1 | p50 Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| `multifieldqa_en` | 10 | 10 | 0 | 0.4197 | 11.60s | 18.25s |
| `hotpotqa` | 10 | 10 | 0 | 0.3026 | 10.48s | 13.49s |
| `multi_news` | 10 | 10 | 0 | 0.2468 | 8.30s | 11.56s |
| `gov_report` | 10 | 10 | 0 | 0.1929 | 14.34s | 16.12s |
| `lcc` | 10 | 10 | 0 | 0.0609 | 10.97s | 18.17s |
| `passage_count` | 10 | 10 | 0 | 0.0000 | 9.93s | 13.49s |
| LongBench subset | 60 | 60 | 0 | 0.2038 | 10.59s | 16.03s |

LongBench scores use the repo's lightweight max token-F1 scorer. This is useful as a same-run regression and retention metric, but it should not be presented as an official LongBench leaderboard number without matching the official LongBench evaluator and prompt templates.

The repo now includes a stricter postprocess step:

```bash
python3 scripts/score_longbench_official.py \
  --samples /home/USER/llm-quant-bench/runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/eval/samples.jsonl \
  --out /home/USER/llm-quant-bench/runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/official-metrics
```

This applies the LongBench v1 task-specific metric mapping to the generated
samples. It is stricter than the earlier single token-F1 metric, but leaderboard
claims still require the exact official LongBench repository revision, prompt
templates, and full task set.

Result from this stricter postprocess:

| Task | Items | Metric | Score |
|---|---:|---|---:|
| multifieldqa_en | 10 | QA F1 | 38.67% |
| hotpotqa | 10 | QA F1 | 30.53% |
| multi_news | 10 | ROUGE-L | 12.90% |
| gov_report | 10 | ROUGE-L | 12.08% |
| lcc | 10 | code similarity | 2.76% |
| passage_count | 10 | count | 0.00% |
| Macro / micro | 60 | mixed LongBench v1 metrics | 16.16% |

## Retention Status

BF16/FP16-vs-AWQ quality retention remains pending.

Required next step:

```bash
python3 scripts/run_quality_retention.py \
  --out runs/qwen72b-awq-vs-bf16-retention \
  --baseline-base-url http://BASELINE_HOST:8001/v1 \
  --baseline-model qwen72b-bf16 \
  --candidate-base-url http://127.0.0.1:8001/v1 \
  --candidate-model qwen72b-awq-l20 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /path/to/cmmlu/test \
  --concurrency 8
```

Formula:

```text
quality_retention = candidate_score / baseline_score
delta_percentage_points = 100 * (candidate_score - baseline_score)
```
