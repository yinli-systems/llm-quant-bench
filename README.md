# LLM Quant Bench

Benchmark toolkit for quantized LLM quality retention, throughput, latency, stability, and sustained serving usability.

The original motivation is simple: verify whether a compressed 70B-class model can stay usable on a single NVIDIA L20 GPU. The toolkit compares a baseline model and a quantized candidate on the same golden prompts, records quality and serving metrics, and writes reproducible `summary.json` and `report.md` outputs.

It works with any OpenAI-compatible `/v1/chat/completions` server, including vLLM, TGI, llama.cpp server, and custom gateways.

## Scope

This project is enough for internal regression testing and for building evidence that a quantized 70B model is usable on constrained hardware. It is not, by itself, enough to claim that a quantized model is lossless.

Public evaluation systems generally require broad coverage, multiple metrics, reproducibility, and explicit limits. HELM emphasizes broad coverage, multi-metric measurement, and standardization. EleutherAI `lm-evaluation-harness` provides standard academic tasks. OpenCompass provides large benchmark configurations. LongBench focuses on long-context understanding. The MT-Bench and Chatbot Arena paper discusses LLM-as-judge biases such as position bias, verbosity bias, and self-enhancement bias. NVIDIA GenAI-Perf tracks serving metrics such as TTFT, ITL, request latency, output token throughput, and request throughput.

Recommended positioning:

- Business golden-set regression.
- Baseline-vs-quantized quality retention.
- 24-72 hour soak tests.
- Real single-GPU serving SLA validation.

## Real L20 Experiment Snapshot

These are measured runs for `Qwen/Qwen2.5-72B-Instruct-AWQ` on one NVIDIA L20. They are serving measurements, not quality-retention claims.

| Model | Quant | GPU | Context | Concurrency | Success Rate | p95 TTFT | tok/s | OOM |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 8192 | 1 | 100% (3/3) | 11.03s | 6.16 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 4096 | 1 | 100% (5/5) | 5.28s | 10.02 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 10 | 100% (36740/36740) | 6.61s | 108.84 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 16 | 100% (1177/1177) | 0.14s | 245.91 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 32 | 100% (408/408) | 2.17s | 390.08 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 48 | 100% (2333/2333) | 0.24s | 488.63 output tok/s | No |
| Qwen2.5-72B-Instruct | Q4 AWQ / AWQ Marlin | L20 48GB (46GB usable) | 1024 | 64 | 100% (461/461) | 3.39s | 432.63 output tok/s | No |

The 8192-context row used a 7,514-token prompt. The 4096-context row used a 3,875-token prompt. The 1024/c10 row is a 24h fixed-shape soak. The other 1024-context rows are short-context throughput sweeps; `c48` was the best tested throughput point, while `c64` was past the useful concurrency peak. See [docs/l20-qwen72b-awq-results.md](docs/l20-qwen72b-awq-results.md) for the full run notes.

## Fixed-Shape Serving Benchmark

This table is closer to external serving benchmarks: unique prompts, average server-side prompt length of 527 tokens, fixed 256 output tokens, streaming usage enabled, and no request failures.

| Shape | Concurrency | Requests | Success Rate | p95 TTFT | p95 Latency | Output tok/s | Req/s | p05 Decode tok/s | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ~512 in / 256 out | 1 | 8 | 100% | 0.76s | 15.88s | 16.21 | 0.063 | 16.92 | No |
| ~512 in / 256 out | 4 | 32 | 100% | 2.94s | 18.63s | 57.10 | 0.223 | 14.99 | No |
| ~512 in / 256 out | 8 | 64 | 100% | 6.06s | 22.20s | 93.38 | 0.365 | 12.13 | No |
| ~512 in / 256 out | 10 | 80 | 100% | 6.71s | 23.69s | 108.70 | 0.425 | 11.26 | No |
| ~512 in / 256 out | 16 | 128 | 100% | 11.01s | 36.29s | 127.70 | 0.499 | 7.66 | No |
| ~512 in / 256 out | 24 | 120 | 100% | 37.26s | 64.08s | 120.51 | 0.471 | 5.25 | No |

This benchmark uses `max_tokens=256`, `min_tokens=256`, and `ignore_eos=true` to make output length comparable across concurrency levels. The c10 row is the closest match to the GigaGPU 10-concurrent-user public table. The c24 run is included as a saturation check; it is slower than c16 and has much worse tail latency, so c16 is the best fixed-shape throughput point tested here.

## Repeated Fixed-Shape CI

The c1/c4/c8/c16 fixed-shape conditions were repeated three times each with the same ~512 input / 256 output workload. The table reports run-level means with two-sided 95% confidence intervals using Student t critical values.

| Shape | Concurrency | Runs | Success Rate | Output tok/s | p95 TTFT | p95 Latency | p95 TPOT | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ~512 in / 256 out | 1 | 3 | 100% +/- 0.00% | 16.57 +/- 1.21 | 0.54s +/- 1.90s | 15.66s +/- 1.94s | 0.059s +/- 0.000s | No |
| ~512 in / 256 out | 4 | 3 | 100% +/- 0.00% | 55.75 +/- 3.54 | 3.01s +/- 0.17s | 18.70s +/- 0.16s | 0.070s +/- 0.000s | No |
| ~512 in / 256 out | 8 | 3 | 100% +/- 0.00% | 93.26 +/- 0.23 | 6.05s +/- 0.06s | 22.16s +/- 0.05s | 0.082s +/- 0.000s | No |
| ~512 in / 256 out | 16 | 3 | 100% +/- 0.00% | 127.22 +/- 12.68 | 11.91s +/- 0.15s | 34.95s +/- 1.00s | 0.125s +/- 0.001s | No |

The repeated c16 mean, 127.22 output tok/s, matches the earlier single c16 screening result of 127.70 output tok/s. The c16 throughput interval is wider because one of the three runs produced 121.32 output tok/s, while the other two produced about 130 output tok/s. All twelve repeated runs completed without request failures or vLLM OOM/error signatures.

## 24h Fixed-Shape Soak

The c10 fixed-shape workload was also run for a full day on the same service. Power was sampled every 10 seconds with `nvidia-smi`, so energy numbers are GPU board power estimates rather than wall-power measurements.

| Shape | Concurrency | Duration | Requests | Success Rate | p95 TTFT | p95 Latency | Output tok/s | Req/s | Avg GPU Power | GPU Energy | Output tok/J | Total tok/J | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ~512 in / 256 out | 10 | 24.00h | 36740 | 100% | 6.61s | 23.54s | 108.84 | 0.425 | 330.15 W | 7.92 kWh | 0.330 | 1.008 | No |

The 24h run generated 9,405,440 output tokens and 28,767,420 total tokens with zero request failures. The vLLM log had no CUDA OOM, traceback, killed-process, or error signatures for this run.

## External Comparisons

These comparisons are directional. Serving benchmarks are sensitive to prompt length, output length, concurrency, batching policy, quantization kernel, sampling settings, and whether the number is per-request decode speed or aggregate server throughput.

| Source | Hardware | Model / Quant | Shape | Reported Result | How to Read It |
|---|---|---|---|---:|---|
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c8 | 93.38 output tok/s | Fixed-shape aggregate throughput with 100% success. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c10 | 108.70 output tok/s | Closest local match to 10-concurrent-user public tables. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c10, 24h soak | 108.84 output tok/s | Day-long sustained throughput with 36,740/36,740 successful requests. |
| This repo | 1x L20 48GB | Qwen2.5-72B AWQ Marlin | ~512 input / 256 output, c16 | 127.70 output tok/s | Higher throughput, but p95 latency rises to 36.29s. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 3090 | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 32 tok/s | Similar fixed-shape benchmark, different GPU and quant/runtime details. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 5090 | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 58-82 tok/s | This L20 run is above the published 5090 range for that table. |
| [GigaGPU Apr 2026](https://gigagpu.com/tokens-sec-benchmark-update-april-2026/) | 1x RTX 6000 Pro | Qwen 2.5 72B Q4 | 512 input / 256 output, 10 concurrent users | 45 tok/s | Same caveat: useful directional comparison, not identical kernel/config. |
| [Qwen official speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html) | 1x A100 80GB | Qwen2.5-72B AWQ, Transformers | input 1 / 6144 / 14336, 2048 output, batch size 1 | 11.50 / 8.17 / 5.57 tok/s | Useful for single-request long-context sanity checks, not aggregate serving throughput. |
| [Qwen official speed benchmark](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html) | 2x A100 80GB | Qwen2.5-72B AWQ, vLLM | input 1 / 6144 / 14336 / 30720, 2048 output, batch size 1 | 44.30 / 40.67 / 36.63 / 30.02 tok/s | Official vLLM baseline uses 2 A100s and batch size 1, so it should not be compared directly with c16 aggregate throughput. |
| [NVIDIA NIM supported models](https://docs.nvidia.com/nim/large-language-models/1.14.0/supported-models.html) | L20 | Qwen2.5 72B Instruct FP8 | Optimized profiles | 4 or 8 GPUs | NVIDIA's optimized L20 profiles are multi-GPU; this repo demonstrates a single-L20 AWQ path outside that conservative profile. |

The practical interpretation is that single-L20 Qwen2.5-72B AWQ serving is feasible, measurable, and stable for at least a day under this fixed-shape workload. The fixed-shape c8/c16 results are strong versus public single-GPU Q4 serving tables, while the long-context c1 rows are mainly capacity evidence. These numbers are not quality-retention evidence and should not be described as lossless.

## Research Follow-Up

The 24h fixed-shape c10 soak test completed with GPU power logging and zero request failures. The follow-up plan for quality retention, additional 70B models, additional runtimes, and AWQ/GPTQ/FP8 ablations is tracked in [docs/research-experiment-plan.md](docs/research-experiment-plan.md).

The next pre-registered experiment is an exact-revision Qwen3.8-27B BF16 vs
official FP8 comparison on a matched 4x RTX 4090 topology. Its standard
GPQA/MMLU-Pro lane, gold-blind GPQA permutation lane, quality-first gates, and
remote staging blocker are documented in
[docs/qwen38-27b-bf16-fp8-pareto-plan.md](docs/qwen38-27b-bf16-fp8-pareto-plan.md).

## Quality Evaluation Snapshot

The first AWQ candidate quality pass completed on the same L20 setup. See [docs/l20-qwen72b-awq-quality-results.md](docs/l20-qwen72b-awq-quality-results.md) for run directories, commands, and caveats.

| Evaluation | Context | Items | OK | Failed | Score | p95 Latency | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| MMLU | 1024 | 14,042 | 14,039 | 3 | 0.8130 | 3.19s | 3 prompts exceeded the 1024-token service limit. |
| CMMLU | 1024 | 11,582 | 11,582 | 0 | 0.8309 | 1.49s | Zero-shot direct-choice scoring. |
| GSM8K | 1024 | 1,319 | 1,319 | 0 | 0.8082 | 16.65s | Final-number exact match. |
| MT-Bench generation | 1024 | 80 | 80 | 0 | pending judge | 31.13s | Answer generation only; no external judge was available. |
| LongBench subset | 8192 | 60 | 60 | 0 | 0.2038 | 16.03s | Lightweight max token-F1, not an official leaderboard score. |
| LongBench official-style metrics | 8192 | 60 | 60 | 0 | 16.16% | n/a | LongBench v1 task-metric mapping over the same generated samples. |

These rows are the earlier single-L20 absolute AWQ snapshot. A newer matched
BF16-versus-AWQ run is recorded separately below.

## Quality Retention Status

The missing matched baseline was completed on ParaCloud. Both arms used the same
26,943-item frozen bundle, prompt construction, parser, temperature, and model
family revision line, with zero failed requests. Full evidence and checksums are
in [results/qwen25-72b-retention-v1/EVIDENCE.md](results/qwen25-72b-retention-v1/EVIDENCE.md).

| Benchmark | BF16 | AWQ | AWQ - BF16 | Retention |
|---|---:|---:|---:|---:|
| MMLU | 0.817262 | 0.812776 | -0.449 pp | 99.451% |
| CMMLU | 0.835521 | 0.830168 | -0.535 pp | 99.359% |
| GSM8K | 0.807430 | 0.792267 | -1.516 pp | 98.122% |

This supports a scoped quality-retention claim for the exact frozen benchmark
bundle. It does not support a latency, throughput, or cost comparison because
BF16 used eight RTX 4090 GPUs with TP=2/PP=4 and AWQ used two RTX 4090 GPUs with
TP=2.

The unsupported claim is:

```text
Qwen2.5-72B-Instruct-AWQ is lossless versus BF16/FP16.
```

The repo includes the executed retention path and scaffolding for the remaining
research pieces:

| Gap | Script / Manifest | Output |
|---|---|---|
| BF16/FP16 baseline retention | `scripts/run_quality_retention.py` | matched baseline/candidate summaries plus `quality_retention.json` |
| MT-Bench judge score | `scripts/score_mt_bench.py` | `judgments.jsonl`, judge `summary.json`, and `report.md` |
| LongBench 8K quality | `scripts/run_longbench_8k.py` | LongBench subset `summary.json` and run manifest |
| LongBench official-style metrics | `scripts/score_longbench_official.py` | task-specific LongBench v1 metric summary from existing samples |
| vLLM vs SGLang vs llama.cpp | `scripts/run_ablation_matrix.py` with `examples/runtime_ablation.example.json` | per-runtime load summaries and `ablation_report.md` |
| AWQ vs GPTQ vs FP8 | `scripts/run_ablation_matrix.py` with `examples/quant_ablation.example.json` | per-quant load/quality summaries and `ablation_report.md` |
| Multi-run confidence intervals | `scripts/run_repeated_load.py`, `scripts/summarize_repeats_ci.py` | repeated run summaries with mean/stddev/95% CI |
| Full matrix readiness | `scripts/check_experiment_readiness.py`, `examples/full_research_matrix.example.json` | explicit ready/blocked table before launching expensive jobs |

Current status: the BF16 baseline is complete on the documented eight-GPU
ParaCloud topology. MT-Bench judging remains blocked on a configured external
judge. A local 72B BF16/FP16 baseline remains infeasible on one L20 because the
weights alone require roughly 144GB before KV cache and runtime overhead.

Example baseline retention run, once a real BF16/FP16 endpoint exists:

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

Example MT-Bench judge pass:

```bash
python3 scripts/score_mt_bench.py \
  --out runs/qwen72b-awq-mt-bench-judge \
  --samples runs/qwen72b-awq-mt-bench/samples.jsonl \
  --judge-base-url http://JUDGE_HOST:8001/v1 \
  --judge-model judge-model-name \
  --concurrency 4
```

Example LongBench 8K subset run:

```bash
python3 scripts/run_longbench_8k.py \
  --out runs/qwen72b-awq-longbench-8k \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen72b-awq-l20 \
  --longbench-dir /path/to/LongBench/data \
  --tasks multifieldqa_en hotpotqa qasper passage_count lcc gov_report \
  --max-per-task 20 \
  --concurrency 1
```

Example LongBench official-style metric postprocess:

```bash
python3 scripts/score_longbench_official.py \
  --samples runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/eval/samples.jsonl \
  --out runs/qwen72b-awq-l20/quality-eval-20260520T084323Z/current-longbench-8k/official-metrics
```

Example repeated load run with confidence intervals:

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

Example readiness check for the full research matrix:

```bash
python3 scripts/check_experiment_readiness.py \
  --manifest examples/full_research_matrix.example.json \
  --out runs/research-matrix-readiness/readiness.json
```

See [docs/strict-experiment-suite.md](docs/strict-experiment-suite.md) for the stricter runbook and claim boundaries.

Recommended external benchmarks to add:

- General capability: `lm-evaluation-harness`, for example MMLU/MMLU-Pro, GPQA, GSM8K/MATH, ARC, HellaSwag, and TruthfulQA.
- Chinese and broad benchmark coverage: OpenCompass, for example C-Eval, CMMLU, AGIEval, BBH, GSM8K, MATH, and HumanEval.
- Long context: LongBench or LongBench v2, plus needle-in-a-haystack or custom 8K/16K/32K retrieval and summarization sets.
- Instruction following and preference: MT-Bench, AlpacaEval, and Arena-Hard. Use answer-order swapping for judge runs and sample human review when possible.
- Code: HumanEval, MBPP, BigCodeBench, and real repository edit tasks.
- Serving capacity: vLLM `bench serve` or NVIDIA GenAI-Perf with fixed request-rate, concurrency, input-length, and output-length matrices for TTFT, ITL, TPOT, throughput, and goodput.

References:

- [HELM by Stanford CRFM](https://crfm.stanford.edu/2022/11/17/helm.html)
- [HELM GitHub](https://github.com/stanford-crfm/helm)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenCompass](https://github.com/open-compass/opencompass)
- [LongBench paper](https://arxiv.org/abs/2308.14508)
- [MT-Bench and Chatbot Arena paper](https://arxiv.org/abs/2306.05685)
- [NVIDIA GenAI-Perf metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2500/user-guide/docs/perf_analyzer/genai-perf/README.html)
- [vLLM serving benchmark](https://docs.vllm.ai/en/latest/cli/bench/serve.html)

## Quick Start

Run an end-to-end smoke test without a real model:

```bash
python3 -m llm_quant_bench demo --out runs/demo
```

The demo starts a local mock OpenAI-compatible server and writes:

- `runs/demo/results.jsonl`
- `runs/demo/summary.json`
- `runs/demo/report.md`

If this command fails, the problem is inside the benchmark tool. If this command works but a real 70B model fails, the issue is usually the model-serving endpoint, OpenAI-compatible protocol behavior, GPU memory, timeout settings, or concurrency settings.

Prepare two services:

- `baseline`: an FP16/BF16 70B model, or another high-quality reference model that you trust.
- `candidate`: the quantized model under test, such as AWQ, GPTQ, GGUF, or EXL2 Q4.

Edit [examples/config.example.json](examples/config.example.json), then run:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --repeats 3 \
  --concurrency 1
```

Outputs:

- `runs/l20-70b-q4/results.jsonl`: raw baseline and candidate results for each prompt.
- `runs/l20-70b-q4/summary.json`: machine-readable aggregate metrics.
- `runs/l20-70b-q4/report.md`: Markdown report ready for internal review.

If the server does not support streaming:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --no-stream
```

For OpenAI-compatible servers that support streaming usage metadata, add:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4 \
  --stream-usage
```

This sends `stream_options.include_usage=true` so prompt and completion token counts can come from the server instead of tokenizer-free estimates.

## Candidate Load Test

The `run` command requests both the baseline and candidate, so its throughput only describes the benchmark run itself. To measure candidate-only serving capacity, use `load`:

```bash
python3 -m llm_quant_bench load \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-load-c1 \
  --concurrency 1 \
  --requests 100
```

Run for a fixed duration:

```bash
python3 -m llm_quant_bench load \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-load-c4-10m \
  --concurrency 4 \
  --duration-seconds 600
```

If the endpoint supports OpenAI streaming usage events, add `--stream-usage` to collect prompt token totals during load tests.

To measure SLO-constrained goodput instead of treating every successful request
as equally useful, set one or more per-request thresholds:

```bash
python3 -m llm_quant_bench load \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-goodput-c16 \
  --concurrency 16 \
  --requests 256 \
  --slo-ttft-ms 7000 \
  --slo-tpot-ms 100 \
  --slo-e2e-ms 25000
```

A request contributes to `request_goodput` and `output_token_goodput` only if
it succeeds and every configured metric is present and within its SLO. Missing
streaming TTFT/TPOT data therefore fails closed. Repeated-run confidence
summaries include these goodput metrics when present.

Outputs:

- `load_results.jsonl`
- `load_summary.json`
- `load_report.md`

This command only load-tests an already running candidate endpoint. It does not load, quantize, or optimize the 70B model. For real 70B serving on an L20, first serve the quantized model with vLLM, llama.cpp, TGI, TensorRT-LLM, or another inference stack.

For one concrete single-L20 Qwen2.5-72B AWQ run, see [docs/l20-qwen72b-awq-results.md](docs/l20-qwen72b-awq-results.md).

Run a long soak test:

```bash
python3 -m llm_quant_bench run \
  --config examples/config.example.json \
  --dataset examples/golden_set.jsonl \
  --out runs/l20-70b-q4-soak-24h \
  --duration-seconds 86400 \
  --concurrency 1
```

## Golden Set Format

Each line is one JSON object. The only required field is `prompt`:

```json
{"id":"json-001","category":"format","severity":"critical","prompt":"Return only valid JSON...","require_json":true,"contains_all":["answer"]}
```

Supported scoring fields:

- `expected`: exact match after normalization.
- `expected_regex`: regular-expression match.
- `contains_all`: output must contain all listed strings.
- `require_json`: output must be valid JSON.
- `expected_number` and `number_tolerance`: numeric answer with tolerance.
- `reference` and `min_similarity`: text similarity against a reference answer.
- `min_chars` and `max_chars`: length constraints.
- `severity: critical|high|normal`: failures on important samples count as severe errors.
- `context_tokens`: explicit context-length marker used for long-context success rate.

Samples with no objective scoring rule and no judge are only checked for non-empty output. They are counted in `weakly_scored_rate`. For external claims such as "near-lossless", target `weakly_scored_rate = 0`.

## Metrics

The goal is not only to prove that a 70B model can load. The goal is to prove that quality stays close to the baseline and that the service remains stable over time.

Quality retention:

```text
quality_retention_raw = mean(candidate_quality_score) / mean(baseline_quality_score)
quality_retention = min(quality_retention_raw, 1.0)
```

Win/tie rate:

```text
win_tie_rate = (candidate_wins + ties) / total_pairs
loss_rate = candidate_losses / total_pairs
```

Stability:

```text
success_rate = successful_candidate_requests / total_candidate_requests
severe_error_rate = severe_candidate_errors / total_candidate_requests
weakly_scored_rate = samples_with_only_non_empty_scoring_and_no_judge / total_pairs
judge_inconsistency_rate = position_swapped_judge_disagreements / judged_pairs
judge_unvalidated_rate = judged_pairs_with_only_one_valid_order / judged_pairs
```

Performance attainment:

```text
performance_attainment =
  candidate_requests_meeting_ttft_itl_tpot_and_tokens_per_second_targets
  / successful_candidate_requests
```

Streaming performance:

```text
time_per_output_token = decode_seconds / output_tokens
inter_token_latency ~= mean_time_between_streaming_chunks
output_token_throughput = total_candidate_output_tokens / benchmark_wall_clock_seconds
request_throughput = successful_candidate_requests / benchmark_wall_clock_seconds
```

Long-context success rate:

```text
context_success_rate =
  successful_candidate_requests_with_context_tokens >= target_context_tokens
  / total_candidate_requests_with_context_tokens >= target_context_tokens
```

Composite usability:

```text
usability_score =
  quality_retention
  * success_rate
  * performance_attainment
  * context_success_rate
```

Perplexity:

```text
PPL = exp(-mean(log p(token_i)))
PPL_delta_percent = (candidate_ppl - baseline_ppl) / baseline_ppl * 100
```

If an external tool has already produced token logprobs, compute perplexity directly:

```bash
python3 -m llm_quant_bench ppl \
  --baseline-logprobs baseline_logprobs.json \
  --candidate-logprobs candidate_logprobs.json
```

`baseline_logprobs.json` can be a JSON array or:

```json
{"token_logprobs":[-0.1,-0.3,-0.2]}
```

## Suggested Gates

A practical definition of "near-lossless sustained usability" for a single-L20 70B Q4 deployment could be:

- `quality_retention >= 0.98`
- `win_tie_rate >= 0.95`
- `success_rate >= 0.995`
- `weakly_scored_rate = 0`
- `judge_inconsistency_rate <= 0.05`
- `judge_unvalidated_rate = 0`
- `severe_error_rate <= 0.005`
- 8K or 16K long-context `context_success_rate >= 0.99`
- 0 OOM or CUDA crashes in a 24-72 hour soak test
- `p95_ttft_s` and `p05_tokens_per_second` meet the product SLA

For stronger evidence, expand `examples/golden_set.jsonl` to 200-1000 samples covering your real traffic: RAG, code, JSON tool calls, long-document summarization, math reasoning, safety refusals, multi-turn dialogue, and high-frequency support questions.

## External Benchmark Commands

This repo also includes a lightweight endpoint-based quality runner for same-prompt
candidate-vs-baseline retention checks. It is useful when the target model is
already served through an OpenAI-compatible endpoint and official harnesses are
too heavy for a first systems pass.

```bash
python3 scripts/run_quality_eval.py \
  --out runs/qwen72b-awq-quality \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen72b-awq-l20 \
  --benchmarks mmlu cmmlu gsm8k \
  --cmmlu-dir /path/to/cmmlu/test \
  --concurrency 8
```

For LongBench, use a long-context service configuration and pass local JSONL
files:

```bash
python3 scripts/run_quality_eval.py \
  --out runs/qwen72b-awq-longbench \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen72b-awq-l20 \
  --benchmarks longbench \
  --longbench-dir /path/to/LongBench/data \
  --longbench-tasks multifieldqa_en hotpotqa qasper \
  --concurrency 1
```

When both baseline and candidate summaries are available, compute retention with:

```bash
python3 scripts/summarize_quality_retention.py \
  --baseline-summary runs/baseline-quality/summary.json \
  --candidate-summary runs/qwen72b-awq-quality/summary.json \
  --out runs/qwen72b-awq-quality/quality_retention.json
```

The formula is:

```text
quality_retention = candidate_score / baseline_score
delta_percentage_points = 100 * (candidate_score - baseline_score)
```

`lm-evaluation-harness` is useful for general capability checks:

```bash
lm-eval run \
  --model local-chat-completions \
  --model_args model=quantized-70b,base_url=http://localhost:8001/v1/chat/completions,num_concurrent=1,max_retries=3 \
  --tasks mmlu_pro,gpqa,gsm8k_cot,mathqa,arc_challenge,hellaswag,truthfulqa_mc2 \
  --apply_chat_template \
  --batch_size auto \
  --output_path runs/external/lm-eval-l20-70b-q4
```

Task names change across `lm-evaluation-harness` versions. Run `lm-eval ls tasks` first to verify the tasks available in your environment.

vLLM serving benchmark is useful for serving-capacity checks:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://localhost:8001 \
  --endpoint /v1/chat/completions \
  --model quantized-70b \
  --dataset-name random \
  --input-len 8192 \
  --output-len 512 \
  --num-prompts 200 \
  --request-rate 1
```

GenAI-Perf is useful for a more standardized serving report:

```bash
genai-perf profile \
  -m quantized-70b \
  --service-kind openai \
  --endpoint-type chat \
  --streaming \
  --url localhost:8001 \
  --synthetic-input-tokens-mean 8192 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 512 \
  --output-tokens-stddev 0 \
  --num-prompts 200 \
  --concurrency 1
```
