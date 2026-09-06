# Qwen3.8 preflight evidence — 2026-09-06

This is pipeline and hardware validation, not Qwen model quality evidence.

- Harness: `b954108c9baaaa934b4ad842033b31a97ee30816`.
- Active evaluation protocol: `qwen38-27b-bf16-fp8-pareto-paracloud-v3`.
- Existing GPQA CSV matched upstream Git blob
  `7589e3e467d69a1dceb126a60c4108d6d4f1d166` at dataset revision
  `633f5ee89ab8ad4522a9f850766b73f62147ffdd` and SHA-256
  `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
- Local dataset staging loaded 198 GPQA train rows and MMLU-Pro 12,032 test
  plus 70 validation rows at `b189ec765aa7ed75c8acfea42df31fdae71f97be`.
  Both reloaded successfully with Hugging Face and datasets offline flags.
- Actual generated task overlays loaded 198 GPQA documents and 14 MMLU-Pro
  subjects totaling 12,032 documents. First/last document prompt and target
  construction succeeded for every task.
- Pinned harness dummy model, one item per task, completed generation,
  filtering, scoring, and sample serialization. The downstream sample loader
  read one GPQA sample and 14 MMLU-Pro samples, including document, target,
  and prompt hashes. Dummy scores are not reported as model evidence.
- Local probe used Python 3.12, datasets 5.0.1, and the pinned harness;
  the remote GPU environment must independently pass the same checks.
- ParaCloud hardware probe job `1560672` returned four RTX 4090 GPUs,
  each 24,564 MiB / SM8.9, driver 580.82.07, kernel 5.15.0-119-generic,
  glibc 2.35. Raw probe logs are retained under
  `/ssd/scxi253/qwen38-27b-pareto-v2/logs/hardware-1560672.out`.

Model inference remains a separate required smoke test. Runtime installation,
model loading, CUDA kernels, generation completeness, and quality retention
are not established by this preflight report.
