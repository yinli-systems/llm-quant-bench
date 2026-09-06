# Qwen2.5-72B BF16 versus AWQ quality retention

Status: complete, evidence-gated ParaCloud run.

## Scope

- Frozen evaluation bundle: 26,943 items across MMLU test, CMMLU test, and
  GSM8K test.
- Frozen item SHA-256:
  `9b65268478b32ad8d077c40f4d914f421055f521ce2eefd832048e2b10ce618b`.
- BF16 model revision:
  `495f39366efef23836d0cfae4fbe635880d2be31`.
- AWQ model revision:
  `698703eae6604af048a3d2f509995dc302088217`.
- BF16 allocation: Slurm job `1559100`, four nodes, two RTX 4090 GPUs per
  node, TP=2, PP=4. The job and its retention dependency completed with exit
  code 0.
- AWQ allocation: Slurm job `1557942`, one node, two RTX 4090 GPUs, TP=2.

Every request succeeded in both arms. The committed summaries, GPU inventory,
model snapshot manifests, and paired retention reduction are covered by
`SHA256SUMS`.

## Primary result

| Benchmark | BF16 | AWQ | AWQ - BF16 | Retention |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 0.817262 | 0.812776 | -0.449 pp | 99.451% |
| CMMLU | 0.835521 | 0.830168 | -0.535 pp | 99.359% |
| GSM8K | 0.807430 | 0.792267 | -1.516 pp | 98.122% |

The paired normal-approximation interval excludes zero for MMLU and CMMLU,
but includes zero for GSM8K. See `quality-retention.json` for paired wins,
losses, ties, and intervals.

## Claim boundary

This establishes quality retention for these exact model revisions, frozen
items, prompts, parser, and serving settings. It does not establish a
like-for-like latency, throughput, or cost comparison: BF16 used eight GPUs
with pipeline parallelism, while AWQ used two GPUs with tensor parallelism.
The raw per-item samples remain in immutable remote custody and are omitted
from Git because they reproduce benchmark prompts and model responses; their
shared item hash and aggregate evidence are committed here.

The launcher now tears down the long-lived Ray worker steps before requesting
the final all-node GPU snapshot. This fixes the post-evaluation Slurm step
deadlock observed after the result had already passed its completeness gate.
