# Qwen3.8-27B BF16 vs FP8 Pareto experiment

Active scoring protocol: **v3**, frozen before the first scored run. It gives
both arms 32,768 generation tokens within 65,536 context. Inspection of the
pinned harness found GPQA otherwise defaults to 256 generation tokens and
MMLU-Pro to 2,048, which can truncate the requested xhigh reasoning. CLI
generation overrides apply the same explicit budget to both tasks and arms;
all upstream prompts, parsers, and deterministic sampling settings remain.
Versions 1 and 2 below preserve the planning and storage history.

Status on 2026-09-06: version 1 remains preserved as the original blocked
preregistration. After explicit authorization, the reproducible old
Qwen2.5-72B BF16 snapshot was removed while the old AWQ snapshot and all run
evidence were preserved. Version 2 changes only storage sequencing: the two
official model snapshots total 86,429,873,704 weight bytes, so it requires 128
GiB before model staging, 48 GiB before runtime staging, and still reserves 32
GiB for execution. Model, dataset, harness, runtime, topology, prompt, seed,
and quality gates are unchanged.

## Why this experiment

The previous Qwen2.5-72B result established a narrow BF16-versus-AWQ quality
retention result, but its different GPU counts make it unsuitable for a speed
or cost claim. This follow-up tests the official Qwen3.8-27B BF16 and FP8
checkpoints on the same four RTX 4090 GPUs and the same software, task, prompt,
seed, and serving contracts.

Qwen's model card describes Qwen3.8-27B as a 27B dense vision-language model,
with thinking enabled by default and `reasoning_effort` values of `xhigh`,
`medium`, and `low`. It reports 89.2 on GPQA Diamond, but does not provide
enough detail on that page to treat a local harness run as a reproduction of
the official score. The local primary claim is therefore FP8 retention versus
the pinned BF16 checkpoint, not agreement with 89.2.

Primary sources used to freeze the protocol:

- Qwen3.8-27B model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3.8-27B FP8 checkpoint: <https://huggingface.co/Qwen/Qwen3.8-27B-FP8>
- lm-evaluation-harness: <https://github.com/EleutherAI/lm-evaluation-harness>
- vLLM FP8/hardware support: <https://docs.vllm.ai/en/stable/features/quantization/>

## Frozen evidence lanes

The standard lane uses `gpqa_diamond_cot_zeroshot` task version 2.2 and
`mmlu_pro` task version 3.1 from lm-evaluation-harness commit
`b954108c9baaaa934b4ad842033b31a97ee30816`. It also pins GPQA dataset commit
`633f5ee89ab8ad4522a9f850766b73f62147ffdd` and MMLU-Pro dataset commit
`b189ec765aa7ed75c8acfea42df31fdae71f97be` through thin task overlays; the
upstream prompts, processors, filters, and metrics remain unchanged. Both tasks are generative, so the
same `xhigh` thinking contract can be applied to both arms. Version 2.2 matters:
it includes the upstream GPQA answer-preprocessing correction.

The robustness lane is deliberately separate. It applies four gold-blind,
balanced answer rotations per GPQA question, reports ties as abstentions, and
uses question-level paired inference. Its result is never labeled as an
official GPQA leaderboard score.

The serving lane runs only after quality passes. It uses the same four GPUs,
TP=4, fixed approximately 512-input/256-output shapes, three repeats at
concurrency 1/4/8/16, exact request accounting, and failure-complete latency
and throughput summaries. A later minimum-GPU run is labeled capacity evidence
and is not folded into the matched-topology speedup.

The active storage-recovery protocol is
`protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v2.json`; version 1 remains an
immutable record of the original blocked plan.

## Local validation

Validate the protocol against the exact harness checkout:

```bash
python3 scripts/validate_qwen38_pareto_protocol.py \
  --protocol protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v2.json \
  --lm-eval-root /path/to/lm-evaluation-harness
```

Inspect the exact command without loading a model:

```bash
python3 scripts/run_qwen38_standard_eval.py \
  --protocol protocols/qwen38_27b_bf16_fp8_pareto_paracloud_v2.json \
  --lm-eval-root /path/to/lm-evaluation-harness \
  --arm baseline \
  --model-path /path/to/qwen38-27b-bf16 \
  --out /tmp/qwen38-dry-run \
  --limit 1 \
  --dry-run
```

The dry run still verifies the protocol, harness commit, and task metadata. A
real run additionally requires `.READY` and `SNAPSHOT_MANIFEST.json` model
receipts.

## Remote staging and launch order

Version 2 requires at least 128 GiB before downloading both exact model
snapshots, at least 48 GiB before building the runtime, and at least 32 GiB
after all immutable receipts exist. These are separate gates because the
official pinned weight blobs total 86,429,873,704 bytes.

After space is available:

1. Create `/ssd/scxi253/qwen38-27b-pareto-v2` and clone this repository at a
   recorded clean commit.
2. Download the two model snapshots at their pinned revisions on a networked
   login node with `scripts/stage_qwen38_models.py`. It hashes every downloaded
   file, records the exact repository and revision, and writes `.READY` only
   after the manifest is complete.
3. Build a fresh environment with `scripts/prepare_qwen38_runtime.sh`. It pins
   vLLM 0.23.0, checks the post-model 48 GiB floor, records `pip freeze`, and
   leaves the completed Qwen2.5 evidence environment untouched. Then run
   `scripts/stage_qwen38_datasets.py`
   in the same networked environment to warm the exact dataset revisions and
   create fingerprints/content receipts. GPQA is gated on Hugging Face, so its
   terms must already be accepted and `HF_TOKEN` must be supplied through the
   environment; the token is never written to a command or receipt.
   Alternatively, `--gpqa-csv /path/to/existing/gpqa_diamond.csv` reuses the
   previously downloaded file only when its Git blob is
   `7589e3e467d69a1dceb126a60c4108d6d4f1d166`, as listed in the official tree
   of the pinned dataset revision, and its SHA-256 is
   `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
   This transport fallback uses the same CSV dataset builder and preserves
   upstream question processing, prompts, and filters. Its local source is
   recorded in the dataset receipt and revalidated before evaluation.
4. Run the static and runtime preflight from the recorded clean repository
   commit.
5. Submit BF16 and FP8 smoke jobs (`MODE=smoke`). Each arm evaluates one item
   per task and exercises model load, chat templating, thinking stripping,
   parsing, and output receipts.
6. If both smokes pass, submit 10% pilots (`MODE=pilot`) for real throughput and
   wall-time estimates. Pilot scores are diagnostic and are not the final claim.
7. Submit full standard runs (`MODE=full`) only if the projected allocation is
   acceptable. Apply the paired quality gates with
   `scripts/summarize_qwen38_quality.py` before the robustness and serving lanes.

Example submission from the staged repository:

```bash
sbatch --export=ALL,ARM=baseline,MODE=smoke \
  cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch
sbatch --export=ALL,ARM=candidate,MODE=smoke \
  cluster/slurm/qwen38_27b_standard_quality_4x4090.sbatch
```

The Slurm script fails closed on wrong GPU count/model/capability, insufficient
execution disk, missing receipts, wrong vLLM version, wrong harness commit, task
version drift, model-content hash drift, and fatal CUDA/runtime log signatures.
