# Pilot completeness observation — 2026-09-06

This is an in-progress integrity observation, not a completed pilot result.

- Job: `1561180`, source `2c0f118`, frozen protocol v4.
- At the 13:58 UTC observation, Slurm was RUNNING on `wqd10nba06g8`.
  The baseline telemetry then advanced from 205 to 206 completed requests;
  the candidate arm had not started.
- One anomalous entry was present: zero-based request/row `152`, prompt
  length 1,922 tokens, output length 988 tokens, `finish_reason="stop"`,
  `stop_reason=null`, `thinking_closed=false`, elapsed 18.7086 seconds.
- There were no length-limited or other non-stop completions, and no observed
  traceback, CUDA OOM, Ninja failure, or engine-initialization failure.
- `thinking_closed` is a literal `</think>` substring check on vLLM's output
  text. Absence of this marker does not by itself establish why the model
  omitted it. Raw sample files had not yet been emitted by the evaluator,
  so the answer text could not yet be audited from persisted artifacts.

The existing runner rejects any nonzero `unclosed_thinking` count after
evaluation. Under that unchanged rule, this baseline cannot pass the
completion gate, even if every remaining request completes normally.
The single-allocation wrapper uses `set -e`, so it will not start FP8 after
that baseline failure. No runtime restart, resubmission, cancellation,
checkpoint deletion, scoring change, or threshold relaxation was performed.

Allow the current evaluation to retain its results, then inspect the raw
affected response and aggregate all integrity/parse failures. Preserve the
original score and failure classification; do not declare quality retention
or start performance evaluation from this pilot.

## 20:00 UTC observation: length-limit failures

- Slurm remained RUNNING after 7h31m54s, with 665/1,229 baseline requests
  recorded. FP8 had not started and no completion receipt existed yet.
- Five requests reached the frozen 32,768-token generation limit:
  zero-based request IDs `413`, `450`, `570`, `591`, and `652`.
  Each reported `finish_reason="length"`, missing `</think>`, and roughly
  613 seconds of generation time. The generation budget was not changed.
- Twelve total responses lacked the thinking-end marker: the five truncated
  responses plus seven responses that reported normal `stop` completion.
  These counts overlap and must not be added as independent failures.
- Instrumented calls totaled 1,398,219 output tokens and 26,222.74 seconds.
  No traceback, CUDA OOM, Ninja failure or engine-initialization failure
  was observed. A subsequent allocation-local check found one vLLM engine,
  four tensor-parallel workers, and four RTX 4090s at 100% utilization,
  with 22,167 MiB used on each device.
- The length-limited responses are a new failure type beyond the earlier
  missing-marker-only cases. Persisted raw answers are still needed to
  distinguish prolonged reasoning, repetitive generation, or other causes;
  a larger token budget is not assumed to be a valid repair.

Continue retaining the current evaluation's artifacts for the final audit.
No model, prompt, parser, generation budget, or completion gate was changed,
and no additional GPU job was submitted.
