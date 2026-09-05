# Quality Evaluation Report

- Total items: 26943
- Successful items: 26943
- Failed items: 0
- Elapsed seconds: 3503.45

## By Benchmark

| Benchmark | Items | OK | Failed | Scored | Mean Score | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|
| cmmlu | 11582 | 11582 | 0 | 11582 | 0.8302 | 0.75s |
| gsm8k | 1319 | 1319 | 0 | 1319 | 0.7923 | 13.14s |
| mmlu | 14042 | 14042 | 0 | 14042 | 0.8128 | 1.64s |

## Notes

- MMLU/CMMLU are zero-shot direct-answer multiple-choice evaluations.
- GSM8K uses exact match on the final extracted number.
- LongBench uses a lightweight max token-F1 scorer and should be used for same-run retention comparisons, not leaderboard claims.
- MT-Bench rows are answer generations only unless scored later by an external judge.
