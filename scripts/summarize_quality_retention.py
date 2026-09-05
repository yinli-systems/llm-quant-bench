#!/usr/bin/env python3
"""Summarize baseline-vs-candidate quality retention from quality summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--baseline-samples")
    parser.add_argument("--candidate-samples")
    parser.add_argument("--baseline-gpus", type=int)
    parser.add_argument("--candidate-gpus", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate_summary).read_text(encoding="utf-8"))
    baseline_hash = baseline.get("items_sha256")
    candidate_hash = candidate.get("items_sha256")
    if not baseline_hash or baseline_hash != candidate_hash:
        raise ValueError(
            f"Frozen item hashes must be present and equal: {baseline_hash!r} != {candidate_hash!r}"
        )
    if set(baseline.get("by_benchmark", {})) != set(candidate.get("by_benchmark", {})):
        raise ValueError("Baseline and candidate benchmark coverage differs")
    if not args.allow_failures and (baseline.get("failed_items") or candidate.get("failed_items")):
        raise ValueError(
            "Quality retention requires zero failed requests unless --allow-failures is explicit"
        )

    paired = None
    if bool(args.baseline_samples) != bool(args.candidate_samples):
        raise ValueError("Provide both --baseline-samples and --candidate-samples, or neither")
    if args.baseline_samples:
        paired = paired_analysis(
            Path(args.baseline_samples), Path(args.candidate_samples)
        )
    rows = []
    for benchmark, base_row in baseline.get("by_benchmark", {}).items():
        cand_row = candidate.get("by_benchmark", {}).get(benchmark)
        if not cand_row:
            raise ValueError(f"Candidate summary missing benchmark {benchmark}")
        if base_row.get("items") != cand_row.get("items"):
            raise ValueError(f"Item-count mismatch for {benchmark}")
        if base_row.get("scored_items") != cand_row.get("scored_items"):
            raise ValueError(f"Scored-item mismatch for {benchmark}")
        base_score = base_row.get("score_mean")
        cand_score = cand_row.get("score_mean")
        retention = compute_retention(cand_score, base_score)
        delta_pp = None
        if cand_score is not None and base_score is not None:
            delta_pp = 100.0 * (cand_score - base_score)
        rows.append(
            {
                "benchmark": benchmark,
                "baseline_score": base_score,
                "candidate_score": cand_score,
                "quality_retention": retention,
                "delta_percentage_points": delta_pp,
                "baseline_items": base_row.get("scored_items"),
                "candidate_items": cand_row.get("scored_items"),
                "baseline_latency_p50_s": base_row.get("latency_p50_s"),
                "candidate_latency_p50_s": cand_row.get("latency_p50_s"),
                "baseline_latency_p95_s": base_row.get("latency_p95_s"),
                "candidate_latency_p95_s": cand_row.get("latency_p95_s"),
            }
        )
    output = {
        "status": "complete",
        "items_sha256": baseline_hash,
        "formula": "quality_retention = candidate_score / baseline_score",
        "delta_percentage_points_formula": "100 * (candidate_score - baseline_score)",
        "rows": rows,
        "paired_analysis": paired,
        "execution": {
            "baseline_elapsed_s": baseline.get("elapsed_s"),
            "candidate_elapsed_s": candidate.get("elapsed_s"),
            "baseline_gpus": args.baseline_gpus,
            "candidate_gpus": args.candidate_gpus,
            "baseline_eval_gpu_seconds": gpu_seconds(
                baseline.get("elapsed_s"), args.baseline_gpus
            ),
            "candidate_eval_gpu_seconds": gpu_seconds(
                candidate.get("elapsed_s"), args.candidate_gpus
            ),
            "scope_note": "GPU-seconds cover evaluation wall time, excluding model download and server startup.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def compute_retention(candidate_score: Any, baseline_score: Any) -> float | None:
    if candidate_score is None or baseline_score in (None, 0):
        return None
    return float(candidate_score) / float(baseline_score)


def gpu_seconds(elapsed_s: Any, gpu_count: int | None) -> float | None:
    if elapsed_s is None or gpu_count is None:
        return None
    return float(elapsed_s) * gpu_count


def load_samples(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["benchmark"]), str(row["task"]), str(row["item_id"]))
            if key in rows:
                raise ValueError(f"Duplicate sample identity in {path}: {key}")
            rows[key] = row
    return rows


def paired_analysis(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = load_samples(baseline_path)
    candidate = load_samples(candidate_path)
    if set(baseline) != set(candidate):
        missing = len(set(baseline) - set(candidate))
        extra = len(set(candidate) - set(baseline))
        raise ValueError(f"Paired sample coverage mismatch: missing={missing}, extra={extra}")

    by_benchmark: dict[str, list[tuple[float, float]]] = {}
    for key in sorted(baseline):
        base_row = baseline[key]
        cand_row = candidate[key]
        if not base_row.get("ok") or not cand_row.get("ok"):
            raise ValueError(f"Paired analysis requires successful requests: {key}")
        if base_row.get("score") is None or cand_row.get("score") is None:
            continue
        by_benchmark.setdefault(key[0], []).append(
            (float(base_row["score"]), float(cand_row["score"]))
        )

    result: dict[str, Any] = {}
    for benchmark, pairs in sorted(by_benchmark.items()):
        differences = [candidate_score - baseline_score for baseline_score, candidate_score in pairs]
        mean_difference = statistics.fmean(differences)
        standard_error = (
            statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1
            else 0.0
        )
        baseline_only = sum(base > cand for base, cand in pairs)
        candidate_only = sum(cand > base for base, cand in pairs)
        result[benchmark] = {
            "paired_items": len(pairs),
            "mean_score_delta": mean_difference,
            "mean_score_delta_percentage_points": 100.0 * mean_difference,
            "normal_approx_delta_95": [
                mean_difference - 1.959963984540054 * standard_error,
                mean_difference + 1.959963984540054 * standard_error,
            ],
            "baseline_only_wins": baseline_only,
            "candidate_only_wins": candidate_only,
            "ties": len(pairs) - baseline_only - candidate_only,
            "interval_note": "Paired normal-approximation interval over per-item score differences.",
        }
    return result


if __name__ == "__main__":
    main()
