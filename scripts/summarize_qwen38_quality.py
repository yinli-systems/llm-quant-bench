#!/usr/bin/env python3
"""Validate paired lm-eval samples and apply the frozen Qwen3.8 quality gates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_qwen38_pareto_protocol import (  # noqa: E402
    load_protocol,
    sha256,
    validate_static,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def contains_nonempty_string(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(contains_nonempty_string(item) for item in value)
    return False


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def load_run_receipt(run_dir: Path, expected_arm: str, protocol_sha256: str) -> dict[str, Any]:
    receipt = json.loads((run_dir / "run_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("arm") != expected_arm:
        raise ValueError(f"wrong arm receipt in {run_dir}: {receipt.get('arm')}")
    if receipt.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"protocol hash mismatch in {run_dir}")
    if receipt.get("dry_run") is not False or receipt.get("limit") is not None:
        raise ValueError(f"not a full scored run: {run_dir}")
    if receipt.get("evaluation_source_dirty") is not False or not receipt.get(
        "evaluation_source_revision"
    ):
        raise ValueError(f"uncommitted or unknown evaluation source in {run_dir}")
    completion = json.loads((run_dir / "completion_check.json").read_text(encoding="utf-8"))
    if (completion.get("requests") != 12230 or completion.get("length_limited") != 0
            or completion.get("unclosed_thinking") != 0):
        raise ValueError(f"incomplete generation evidence in {run_dir}: {completion}")
    return receipt


def load_task_samples(run_dir: Path, task_name: str, filter_name: str) -> dict[int, dict[str, Any]]:
    sample_path = find_one(run_dir, f"**/samples_{task_name}_*.jsonl")
    rows: dict[int, dict[str, Any]] = {}
    with sample_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("filter") != filter_name:
                continue
            doc_id = int(row["doc_id"])
            if doc_id in rows:
                raise ValueError(f"duplicate {task_name} doc_id {doc_id}")
            if not contains_nonempty_string(row.get("resps")):
                raise ValueError(f"empty raw response for {task_name} doc_id {doc_id}")
            if "exact_match" not in row:
                raise ValueError(f"missing exact_match for {task_name} doc_id {doc_id}")
            rows[doc_id] = row
    return rows


def load_samples(protocol: dict[str, Any], run_dir: Path) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    output: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    specs = {task["name"]: task for task in protocol["quality"]["standard_lane"]["tasks"]}
    gpqa = specs["gpqa_diamond_cot_zeroshot"]
    gpqa_rows = load_task_samples(run_dir, gpqa["execution_name"], gpqa["primary_filter"])
    output["gpqa_diamond_cot_zeroshot"] = {
        ("gpqa_diamond", doc_id): row for doc_id, row in gpqa_rows.items()
    }

    mmlu = specs["mmlu_pro"]
    mmlu_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for subtask in mmlu["subtasks"]:
        execution_name = f"qwen38_mmlu_pro_{subtask}_pinned"
        rows = load_task_samples(run_dir, execution_name, mmlu["primary_filter"])
        for doc_id, row in rows.items():
            mmlu_rows[(subtask, doc_id)] = row
    output["mmlu_pro"] = mmlu_rows
    return output


def paired_interval(differences: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, mean
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    return mean - 1.96 * standard_error, mean + 1.96 * standard_error


def analyze_task(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    expected_items: int,
) -> dict[str, Any]:
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            f"paired item mismatch; missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    if len(baseline) != expected_items:
        raise ValueError(f"expected {expected_items} paired items, found {len(baseline)}")

    baseline_scores = []
    candidate_scores = []
    differences = []
    disagreements = 0
    for key in sorted(baseline):
        base = baseline[key]
        cand = candidate[key]
        if (
            base.get("doc_hash") != cand.get("doc_hash")
            or base.get("target_hash") != cand.get("target_hash")
            or base.get("prompt_hash") != cand.get("prompt_hash")
        ):
            raise ValueError(f"document/target/prompt hash mismatch for {key}")
        base_score = float(base["exact_match"])
        cand_score = float(cand["exact_match"])
        if base_score not in (0.0, 1.0) or cand_score not in (0.0, 1.0):
            raise ValueError(f"non-binary exact_match for {key}")
        baseline_scores.append(base_score)
        candidate_scores.append(cand_score)
        differences.append(cand_score - base_score)
        disagreements += base_score != cand_score

    baseline_score = statistics.fmean(baseline_scores)
    candidate_score = statistics.fmean(candidate_scores)
    delta = candidate_score - baseline_score
    ci_low, ci_high = paired_interval(differences)
    return {
        "paired_items": len(baseline_scores),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "candidate_minus_baseline": delta,
        "candidate_minus_baseline_percentage_points": delta * 100,
        "quality_retention": candidate_score / baseline_score if baseline_score else None,
        "paired_normal_95_ci": [ci_low, ci_high],
        "disagreements": disagreements,
    }


def apply_gates(protocol: dict[str, Any], results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    gates = protocol["gates"]
    checks = []
    for task, max_drop_pp in (
        ("gpqa_diamond_cot_zeroshot", gates["maximum_gpqa_absolute_drop_percentage_points"]),
        ("mmlu_pro", gates["maximum_mmlu_pro_absolute_drop_percentage_points"]),
    ):
        result = results[task]
        retention = result["quality_retention"]
        checks.append(
            {
                "name": f"{task} retention",
                "passed": retention is not None and retention >= gates["minimum_quality_retention"],
                "actual": retention,
                "threshold": gates["minimum_quality_retention"],
            }
        )
        checks.append(
            {
                "name": f"{task} absolute drop",
                "passed": result["candidate_minus_baseline_percentage_points"] >= -max_drop_pp,
                "actual": result["candidate_minus_baseline_percentage_points"],
                "threshold": -max_drop_pp,
            }
        )
    return checks


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.8-27B BF16 vs FP8 standard quality gate",
        "",
        f"- Status: **{summary['status']}**",
        f"- Protocol SHA-256: `{summary['protocol_sha256']}`",
        "",
        "| Task | Items | BF16 | FP8 | Delta (pp) | Retention | 95% paired CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, result in summary["tasks"].items():
        ci = result["paired_normal_95_ci"]
        lines.append(
            f"| {task} | {result['paired_items']} | {result['baseline_score']:.4%} | "
            f"{result['candidate_score']:.4%} | {result['candidate_minus_baseline_percentage_points']:+.3f} | "
            f"{result['quality_retention']:.4%} | [{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}] pp |"
        )
    lines.extend(["", "## Gates", ""])
    for gate in summary["gates"]:
        lines.append(
            f"- {'PASS' if gate['passed'] else 'FAIL'} — {gate['name']}: "
            f"actual={gate['actual']}, threshold={gate['threshold']}"
        )
    lines.extend(
        [
            "",
            "This is a checkpoint-specific matched evaluation. It is not a "
            "reproduction of Qwen's official GPQA score.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if any(not check["ok"] for check in validate_static(protocol)):
        raise SystemExit("static protocol validation failed")
    protocol_hash = sha256(args.protocol)
    baseline_receipt = load_run_receipt(args.baseline_dir, "baseline", protocol_hash)
    candidate_receipt = load_run_receipt(args.candidate_dir, "candidate", protocol_hash)
    if baseline_receipt["tasks"] != candidate_receipt["tasks"]:
        raise SystemExit("arm task lists differ")
    if baseline_receipt["evaluation_source_revision"] != candidate_receipt["evaluation_source_revision"]:
        raise SystemExit("arm evaluation source revisions differ")
    for arm_name, receipt in (("baseline", baseline_receipt), ("candidate", candidate_receipt)):
        arm = protocol["arms"][arm_name]
        if receipt.get("model_id") != arm["model_id"] or receipt.get("model_revision") != arm["revision"]:
            raise SystemExit(f"{arm_name} model receipt differs from protocol")

    baseline = load_samples(protocol, args.baseline_dir)
    candidate = load_samples(protocol, args.candidate_dir)
    tasks = protocol["quality"]["standard_lane"]["tasks"]
    task_specs = {task["name"]: task for task in tasks}
    task_results = {
        "gpqa_diamond_cot_zeroshot": analyze_task(
            baseline["gpqa_diamond_cot_zeroshot"],
            candidate["gpqa_diamond_cot_zeroshot"],
            task_specs["gpqa_diamond_cot_zeroshot"]["expected_split_rows"]["train"],
        ),
        "mmlu_pro": analyze_task(
            baseline["mmlu_pro"],
            candidate["mmlu_pro"],
            task_specs["mmlu_pro"]["expected_split_rows"]["test"],
        ),
    }
    gate_results = apply_gates(protocol, task_results)
    passed = all(gate["passed"] for gate in gate_results)
    summary = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_hash,
        "status": "pass" if passed else "fail",
        "tasks": task_results,
        "gates": gate_results,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "quality_gate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.out / "quality_gate.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
