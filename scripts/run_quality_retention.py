#!/usr/bin/env python3
"""Run matched baseline and candidate quality evaluations, then compute retention.

This is the wrapper to use when a real FP16/BF16 baseline endpoint exists. It
invokes scripts/run_quality_eval.py twice with the same benchmark data and then
invokes scripts/summarize_quality_retention.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_SCRIPT = REPO_ROOT / "scripts" / "run_quality_eval.py"
RETENTION_SCRIPT = REPO_ROOT / "scripts" / "summarize_quality_retention.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-base-url", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-base-url", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--items-file")
    parser.add_argument("--cmmlu-dir")
    parser.add_argument("--longbench-dir")
    parser.add_argument("--longbench-tasks", nargs="*")
    parser.add_argument("--mt-bench-file")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-per-task", type=int)
    parser.add_argument("--baseline-api-key")
    parser.add_argument("--candidate-api-key")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_out = args.out / "baseline"
    candidate_out = args.out / "candidate"
    retention_out = args.out / "quality_retention.json"

    baseline_cmd = quality_eval_command(
        out=baseline_out,
        base_url=args.baseline_base_url,
        model=args.baseline_model,
        api_key=args.baseline_api_key,
        args=args,
    )
    candidate_cmd = quality_eval_command(
        out=candidate_out,
        base_url=args.candidate_base_url,
        model=args.candidate_model,
        api_key=args.candidate_api_key,
        args=args,
    )
    retention_cmd = [
        sys.executable,
        str(RETENTION_SCRIPT),
        "--baseline-summary",
        str(baseline_out / "summary.json"),
        "--candidate-summary",
        str(candidate_out / "summary.json"),
        "--baseline-samples",
        str(baseline_out / "samples.jsonl"),
        "--candidate-samples",
        str(candidate_out / "samples.jsonl"),
        "--out",
        str(retention_out),
    ]

    manifest = {
        "baseline_command": baseline_cmd,
        "candidate_command": candidate_cmd,
        "retention_command": retention_cmd,
        "formula": "quality_retention = candidate_score / baseline_score",
    }
    (args.out / "retention_run_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    run_logged(baseline_cmd, args.out / "baseline.log")
    run_logged(candidate_cmd, args.out / "candidate.log")
    run_logged(retention_cmd, args.out / "retention.log")
    return 0


def quality_eval_command(
    *,
    out: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(QUALITY_SCRIPT),
        "--out",
        str(out),
        "--base-url",
        base_url,
        "--model",
        model,
        "--benchmarks",
        *args.benchmarks,
        "--concurrency",
        str(args.concurrency),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if args.cmmlu_dir:
        cmd.extend(["--cmmlu-dir", args.cmmlu_dir])
    if getattr(args, "items_file", None):
        cmd.extend(["--items-file", args.items_file])
    if args.longbench_dir:
        cmd.extend(["--longbench-dir", args.longbench_dir])
    if args.longbench_tasks:
        cmd.extend(["--longbench-tasks", *args.longbench_tasks])
    if args.mt_bench_file:
        cmd.extend(["--mt-bench-file", args.mt_bench_file])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.max_per_task is not None:
        cmd.extend(["--max-per-task", str(args.max_per_task)])
    return cmd


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
