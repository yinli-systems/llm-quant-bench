#!/usr/bin/env python3
"""Run one frozen lm-eval quality arm for the Qwen3.8 BF16/FP8 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_qwen38_pareto_protocol import (  # noqa: E402
    load_protocol,
    validate_lm_eval,
    validate_static,
    validate_gpqa_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lm-eval-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("baseline", "candidate"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", default=["gpqa_diamond_cot_zeroshot", "mmlu_pro"])
    parser.add_argument("--limit", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpqa-csv", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state(root: Path) -> tuple[str | None, bool]:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=False,
    )
    return (head.stdout.strip() if head.returncode == 0 else None, bool(status.stdout.strip()))


def require_valid(checks: list[dict[str, Any]]) -> None:
    failures = [check for check in checks if not check["ok"]]
    if failures:
        rendered = "; ".join(f"{check['name']}: {check['detail']}" for check in failures)
        raise SystemExit(f"preflight failed: {rendered}")


def build_command(
    protocol: dict[str, Any],
    *,
    model_path: Path,
    out: Path,
    tasks: list[str],
    include_path: Path,
    execution_tasks: list[str],
    limit: float | None,
) -> list[str]:
    topology = protocol["matched_topology"]
    quality = protocol["quality"]
    allowed_tasks = {task["name"] for task in quality["standard_lane"]["tasks"]}
    unexpected = sorted(set(tasks) - allowed_tasks)
    if unexpected:
        raise ValueError(f"tasks are not frozen in the protocol: {unexpected}")

    model_args = {
        "pretrained": str(model_path),
        "dtype": "bfloat16",
        "tensor_parallel_size": topology["tensor_parallel_size"],
        "max_model_len": topology["max_model_len"],
        "gpu_memory_utilization": topology["gpu_memory_utilization"],
        "enable_thinking": quality["enable_thinking"],
        "think_end_token": quality["think_end_token"],
        "chat_template_args": {
            "reasoning_effort": quality["reasoning_effort"],
            "preserve_thinking": True,
        },
    }
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        quality["backend"],
        "--model_args",
        json.dumps(model_args, separators=(",", ":")),
        "--tasks",
        ",".join(execution_tasks),
        "--batch_size",
        str(quality["batch_size"]),
        "--seed",
        quality["seed"],
        "--apply_chat_template",
        "--show_config",
        "--log_samples",
        "--output_path",
        str(out),
        "--include_path",
        str(include_path),
    ]
    if "max_gen_toks" in quality:
        command.extend(["--gen_kwargs", json.dumps({"max_gen_toks": quality["max_gen_toks"]})])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def yaml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def validate_task_loading(include_path: Path, execution_tasks: list[str]) -> dict[str, Any]:
    """Load the actual overlays and validate complete document coverage before GPU initialization."""
    import random
    import numpy as np
    from lm_eval.tasks import TaskManager

    random.seed(1234)
    np.random.seed(1234)
    loaded = TaskManager(include_path=str(include_path)).load(execution_tasks)
    rows = {}
    for name, task in loaded["tasks"].items():
        docs = task.eval_docs
        if len(docs) == 0:
            raise ValueError(f"empty evaluation task: {name}")
        for index in (0, len(docs) - 1):
            prompt, target = task.doc_to_text(docs[index]), task.doc_to_target(docs[index])
            if not isinstance(prompt, str) or not prompt or target is None:
                raise ValueError(f"invalid prompt/target in {name} at {index}")
        rows[name] = len(docs)
    if "qwen38_gpqa_diamond_cot_zeroshot_pinned" in execution_tasks:
        if rows.get("qwen38_gpqa_diamond_cot_zeroshot_pinned") != 198:
            raise ValueError("GPQA task coverage mismatch")
    if "qwen38_mmlu_pro_pinned" in execution_tasks:
        mmlu = {name: count for name, count in rows.items() if name.startswith("qwen38_mmlu_pro_")}
        if len(mmlu) != 14 or sum(mmlu.values()) != 12032:
            raise ValueError("MMLU-Pro task coverage mismatch")
    return {"status": "passed", "task_rows": rows}


def prepare_task_overlays(
    protocol: dict[str, Any],
    *,
    lm_eval_root: Path,
    out: Path,
    tasks: list[str],
    gpqa_csv: Path | None = None,
) -> tuple[Path, list[str]]:
    """Create thin task overlays that pin datasets without copying task logic."""
    overlay_root = out / "task_overrides"
    overlay_root.mkdir(parents=True, exist_ok=True)
    task_specs = {task["name"]: task for task in protocol["quality"]["standard_lane"]["tasks"]}
    execution_tasks: list[str] = []

    for name in tasks:
        spec = task_specs[name]
        if name == "gpqa_diamond_cot_zeroshot":
            source = (lm_eval_root / spec["task_config_relative_path"]).resolve()
            content = (
                f"include: {yaml_string(source)}\n"
                f"task: {yaml_string(spec['execution_name'])}\n"
                "dataset_kwargs:\n"
                f"  revision: {yaml_string(spec['dataset_revision'])}\n"
            )
            if gpqa_csv is not None:
                if not validate_gpqa_csv(gpqa_csv):
                    raise ValueError("GPQA CSV does not match pinned upstream content")
                content = (
                    f"include: {yaml_string(source)}\n"
                    f"task: {yaml_string(spec['execution_name'])}\n"
                    "dataset_path: csv\n"
                    "dataset_name: gpqa_diamond\n"
                    "dataset_kwargs:\n"
                    "  data_files:\n"
                    f"    train: {yaml_string(gpqa_csv.resolve())}\n"
                )
            (overlay_root / "gpqa_diamond_pinned.yaml").write_text(content, encoding="utf-8")
            execution_tasks.append(spec["execution_name"])
            continue

        if name == "mmlu_pro":
            pinned_subtasks = []
            for subtask in spec["subtasks"]:
                source = (lm_eval_root / f"lm_eval/tasks/mmlu_pro/mmlu_pro_{subtask}.yaml").resolve()
                execution_name = f"qwen38_mmlu_pro_{subtask}_pinned"
                pinned_subtasks.append(execution_name)
                content = (
                    f"include: {yaml_string(source)}\n"
                    f"task: {yaml_string(execution_name)}\n"
                    "dataset_kwargs:\n"
                    f"  revision: {yaml_string(spec['dataset_revision'])}\n"
                )
                (overlay_root / f"mmlu_pro_{subtask}_pinned.yaml").write_text(
                    content,
                    encoding="utf-8",
                )
            group_lines = [
                f"group: {yaml_string(spec['execution_name'])}",
                "task:",
                *(f"  - {yaml_string(task)}" for task in pinned_subtasks),
                "aggregate_metric_list:",
                "  - aggregation: mean",
                "    metric: exact_match",
                "    weight_by_size: true",
                "    filter_list: custom-extract",
                "metadata:",
                f"  version: {yaml_string(spec['group_metadata_version'])}",
                "",
            ]
            (overlay_root / "mmlu_pro_group_pinned.yaml").write_text(
                "\n".join(group_lines),
                encoding="utf-8",
            )
            execution_tasks.append(spec["execution_name"])
            continue

        raise ValueError(f"no overlay builder for task: {name}")
    return overlay_root, execution_tasks


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    require_valid(validate_static(protocol) + validate_lm_eval(protocol, args.lm_eval_root))
    arm = protocol["arms"][args.arm]
    source_revision, source_dirty = source_state(REPO_ROOT)

    if not args.dry_run:
        if source_revision is None or source_dirty:
            raise SystemExit("scored runs require a clean Git checkout with a recorded revision")
        marker = args.model_path / protocol["resource_preflight"]["model_ready_marker"]
        manifest = args.model_path / "SNAPSHOT_MANIFEST.json"
        if not marker.is_file() or not manifest.is_file():
            raise SystemExit(f"model is not receipt-complete: {args.model_path}")

    args.out.mkdir(parents=True, exist_ok=True)
    include_path, execution_tasks = prepare_task_overlays(
        protocol,
        lm_eval_root=args.lm_eval_root,
        out=args.out,
        tasks=args.tasks,
        gpqa_csv=args.gpqa_csv,
    )
    command = build_command(
        protocol,
        model_path=args.model_path,
        out=args.out,
        tasks=args.tasks,
        include_path=include_path,
        execution_tasks=execution_tasks,
        limit=args.limit,
    )
    receipt = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(args.protocol),
        "arm": args.arm,
        "model_id": arm["model_id"],
        "model_revision": arm["revision"],
        "model_path": str(args.model_path),
        "lm_eval_commit": protocol["source_locks"]["lm_eval"]["commit"],
        "evaluation_source_revision": source_revision,
        "evaluation_source_dirty": source_dirty,
        "tasks": args.tasks,
        "execution_tasks": execution_tasks,
        "dataset_revisions": {
            task["name"]: task["dataset_revision"]
            for task in protocol["quality"]["standard_lane"]["tasks"]
            if task["name"] in args.tasks
        },
        "limit": args.limit,
        "dry_run": args.dry_run,
        "command": command,
    }
    (args.out / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if args.dry_run:
        return 0

    integrity = validate_task_loading(include_path, execution_tasks)
    (args.out / "task_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(args.lm_eval_root) + (os.pathsep + existing if existing else "")
    log_path = args.out / "lm_eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=args.lm_eval_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise SystemExit(f"lm-eval failed with exit code {result.returncode}; see {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
