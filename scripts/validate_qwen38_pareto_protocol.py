#!/usr/bin/env python3
"""Validate the frozen Qwen3.8 BF16/FP8 protocol and runtime prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_VERSIONS = {
    "qwen38-27b-bf16-fp8-pareto-paracloud-v1",
    "qwen38-27b-bf16-fp8-pareto-paracloud-v2",
}
EXPECTED_HARNESS_COMMIT = "b954108c9baaaa934b4ad842033b31a97ee30816"
EXPECTED_VLLM_VERSION = "0.23.0"
EXPECTED_TASKS = {
    "gpqa_diamond_cot_zeroshot": "2.2",
    "mmlu_pro": "3.1",
}
EXPECTED_DATASETS = {
    "gpqa_diamond_cot_zeroshot": (
        "Idavidrein/gpqa",
        "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
    ),
    "mmlu_pro": (
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    ),
}
MINIMUM_SAFE_MODEL_STAGING_DISK_GIB = 128
MINIMUM_SAFE_RUNTIME_STAGING_DISK_GIB = 48
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--lm-eval-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--check-gpus", action="store_true")
    parser.add_argument("--verify-model-content", action="store_true")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail})


def load_protocol(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_static(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "protocol version",
        protocol.get("protocol_version") in EXPECTED_PROTOCOL_VERSIONS,
        str(protocol.get("protocol_version")),
    )
    add_check(checks, "protocol frozen", protocol.get("frozen") is True, str(protocol.get("frozen")))

    source_locks = protocol.get("source_locks") or {}
    harness_commit = (source_locks.get("lm_eval") or {}).get("commit")
    add_check(
        checks,
        "lm-eval commit lock",
        harness_commit == EXPECTED_HARNESS_COMMIT,
        str(harness_commit),
    )
    vllm_version = (source_locks.get("vllm") or {}).get("version")
    add_check(
        checks,
        "vLLM version lock",
        vllm_version == EXPECTED_VLLM_VERSION,
        str(vllm_version),
    )

    arms = protocol.get("arms") or {}
    baseline = arms.get("baseline") or {}
    candidate = arms.get("candidate") or {}
    for arm_name, arm in (("baseline", baseline), ("candidate", candidate)):
        revision = str(arm.get("revision") or "")
        add_check(
            checks,
            f"{arm_name} immutable model revision",
            bool(HEX40_RE.fullmatch(revision)),
            revision or "missing",
        )
        add_check(
            checks,
            f"{arm_name} runtime path",
            bool(arm.get("runtime_relative_path")),
            str(arm.get("runtime_relative_path")),
        )
    add_check(
        checks,
        "distinct checkpoints",
        baseline.get("model_id") != candidate.get("model_id"),
        f"{baseline.get('model_id')} vs {candidate.get('model_id')}",
    )

    topology = protocol.get("matched_topology") or {}
    expected_gpu_count = topology.get("nodes", 0) * topology.get("gpus_per_node", 0)
    parallel_gpu_count = topology.get("tensor_parallel_size", 0) * topology.get(
        "pipeline_parallel_size", 0
    )
    add_check(
        checks,
        "parallel topology consumes allocation",
        expected_gpu_count == parallel_gpu_count == 4,
        f"allocated={expected_gpu_count}, parallel={parallel_gpu_count}",
    )
    add_check(
        checks,
        "Ada FP8 hardware target",
        topology.get("compute_capability") == "8.9" and "4090" in str(topology.get("gpu_model")),
        f"{topology.get('gpu_model')} sm={topology.get('compute_capability')}",
    )

    quality = protocol.get("quality") or {}
    add_check(
        checks,
        "reasoning contract",
        quality.get("enable_thinking") is True
        and quality.get("reasoning_effort") == "xhigh"
        and quality.get("think_end_token") == "</think>"
        and quality.get("temperature") == 0.0,
        "thinking=true, effort=xhigh, delimiter=</think>, temperature=0",
    )
    tasks = {
        task.get("name"): str(task.get("task_metadata_version"))
        for task in ((quality.get("standard_lane") or {}).get("tasks") or [])
    }
    add_check(checks, "task/version locks", tasks == EXPECTED_TASKS, json.dumps(tasks, sort_keys=True))
    datasets = {
        task.get("name"): (task.get("dataset_id"), task.get("dataset_revision"))
        for task in ((quality.get("standard_lane") or {}).get("tasks") or [])
    }
    add_check(
        checks,
        "dataset revision locks",
        datasets == EXPECTED_DATASETS
        and all(HEX40_RE.fullmatch(str(revision or "")) for _, revision in datasets.values()),
        json.dumps(datasets, sort_keys=True),
    )
    robustness = quality.get("robustness_lane") or {}
    add_check(
        checks,
        "gold-blind permutation lane",
        robustness.get("permutations_per_question") == 4
        and "gold-blind" in str(robustness.get("permutation_policy"))
        and "ties abstain" in str(robustness.get("aggregation")),
        f"permutations={robustness.get('permutations_per_question')}; {robustness.get('aggregation')}",
    )

    gates = protocol.get("gates") or {}
    gates_ok = (
        gates.get("zero_failed_requests") is True
        and gates.get("minimum_quality_retention", 0) >= 0.98
        and gates.get("maximum_gpqa_absolute_drop_percentage_points", 99) <= 2.0
        and gates.get("maximum_mmlu_pro_absolute_drop_percentage_points", 99) <= 1.0
        and gates.get("minimum_peak_gpu_memory_reduction_fraction", 0) >= 0.35
        and gates.get("minimum_output_throughput_increase_fraction", 0) >= 0.2
    )
    add_check(checks, "pre-registered acceptance gates", gates_ok, json.dumps(gates, sort_keys=True))

    preflight = protocol.get("resource_preflight") or {}
    model_staging_disk = preflight.get(
        "minimum_free_disk_gib_before_model_staging",
        preflight.get("minimum_free_disk_gib_before_staging", 0),
    )
    runtime_staging_disk = preflight.get(
        "minimum_free_disk_gib_before_runtime_staging",
        preflight.get("minimum_free_disk_gib_before_staging", 0),
    )
    execution_disk = preflight.get("minimum_free_disk_gib_for_execution", 0)
    add_check(
        checks,
        "safe disk floor",
        model_staging_disk >= MINIMUM_SAFE_MODEL_STAGING_DISK_GIB
        and runtime_staging_disk >= MINIMUM_SAFE_RUNTIME_STAGING_DISK_GIB
        and execution_disk >= 32,
        (
            f"model staging={model_staging_disk} GiB, "
            f"runtime staging={runtime_staging_disk} GiB, execution={execution_disk} GiB"
        ),
    )
    add_check(
        checks,
        "quality gates performance",
        (protocol.get("performance") or {}).get("run_only_after_quality_gate_passes") is True,
        "performance is skipped unless quality passes",
    )
    return checks


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def metadata_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = re.search(
        r"(?m)^metadata:\s*\n(?:^[ \t]+.*\n)*?^[ \t]+version:\s*['\"]?([^'\"\s#]+)",
        path.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


def validate_lm_eval(protocol: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    expected_head = protocol["source_locks"]["lm_eval"]["commit"]
    actual_head = git_head(root)
    add_check(checks, "lm-eval checkout HEAD", actual_head == expected_head, actual_head or "not a git checkout")
    tasks = protocol["quality"]["standard_lane"]["tasks"]
    for task in tasks:
        config = root / task["config_relative_path"]
        actual = metadata_version(config)
        expected = str(task["task_metadata_version"])
        add_check(checks, f"{task['name']} metadata", actual == expected, f"{actual or 'missing'} at {config}")
        group_path = task.get("group_config_relative_path")
        if group_path:
            actual_group = metadata_version(root / group_path)
            expected_group = str(task["group_metadata_version"])
            add_check(
                checks,
                f"{task['name']} group metadata",
                actual_group == expected_group,
                f"{actual_group or 'missing'} at {root / group_path}",
            )
    return checks


def python_package_version(python: Path, package: str) -> str | None:
    code = (
        "import importlib.metadata as m; "
        f"print(m.version({package!r}))"
    )
    result = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_receipt(
    model_path: Path,
    arm: dict[str, Any],
    *,
    marker_name: str,
    require_hashes: bool,
    verify_content: bool = False,
) -> tuple[bool, str]:
    marker = model_path / marker_name
    manifest_path = model_path / "SNAPSHOT_MANIFEST.json"
    if not marker.is_file() or not manifest_path.is_file():
        return False, f"missing {marker} or {manifest_path}"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    if manifest.get("repo") != arm.get("model_id"):
        return False, f"repo mismatch: {manifest.get('repo')}"
    if manifest.get("revision") != arm.get("revision"):
        return False, f"revision mismatch: {manifest.get('revision')}"

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False, "manifest has no files"
    declared_total = 0
    for item in files:
        if not isinstance(item, dict):
            return False, "non-object file receipt"
        relative = Path(str(item.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            return False, f"unsafe file path: {relative}"
        if not isinstance(item.get("bytes"), int) or item["bytes"] < 0:
            return False, f"invalid byte count: {relative}"
        digest = str(item.get("sha256") or "")
        if require_hashes and not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False, f"missing SHA-256: {relative}"
        actual_path = model_path / relative
        if not actual_path.is_file() or actual_path.stat().st_size != item["bytes"]:
            return False, f"missing or size-mismatched file: {relative}"
        if verify_content and sha256(actual_path) != digest:
            return False, f"content hash mismatch: {relative}"
        declared_total += item["bytes"]
    if manifest.get("file_count") != len(files) or manifest.get("total_bytes") != declared_total:
        return False, "manifest counts do not reconcile"
    tree_payload = json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    if manifest.get("tree_sha256") != hashlib.sha256(tree_payload).hexdigest():
        return False, "tree receipt mismatch"
    expected_marker = hashlib.sha256(manifest_bytes).hexdigest()
    if marker.read_text(encoding="utf-8").strip() != expected_marker:
        return False, "ready marker does not bind the manifest"
    return True, f"{len(files)} files, {declared_total} bytes, revision {arm['revision']}"


def validate_dataset_receipt(
    root: Path,
    protocol: dict[str, Any],
    *,
    verify_content: bool = False,
) -> tuple[bool, str]:
    preflight = protocol["resource_preflight"]
    receipt_path = root / preflight["dataset_receipt_relative_path"]
    marker_path = root / preflight["dataset_ready_marker_relative_path"]
    if not receipt_path.is_file() or not marker_path.is_file():
        return False, f"missing {receipt_path} or {marker_path}"
    try:
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"invalid dataset receipt: {exc}"
    if marker_path.read_text(encoding="utf-8").strip() != hashlib.sha256(receipt_bytes).hexdigest():
        return False, "dataset ready marker does not bind the receipt"
    if receipt.get("protocol_version") != protocol["protocol_version"]:
        return False, "dataset receipt protocol mismatch"
    if receipt.get("offline_reload_verified") is not True:
        return False, "dataset cache was not verified offline"

    expected = {
        task["name"]: task for task in protocol["quality"]["standard_lane"]["tasks"]
    }
    rows = receipt.get("datasets")
    if not isinstance(rows, list) or {row.get("name") for row in rows} != set(expected):
        return False, "dataset task coverage mismatch"
    verified_files = 0
    for row in rows:
        spec = expected[row["name"]]
        if row.get("dataset_id") != spec["dataset_id"]:
            return False, f"dataset id mismatch: {row['name']}"
        if row.get("revision") != spec["dataset_revision"]:
            return False, f"dataset revision mismatch: {row['name']}"
        if row.get("config") != spec.get("dataset_config"):
            return False, f"dataset config mismatch: {row['name']}"
        actual_splits = {
            split: info.get("rows") for split, info in (row.get("splits") or {}).items()
        }
        if actual_splits != spec["expected_split_rows"]:
            return False, f"dataset split rows mismatch: {row['name']} {actual_splits}"
        for info in (row.get("splits") or {}).values():
            if not info.get("fingerprint"):
                return False, f"missing dataset fingerprint: {row['name']}"
            for item in info.get("cache_files") or []:
                relative = Path(str(item.get("path") or ""))
                digest = str(item.get("sha256") or "")
                if relative.is_absolute() or ".." in relative.parts:
                    return False, f"unsafe dataset cache path: {relative}"
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    return False, f"missing dataset cache SHA-256: {relative}"
                actual = root / relative
                if not actual.is_file() or actual.stat().st_size != item.get("bytes"):
                    return False, f"missing or size-mismatched dataset cache: {relative}"
                if verify_content and sha256(actual) != digest:
                    return False, f"dataset content hash mismatch: {relative}"
                verified_files += 1
    if verified_files == 0:
        return False, "dataset receipt contains no cache files"
    return True, f"{len(rows)} datasets, {verified_files} cache files"


def validate_runtime(
    protocol: dict[str, Any],
    root: Path,
    *,
    verify_model_content: bool = False,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    preflight = protocol["resource_preflight"]
    if not root.exists():
        add_check(checks, "runtime root", False, f"missing: {root}")
        return checks

    python = root / preflight["runtime_python_relative_path"]
    add_check(checks, "runtime Python", python.is_file(), str(python))
    if python.is_file():
        actual_vllm = python_package_version(python, "vllm")
        expected_vllm = protocol["source_locks"]["vllm"]["version"]
        add_check(checks, "runtime vLLM", actual_vllm == expected_vllm, actual_vllm or "not installed")

    model_receipts_complete = True
    for arm_name, arm in protocol["arms"].items():
        model_path = root / arm["runtime_relative_path"]
        marker = model_path / preflight["model_ready_marker"]
        manifest = model_path / "SNAPSHOT_MANIFEST.json"
        receipt_ok, receipt_detail = validate_model_receipt(
            model_path,
            arm,
            marker_name=preflight["model_ready_marker"],
            require_hashes=preflight.get("require_per_file_sha256", False),
            verify_content=verify_model_content,
        )
        model_receipts_complete = model_receipts_complete and receipt_ok
        add_check(checks, f"{arm_name} model ready marker", marker.is_file(), str(marker))
        add_check(
            checks,
            f"{arm_name} snapshot manifest",
            (not preflight["require_snapshot_manifest"]) or manifest.is_file(),
            str(manifest),
        )
        add_check(checks, f"{arm_name} immutable model receipt", receipt_ok, receipt_detail)
    dataset_ok, dataset_detail = validate_dataset_receipt(
        root,
        protocol,
        verify_content=verify_model_content,
    )
    add_check(checks, "immutable dataset receipt", dataset_ok, dataset_detail)
    model_receipts_complete = model_receipts_complete and dataset_ok
    free_gib = shutil.disk_usage(root).free / (1024**3)
    disk_gate = (
        "minimum_free_disk_gib_for_execution"
        if model_receipts_complete
        else "minimum_free_disk_gib_before_model_staging"
    )
    minimum = float(
        preflight.get(disk_gate, preflight["minimum_free_disk_gib_before_staging"])
    )
    add_check(
        checks,
        "runtime free disk",
        free_gib >= minimum,
        f"{free_gib:.1f} GiB free; need {minimum:.1f} GiB ({disk_gate})",
    )
    return checks


def query_gpus() -> tuple[list[dict[str, str]], str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return [], result.stderr.strip() or "nvidia-smi failed"
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 4:
            rows.append(dict(zip(("name", "uuid", "compute_capability", "memory_mib"), values)))
    return rows, None


def validate_gpus(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    rows, error = query_gpus()
    topology = protocol["matched_topology"]
    expected_count = topology["nodes"] * topology["gpus_per_node"]
    add_check(
        checks,
        "allocated GPU count",
        error is None and len(rows) == expected_count,
        error or f"found {len(rows)}",
    )
    expected_model = topology["gpu_model"]
    expected_capability = topology["compute_capability"]
    add_check(
        checks,
        "allocated GPU model/capability",
        bool(rows)
        and all(row["name"] == expected_model and row["compute_capability"] == expected_capability for row in rows),
        json.dumps(rows, sort_keys=True),
    )
    return checks


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    checks = validate_static(protocol)
    checked = ["protocol"]
    if args.lm_eval_root:
        checks.extend(validate_lm_eval(protocol, args.lm_eval_root))
        checked.append("lm-eval")
    if args.runtime_root:
        checks.extend(
            validate_runtime(
                protocol,
                args.runtime_root,
                verify_model_content=args.verify_model_content,
            )
        )
        checked.append("runtime")
    if args.check_gpus:
        checks.extend(validate_gpus(protocol))
        checked.append("gpus")

    failures = [check for check in checks if not check["ok"]]
    report = {
        "protocol": str(args.protocol),
        "checked": checked,
        "status": "ready" if not failures else "blocked",
        "passed": len(checks) - len(failures),
        "failed": len(failures),
        "checks": checks,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
