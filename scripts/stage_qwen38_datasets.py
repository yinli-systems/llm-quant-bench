#!/usr/bin/env python3
"""Warm exact dataset revisions and create non-raw, content-hashed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_qwen38_pareto_protocol import (  # noqa: E402
    load_protocol,
    sha256,
    validate_dataset_receipt,
    validate_static,
    validate_gpqa_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--gpqa-csv", type=Path, help="Reuse an existing copy of the exact pinned GPQA CSV")
    return parser.parse_args()


def relative_cache_file(path: Path, root: Path) -> dict[str, Any]:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"dataset cache escaped runtime root: {resolved_path}") from exc
    return {
        "path": str(relative),
        "bytes": resolved_path.stat().st_size,
        "sha256": sha256(resolved_path),
    }


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if any(not check["ok"] for check in validate_static(protocol)):
        raise SystemExit("static protocol validation failed")
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    preflight = protocol["resource_preflight"]
    ready, detail = validate_dataset_receipt(
        args.runtime_root,
        protocol,
        verify_content=True,
    )
    if ready:
        print(json.dumps({"status": "already_ready", "detail": detail}))
        return 0
    marker_path = args.runtime_root / preflight["dataset_ready_marker_relative_path"]
    if marker_path.exists():
        raise SystemExit(f"refusing to replace an invalid ready dataset cache: {detail}")

    cache_dir = args.runtime_root / preflight["dataset_cache_relative_path"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.runtime_root / "hf-cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    try:
        import datasets
        import huggingface_hub.constants as hub_constants
    except ImportError as exc:
        raise SystemExit("datasets and huggingface_hub are required for dataset staging") from exc
    receipts = []
    load_specs = []
    gpqa_csv = None
    if args.gpqa_csv:
        if not validate_gpqa_csv(args.gpqa_csv):
            raise SystemExit("GPQA CSV does not match the pinned upstream Git blob and SHA-256")
        gpqa_csv = args.runtime_root / "data/gpqa_diamond.csv"
        gpqa_csv.parent.mkdir(parents=True, exist_ok=True)
        if gpqa_csv.exists() and not validate_gpqa_csv(gpqa_csv):
            raise SystemExit("refusing to overwrite a different local GPQA CSV")
        if args.gpqa_csv.resolve() != gpqa_csv.resolve():
            shutil.copyfile(args.gpqa_csv, gpqa_csv)
    for task in protocol["quality"]["standard_lane"]["tasks"]:
        kwargs: dict[str, Any] = {
            "path": task["dataset_id"],
            "revision": task["dataset_revision"],
            "cache_dir": str(cache_dir),
        }
        if task.get("dataset_config"):
            kwargs["name"] = task["dataset_config"]
        if task["name"] == "gpqa_diamond_cot_zeroshot" and gpqa_csv:
            kwargs = {
                "path": "csv", "name": "gpqa_diamond",
                "data_files": {"train": str(gpqa_csv.resolve())},
                "cache_dir": str(cache_dir),
            }
        load_specs.append((task, kwargs))
        try:
            dataset = datasets.load_dataset(**kwargs)
        except Exception as exc:
            raise SystemExit(
                f"failed to stage {task['dataset_id']} at {task['dataset_revision']}; "
                "for gated GPQA, accept the dataset terms and provide HF_TOKEN in the environment"
            ) from exc

        split_receipts = {}
        actual_counts = {split: len(value) for split, value in dataset.items()}
        if actual_counts != task["expected_split_rows"]:
            raise SystemExit(
                f"unexpected split counts for {task['dataset_id']}: {actual_counts}; "
                f"expected {task['expected_split_rows']}"
            )
        for split, value in dataset.items():
            cache_files = [
                relative_cache_file(Path(item["filename"]), args.runtime_root)
                for item in value.cache_files
            ]
            split_receipts[split] = {
                "rows": len(value),
                "fingerprint": value._fingerprint,
                "cache_files": cache_files,
            }
        receipts.append(
            {
                "name": task["name"],
                "dataset_id": task["dataset_id"],
                "revision": task["dataset_revision"],
                "config": task.get("dataset_config"),
                "splits": split_receipts,
            }
        )

    datasets.config.HF_DATASETS_OFFLINE = True
    hub_constants.HF_HUB_OFFLINE = True
    for task, kwargs in load_specs:
        try:
            offline_dataset = datasets.load_dataset(**kwargs)
        except Exception as exc:
            raise SystemExit(
                f"offline reload failed for {task['dataset_id']} at {task['dataset_revision']}"
            ) from exc
        offline_counts = {split: len(value) for split, value in offline_dataset.items()}
        if offline_counts != task["expected_split_rows"]:
            raise SystemExit(f"offline split counts changed for {task['dataset_id']}: {offline_counts}")

    receipt = {
        "protocol_version": protocol["protocol_version"],
        "offline_reload_verified": True,
        "datasets": receipts,
    }
    if gpqa_csv:
        receipt["local_gpqa_csv"] = str(gpqa_csv.relative_to(args.runtime_root))
    receipt_path = args.runtime_root / preflight["dataset_receipt_relative_path"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_bytes = (json.dumps(receipt, indent=2) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    marker_path.write_text(hashlib.sha256(receipt_bytes).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
