#!/usr/bin/env python3
"""Stage exact Qwen3.8 model snapshots and create content-hashed receipts."""

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
    validate_model_receipt,
    validate_static,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    excluded = {".READY", "SNAPSHOT_MANIFEST.json"}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative.name in excluded or ".cache" in relative.parts:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(relative), "bytes": size, "sha256": sha256(path)})
    tree_sha256 = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": tree_sha256,
    }


def nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def write_receipt(model_path: Path, arm: dict[str, Any]) -> dict[str, Any]:
    file_receipt = build_file_manifest(model_path)
    if not file_receipt["files"]:
        raise RuntimeError(f"download produced no files: {model_path}")
    manifest = {
        "name": model_path.name,
        "repo": arm["model_id"],
        "revision": arm["revision"],
        "local_dir": str(model_path),
        **file_receipt,
    }
    manifest_path = model_path / "SNAPSHOT_MANIFEST.json"
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    manifest_path.write_bytes(manifest_bytes)
    (model_path / ".READY").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    failures = [check for check in validate_static(protocol) if not check["ok"]]
    if failures:
        raise SystemExit("static protocol validation failed")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive")

    preflight = protocol["resource_preflight"]
    disk_anchor = nearest_existing_parent(args.runtime_root)
    free_gib = shutil.disk_usage(disk_anchor).free / (1024**3)
    minimum = float(
        preflight.get(
            "minimum_free_disk_gib_before_model_staging",
            preflight["minimum_free_disk_gib_before_staging"],
        )
    )
    if free_gib < minimum:
        raise SystemExit(f"insufficient disk before staging: {free_gib:.1f} GiB free; need {minimum:.1f} GiB")

    args.runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(args.runtime_root / "hf-cache"))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for model staging") from exc
    for arm_name in ("baseline", "candidate"):
        arm = protocol["arms"][arm_name]
        model_path = args.runtime_root / arm["runtime_relative_path"]
        ready, detail = validate_model_receipt(
            model_path,
            arm,
            marker_name=preflight["model_ready_marker"],
            require_hashes=preflight["require_per_file_sha256"],
            verify_content=True,
        )
        if ready:
            print(json.dumps({"arm": arm_name, "status": "already_ready", "detail": detail}), flush=True)
            continue
        if (model_path / preflight["model_ready_marker"]).exists():
            raise SystemExit(f"refusing to replace an invalid ready snapshot: {detail}")
        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=arm["model_id"],
            revision=arm["revision"],
            local_dir=model_path,
            endpoint=args.endpoint,
            max_workers=args.max_workers,
        )
        manifest = write_receipt(model_path, arm)
        print(
            json.dumps(
                {
                    "arm": arm_name,
                    "status": "ready",
                    "files": manifest["file_count"],
                    "bytes": manifest["total_bytes"],
                    "tree_sha256": manifest["tree_sha256"],
                }
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
