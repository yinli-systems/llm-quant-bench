#!/usr/bin/env python3
"""Read-only, aggregate-only audit of a completed pilot; never re-score answers."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    return []


def read_rows(path):
    # Physical JSONL lines only: splitlines() also splits Unicode inside strings.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit(run):
    ev = run / "eval"
    receipt = json.loads((ev / "run_receipt.json").read_text())
    integrity = json.loads((ev / "task_integrity.json").read_text())
    assert receipt["limit"] == 0.1 and receipt["arm"] == "baseline"
    assert receipt["protocol_sha256"] == digest(run / "protocol.json")
    assert receipt["evaluation_source_dirty"] is False
    assert integrity["status"] == "passed"
    telemetry = read_rows(ev / "generation_telemetry.jsonl")
    completion = json.loads((ev / "completion_check.json").read_text())
    computed = {
        "requests": len(telemetry),
        "length_limited": sum(r["finish_reason"] == "length" for r in telemetry),
        "unclosed_thinking": sum(not r["thinking_closed"] for r in telemetry),
        "output_tokens": sum(r["output_tokens"] for r in telemetry),
    }
    assert completion == computed
    assert len({r["request_id"] for r in telemetry}) == 1229 == len(telemetry)
    result_files = list(ev.rglob("results_*.json"))
    assert len(result_files) == 1
    results = json.loads(result_files[0].read_text())
    tasks, hashes = {}, []
    for path in sorted(ev.rglob("samples_*.jsonl")):
        task = path.name.split("_2026-")[0].removeprefix("samples_")
        primary = "flexible-extract" if "gpqa" in task else "custom-extract"
        all_rows = read_rows(path)
        rows = [r for r in all_rows if r["filter"] == primary]
        expected = math.ceil(integrity["task_rows"][task] * 0.1)
        assert len(rows) == expected == len({r["doc_id"] for r in rows})
        correct = sum(r["exact_match"] for r in rows)
        assert all(r["exact_match"] in (0, 1) for r in rows)
        assert abs(correct / len(rows) - results["results"][task]["exact_match," + primary]) < 1e-12
        invalid, boxed_invalid, empty, marker = 0, 0, 0, 0
        for row in rows:
            raw = "\n".join(strings(row["resps"]))
            parsed = strings(row["filtered_resps"])
            bad = not parsed or any(not s.strip() or s == "[invalid]" for s in parsed)
            invalid += bad
            boxed_invalid += bad and "\\boxed" in raw
            empty += not raw.strip()
            marker += "</think>" in raw
            item = {k: row[k] for k in ["doc_id", "doc_hash", "target_hash", "prompt_hash"]}
            assert all(isinstance(item[k], str) and len(item[k]) == 64 for k in ["doc_hash", "target_hash", "prompt_hash"])
            hashes.append({"task": task, **item})
        tasks[task] = {
            "primary_filter": primary, "count": len(rows), "correct": int(correct),
            "score": correct / len(rows), "invalid": invalid,
            "invalid_rate": invalid / len(rows), "invalid_with_boxed_text": boxed_invalid,
            "empty_saved_response": empty, "saved_response_with_think_end": marker,
            "all_filter_rows": len(all_rows),
        }
    assert len(tasks) == 15 and sum(t["count"] for t in tasks.values()) == 1229
    families = {}
    for family in ["gpqa", "mmlu_pro"]:
        selected = [t for k, t in tasks.items() if family in k]
        total = {k: sum(t[k] for t in selected) for k in ["count", "correct", "invalid", "invalid_with_boxed_text", "empty_saved_response"]}
        total.update(score=total["correct"] / total["count"], invalid_rate=total["invalid"] / total["count"])
        families[family] = total
    candidate = run.parents[2] / "candidate" / "pilot" / run.name
    assert not candidate.exists()
    log = (ev / "lm_eval.log").read_text()
    return {
        "status": "FAILED_BASELINE_COMPLETENESS_CANDIDATE_NOT_RUN",
        "job_id": 1561180, "run_dir": str(run), "receipt": receipt,
        "completion": completion, "tasks": tasks, "families": families,
        "generation_seconds": sum(r["batch_elapsed_seconds"] for r in telemetry),
        "length_and_missing_marker_overlap": sum(r["finish_reason"] == "length" and not r["thinking_closed"] for r in telemetry),
        "anomalies": [r for r in telemetry if r["finish_reason"] != "stop" or not r["thinking_closed"]],
        "paired_sample_hash_status": "NOT_ASSESSABLE_CANDIDATE_ABSENT",
        "sample_provenance_digest": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "gpu_start_rows": (run / "gpus-start.csv").read_text().splitlines(),
        "gpu_final_available": (run / "gpus-final.csv").exists(),
        "original_sha256sums_available": (run / "SHA256SUMS").exists(),
        "fatal_signatures": {s: log.count(s) for s in ["Traceback (most recent call last)", "CUDA out of memory", "Ninja build failed", "Engine core initialization failed"]},
        "artifact_sha256": {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob("*")) if p.is_file()},
        "claim_boundary": "Failed pilot diagnostic only; original scores unchanged; no retention, official score, or speedup claim. Hash inventory is post-run, not the original success receipt.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2, sort_keys=True))
