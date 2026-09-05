#!/usr/bin/env python3
"""Summarize repeated benchmark runs with confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

METRICS: dict[str, tuple[str, ...]] = {
    "success_rate": ("requests", "success_rate"),
    "request_throughput": ("requests", "request_throughput"),
    "output_token_throughput": ("tokens", "output_token_throughput"),
    "p95_latency_s": ("latency", "p95_s"),
    "p95_ttft_s": ("ttft", "p95_s"),
    "p95_itl_s": ("inter_token_latency", "p95_s"),
    "p95_tpot_s": ("time_per_output_token", "p95_s"),
    "goodput_compliance_rate": ("goodput", "compliance_rate"),
    "request_goodput": ("goodput", "request_goodput"),
    "output_token_goodput": ("goodput", "output_token_goodput"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--group-by", default="concurrency")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    output = summarize_repeats(summaries, group_by=args.group_by)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    report_path = args.out.with_suffix(".md")
    report_path.write_text(render_report(output), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


def summarize_repeats(rows: list[dict[str, Any]], *, group_by: str = "concurrency") -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_by, "all")), []).append(row)
    return {
        "group_by": group_by,
        "groups": {
            group: summarize_group(group_rows)
            for group, group_rows in sorted(groups.items(), key=lambda item: item[0])
        },
        "confidence_interval": "two-sided 95% CI over run-level summaries using Student t critical values",
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"runs": len(rows), "metrics": {}}
    for name, path in METRICS.items():
        values = [value for row in rows if (value := nested_float(row, path)) is not None]
        if values:
            output["metrics"][name] = describe(values)
    return output


def nested_float(row: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        if value is None:
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def describe(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values)
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "stddev": 0.0,
            "ci95_half_width": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "values": values,
        }
    stddev = statistics.stdev(values)
    half_width = t_critical_95(n - 1) * stddev / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "stddev": stddev,
        "ci95_half_width": half_width,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": values,
    }


def t_critical_95(degrees_of_freedom: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom < 25:
        return table[20]
    if degrees_of_freedom < 30:
        return table[25]
    if degrees_of_freedom < 40:
        return table[30]
    if degrees_of_freedom < 60:
        return table[40]
    if degrees_of_freedom < 120:
        return table[60]
    return 1.96


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Repeated Run Confidence Intervals",
        "",
        f"- Group by: {summary['group_by']}",
        f"- CI: {summary['confidence_interval']}",
        "",
        "| Group | Runs | Metric | Mean | 95% CI | Stddev |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group, group_row in summary["groups"].items():
        for metric, row in group_row["metrics"].items():
            lines.append(
                f"| {group} | {group_row['runs']} | {metric} | "
                f"{row['mean']:.4f} | +/- {row['ci95_half_width']:.4f} | {row['stddev']:.4f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
