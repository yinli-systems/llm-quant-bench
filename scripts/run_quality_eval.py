#!/usr/bin/env python3
"""Run lightweight quality evaluations against an OpenAI-compatible chat endpoint.

This runner is intentionally dependency-light. It can run MMLU, CMMLU, GSM8K,
LongBench, and MT-Bench style answer generation from local dataset files/caches.
The metrics are useful for same-prompt baseline-vs-candidate retention checks;
they are not a drop-in replacement for official benchmark leaderboards unless
the same prompt templates and scorers are used for every compared model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import string
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_quant_bench.client import ModelConfig, OpenAIChatClient

CHOICE_RE = re.compile(
    r"(?:^|\b)(?:answer|答案|选项|option)?\s*(?:is|是|:|：)?\s*[\(\[]?\s*([ABCD])\s*[\)\].,，。]?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True)
class EvalItem:
    benchmark: str
    task: str
    item_id: str
    prompt: str
    expected: Any
    metric: str
    max_tokens: int


def item_payload(item: EvalItem) -> dict[str, Any]:
    return {
        "schema": "llm-quant-bench-eval-item-v1",
        "benchmark": item.benchmark,
        "task": item.task,
        "item_id": item.item_id,
        "prompt": item.prompt,
        "expected": item.expected,
        "metric": item.metric,
        "max_tokens": item.max_tokens,
    }


def serialize_items(items: list[EvalItem]) -> bytes:
    lines = [
        json.dumps(item_payload(item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in items
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def load_items_file(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("schema") != "llm-quant-bench-eval-item-v1":
                raise ValueError(f"Unsupported item schema at {path}:{line_number}")
            items.append(
                EvalItem(
                    benchmark=str(payload["benchmark"]),
                    task=str(payload["task"]),
                    item_id=str(payload["item_id"]),
                    prompt=str(payload["prompt"]),
                    expected=payload.get("expected"),
                    metric=str(payload["metric"]),
                    max_tokens=int(payload["max_tokens"]),
                )
            )
    identities = [(item.benchmark, item.task, item.item_id) for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate benchmark/task/item_id entries in {path}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen72b-awq-l20")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--benchmarks", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Per-benchmark sample limit.")
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--mmlu-config", default="all")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--cmmlu-dir", default=None)
    parser.add_argument("--longbench-dir", default=None)
    parser.add_argument("--longbench-tasks", nargs="*", default=None)
    parser.add_argument("--longbench-max-context-chars", type=int, default=24000)
    parser.add_argument("--mt-bench-file", default=None)
    parser.add_argument("--mt-bench-max-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-max-tokens", type=int, default=256)
    parser.add_argument("--longbench-max-tokens", type=int, default=128)
    parser.add_argument("--sample-log-every", type=int, default=100)
    parser.add_argument(
        "--items-file",
        default=None,
        help="Read an exact, previously frozen EvalItem JSONL bundle instead of loading datasets.",
    )
    parser.add_argument(
        "--write-items-file",
        default=None,
        help="Write the exact post-filter EvalItem JSONL bundle used by this run.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and hash the item bundle without calling a model endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.items_file:
        items = load_items_file(Path(args.items_file))
        requested = {name.lower() for name in args.benchmarks}
        items = [item for item in items if item.benchmark.lower() in requested]
    else:
        items = []
        for benchmark in args.benchmarks:
            name = benchmark.lower()
            if name == "mmlu":
                items.extend(load_mmlu(args))
            elif name == "cmmlu":
                items.extend(load_cmmlu(args))
            elif name == "gsm8k":
                items.extend(load_gsm8k(args))
            elif name == "longbench":
                items.extend(load_longbench(args))
            elif name in {"mt-bench", "mtbench"}:
                items.extend(load_mt_bench(args))
            else:
                raise SystemExit(f"Unsupported benchmark: {benchmark}")

    if args.max_per_task is not None:
        items = cap_per_key(items, "task", args.max_per_task)

    if args.limit is not None:
        limited: list[EvalItem] = []
        seen: dict[str, int] = {}
        for item in items:
            count = seen.get(item.benchmark, 0)
            if count < args.limit:
                limited.append(item)
                seen[item.benchmark] = count + 1
        items = limited

    item_bytes = serialize_items(items)
    items_sha256 = hashlib.sha256(item_bytes).hexdigest()
    if args.write_items_file:
        frozen_path = Path(args.write_items_file)
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_bytes(item_bytes)
    if args.prepare_only and not args.write_items_file:
        raise SystemExit("--prepare-only requires --write-items-file")

    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "benchmarks": args.benchmarks,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "max_per_task": args.max_per_task,
        "temperature": args.temperature,
        "created_at_unix": time.time(),
        "num_items": len(items),
        "items_sha256": items_sha256,
        "items_file": args.items_file,
        "frozen_items_output": args.write_items_file,
        "item_schema": "llm-quant-bench-eval-item-v1",
        "prepare_only": args.prepare_only,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return

    config = ModelConfig(
        name="candidate",
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=1,
        timeout_s=args.timeout_s,
        extra_body={"ignore_eos": False},
    )

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    samples_path = out_dir / "samples.jsonl"
    with (
        samples_path.open("w", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        futures = [pool.submit(run_one, item, config) for item in items]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if args.sample_log_every and idx % args.sample_log_every == 0:
                print(f"completed {idx}/{len(items)}", flush=True)

    summary = summarize(results, time.perf_counter() - started)
    summary["items_sha256"] = items_sha256
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_one(item: EvalItem, base_config: ModelConfig) -> dict[str, Any]:
    config = ModelConfig(
        name=base_config.name,
        base_url=base_config.base_url,
        model=base_config.model,
        api_key=base_config.api_key,
        temperature=base_config.temperature,
        max_tokens=item.max_tokens,
        timeout_s=base_config.timeout_s,
        extra_body=base_config.extra_body,
    )
    client = OpenAIChatClient(config)
    result = client.generate(item.prompt, stream=False)
    score, prediction = score_response(item, result.text)
    return {
        "benchmark": item.benchmark,
        "task": item.task,
        "item_id": item.item_id,
        "ok": result.ok,
        "metric": item.metric,
        "score": score if result.ok else 0.0,
        "prediction": prediction,
        "expected": item.expected,
        "latency_s": result.latency_s,
        "output_tokens": result.output_tokens,
        "prompt_tokens": result.prompt_tokens,
        "error": result.error,
        "response": result.text,
    }


def load_mmlu(args: argparse.Namespace) -> list[EvalItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("MMLU requires the datasets package.") from exc
    dataset = load_dataset("cais/mmlu", args.mmlu_config, split=args.mmlu_split)
    items: list[EvalItem] = []
    for idx, row in enumerate(dataset):
        choices = list(row["choices"])
        answer = "ABCD"[int(row["answer"])]
        prompt = format_mcq_prompt(row["question"], choices, language="en")
        items.append(
            EvalItem(
                benchmark="mmlu",
                task=str(row.get("subject") or args.mmlu_config),
                item_id=f"mmlu-{idx}",
                prompt=prompt,
                expected=answer,
                metric="choice_accuracy",
                max_tokens=4,
            )
        )
    return items


def load_cmmlu(args: argparse.Namespace) -> list[EvalItem]:
    if not args.cmmlu_dir:
        raise SystemExit("CMMLU requires --cmmlu-dir pointing at CMMLU test CSV files.")
    root = Path(args.cmmlu_dir)
    items: list[EvalItem] = []
    for path in sorted(root.glob("*.csv")):
        if path.name.startswith("."):
            continue
        task = path.stem
        with path.open(newline="", encoding="utf-8") as handle:
            for row_idx, row in enumerate(csv.DictReader(handle)):
                answer = str(row.get("Answer") or "").strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                choices = [row["A"], row["B"], row["C"], row["D"]]
                prompt = format_mcq_prompt(row["Question"], choices, language="zh")
                items.append(
                    EvalItem(
                        benchmark="cmmlu",
                        task=task,
                        item_id=f"{task}-{row_idx}",
                        prompt=prompt,
                        expected=answer,
                        metric="choice_accuracy",
                        max_tokens=4,
                    )
                )
    return items


def load_gsm8k(args: argparse.Namespace) -> list[EvalItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("GSM8K requires the datasets package.") from exc
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    items: list[EvalItem] = []
    for idx, row in enumerate(dataset):
        expected = extract_gsm8k_answer(row["answer"])
        prompt = (
            "Solve the grade-school math problem. Show concise reasoning, then end with "
            "'Final answer: <number>'.\n\n"
            f"Problem: {row['question']}"
        )
        items.append(
            EvalItem(
                benchmark="gsm8k",
                task="main",
                item_id=f"gsm8k-{idx}",
                prompt=prompt,
                expected=expected,
                metric="number_exact_match",
                max_tokens=args.gsm8k_max_tokens,
            )
        )
    return items


def load_longbench(args: argparse.Namespace) -> list[EvalItem]:
    if not args.longbench_dir:
        raise SystemExit("LongBench requires --longbench-dir pointing at JSONL files.")
    root = Path(args.longbench_dir)
    task_names = set(args.longbench_tasks or [])
    paths = sorted(root.glob("*.jsonl"))
    if task_names:
        paths = [path for path in paths if path.stem in task_names]
    items: list[EvalItem] = []
    for path in paths:
        task = path.stem
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                row = json.loads(line)
                context = str(row.get("context") or "")
                if len(context) > args.longbench_max_context_chars:
                    half = args.longbench_max_context_chars // 2
                    context = context[:half] + "\n...\n" + context[-half:]
                question = str(row.get("input") or "")
                language = row.get("language")
                prompt = format_longbench_prompt(task, context, question, language)
                items.append(
                    EvalItem(
                        benchmark="longbench",
                        task=task,
                        item_id=str(row.get("_id") or f"{task}-{idx}"),
                        prompt=prompt,
                        expected=row.get("answers") or [],
                        metric="max_token_f1",
                        max_tokens=args.longbench_max_tokens,
                    )
                )
    return items


def load_mt_bench(args: argparse.Namespace) -> list[EvalItem]:
    if not args.mt_bench_file:
        raise SystemExit("MT-Bench generation requires --mt-bench-file.")
    items: list[EvalItem] = []
    with Path(args.mt_bench_file).open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            turns = row.get("turns") or []
            if not turns:
                continue
            prompt = "\n\n".join(f"User turn {i + 1}: {turn}" for i, turn in enumerate(turns))
            prompt += "\n\nAnswer the user turns in order."
            items.append(
                EvalItem(
                    benchmark="mt-bench",
                    task=str(row.get("category") or "unknown"),
                    item_id=str(row.get("question_id") or idx),
                    prompt=prompt,
                    expected=None,
                    metric="generation_only_requires_external_judge",
                    max_tokens=args.mt_bench_max_tokens,
                )
            )
    return items


def format_mcq_prompt(question: str, choices: list[str], *, language: str) -> str:
    labels = ["A", "B", "C", "D"]
    rendered = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
    if language == "zh":
        return (
            "请回答下面的单项选择题。只输出一个大写字母 A、B、C 或 D，不要解释。\n\n"
            f"题目：{question}\n{rendered}\n答案："
        )
    return (
        "Answer the multiple-choice question. Output only one uppercase letter: A, B, C, or D.\n\n"
        f"Question: {question}\n{rendered}\nAnswer:"
    )


def format_longbench_prompt(task: str, context: str, question: str, language: str | None) -> str:
    if language == "zh":
        return (
            "请根据下面的长上下文回答问题。答案要简洁，直接给出最终答案。\n\n"
            f"任务：{task}\n上下文：\n{context}\n\n问题：{question}\n答案："
        )
    if not question:
        question = "Complete the task implied by the context."
    return (
        "Answer the question using only the long context below. Be concise and give the final answer directly.\n\n"
        f"Task: {task}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def score_response(item: EvalItem, text: str) -> tuple[float | None, Any]:
    if item.metric == "choice_accuracy":
        pred = extract_choice(text)
        return (1.0 if pred == item.expected else 0.0), pred
    if item.metric == "number_exact_match":
        pred = extract_number(text)
        return (1.0 if pred == item.expected else 0.0), pred
    if item.metric == "max_token_f1":
        answers = item.expected if isinstance(item.expected, list) else [item.expected]
        scores = [token_f1(text, str(answer)) for answer in answers if answer is not None]
        return (max(scores) if scores else None), text
    if item.metric == "generation_only_requires_external_judge":
        return None, text
    return None, text


def extract_choice(text: str) -> str | None:
    match = CHOICE_RE.search(text.strip().upper())
    if match:
        return match.group(1).upper()
    for char in text.strip().upper():
        if char in {"A", "B", "C", "D"}:
            return char
    return None


def extract_gsm8k_answer(answer: str) -> str | None:
    marker = "####"
    if marker in answer:
        return normalize_number(answer.split(marker, 1)[1])
    return extract_number(answer)


def extract_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    return normalize_number(matches[-1])


def normalize_number(value: str) -> str | None:
    try:
        number = float(value.strip().replace(",", ""))
    except ValueError:
        return None
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    common: dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in answer_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def summarize(rows: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    for key, group_rows in groupby(rows, "benchmark").items():
        by_benchmark[key] = summarize_group(group_rows)
    for key, group_rows in groupby(rows, "task").items():
        by_task[key] = summarize_group(group_rows)
    return {
        "elapsed_s": elapsed_s,
        "total_items": len(rows),
        "ok_items": sum(1 for row in rows if row["ok"]),
        "failed_items": sum(1 for row in rows if not row["ok"]),
        "by_benchmark": by_benchmark,
        "by_task": by_task,
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("score") is not None and row.get("ok")]
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
    return {
        "items": len(rows),
        "ok": sum(1 for row in rows if row["ok"]),
        "failed": sum(1 for row in rows if not row["ok"]),
        "scored_items": len(scored),
        "score_mean": statistics.fmean(float(row["score"]) for row in scored) if scored else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
    }


def groupby(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return groups


def cap_per_key(items: Iterable[EvalItem], attr: str, limit: int) -> list[EvalItem]:
    capped: list[EvalItem] = []
    seen: dict[str, int] = {}
    for item in items:
        key = str(getattr(item, attr))
        count = seen.get(key, 0)
        if count < limit:
            capped.append(item)
            seen[key] = count + 1
    return capped


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Quality Evaluation Report",
        "",
        f"- Total items: {summary['total_items']}",
        f"- Successful items: {summary['ok_items']}",
        f"- Failed items: {summary['failed_items']}",
        f"- Elapsed seconds: {summary['elapsed_s']:.2f}",
        "",
        "## By Benchmark",
        "",
        "| Benchmark | Items | OK | Failed | Scored | Mean Score | p95 Latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(summary["by_benchmark"].items()):
        score = row["score_mean"]
        p95 = row["latency_p95_s"]
        lines.append(
            f"| {name} | {row['items']} | {row['ok']} | {row['failed']} | {row['scored_items']} | "
            f"{score:.4f} | {p95:.2f}s |"
            if score is not None and p95 is not None
            else f"| {name} | {row['items']} | {row['ok']} | {row['failed']} | {row['scored_items']} | n/a | n/a |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MMLU/CMMLU are zero-shot direct-answer multiple-choice evaluations.",
            "- GSM8K uses exact match on the final extracted number.",
            "- LongBench uses a lightweight max token-F1 scorer and should be used for same-run retention comparisons, not leaderboard claims.",
            "- MT-Bench rows are answer generations only unless scored later by an external judge.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
