from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .client import ModelConfig, OpenAIChatClient
from .metrics import quantile, safe_ratio


def run_load_test(
    *,
    model: ModelConfig,
    prompts: list[str],
    out_dir: Path,
    concurrency: int,
    stream: bool,
    requests: int | None = None,
    duration_s: float | None = None,
    slo_ttft_s: float | None = None,
    slo_tpot_s: float | None = None,
    slo_e2e_s: float | None = None,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if requests is None and duration_s is None:
        requests = 100
    if requests is not None and requests < 1:
        raise ValueError("requests must be >= 1")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    for name, value in (
        ("slo_ttft_s", slo_ttft_s),
        ("slo_tpot_s", slo_tpot_s),
        ("slo_e2e_s", slo_e2e_s),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be > 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "load_results.jsonl"
    report_path = out_dir / "load_report.md"

    lock = threading.Lock()
    next_request_id = 0
    records: list[dict[str, Any]] = []
    started_wall = time.time()
    started = time.monotonic()

    def should_stop() -> bool:
        if duration_s is not None and time.monotonic() - started >= duration_s:
            return True
        return requests is not None and next_request_id >= requests

    def claim_request() -> tuple[int, str] | None:
        nonlocal next_request_id
        with lock:
            if should_stop():
                return None
            request_id = next_request_id
            next_request_id += 1
            prompt = prompts[request_id % len(prompts)]
            return request_id, prompt

    def worker() -> list[dict[str, Any]]:
        client = OpenAIChatClient(model)
        local_records: list[dict[str, Any]] = []
        while True:
            claimed = claim_request()
            if claimed is None:
                return local_records
            request_id, prompt = claimed
            result = client.generate(prompt, stream=stream)
            record = {
                "request_id": request_id,
                "ok": result.ok,
                "error": result.error,
                "latency_s": result.latency_s,
                "ttft_s": result.ttft_s,
                "inter_token_latency_s": result.inter_token_latency_s,
                "inter_chunk_latency_s": result.inter_chunk_latency_s,
                "time_per_output_token_s": result.time_per_output_token_s,
                "output_tokens": result.output_tokens,
                "prompt_tokens": result.prompt_tokens,
                "tokens_per_second": result.tokens_per_second,
                "stream_completed": result.stream_completed,
                "token_count_source": result.token_count_source,
            }
            local_records.append(record)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())

    finished_wall = time.time()
    duration = max(time.monotonic() - started, 1e-9)
    records.sort(key=lambda item: item["request_id"])
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize_load_records(
        records,
        benchmark_duration_s=duration,
        concurrency=concurrency,
        model_name=model.name,
        stream=stream,
        slo_ttft_s=slo_ttft_s,
        slo_tpot_s=slo_tpot_s,
        slo_e2e_s=slo_e2e_s,
    )
    summary["started_at_epoch_s"] = started_wall
    summary["finished_at_epoch_s"] = finished_wall
    (out_dir / "load_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_load_report(summary), encoding="utf-8")
    return summary


def summarize_load_records(
    records: list[dict[str, Any]],
    *,
    benchmark_duration_s: float,
    concurrency: int,
    model_name: str,
    stream: bool,
    slo_ttft_s: float | None = None,
    slo_tpot_s: float | None = None,
    slo_e2e_s: float | None = None,
) -> dict[str, Any]:
    total = len(records)
    ok_records = [record for record in records if record.get("ok")]
    errors = Counter(str(record.get("error")) for record in records if not record.get("ok"))

    latency = _present([record.get("latency_s") for record in ok_records])
    ttft = _present([record.get("ttft_s") for record in ok_records])
    itl = _present([record.get("inter_token_latency_s") for record in ok_records])
    icl = _present([record.get("inter_chunk_latency_s") for record in ok_records])
    tpot = _present([record.get("time_per_output_token_s") for record in ok_records])
    tps = _present([record.get("tokens_per_second") for record in ok_records])
    output_tokens = sum(int(record.get("output_tokens") or 0) for record in ok_records)
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in ok_records)

    summary = {
        "model": model_name,
        "stream": stream,
        "concurrency": concurrency,
        "benchmark_duration_s": benchmark_duration_s,
        "requests": {
            "total": total,
            "successful": len(ok_records),
            "failed": total - len(ok_records),
            "success_rate": safe_ratio(len(ok_records), total),
            "request_throughput": len(ok_records) / benchmark_duration_s,
            "attempted_request_throughput": total / benchmark_duration_s,
        },
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "output_token_throughput": output_tokens / benchmark_duration_s,
        },
        "latency": {
            "p50_s": quantile(latency, 0.50),
            "p95_s": quantile(latency, 0.95),
            "p99_s": quantile(latency, 0.99),
        },
        "ttft": {
            "p50_s": quantile(ttft, 0.50),
            "p95_s": quantile(ttft, 0.95),
            "p99_s": quantile(ttft, 0.99),
        },
        "inter_token_latency": {
            "p50_s": quantile(itl, 0.50),
            "p95_s": quantile(itl, 0.95),
            "p99_s": quantile(itl, 0.99),
        },
        "inter_chunk_latency": {
            "p50_s": quantile(icl, 0.50),
            "p95_s": quantile(icl, 0.95),
            "p99_s": quantile(icl, 0.99),
            "definition": "Spacing between non-empty SSE content chunks; chunks are not tokens.",
        },
        "time_per_output_token": {
            "p50_s": quantile(tpot, 0.50),
            "p95_s": quantile(tpot, 0.95),
            "p99_s": quantile(tpot, 0.99),
        },
        "per_request_decode_speed": {
            "p05_tokens_per_second": quantile(tps, 0.05),
            "p50_tokens_per_second": quantile(tps, 0.50),
            "p95_tokens_per_second": quantile(tps, 0.95),
        },
        "errors": dict(errors.most_common()),
    }
    summary["goodput"] = summarize_goodput(
        records,
        benchmark_duration_s=benchmark_duration_s,
        slo_ttft_s=slo_ttft_s,
        slo_tpot_s=slo_tpot_s,
        slo_e2e_s=slo_e2e_s,
    )
    return summary


def summarize_goodput(
    records: list[dict[str, Any]],
    *,
    benchmark_duration_s: float,
    slo_ttft_s: float | None = None,
    slo_tpot_s: float | None = None,
    slo_e2e_s: float | None = None,
) -> dict[str, Any] | None:
    constraints = {
        "ttft_s": slo_ttft_s,
        "time_per_output_token_s": slo_tpot_s,
        "latency_s": slo_e2e_s,
    }
    active = {metric: limit for metric, limit in constraints.items() if limit is not None}
    if not active:
        return None
    if benchmark_duration_s <= 0:
        raise ValueError("benchmark_duration_s must be positive")

    compliant: list[dict[str, Any]] = []
    violations: Counter[str] = Counter()
    for record in records:
        if not record.get("ok"):
            violations["request_failed"] += 1
            continue
        record_compliant = True
        for metric, limit in active.items():
            value = record.get(metric)
            if value is None:
                violations[f"{metric}_missing"] += 1
                record_compliant = False
            elif float(value) > float(limit):
                violations[f"{metric}_exceeded"] += 1
                record_compliant = False
        if record_compliant:
            compliant.append(record)

    output_tokens = sum(int(record.get("output_tokens") or 0) for record in compliant)
    return {
        "slo": active,
        "compliant_requests": len(compliant),
        "total_requests": len(records),
        "compliance_rate": safe_ratio(len(compliant), len(records)),
        "request_goodput": len(compliant) / benchmark_duration_s,
        "output_token_goodput": output_tokens / benchmark_duration_s,
        "violations": dict(violations.most_common()),
        "definition": (
            "A request contributes to goodput only if it succeeds and every configured "
            "per-request SLO metric is present and at or below its threshold."
        ),
    }


def render_load_report(summary: dict[str, Any]) -> str:
    requests = summary["requests"]
    tokens = summary["tokens"]
    latency = summary["latency"]
    ttft = summary["ttft"]
    itl = summary["inter_token_latency"]
    icl = summary["inter_chunk_latency"]
    tpot = summary["time_per_output_token"]
    decode = summary["per_request_decode_speed"]
    goodput = summary.get("goodput")

    def num(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None else f"{value:.3f}{suffix}"

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    lines = [
            "# Candidate Load Test Report",
            "",
            "## Scope",
            "",
            f"- Model: {summary['model']}",
            f"- Stream: {summary['stream']}",
            f"- Concurrency: {summary['concurrency']}",
            f"- Duration: {num(summary['benchmark_duration_s'], 's')}",
            "",
            "## Throughput",
            "",
            f"- Successful requests: {requests['successful']} / {requests['total']}",
            f"- Success rate: {pct(requests['success_rate'])}",
            f"- Request throughput: {num(requests['request_throughput'], ' req/s')}",
            f"- Attempted request throughput: {num(requests['attempted_request_throughput'], ' req/s')}",
            f"- Output token throughput: {num(tokens['output_token_throughput'], ' tok/s')}",
            "",
            "## Latency",
            "",
            f"- p95 request latency: {num(latency['p95_s'], 's')}",
            f"- p95 TTFT: {num(ttft['p95_s'], 's')}",
            f"- p95 inter-token latency: {num(itl['p95_s'], 's')}",
            f"- p95 inter-chunk latency: {num(icl['p95_s'], 's')}",
            f"- p95 time per output token: {num(tpot['p95_s'], 's')}",
            f"- p05 per-request decode speed: {num(decode['p05_tokens_per_second'], ' tok/s')}",
        ]
    if goodput is not None:
        lines.extend(
            [
                "",
                "## SLO Goodput",
                "",
                f"- SLOs: {json.dumps(goodput['slo'], sort_keys=True)}",
                (
                    f"- Compliant requests: {goodput['compliant_requests']} / "
                    f"{goodput['total_requests']}"
                ),
                f"- Compliance rate: {pct(goodput['compliance_rate'])}",
                f"- Request goodput: {num(goodput['request_goodput'], ' req/s')}",
                f"- Output token goodput: {num(goodput['output_token_goodput'], ' tok/s')}",
                f"- Violations: {json.dumps(goodput['violations'], sort_keys=True)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This tests an already-running candidate endpoint. It does not load, quantize, or optimize a 70B model by itself.",
            "- For publishable L20 capacity numbers, also run vLLM bench serve or NVIDIA GenAI-Perf with fixed input/output lengths and request rates.",
        ]
    )
    return "\n".join(lines) + "\n"


def prompts_from_dataset(dataset: list[dict[str, Any]]) -> list[str]:
    return [str(sample["prompt"]) for sample in dataset]


def _present(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None]
