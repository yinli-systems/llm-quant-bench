from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .client import ModelConfig
from .demo import run_demo
from .load import prompts_from_dataset, run_load_test
from .metrics import perplexity_delta_percent, perplexity_from_token_logprobs
from .runner import (
    BenchmarkConfig,
    BenchmarkTargets,
    load_dataset,
    render_report,
    run_benchmark,
    summarize_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llm-quant-bench",
        description="Benchmark quantized LLM quality, stability, and service usability.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run baseline vs candidate benchmark.")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--dataset", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--repeats", type=int, default=1)
    run_parser.add_argument("--concurrency", type=int, default=1)
    run_parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Run full dataset waves until this many seconds elapse. Overrides repeats as the stop condition.",
    )
    run_parser.add_argument("--no-stream", action="store_true", help="Disable streaming TTFT measurement.")
    run_parser.add_argument(
        "--stream-usage",
        action="store_true",
        help="Request streaming usage metadata with stream_options.include_usage=true.",
    )

    summary_parser = subparsers.add_parser("summarize", help="Summarize an existing results.jsonl.")
    summary_parser.add_argument("--results", required=True, type=Path)
    summary_parser.add_argument("--targets", type=Path, help="Optional config JSON containing targets.")
    summary_parser.add_argument("--out", type=Path, help="Optional report.md output path.")

    ppl_parser = subparsers.add_parser("ppl", help="Compute perplexity formulas from token logprobs JSON.")
    ppl_parser.add_argument("--baseline-logprobs", required=True, type=Path)
    ppl_parser.add_argument("--candidate-logprobs", required=True, type=Path)

    demo_parser = subparsers.add_parser("demo", help="Run a local end-to-end smoke test.")
    demo_parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    demo_parser.add_argument("--no-stream", action="store_true")

    load_parser = subparsers.add_parser("load", help="Run candidate-only load test.")
    load_parser.add_argument("--config", required=True, type=Path)
    load_parser.add_argument("--dataset", required=True, type=Path)
    load_parser.add_argument("--out", required=True, type=Path)
    load_parser.add_argument("--concurrency", type=int, default=1)
    load_parser.add_argument("--requests", type=int)
    load_parser.add_argument("--duration-seconds", type=float)
    load_parser.add_argument(
        "--slo-ttft-ms", type=float, help="Count goodput only when per-request TTFT meets this SLO."
    )
    load_parser.add_argument(
        "--slo-tpot-ms", type=float, help="Count goodput only when per-request TPOT meets this SLO."
    )
    load_parser.add_argument(
        "--slo-e2e-ms", type=float, help="Count goodput only when end-to-end latency meets this SLO."
    )
    load_parser.add_argument("--no-stream", action="store_true")
    load_parser.add_argument(
        "--stream-usage",
        action="store_true",
        help="Request streaming usage metadata with stream_options.include_usage=true.",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        config = BenchmarkConfig.from_path(args.config)
        if args.stream_usage:
            config = _with_stream_usage(config)
        dataset = load_dataset(args.dataset)
        summary = run_benchmark(
            config=config,
            dataset=dataset,
            out_dir=args.out,
            repeats=args.repeats,
            concurrency=args.concurrency,
            stream=not args.no_stream,
            duration_s=args.duration_seconds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "summarize":
        targets = _load_targets(args.targets)
        summary = summarize_file(args.results, targets)
        report = render_report(summary, targets)
        if args.out:
            args.out.write_text(report, encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ppl":
        baseline = _read_logprobs(args.baseline_logprobs)
        candidate = _read_logprobs(args.candidate_logprobs)
        baseline_ppl = perplexity_from_token_logprobs(baseline)
        candidate_ppl = perplexity_from_token_logprobs(candidate)
        print(
            json.dumps(
                {
                    "baseline_ppl": baseline_ppl,
                    "candidate_ppl": candidate_ppl,
                    "ppl_delta_percent": perplexity_delta_percent(
                        baseline_ppl, candidate_ppl
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "demo":
        summary = run_demo(args.out, stream=not args.no_stream)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "load":
        config = BenchmarkConfig.from_path(args.config)
        model = (
            _with_stream_usage_model(config.candidate)
            if args.stream_usage
            else config.candidate
        )
        dataset = load_dataset(args.dataset)
        summary = run_load_test(
            model=model,
            prompts=prompts_from_dataset(dataset),
            out_dir=args.out,
            concurrency=args.concurrency,
            stream=not args.no_stream,
            requests=args.requests,
            duration_s=args.duration_seconds,
            slo_ttft_s=_milliseconds_to_seconds(args.slo_ttft_ms),
            slo_tpot_s=_milliseconds_to_seconds(args.slo_tpot_ms),
            slo_e2e_s=_milliseconds_to_seconds(args.slo_e2e_ms),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 2


def _load_targets(path: Path | None) -> BenchmarkTargets:
    if not path:
        return BenchmarkTargets()
    data = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkTargets.from_dict(data.get("targets", data))


def _read_logprobs(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [float(item) for item in data]
    if isinstance(data, dict) and isinstance(data.get("token_logprobs"), list):
        return [float(item) for item in data["token_logprobs"]]
    raise ValueError(f"{path} must be a JSON list or object with token_logprobs")


def _with_stream_usage(config: BenchmarkConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        baseline=_with_stream_usage_model(config.baseline),
        candidate=_with_stream_usage_model(config.candidate),
        targets=config.targets,
        judge=config.judge,
    )


def _with_stream_usage_model(model: ModelConfig) -> ModelConfig:
    stream_options = dict(model.stream_options or {})
    stream_options["include_usage"] = True
    return replace(model, stream_options=stream_options)


def _milliseconds_to_seconds(value: float | None) -> float | None:
    return None if value is None else value / 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
