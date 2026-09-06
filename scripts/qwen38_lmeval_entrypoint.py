#!/usr/bin/env python3
"""Run the pinned harness while retaining vLLM completion accounting."""
import functools
import json
import os
import runpy
import time
from pathlib import Path


def main():
    from vllm import LLM

    telemetry = Path(os.environ["QWEN38_GENERATION_TELEMETRY"])
    original = LLM.generate

    @functools.wraps(original)
    def recorded_generate(self, *args, **kwargs):
        started = time.monotonic()
        outputs = original(self, *args, **kwargs)
        elapsed = time.monotonic() - started
        with telemetry.open("a", encoding="utf-8") as stream:
            for request in outputs:
                for completion in request.outputs:
                    row = {
                        "request_id": request.request_id,
                        "prompt_tokens": len(request.prompt_token_ids or []),
                        "output_tokens": len(completion.token_ids),
                        "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason,
                        "thinking_closed": "</think>" in completion.text,
                        "batch_elapsed_seconds": elapsed,
                    }
                    stream.write(json.dumps(row) + "\n")
        return outputs

    LLM.generate = recorded_generate
    runpy.run_module("lm_eval", run_name="__main__")


if __name__ == "__main__":
    main()
