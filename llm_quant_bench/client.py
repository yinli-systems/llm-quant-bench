from __future__ import annotations

import json
import ipaddress
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 120.0
    extra_body: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], fallback_name: str) -> "ModelConfig":
        return cls(
            name=str(data.get("name") or fallback_name),
            base_url=str(data["base_url"]).rstrip("/"),
            model=str(data["model"]),
            api_key=data.get("api_key"),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 512)),
            timeout_s=float(data.get("timeout_s", 120.0)),
            extra_body=data.get("extra_body"),
            stream_options=data.get("stream_options"),
        )


@dataclass
class GenerationResult:
    ok: bool
    text: str
    latency_s: float
    ttft_s: float | None
    inter_token_latency_s: float | None
    time_per_output_token_s: float | None
    output_tokens: int | None
    prompt_tokens: int | None
    tokens_per_second: float | None
    error: str | None = None
    raw_usage: dict[str, Any] | None = None


class OpenAIChatClient:
    """Minimal OpenAI-compatible chat client with optional streaming TTFT timing."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._opener = _build_opener(config.base_url)

    def generate(self, prompt: str, *, stream: bool) -> GenerationResult:
        started = time.perf_counter()
        try:
            body = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": stream,
            }
            if stream and self.config.stream_options:
                body["stream_options"] = self.config.stream_options
            if self.config.extra_body:
                body.update(self.config.extra_body)

            if stream:
                return self._generate_streaming(body, started)
            return self._generate_non_streaming(body, started)
        except Exception as exc:
            latency_s = time.perf_counter() - started
            return GenerationResult(
                ok=False,
                text="",
                latency_s=latency_s,
                ttft_s=None,
                inter_token_latency_s=None,
                time_per_output_token_s=None,
                output_tokens=None,
                prompt_tokens=None,
                tokens_per_second=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _generate_non_streaming(
        self, body: dict[str, Any], started: float
    ) -> GenerationResult:
        req = self._request(body)
        try:
            with self._opener.open(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        latency_s = time.perf_counter() - started
        text = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        usage = payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        output_tokens = _safe_int(completion_tokens) or estimate_tokens(text)
        tps = output_tokens / latency_s if latency_s > 0 and output_tokens else None
        tpot = latency_s / output_tokens if output_tokens else None
        return GenerationResult(
            ok=True,
            text=text,
            latency_s=latency_s,
            ttft_s=None,
            inter_token_latency_s=None,
            time_per_output_token_s=tpot,
            output_tokens=output_tokens,
            prompt_tokens=_safe_int(prompt_tokens),
            tokens_per_second=tps,
            raw_usage=usage,
        )

    def _generate_streaming(
        self, body: dict[str, Any], started: float
    ) -> GenerationResult:
        req = self._request(body)
        chunks: list[str] = []
        ttft_s: float | None = None
        chunk_times: list[float] = []
        usage: dict[str, Any] | None = None

        try:
            with self._opener.open(req, timeout=self.config.timeout_s) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            now = time.perf_counter()
                            if ttft_s is None:
                                ttft_s = now - started
                            chunk_times.append(now)
                            chunks.append(content)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        latency_s = time.perf_counter() - started
        text = "".join(chunks)
        usage = usage or {}
        completion_tokens = _safe_int(usage.get("completion_tokens")) or estimate_tokens(text)
        prompt_tokens = _safe_int(usage.get("prompt_tokens"))
        decode_s = max(latency_s - (ttft_s or 0.0), 1e-9)
        tps = completion_tokens / decode_s if completion_tokens else None
        tpot = decode_s / completion_tokens if completion_tokens else None
        if len(chunk_times) > 1:
            inter_token_latency_s = (chunk_times[-1] - chunk_times[0]) / (
                len(chunk_times) - 1
            )
        else:
            inter_token_latency_s = tpot
        return GenerationResult(
            ok=True,
            text=text,
            latency_s=latency_s,
            ttft_s=ttft_s,
            inter_token_latency_s=inter_token_latency_s,
            time_per_output_token_s=tpot,
            output_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            tokens_per_second=tps,
            raw_usage=usage,
        )


def estimate_tokens(text: str) -> int:
    """Rough tokenizer-free estimate for dashboards when API usage is absent."""
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, round(ascii_chars / 4 + non_ascii_chars / 1.6))


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_opener(base_url: str) -> urllib.request.OpenerDirector:
    if _is_loopback_url(base_url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _is_loopback_url(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
