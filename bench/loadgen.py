"""Poisson request generator against a vLLM OpenAI-compatible server (Track C).

Fires requests whose interarrival times are exponential (→ a Poisson arrival
process) at a target ``request_rate`` (req/s), streaming each response to
capture send / first-token / finish timestamps into a
:class:`~bench.metrics.RequestRecord`.

Design note: the **scheduler** (when each request launches) is separated from
the **sender** (the HTTP call). ``run_load`` takes a ``sender`` callable, so the
Poisson scheduling logic is fully CPU-testable with a mock sender — no server,
no GPU. The default sender talks to ``/v1/chat/completions`` (template applied
server-side, avoiding double-BOS issues) and streams with usage accounting.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np

from bench.datasets import Request
from bench.metrics import RequestRecord

Sender = Callable[[Request], Awaitable[RequestRecord]]


@dataclass
class SamplingConfig:
    """Generation knobs sent with every request."""

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False  # True → every request generates exactly max_tokens
    use_reference_len: bool = False  # per-request: generate exactly its reference
    # length (n_reference_tokens, capped at max_tokens) with ignore_eos. Gives a
    # VARIABLE but MODEL-INDEPENDENT load — fair across fp16/fp8 and gives LTR a
    # length spread to reorder. This is the paper's fixed-real-length methodology.


def _req_output(req: "Request", cfg: SamplingConfig):
    """(max_tokens, ignore_eos) for this request."""
    if cfg.use_reference_len and req.n_reference_tokens:
        return max(1, min(req.n_reference_tokens, cfg.max_tokens)), True
    return cfg.max_tokens, cfg.ignore_eos


# --------------------------------------------------------------------------- #
# Poisson scheduling (pure, CPU-testable)
# --------------------------------------------------------------------------- #
def interarrival_times(n: int, request_rate: float, *, seed: int = 0) -> list[float]:
    """``n`` interarrival gaps (s). Exponential(mean=1/rate); all-zero if infinite.

    ``request_rate <= 0`` or ``inf`` means "fire all at once" (the README's
    ``--request-rate inf``), i.e. zero gaps.
    """
    if n <= 0:
        return []
    if request_rate <= 0 or math.isinf(request_rate):
        return [0.0] * n
    rng = np.random.default_rng(seed)
    return rng.exponential(1.0 / request_rate, size=n).tolist()


# --------------------------------------------------------------------------- #
# Load driver
# --------------------------------------------------------------------------- #
async def run_load(
    requests: Sequence[Request],
    *,
    request_rate: float,
    sender: Sender,
    seed: int = 0,
) -> list[RequestRecord]:
    """Launch ``requests`` on a Poisson schedule and gather their records.

    Each request is launched (non-blocking) after its interarrival gap, so a
    slow response never delays subsequent arrivals — arrivals stay Poisson
    regardless of server latency. Records are returned in submission order.
    """
    gaps = interarrival_times(len(requests), request_rate, seed=seed)
    tasks: list[asyncio.Task[RequestRecord]] = []
    for gap, req in zip(gaps, requests, strict=True):
        if gap > 0:
            await asyncio.sleep(gap)
        tasks.append(asyncio.create_task(sender(req)))
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))


# --------------------------------------------------------------------------- #
# Real HTTP sender (streaming, OpenAI-compatible)
# --------------------------------------------------------------------------- #
def _chat_payload(req: Request, model: str, cfg: SamplingConfig) -> dict:
    mt, ieos = _req_output(req, cfg)
    payload = {
        "model": model,
        "messages": req.messages,
        "max_tokens": mt,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ieos:
        payload["ignore_eos"] = True  # vLLM extension
    if req.priority is not None:
        payload["priority"] = req.priority  # B1: LTR ranking drives priority scheduling
    return payload


def _completions_payload(req: Request, model: str, cfg: SamplingConfig) -> dict:
    mt, ieos = _req_output(req, cfg)
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": mt,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,  # prompt is already chat-templated
    }
    if ieos:
        payload["ignore_eos"] = True
    if req.priority is not None:
        payload["priority"] = req.priority  # B1: LTR ranking drives priority scheduling
    return payload


def _extract_delta(chunk: dict, endpoint: str) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    if endpoint == "chat":
        return (choices[0].get("delta") or {}).get("content") or ""
    return choices[0].get("text") or ""


def make_http_sender(
    session,
    base_url: str,
    model: str,
    cfg: SamplingConfig,
    *,
    endpoint: str = "chat",
) -> Sender:
    """Build a streaming HTTP sender bound to an aiohttp ``session``.

    ``endpoint`` is ``"chat"`` (``/v1/chat/completions``, messages) or
    ``"completions"`` (``/v1/completions``, pre-templated prompt).
    """
    if endpoint == "chat":
        url = base_url.rstrip("/") + "/v1/chat/completions"
        build = _chat_payload
    elif endpoint == "completions":
        url = base_url.rstrip("/") + "/v1/completions"
        build = _completions_payload
    else:
        raise ValueError(f"unknown endpoint: {endpoint!r}")

    async def sender(req: Request) -> RequestRecord:
        rec = RequestRecord(id=req.id, start_time=time.perf_counter(), n_prompt_tokens=req.n_prompt_tokens)
        counted = 0
        usage_tokens: int | None = None
        try:
            async with session.post(url, json=build(req, model, cfg)) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    rec.success = False
                    rec.error = f"HTTP {resp.status}: {body}"
                    rec.end_time = time.perf_counter()
                    return rec
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if (usage := chunk.get("usage")) and usage.get("completion_tokens") is not None:
                        usage_tokens = int(usage["completion_tokens"])
                    delta = _extract_delta(chunk, endpoint)
                    if delta:
                        if rec.first_token_time is None:
                            rec.first_token_time = time.perf_counter()
                        counted += 1
            rec.end_time = time.perf_counter()
            rec.n_output_tokens = usage_tokens if usage_tokens is not None else counted
            rec.success = rec.first_token_time is not None and rec.n_output_tokens > 0
            if not rec.success and rec.error is None:
                rec.error = "no tokens generated"
        except Exception as exc:  # noqa: BLE001 - record any transport error as a failure
            rec.success = False
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.end_time = rec.end_time or time.perf_counter()
        return rec

    return sender


async def run_http_load(
    requests: Sequence[Request],
    *,
    base_url: str,
    model: str,
    request_rate: float,
    sampling: SamplingConfig,
    endpoint: str = "chat",
    seed: int = 0,
) -> list[RequestRecord]:
    """Convenience wrapper: open an aiohttp session and drive a real server."""
    import aiohttp

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=3600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sender = make_http_sender(session, base_url, model, sampling, endpoint=endpoint)
        return await run_load(requests, request_rate=request_rate, sender=sender, seed=seed)
