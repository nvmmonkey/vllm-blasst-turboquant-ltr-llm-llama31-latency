"""CPU tests for the Poisson load generator (Track C).

Covers what README §6 calls out — mean interarrival ≈ 1/rate, exactly N
requests emitted, monotonic timestamps — plus a mocked streaming HTTP call to
exercise the SSE parser without a live server.
"""

from __future__ import annotations

import math

import pytest

from bench.datasets import Request
from bench.loadgen import (
    SamplingConfig,
    interarrival_times,
    make_http_sender,
    run_load,
)
from bench.metrics import RequestRecord


def _reqs(n: int) -> list[Request]:
    return [Request(id=f"r{i}", messages=[{"role": "user", "content": f"q{i}"}], prompt=f"q{i}") for i in range(n)]


# --------------------------------------------------------------------------- #
# Interarrival distribution
# --------------------------------------------------------------------------- #
def test_interarrival_mean_matches_inverse_rate():
    rate = 20.0
    gaps = interarrival_times(20_000, rate, seed=0)
    assert len(gaps) == 20_000
    assert sum(gaps) / len(gaps) == pytest.approx(1.0 / rate, rel=0.05)


def test_interarrival_emits_exactly_n():
    assert len(interarrival_times(37, 10.0, seed=1)) == 37
    assert interarrival_times(0, 10.0) == []


def test_interarrival_infinite_and_zero_rate_fire_at_once():
    assert interarrival_times(5, math.inf) == [0.0] * 5
    assert interarrival_times(5, 0.0) == [0.0] * 5


def test_interarrival_is_deterministic_under_seed():
    assert interarrival_times(100, 15.0, seed=3) == interarrival_times(100, 15.0, seed=3)
    assert interarrival_times(100, 15.0, seed=3) != interarrival_times(100, 15.0, seed=4)


# --------------------------------------------------------------------------- #
# run_load scheduling (mock sender — no HTTP)
# --------------------------------------------------------------------------- #
async def test_run_load_emits_exactly_n_records():
    reqs = _reqs(12)

    async def mock_sender(req: Request) -> RequestRecord:
        return RequestRecord(id=req.id, start_time=0.0, first_token_time=0.1, end_time=0.5, n_output_tokens=10)

    records = await run_load(reqs, request_rate=math.inf, sender=mock_sender, seed=0)
    assert len(records) == 12
    assert {r.id for r in records} == {r.id for r in reqs}


async def test_run_load_timestamps_are_monotonic():
    import time as _time

    reqs = _reqs(15)

    async def mock_sender(req: Request) -> RequestRecord:
        return RequestRecord(id=req.id, start_time=_time.perf_counter(), first_token_time=None, end_time=None, n_output_tokens=1)

    records = await run_load(reqs, request_rate=200.0, sender=mock_sender, seed=0)
    starts = [r.start_time for r in records]
    assert starts == sorted(starts)  # launched in order → non-decreasing


async def test_run_load_empty_returns_empty():
    async def mock_sender(req):  # pragma: no cover - never called
        raise AssertionError("should not be called")

    assert await run_load([], request_rate=10.0, sender=mock_sender) == []


# --------------------------------------------------------------------------- #
# Streaming HTTP sender (mocked aiohttp) — exercises the SSE parser
# --------------------------------------------------------------------------- #
class _FakeContent:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line

        return gen()


class _FakeResponse:
    def __init__(self, lines, status=200):
        self.status = status
        self.content = _FakeContent(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return "error-body"


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls: list[dict] = []

    def post(self, url, json):
        self.calls.append({"url": url, "json": json})
        return self._response


async def test_http_sender_parses_stream_and_counts_tokens():
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',   # role only, no content
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',    # first token
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n',   # second token
        b'data: {"choices":[],"usage":{"completion_tokens":2,"prompt_tokens":5,"total_tokens":7}}\n',
        b"data: [DONE]\n",
    ]
    session = _FakeSession(_FakeResponse(lines))
    sender = make_http_sender(session, "http://x:8000", "test-model", SamplingConfig(), endpoint="chat")

    rec = await sender(Request(id="r0", messages=[{"role": "user", "content": "hi"}], prompt="hi", n_prompt_tokens=5))
    assert rec.success is True
    assert rec.n_output_tokens == 2                 # from usage.completion_tokens
    assert rec.first_token_time is not None          # set on the "Hello" chunk
    assert rec.n_prompt_tokens == 5
    assert session.calls[0]["url"].endswith("/v1/chat/completions")
    assert session.calls[0]["json"]["stream"] is True


async def test_http_sender_marks_http_error_as_failure():
    session = _FakeSession(_FakeResponse([], status=503))
    sender = make_http_sender(session, "http://x:8000", "m", SamplingConfig())
    rec = await sender(Request(id="r1", messages=[{"role": "user", "content": "hi"}], prompt="hi"))
    assert rec.success is False
    assert "503" in rec.error


async def test_http_sender_falls_back_to_counted_tokens_without_usage():
    lines = [
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"c"}}]}\n',
        b"data: [DONE]\n",
    ]
    session = _FakeSession(_FakeResponse(lines))
    sender = make_http_sender(session, "http://x:8000", "m", SamplingConfig(), endpoint="chat")
    rec = await sender(Request(id="r2", messages=[{"role": "user", "content": "hi"}], prompt="hi"))
    assert rec.success is True
    assert rec.n_output_tokens == 3  # no usage chunk → counted content deltas


async def test_http_sender_includes_priority_when_set():
    session = _FakeSession(_FakeResponse([b"data: [DONE]\n"]))
    sender = make_http_sender(session, "http://x:8000", "m", SamplingConfig(), endpoint="chat")
    await sender(Request(id="r", messages=[{"role": "user", "content": "hi"}], prompt="hi", priority=7))
    assert session.calls[0]["json"]["priority"] == 7  # B1: LTR priority in the body


async def test_http_sender_omits_priority_when_unset():
    session = _FakeSession(_FakeResponse([b"data: [DONE]\n"]))
    sender = make_http_sender(session, "http://x:8000", "m", SamplingConfig(), endpoint="chat")
    await sender(Request(id="r", messages=[{"role": "user", "content": "hi"}], prompt="hi"))
    assert "priority" not in session.calls[0]["json"]  # B0: no priority field
