"""CPU tests for the sweep orchestrator (Track C).

Mocks the live driver and asserts the file layout and CSV schema the README §6
requires: one JSON per (config, rate) plus a combined CSV.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from bench.datasets import Request
from bench.metrics import RequestRecord
from bench.run_sweep import (
    RateResult,
    assemble_summary,
    build_consolidated,
    rate_label,
    run_sweep,
)


def _reqs(n: int) -> list[Request]:
    return [Request(id=f"r{i}", messages=[{"role": "user", "content": "q"}], prompt="q") for i in range(n)]


def _make_fake_driver(calls: list | None = None):
    async def driver(requests, rate):
        if calls is not None:
            calls.append(rate)
        recs = [
            RequestRecord(id=f"{rate}-{i}", start_time=0.0, first_token_time=0.1, end_time=1.0, n_output_tokens=10)
            for i in range(len(requests))
        ]
        preempt = int(rate) if math.isfinite(rate) else 0  # warmup passes inf
        return RateResult(records=recs, kv_peak=0.5, preemptions=preempt, duration_s=2.0)

    return driver


def test_rate_label():
    assert rate_label(math.inf) == "inf"
    assert rate_label(5.0) == "5"
    assert rate_label(12) == "12"


async def test_run_sweep_writes_one_consolidated_file_per_config(tmp_path):
    raw = tmp_path / "raw"
    result = await run_sweep(
        config="b0",
        requests=_reqs(3),
        driver=_make_fake_driver(),
        rates=[5, 10],
        out_dir=tmp_path,
        raw_dir=raw,
        model="meta-llama/Llama-3.1-8B-Instruct",
        vllm_version="0.24.0",
        gpu="RTX3090-24GB",
    )
    # run_sweep returns the consolidated object
    assert result["config"] == "b0"
    assert result["gpu"] == "RTX3090-24GB"
    assert len(result["rates"]) == 2

    # ONE consolidated file per config (shared metadata once, all rates inside)
    consolidated = json.loads((tmp_path / "b0.json").read_text())
    assert consolidated["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert consolidated["n_requests"] == 3
    assert [r["request_rate"] for r in consolidated["rates"]] == ["5", "10"]
    r5 = consolidated["rates"][0]
    assert r5["preemptions"] == 5
    assert r5["kv"]["peak_usage_frac"] == 0.5
    assert set(r5["ttft_ms"]) == {"mean", "p25", "p50", "p75", "p90", "p99"}

    # combined CSV (also one file per config)
    df = pd.read_csv(tmp_path / "b0_sweep.csv")
    assert len(df) == 2
    for col in ["config", "request_rate", "ttft_mean_ms", "e2e_p99_ms", "output_tok_s", "preemptions"]:
        assert col in df.columns

    # per-rate detail goes to raw/ (git-ignored), NOT scattered in summaries/
    assert (raw / "b0" / "b0_5.json").exists()
    assert (raw / "b0" / "b0_10.json").exists()
    assert not (tmp_path / "b0_5.json").exists()


async def test_run_sweep_runs_warmup_as_discarded_call(tmp_path):
    calls: list[float] = []
    await run_sweep(
        config="b0",
        requests=_reqs(2),
        driver=_make_fake_driver(calls),
        rates=[5],
        out_dir=tmp_path,
        raw_dir=tmp_path / "raw",
        warmup_requests=_reqs(2),
    )
    # first call is the warmup at inf, then the measured rate
    assert calls[0] == math.inf
    assert 5.0 in calls
    assert (tmp_path / "b0.json").exists()


def test_build_consolidated_shape():
    s = assemble_summary(
        "b0", 5.0,
        RateResult(records=[RequestRecord(id="a", start_time=0.0, first_token_time=0.1, end_time=1.0, n_output_tokens=8)], kv_peak=0.3, preemptions=2, duration_s=1.0),
        model="m", vllm_version="0.24.0", gpu="RTX3090", seed=0, n_requests=1,
    )
    c = build_consolidated("b0", [s], model="m", vllm_version="0.24.0", gpu="RTX3090", seed=0)
    assert c["config"] == "b0"
    assert c["n_requests"] == 1
    assert len(c["rates"]) == 1
    # shared metadata is NOT duplicated inside each rate block
    assert "model" not in c["rates"][0]
    assert c["rates"][0]["request_rate"] == "5"


def test_assemble_summary_shape():
    result = RateResult(
        records=[RequestRecord(id="a", start_time=0.0, first_token_time=0.1, end_time=1.0, n_output_tokens=5)],
        kv_peak=0.42,
        preemptions=3,
        duration_s=1.0,
    )
    s = assemble_summary(
        "b1", 30.0, result, model="m", vllm_version="0.24.0", gpu="A100-40GB", seed=0, n_requests=1
    )
    assert s["config"] == "b1"
    assert s["request_rate"] == "30"
    assert s["preemptions"] == 3
    assert s["kv"]["peak_usage_frac"] == 0.42
    assert s["throughput"]["total_output_tokens"] == 5
