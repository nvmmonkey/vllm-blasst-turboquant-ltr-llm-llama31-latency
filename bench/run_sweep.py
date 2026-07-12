"""Request-rate sweep runner (Track C).

For each rate in the sweep, warm up, drive ``N`` requests, scrape server-side
preemption/KV gauges around the run, and write one ``results/summaries/
<config>_<rate>.json`` (the README §9 schema) plus one combined CSV.

The live work (serving + ``/metrics`` scraping) is a ``RateDriver`` injected
into :func:`run_sweep`, so the orchestration — rate loop, file layout, CSV
schema — is CPU-testable with a mock driver and no server. :func:`main` wires
the real HTTP driver for runs on the GPU host.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bench.datasets import Request
from bench.loadgen import SamplingConfig, run_http_load
from bench.metrics import (
    KvUsageTracker,
    RequestRecord,
    kv_bytes_per_token,
    kv_pool_gb,
    parse_cache_config,
    preemption_delta,
    summarize_latency,
)

DEFAULT_RATES = (5, 10, 20, 30, 40, 50, 60)


@dataclass
class RateResult:
    """What a driver returns for one rate: records + server-side gauges."""

    records: list[RequestRecord]
    kv_peak: float = 0.0
    preemptions: int = 0
    duration_s: float | None = None
    peak_batch_size: float = 0.0          # peak vllm:num_requests_running (observed max batch)
    kv_peak_gb: float | None = None       # peak KV usage in GB (frac × pool size)
    max_concurrency: float | None = None  # vLLM's kv_cache_max_concurrency estimate
    extra: dict = field(default_factory=dict)


# (requests, rate) -> RateResult
RateDriver = Callable[[Sequence[Request], float], Awaitable[RateResult]]


def rate_label(rate: float) -> str:
    """Filename/label token for a rate (``inf`` or the integer/float value)."""
    if math.isinf(rate):
        return "inf"
    return str(int(rate)) if float(rate).is_integer() else str(rate)


def assemble_summary(
    config: str,
    rate: float,
    result: RateResult,
    *,
    model: str,
    vllm_version: str,
    gpu: str,
    seed: int,
    n_requests: int,
) -> dict:
    """Merge latency + server gauges + metadata into the README §9 schema."""
    lat = summarize_latency(result.records, duration_s=result.duration_s)
    return {
        "config": config,
        "request_rate": rate_label(rate),
        "n_requests": n_requests,
        "n_success": lat["n_success"],
        "n_failed": lat["n_failed"],
        "seed": seed,
        "ttft_ms": lat["ttft_ms"],
        "tpot_ms": lat["tpot_ms"],
        "e2e_ms": lat["e2e_ms"],
        "throughput": lat["throughput"],
        "kv": {"peak_usage_frac": result.kv_peak, "peak_usage_gb": result.kv_peak_gb},
        "batch": {"peak_size": result.peak_batch_size, "max_concurrency_est": result.max_concurrency},
        "preemptions": result.preemptions,
        "model": model,
        "vllm_version": vllm_version,
        "gpu": gpu,
    }


def _flatten_for_csv(summary: dict) -> dict:
    """One flat CSV row from a nested summary."""
    return {
        "config": summary["config"],
        "request_rate": summary["request_rate"],
        "n_requests": summary["n_requests"],
        "n_success": summary["n_success"],
        "n_failed": summary["n_failed"],
        "ttft_mean_ms": summary["ttft_ms"]["mean"],
        "ttft_p25_ms": summary["ttft_ms"]["p25"],
        "ttft_p50_ms": summary["ttft_ms"]["p50"],
        "ttft_p75_ms": summary["ttft_ms"]["p75"],
        "ttft_p90_ms": summary["ttft_ms"]["p90"],
        "ttft_p99_ms": summary["ttft_ms"]["p99"],
        "tpot_mean_ms": summary["tpot_ms"]["mean"],
        "tpot_p50_ms": summary["tpot_ms"]["p50"],
        "tpot_p90_ms": summary["tpot_ms"]["p90"],
        "tpot_p99_ms": summary["tpot_ms"]["p99"],
        "e2e_mean_ms": summary["e2e_ms"]["mean"],
        "e2e_p25_ms": summary["e2e_ms"]["p25"],
        "e2e_p50_ms": summary["e2e_ms"]["p50"],
        "e2e_p75_ms": summary["e2e_ms"]["p75"],
        "e2e_p90_ms": summary["e2e_ms"]["p90"],
        "e2e_p99_ms": summary["e2e_ms"]["p99"],
        "output_tok_s": summary["throughput"]["output_tok_s"],
        "req_s": summary["throughput"]["req_s"],
        "kv_peak_frac": summary["kv"]["peak_usage_frac"],
        "kv_peak_gb": summary["kv"].get("peak_usage_gb"),
        "peak_batch": summary["batch"]["peak_size"],
        "max_concurrency": summary["batch"]["max_concurrency_est"],
        "preemptions": summary["preemptions"],
    }


def build_consolidated(
    config: str,
    summaries: Sequence[dict],
    *,
    model: str,
    vllm_version: str,
    gpu: str,
    seed: int,
) -> dict:
    """Compact ALL rates for one config into a single self-describing object.

    Shared metadata lives once at the top; per-rate metrics go under ``rates``.
    """
    rate_blocks = [
        {
            "request_rate": s["request_rate"],
            "n_success": s["n_success"],
            "n_failed": s["n_failed"],
            "ttft_ms": s["ttft_ms"],
            "tpot_ms": s["tpot_ms"],
            "e2e_ms": s["e2e_ms"],
            "throughput": s["throughput"],
            "kv": s["kv"],
            "batch": s["batch"],
            "preemptions": s["preemptions"],
        }
        for s in summaries
    ]
    return {
        "config": config,
        "model": model,
        "vllm_version": vllm_version,
        "gpu": gpu,
        "seed": seed,
        "n_requests": summaries[0]["n_requests"] if summaries else 0,
        "rates": rate_blocks,
    }


async def run_sweep(
    *,
    config: str,
    requests: Sequence[Request],
    driver: RateDriver,
    rates: Sequence[float] = DEFAULT_RATES,
    out_dir: str | Path = "results/summaries",
    raw_dir: str | Path = "results/raw",
    model: str = "unknown",
    vllm_version: str = "unknown",
    gpu: str = "unknown",
    seed: int = 0,
    warmup_requests: Sequence[Request] = (),
) -> dict:
    """Drive every rate and write ONE consolidated file per config.

    Committed, single-file-per-config artifacts in ``out_dir``:
      * ``<config>.json`` — all rates + shared metadata in one object, and
      * ``<config>_sweep.csv`` — one flat row per rate.
    Per-rate detail JSONs go to ``<raw_dir>/<config>/`` (git-ignored) for
    debugging. Returns the consolidated object.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_dir) / config
    raw.mkdir(parents=True, exist_ok=True)

    if warmup_requests:
        await driver(list(warmup_requests), math.inf)  # discarded

    summaries: list[dict] = []
    for rate in rates:
        result = await driver(list(requests), float(rate))
        summary = assemble_summary(
            config, float(rate), result,
            model=model, vllm_version=vllm_version, gpu=gpu,
            seed=seed, n_requests=len(requests),
        )
        # per-rate detail -> raw/ (git-ignored)
        (raw / f"{config}_{rate_label(float(rate))}.json").write_text(
            json.dumps(summary, indent=2)
        )
        summaries.append(summary)

    # consolidated single file per config -> summaries/ (committed)
    consolidated = build_consolidated(
        config, summaries, model=model, vllm_version=vllm_version, gpu=gpu, seed=seed
    )
    (out / f"{config}.json").write_text(json.dumps(consolidated, indent=2))
    pd.DataFrame([_flatten_for_csv(s) for s in summaries]).to_csv(
        out / f"{config}_sweep.csv", index=False
    )
    return consolidated


# --------------------------------------------------------------------------- #
# Real HTTP driver (live GPU host) — not unit-tested (needs a server)
# --------------------------------------------------------------------------- #
def make_http_driver(
    base_url: str,
    model: str,
    sampling: SamplingConfig,
    *,
    endpoint: str = "chat",
    kv_sample_interval: float = 0.25,
    seed: int = 0,
):  # pragma: no cover - exercised by the live smoke test, not unit tests
    """Build a driver that serves a rate and scrapes preemption/KV gauges."""
    import aiohttp

    metrics_url = base_url.rstrip("/") + "/metrics"
    bytes_per_token = kv_bytes_per_token(model)  # computed once; model is fixed

    async def _scrape(session) -> str:
        async with session.get(metrics_url) as r:
            return await r.text()

    async def driver(requests: Sequence[Request], rate: float) -> RateResult:
        tracker = KvUsageTracker()
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as mon:
            try:
                before = await _scrape(mon)
            except Exception:
                before = ""

            stop = asyncio.Event()

            async def sample():
                while not stop.is_set():
                    try:
                        tracker.observe(await _scrape(mon))
                    except Exception:
                        pass
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=kv_sample_interval)
                    except asyncio.TimeoutError:
                        pass

            sampler = asyncio.create_task(sample())
            t0 = time.perf_counter()
            records = await run_http_load(
                requests, base_url=base_url, model=model,
                request_rate=rate, sampling=sampling, endpoint=endpoint, seed=seed,
            )
            duration = time.perf_counter() - t0
            stop.set()
            await sampler
            try:
                after = await _scrape(mon)
            except Exception:
                after = before

        cache_cfg = parse_cache_config(after or before)
        pool_gb = kv_pool_gb(cache_cfg, bytes_per_token)
        return RateResult(
            records=records,
            kv_peak=tracker.peak,
            preemptions=preemption_delta(before, after) if before and after else 0,
            duration_s=duration,
            peak_batch_size=tracker.peak_running,
            kv_peak_gb=(tracker.peak * pool_gb) if pool_gb is not None else None,
            max_concurrency=cache_cfg.get("kv_cache_max_concurrency"),
        )

    return driver


def _vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def main() -> None:  # pragma: no cover - CLI entry for the GPU host
    ap = argparse.ArgumentParser(description="Drive a request-rate sweep against vLLM.")
    ap.add_argument("--config", required=True, help="config label, e.g. b0 or b1")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--rates", default=",".join(map(str, DEFAULT_RATES)),
                    help="comma-separated req/s (use 'inf' for all-at-once)")
    ap.add_argument("-n", "--n-requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="generate exactly max_tokens per request (uniform load)")
    ap.add_argument("--use-reference-len", action="store_true",
                    help="generate exactly each request's reference length (capped at "
                         "max_tokens): a VARIABLE but model-independent load — fair "
                         "across fp16/fp8 and gives LTR a length spread to reorder")
    ap.add_argument("--source", default="lmsys", choices=["lmsys", "synthetic", "longctx"])
    ap.add_argument("--endpoint", default="chat", choices=["chat", "completions"])
    ap.add_argument("--out-dir", default="results/summaries")
    ap.add_argument("--gpu", default="unknown", help="GPU label recorded in the summary")
    ap.add_argument("--ltr-ranker", default=None,
                    help="path to a trained LTR ranker dir; stamps per-request priority (B1). "
                         "Serve with --scheduling-policy priority.")
    ap.add_argument("--device", default="cuda", help="device for LTR ranker scoring")
    args = ap.parse_args()

    from bench.datasets import iter_requests

    rates = [math.inf if r.strip() == "inf" else float(r) for r in args.rates.split(",")]
    pool = list(iter_requests(
        args.n_requests + args.warmup, seed=args.seed, source=args.source, model=args.model
    ))
    warmup = pool[: args.warmup]
    requests = pool[args.warmup : args.warmup + args.n_requests]

    if args.ltr_ranker:  # B1: score with the LTR ranker and stamp priorities
        from ltr.scheduler.assign import assign_priorities

        requests = assign_priorities(requests, args.ltr_ranker, device=args.device)
        print(f"assigned LTR priorities from {args.ltr_ranker}")

    sampling = SamplingConfig(
        max_tokens=args.max_tokens, temperature=args.temperature,
        ignore_eos=args.ignore_eos, use_reference_len=args.use_reference_len,
    )
    driver = make_http_driver(args.base_url, args.model, sampling, endpoint=args.endpoint, seed=args.seed)

    consolidated = asyncio.run(run_sweep(
        config=args.config, requests=requests, driver=driver, rates=rates,
        out_dir=args.out_dir, model=args.model, vllm_version=_vllm_version(),
        gpu=args.gpu, seed=args.seed, warmup_requests=warmup,
    ))
    n = len(consolidated["rates"])
    print(f"wrote {args.out_dir}/{args.config}.json ({n} rates) + {args.config}_sweep.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
