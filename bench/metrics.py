"""Latency/throughput computation + vLLM ``/metrics`` parsing (Track C).

Two responsibilities:

1. **Client-side latency.** From per-request timestamps recorded by the load
   generator, compute TTFT, TPOT, end-to-end latency, and throughput. TPOT is
   ``(e2e - ttft) / (out_tokens - 1)`` per the README §6.
2. **Server-side gauges.** Scrape the vLLM Prometheus ``/metrics`` endpoint for
   things the client cannot see: the preemption counter
   (``vllm:num_preemptions_total``) and KV-cache usage
   (``vllm:gpu_cache_usage_perc``). Preemptions are read as a *delta* around a
   run; KV usage is a gauge sampled over time (peak taken by the sweep).

``RequestRecord`` is the shared schema the load generator produces and this
module consumes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

# vLLM metric names (the ``vllm:`` prefix is part of the metric name). vLLM 0.24
# (v1) exposes ``vllm:kv_cache_usage_perc``; older releases used
# ``vllm:gpu_cache_usage_perc``. Both are tried so the harness is version-robust.
VLLM_PREEMPTIONS = "vllm:num_preemptions_total"
VLLM_KV_USAGE = "vllm:kv_cache_usage_perc"
VLLM_KV_USAGE_ALTS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
VLLM_NUM_RUNNING = "vllm:num_requests_running"
# Sizing labels on the vllm:cache_config_info series (used to turn the KV
# usage fraction into GB and to read vLLM's own max-concurrency estimate).
_CACHE_INFO_KEYS = ("kv_cache_size_tokens", "num_gpu_blocks", "block_size", "kv_cache_max_concurrency")

_PCTILES = (25, 50, 75, 90, 99)


@dataclass
class RequestRecord:
    """Per-request timing produced by the load generator.

    All timestamps are ``time.perf_counter()`` seconds (monotonic). A record is
    ``success=False`` if the request errored or never produced a token.
    """

    id: str
    start_time: float
    first_token_time: float | None = None
    end_time: float | None = None
    n_output_tokens: int = 0
    n_prompt_tokens: int | None = None
    success: bool = True
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token (s)."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.start_time

    @property
    def e2e(self) -> float | None:
        """End-to-end latency (s)."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def tpot(self) -> float | None:
        """Time per output token (s): decode span / inter-token intervals."""
        if (
            self.first_token_time is None
            or self.end_time is None
            or self.n_output_tokens < 2
        ):
            return None
        return (self.end_time - self.first_token_time) / (self.n_output_tokens - 1)


# --------------------------------------------------------------------------- #
# Percentile helpers
# --------------------------------------------------------------------------- #
def percentile_summary(values: Sequence[float], pctiles: Sequence[int] = _PCTILES) -> dict:
    """Mean + requested percentiles of ``values`` (linear interpolation)."""
    if len(values) == 0:
        base = {"mean": 0.0}
        base.update({f"p{p}": 0.0 for p in pctiles})
        return base
    arr = np.asarray(values, dtype=float)
    out = {"mean": float(arr.mean())}
    for p in pctiles:
        out[f"p{p}"] = float(np.percentile(arr, p))
    return out


def _ms(summary: dict) -> dict:
    """Convert a seconds summary to milliseconds."""
    return {k: v * 1000.0 for k, v in summary.items()}


# --------------------------------------------------------------------------- #
# Latency / throughput
# --------------------------------------------------------------------------- #
def summarize_latency(
    records: Sequence[RequestRecord], *, duration_s: float | None = None
) -> dict:
    """Aggregate per-request records into the README §9 metric block.

    ``duration_s`` is the wall-clock window used for throughput; if omitted it
    is derived from the records (last finish − first send).
    """
    ok = [r for r in records if r.success and r.e2e is not None]
    n_total = len(records)
    n_success = len(ok)

    ttft = [r.ttft for r in ok if r.ttft is not None]
    tpot = [r.tpot for r in ok if r.tpot is not None]
    e2e = [r.e2e for r in ok if r.e2e is not None]

    if duration_s is None and ok:
        start = min(r.start_time for r in ok)
        end = max(r.end_time for r in ok if r.end_time is not None)
        duration_s = max(end - start, 1e-9)
    duration_s = duration_s or 1e-9

    total_out = sum(r.n_output_tokens for r in ok)

    return {
        "n_requests": n_total,
        "n_success": n_success,
        "n_failed": n_total - n_success,
        "ttft_ms": _ms(percentile_summary(ttft)),
        "tpot_ms": _ms(percentile_summary(tpot)),
        "e2e_ms": _ms(percentile_summary(e2e)),
        "throughput": {
            "output_tok_s": total_out / duration_s,
            "req_s": n_success / duration_s,
            "total_output_tokens": total_out,
            "duration_s": duration_s,
        },
    }


# --------------------------------------------------------------------------- #
# Prometheus (/metrics) parsing
# --------------------------------------------------------------------------- #
def parse_prometheus(text: str, names: Iterable[str]) -> dict[str, float]:
    """Sum the values of the requested metric ``names`` in a text exposition.

    Handles ``name{labels} value`` and ``name value`` lines, ignoring ``#``
    comments. Values across label sets are summed (one series per model in our
    single-model setup, so this is just that series' value). Missing metrics
    return ``0.0``. Non-numeric values (``NaN``/``+Inf`` histogram edges) are
    skipped.
    """
    wanted = set(names)
    out: dict[str, float] = dict.fromkeys(wanted, 0.0)
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue
        metric, value = parts
        name = metric.split("{", 1)[0].strip()
        if name in wanted:
            try:
                out[name] += float(value)
            except ValueError:
                continue
    return out


def parse_preemptions(text: str) -> float:
    """Read the cumulative preemption counter from a metrics blob."""
    return parse_prometheus(text, [VLLM_PREEMPTIONS])[VLLM_PREEMPTIONS]


def parse_kv_usage(text: str) -> float:
    """Read instantaneous KV-cache usage fraction ``[0, 1]`` (tolerant of alts)."""
    vals = parse_prometheus(text, VLLM_KV_USAGE_ALTS)
    for name in VLLM_KV_USAGE_ALTS:
        if vals.get(name):
            return vals[name]
    return 0.0


def preemption_delta(before_text: str, after_text: str) -> int:
    """Preemptions that occurred between two scrapes (``after − before``)."""
    delta = parse_preemptions(after_text) - parse_preemptions(before_text)
    return int(round(delta))


def parse_num_running(text: str) -> float:
    """Instantaneous number of running (batched) requests."""
    return parse_prometheus(text, [VLLM_NUM_RUNNING])[VLLM_NUM_RUNNING]


def parse_cache_config(text: str) -> dict[str, float]:
    """Read KV-pool sizing from the ``vllm:cache_config_info`` label set.

    Returns any of ``kv_cache_size_tokens``, ``num_gpu_blocks``, ``block_size``,
    ``kv_cache_max_concurrency`` that are present and numeric.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        if "cache_config_info" not in line or line.lstrip().startswith("#"):
            continue
        for key in _CACHE_INFO_KEYS:
            m = re.search(rf'{key}="([0-9.]+)"', line)
            if m:
                try:
                    out[key] = float(m.group(1))
                except ValueError:
                    pass
    return out


def kv_bytes_per_token(model_id: str, *, dtype_bytes: int = 2) -> int | None:
    """KV bytes/token = 2·layers·kv_heads·head_dim·dtype_bytes (lazy HF config).

    Returns ``None`` if the config can't be loaded or lacks the needed fields,
    so callers degrade to reporting only the usage fraction.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
    except Exception:
        return None
    layers = getattr(cfg, "num_hidden_layers", None)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", None)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hs, nh = getattr(cfg, "hidden_size", None), getattr(cfg, "num_attention_heads", None)
        head_dim = (hs // nh) if hs and nh else None
    if not (layers and kv_heads and head_dim):
        return None
    return 2 * int(layers) * int(kv_heads) * int(head_dim) * dtype_bytes


def kv_pool_gb(cache_cfg: dict[str, float], bytes_per_token: int | None) -> float | None:
    """Total KV-cache pool size in GB from tokens × bytes/token."""
    tokens = cache_cfg.get("kv_cache_size_tokens")
    if not tokens or not bytes_per_token:
        return None
    return tokens * bytes_per_token / 1e9


@dataclass
class KvUsageTracker:
    """Accumulates per-scrape gauges across a run to report peaks.

    Tracks both KV-cache usage fraction and the number of running requests, so
    a run yields peak KV usage *and* the peak batch size (max concurrency
    actually reached at fixed memory).
    """

    samples: list[float] = field(default_factory=list)
    running_samples: list[float] = field(default_factory=list)

    def observe(self, metrics_text: str) -> float:
        v = parse_kv_usage(metrics_text)
        self.samples.append(v)
        self.running_samples.append(parse_num_running(metrics_text))
        return v

    @property
    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def peak_running(self) -> float:
        return max(self.running_samples) if self.running_samples else 0.0
