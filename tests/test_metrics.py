"""CPU tests for latency/throughput math and vLLM /metrics parsing (Track C).

Covers the three things README §6 calls out: percentile math on a known array,
the TPOT formula, and preemption-delta parsing from a sample metrics blob.
"""

from __future__ import annotations

import pytest

from bench.metrics import (
    KvUsageTracker,
    RequestRecord,
    kv_pool_gb,
    parse_cache_config,
    parse_kv_usage,
    parse_num_running,
    parse_preemptions,
    parse_prometheus,
    percentile_summary,
    preemption_delta,
    summarize_latency,
)

SAMPLE_METRICS = """\
# HELP vllm:num_preemptions_total Cumulative number of preemption events.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{model_name="meta-llama/Llama-3.1-8B-Instruct"} 42.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage fraction.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.35
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct"} 8.0
"""


# --------------------------------------------------------------------------- #
# Percentiles
# --------------------------------------------------------------------------- #
def test_percentile_summary_known_array():
    s = percentile_summary([10, 20, 30, 40])
    assert s["mean"] == pytest.approx(25.0)
    assert s["p50"] == pytest.approx(25.0)
    assert s["p90"] == pytest.approx(37.0)   # 30 + 0.7*(40-30)
    assert s["p99"] == pytest.approx(39.7)   # 30 + 0.97*(40-30)


def test_percentile_summary_empty_is_safe():
    s = percentile_summary([])
    assert s == {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0}


# --------------------------------------------------------------------------- #
# RequestRecord latency properties
# --------------------------------------------------------------------------- #
def test_request_record_ttft_e2e_tpot():
    r = RequestRecord(id="x", start_time=0.0, first_token_time=1.0, end_time=5.0, n_output_tokens=5)
    assert r.ttft == pytest.approx(1.0)
    assert r.e2e == pytest.approx(5.0)
    # TPOT = (e2e - ttft) / (out_tokens - 1) = (5-1)/(5-1) = 1.0
    assert r.tpot == pytest.approx(1.0)


def test_tpot_formula_matches_readme():
    # e2e - ttft = decode span; divided by (out-1) inter-token gaps
    r = RequestRecord(id="y", start_time=10.0, first_token_time=10.5, end_time=12.5, n_output_tokens=21)
    assert r.ttft == pytest.approx(0.5)
    assert r.tpot == pytest.approx((12.5 - 10.5) / (21 - 1))  # 2.0/20 = 0.1


def test_tpot_none_when_fewer_than_two_tokens():
    r = RequestRecord(id="z", start_time=0.0, first_token_time=1.0, end_time=2.0, n_output_tokens=1)
    assert r.tpot is None


def test_incomplete_record_has_none_latencies():
    r = RequestRecord(id="w", start_time=0.0, first_token_time=None, end_time=None, success=False)
    assert r.ttft is None and r.e2e is None and r.tpot is None


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_summarize_latency_aggregates_and_converts_to_ms():
    records = [
        RequestRecord(id="a", start_time=0.0, first_token_time=0.1, end_time=1.1, n_output_tokens=11),
        RequestRecord(id="b", start_time=0.0, first_token_time=0.2, end_time=2.2, n_output_tokens=21),
        RequestRecord(id="c", start_time=0.0, success=False, error="boom"),
    ]
    s = summarize_latency(records, duration_s=2.2)
    assert s["n_requests"] == 3
    assert s["n_success"] == 2
    assert s["n_failed"] == 1
    assert s["ttft_ms"]["mean"] == pytest.approx(150.0)      # (100 + 200) / 2
    assert s["e2e_ms"]["mean"] == pytest.approx(1650.0)      # (1100 + 2200) / 2
    assert s["tpot_ms"]["mean"] == pytest.approx(100.0)      # both 0.1 s/token
    assert s["throughput"]["total_output_tokens"] == 32
    assert s["throughput"]["output_tok_s"] == pytest.approx(32 / 2.2)
    assert s["throughput"]["req_s"] == pytest.approx(2 / 2.2)


def test_summarize_latency_derives_duration_when_absent():
    records = [
        RequestRecord(id="a", start_time=100.0, first_token_time=100.1, end_time=101.0, n_output_tokens=5),
        RequestRecord(id="b", start_time=100.5, first_token_time=100.7, end_time=104.0, n_output_tokens=9),
    ]
    s = summarize_latency(records)
    # duration derived from min(start)=100.0 to max(end)=104.0 -> 4.0 s
    assert s["throughput"]["duration_s"] == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Prometheus parsing
# --------------------------------------------------------------------------- #
def test_parse_prometheus_reads_labeled_series():
    vals = parse_prometheus(SAMPLE_METRICS, ["vllm:num_preemptions_total", "vllm:num_requests_running"])
    assert vals["vllm:num_preemptions_total"] == pytest.approx(42.0)
    assert vals["vllm:num_requests_running"] == pytest.approx(8.0)


def test_parse_prometheus_missing_metric_is_zero():
    vals = parse_prometheus(SAMPLE_METRICS, ["vllm:does_not_exist"])
    assert vals["vllm:does_not_exist"] == 0.0


def test_parse_preemptions_and_kv_usage():
    assert parse_preemptions(SAMPLE_METRICS) == pytest.approx(42.0)
    assert parse_kv_usage(SAMPLE_METRICS) == pytest.approx(0.35)


def test_parse_kv_usage_prefers_vllm_0_24_name():
    # vLLM 0.24 (v1) exposes vllm:kv_cache_usage_perc (validated against a live
    # server); the parser must read it, not only the older gpu_cache name.
    blob = 'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.42\n'
    assert parse_kv_usage(blob) == pytest.approx(0.42)


def test_preemption_delta_from_two_scrapes():
    before = SAMPLE_METRICS
    after = SAMPLE_METRICS.replace(
        "vllm:num_preemptions_total{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 42.0",
        "vllm:num_preemptions_total{model_name=\"meta-llama/Llama-3.1-8B-Instruct\"} 49.0",
    )
    assert preemption_delta(before, after) == 7


def test_kv_usage_tracker_reports_peak():
    tracker = KvUsageTracker()
    tracker.observe(SAMPLE_METRICS.replace("0.35", "0.20"))
    tracker.observe(SAMPLE_METRICS.replace("0.35", "0.80"))
    tracker.observe(SAMPLE_METRICS.replace("0.35", "0.55"))
    assert tracker.peak == pytest.approx(0.80)


# --------------------------------------------------------------------------- #
# Batch size + KV-in-GB (presentation metrics)
# --------------------------------------------------------------------------- #
SAMPLE_CACHE_INFO = (
    'vllm:cache_config_info{block_size="16",engine="0",'
    'kv_cache_max_concurrency="120.2",kv_cache_size_tokens="492400",'
    'num_gpu_blocks="30775"} 1.0\n'
)


def test_parse_num_running():
    assert parse_num_running(SAMPLE_METRICS) == pytest.approx(8.0)


def test_parse_cache_config():
    c = parse_cache_config(SAMPLE_CACHE_INFO)
    assert c["kv_cache_size_tokens"] == pytest.approx(492400)
    assert c["num_gpu_blocks"] == pytest.approx(30775)
    assert c["kv_cache_max_concurrency"] == pytest.approx(120.2)
    assert c["block_size"] == pytest.approx(16)


def test_kv_pool_gb_math():
    cfg = {"kv_cache_size_tokens": 492400.0}
    assert kv_pool_gb(cfg, 12288) == pytest.approx(492400 * 12288 / 1e9)  # ~6.05 GB
    assert kv_pool_gb(cfg, None) is None  # unknown bytes/token -> None
    assert kv_pool_gb({}, 12288) is None  # no size in config -> None


def test_tracker_reports_peak_batch_size():
    tracker = KvUsageTracker()
    tracker.observe(SAMPLE_METRICS.replace("} 8.0", "} 3.0"))  # 3 running
    tracker.observe(SAMPLE_METRICS)  # 8 running
    tracker.observe(SAMPLE_METRICS.replace("} 8.0", "} 5.0"))  # 5 running
    assert tracker.peak_running == pytest.approx(8.0)
