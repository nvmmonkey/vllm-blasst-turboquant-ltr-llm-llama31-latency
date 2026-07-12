"""CPU tests for plot generation on synthetic data (Track C).

Verifies the figures render and write non-empty PNGs without a display or GPU.
"""

from __future__ import annotations

import pandas as pd

from bench.plots import plot_latency_vs_rate, plot_metric_bars, read_config_summaries


def _synthetic_df(base: float) -> pd.DataFrame:
    rates = [5, 10, 20, 30, 40, 50, 60]
    return pd.DataFrame(
        {
            "request_rate": rates,
            "e2e_mean_ms": [base + r * 10 for r in rates],
            "e2e_p99_ms": [base + r * 25 for r in rates],
            "preemptions": [0, 0, 1, 5, 12, 30, 60],
        }
    )


def test_plot_latency_vs_rate_writes_png(tmp_path):
    dfs = {"b0": _synthetic_df(200), "b1": _synthetic_df(120)}
    out = plot_latency_vs_rate(dfs, metric="e2e_mean_ms", out_path=tmp_path / "lat.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_metric_bars_writes_png(tmp_path):
    dfs = {"b0": _synthetic_df(200), "b1": _synthetic_df(120)}
    out = plot_metric_bars(dfs, metric="preemptions", rate=60, out_path=tmp_path / "bars.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_read_config_summaries_roundtrip(tmp_path):
    _synthetic_df(200).to_csv(tmp_path / "b0_sweep.csv", index=False)
    dfs = read_config_summaries(tmp_path, ["b0", "b1"])  # b1 file absent
    assert set(dfs) == {"b0"}
    assert list(dfs["b0"]["request_rate"]) == [5, 10, 20, 30, 40, 50, 60]
