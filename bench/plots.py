"""Figures from committed sweep summaries (Track C).

Two figure types the milestone needs:
  * latency-vs-request-rate line charts (one line per config), and
  * a grouped bar chart for the final B0/B1/C comparison at a fixed rate.

Everything is generated from the committed ``results/summaries/<config>_sweep.csv``
files, so plots are reproducible and CPU-only (Agg backend, no display, no GPU).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Friendly axis labels for the CSV columns emitted by run_sweep.
METRIC_LABELS = {
    "e2e_mean_ms": "End-to-end latency (ms, mean)",
    "e2e_p99_ms": "End-to-end latency (ms, p99)",
    "ttft_mean_ms": "TTFT (ms, mean)",
    "ttft_p99_ms": "TTFT (ms, p99)",
    "tpot_mean_ms": "TPOT (ms, mean)",
    "output_tok_s": "Output throughput (tok/s)",
    "preemptions": "Preemptions (count)",
}


def read_config_summaries(
    summaries_dir: str | Path, configs: Sequence[str]
) -> dict[str, pd.DataFrame]:
    """Load ``<config>_sweep.csv`` for each config, sorted by request rate."""
    base = Path(summaries_dir)
    out: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        path = base / f"{cfg}_sweep.csv"
        if path.exists():
            out[cfg] = pd.read_csv(path).sort_values("request_rate")
    return out


def plot_latency_vs_rate(
    dfs: Mapping[str, pd.DataFrame],
    *,
    metric: str = "e2e_mean_ms",
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Line chart of ``metric`` vs request rate, one line per config."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg, df in dfs.items():
        ax.plot(df["request_rate"], df[metric], marker="o", label=cfg.upper())
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} vs request rate")
    ax.grid(True, alpha=0.3)
    if len(dfs) > 1:
        ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_metric_bars(
    dfs: Mapping[str, pd.DataFrame],
    *,
    metric: str,
    rate: float,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Grouped bar chart of ``metric`` at a fixed ``rate`` across configs."""
    labels, values = [], []
    for cfg, df in dfs.items():
        row = df[df["request_rate"] == rate]
        if not row.empty:
            labels.append(cfg.upper())
            values.append(float(row[metric].iloc[0]))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(labels, values, color=plt.cm.viridis([i / max(len(labels), 1) for i in range(len(labels))]))
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(title or f"{METRIC_LABELS.get(metric, metric)} at {rate} req/s")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Generate figures from sweep summaries.")
    ap.add_argument("--summaries-dir", default="results/summaries")
    ap.add_argument("--configs", default="b0,b1", help="comma-separated config labels")
    ap.add_argument("--metric", default="e2e_mean_ms")
    ap.add_argument("--bar-rate", type=float, default=60.0)
    ap.add_argument("--out-dir", default="results/summaries")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    dfs = read_config_summaries(args.summaries_dir, configs)
    if not dfs:
        print(f"no sweep CSVs found in {args.summaries_dir} for {configs}")
        return
    out = Path(args.out_dir)
    p1 = plot_latency_vs_rate(dfs, metric=args.metric, out_path=out / f"latency_vs_rate_{args.metric}.png")
    p2 = plot_metric_bars(dfs, metric=args.metric, rate=args.bar_rate, out_path=out / f"bars_{args.metric}_{int(args.bar_rate)}.png")
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":  # pragma: no cover
    main()
