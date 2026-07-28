"""Presentation-sized versions of the report figures.

The report figures are built for a 3.5-inch IEEE column at 10 pt, so stretching
them onto a 16:9 slide softens the type. These are re-plotted from the same
committed run summaries at slide geometry: sized in inches for where they land on
the slide, 15-16 pt type, 300 dpi. Same data and colours as the report, so a
number on a slide always matches the number in the paper.

Run:  .venv/bin/python scripts/slide_figures.py
Out:  results/figures/slides/*.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _style import STYLE, SUMMARIES, load

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "figures", "slides")
RATES = [4, 8, 16, 32, 64]
R64 = 64


def slide_style() -> None:
    """Slide type: bigger than the paper, and sized for the placement width."""
    plt.rcParams.update({
        "font.family": "sans-serif",          # cleaner than serif on a projector
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
    })


def save(fig, name: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:26s} {w}x{h} px  ({w/300:.1f} x {h/300:.1f} in at 300 dpi)")


ARMS = [("B0", "b0_triton"), ("B1", "b1_triton"), ("+C2", "c2_bf16"),
        ("+C1", "c1_tq4"), ("+C1+C2", "c1c2_tq4")]


def headline() -> None:
    """Five configurations at 64 req/s: TTFT, TPOT, end-to-end. THE result slide."""
    panels = [("ttft_ms", "Mean TTFT (ms)", "Time to first token"),
              ("tpot_ms", "Mean TPOT (ms/token)", "Per-token latency"),
              ("e2e_ms", "Mean end-to-end (ms)", "End-to-end latency")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, (metric, ylab, title) in zip(axes, panels):
        labels = [a for a, _ in ARMS]
        vals = [load(c)[R64][metric]["mean"] for _, c in ARMS]
        colors = [STYLE[a]["color"] for a in labels]
        # hatched = four-bit cache, so solid -> hatched IS the compression step
        hatches = ["", "", "", "//", "//"]
        bars = ax.bar(labels, vals, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=1.2)
        # Neighbouring bars of near-equal height put their value labels at the
        # same y and the text collides (6488 vs 6537). Lift the second of any
        # such pair so both stay readable.
        span = max(vals)
        for i, (b, v) in enumerate(zip(bars, vals)):
            lift = 3
            if i and abs(v - vals[i - 1]) < 0.08 * span:
                lift = 22
            ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=13,
                        xytext=(0, lift), textcoords="offset points")
        ax.set_ylabel(ylab)
        ax.set_title(title, pad=8)
        ax.set_ylim(0, max(vals) * 1.20)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(pad=0.6, w_pad=1.6)
    save(fig, "slide_headline")


def signflip() -> None:
    """The novel finding: one algorithm, two kernels, opposite signs."""
    groups = ["bf16 unified\n(grouped-query)", "TQ4 per-head\ndecode kernel"]
    tpot = [4.9, -22.1]
    p99 = [15.0, -46.8]
    good, bad = "#146b8c", "#b4531f"
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = [0, 1]
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], tpot, w, label="Mean TPOT",
                color=[bad if v > 0 else good for v in tpot], edgecolor="white")
    b2 = ax.bar([i + w / 2 for i in x], p99, w, label="P99 TPOT", hatch="//",
                color=[bad if v > 0 else good for v in p99], edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v > 0 else "top",
                        fontsize=13, fontweight="bold",
                        xytext=(0, 4 if v > 0 else -6), textcoords="offset points")
    ax.axhline(0, color="#333", linewidth=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Change from adding sparsity (%)")
    ax.set_ylim(-58, 26)
    ax.legend(loc="lower left", frameon=True, framealpha=1.0)
    # Each verdict sits in the empty half of its own group: the bf16 group has no
    # bars below zero, the TQ4 group none above it. Placed beside the bars they
    # collided with the value labels.
    ax.annotate("HURTS", (0, -26), ha="center", fontsize=17, color=bad, fontweight="bold")
    ax.annotate("HELPS", (1, 13), ha="center", fontsize=17, color=good, fontweight="bold")
    fig.tight_layout(pad=0.5)
    save(fig, "slide_signflip")


def tail() -> None:
    """TTFT percentiles at 64 req/s -- compression flattens the whole distribution."""
    pcts = ["p25", "p50", "p75", "p90", "p99"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for label, cfg in ARMS:
        e = load(cfg)[R64]
        ax.plot(range(len(pcts)), [e["ttft_ms"][p] for p in pcts],
                label=label, **STYLE[label])
    ax.set_yscale("log")
    ax.set_xticks(range(len(pcts)))
    ax.set_xticklabels([p.upper() for p in pcts])
    ax.set_xlabel("TTFT percentile at 64 req/s")
    ax.set_ylabel("TTFT (ms, log scale)")
    # Below the axes, never inside: every curve rises left to right, so an opaque
    # in-panel legend does not dim the lines under it, it deletes them (it was
    # covering B0 from P75 on).
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=5,
               fontsize=13, frameon=False, columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout(pad=0.5, rect=(0, 0.10, 1, 1))
    save(fig, "slide_tail")


def preempt() -> None:
    """The mechanism: compression, not ordering, removes preemptions."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for label, cfg in ARMS:
        d = load(cfg)
        ax.plot(RATES, [d[r]["preemptions"] for r in RATES], label=label, **STYLE[label])
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Preemptions per run")
    ax.set_title("Preemptions: compression is the fix", pad=8)
    ax.set_ylim(-10, 300)
    ax.legend(ncol=2, fontsize=13, framealpha=1.0)

    UNC, COM = "#8a6a1a", "#0f5f8f"
    subs = [("b1a_04p", UNC, "s", (0, (1, 1.6))), ("b1_v1p", UNC, "o", (0, (4, 2.2))),
            ("c1_04p", COM, "s", (0, (1, 1.6))), ("c1_v1p", COM, "o", "-")]
    for cfg, colour, marker, dash in subs:
        d = load(cfg)
        ax2.plot(RATES, [100 * d[r]["kv"]["peak_usage_frac"] for r in RATES],
                 color=colour, marker=marker, linestyle=dash)
    ax2.axhline(100, color="#9a3324", linewidth=1.2, linestyle=(0, (2, 2)))
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(RATES)
    ax2.set_xticklabels([str(r) for r in RATES])
    ax2.set_xlabel("Request rate (req/s)")
    ax2.set_ylabel("Peak KV occupancy (%)")
    ax2.set_title("Cache occupancy: the pool stops filling", pad=8)
    ax2.set_ylim(28, 135)
    ax2.text(4.2, 101, "pool full", fontsize=13, color="#9a3324", va="bottom")
    ax2.annotate("uncompressed", xy=(16, 100), xytext=(0, 16), textcoords="offset points",
                 ha="center", fontsize=14, color=UNC, fontweight="bold")
    ax2.annotate("compressed", xy=(16, 78), xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=14, color=COM, fontweight="bold")
    fig.tight_layout(pad=0.6, w_pad=1.8)
    save(fig, "slide_preempt")


def repro() -> None:
    """Reproduction of the prior LTR result on its own engine and trace."""
    with open(os.path.join(SUMMARIES, "reference_ladder.json")) as fh:
        doc = json.load(fh)
    arms = [("B0 (FCFS)", "b0"), ("B1 (LTR)", "b1a"), ("+C1 (ours)", "c1")]
    key = {"B0 (FCFS)": "B0", "B1 (LTR)": "B1", "+C1 (ours)": "+C1"}
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for label, arm in arms:
        e = sorted(doc["arms"][arm], key=lambda z: z["rate"])
        ax.plot([z["rate"] for z in e], [z["nlat_p99"] for z in e],
                label=label, **STYLE[key[label]])
    ax.set_xscale("log", base=2)
    rates = sorted({z["rate"] for z in doc["arms"]["b0"]})
    ax.set_xticks(rates)
    ax.set_xticklabels([str(r) for r in rates])
    ax.set_yscale("log")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("P99 normalized latency\n(ms/token, log scale)")
    ax.set_title("Prior LTR result reproduced on its own engine", pad=8)
    ax.legend(framealpha=1.0, fontsize=13)
    fig.tight_layout(pad=0.5)
    save(fig, "slide_repro")


def engines() -> None:
    """Backup slide: the same shape on three engine versions."""
    panels = [("vLLM 0.4.1 (swap)", [("B0", "b0_04p"), ("B1", "b1a_04p"), ("+C1", "c1_04p")]),
              ("vLLM 0.8.5 (recompute)", [("B0", "b0_v1p"), ("B1", "b1_v1p"), ("+C1", "c1_v1p")]),
              ("vLLM 0.25 (TurboQuant)", [("B0", "b0_triton"), ("B1", "b1_triton"), ("+C1", "c1_tq4")])]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, (title, arms) in zip(axes, panels):
        for label, cfg in arms:
            d = load(cfg)
            ax.plot(RATES, [d[r]["ttft_ms"]["mean"] for r in RATES],
                    label=label, **STYLE[label])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(RATES)
        ax.set_xticklabels([str(r) for r in RATES])
        ax.set_xlabel("Request rate (req/s)")
        ax.set_title(title, pad=8)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].set_ylabel("Mean TTFT (ms, log)")
    axes[0].legend(fontsize=13, framealpha=1.0)
    fig.tight_layout(pad=0.6, w_pad=1.2)
    save(fig, "slide_engines")


def density() -> None:
    """Fig. 7(b) at slide size: the benefit follows batch size, not context length."""
    ctx = [512, 2048, 4096, 7168]
    delta = [-20.4, -10.3, 0.6, 2.8]
    batch = [40, 21, 10, 6]
    good, bad = "#146b8c", "#b4531f"
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bars = ax.bar([str(c) for c in ctx], delta,
                  color=[bad if v > 0 else good for v in delta], edgecolor="white", width=0.6)
    for b, v, n in zip(bars, delta, batch):
        ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom" if v > 0 else "top", fontsize=14,
                    fontweight="bold", xytext=(0, 5 if v > 0 else -6),
                    textcoords="offset points")
        ax.annotate(f"batch {n}", (b.get_x() + b.get_width() / 2, 5.5),
                    ha="center", fontsize=13, color="#5a6672")
    ax.axhline(0, color="#333", linewidth=1.4)
    ax.set_xlabel("Context length (tokens), fixed 6 req/s")
    ax.set_ylabel("Change in TPOT (%)")
    ax.set_ylim(-26, 9)
    fig.tight_layout(pad=0.5)
    save(fig, "slide_density")


def schematics() -> None:
    """Rasterise the two Background schematics from their vector PDFs, so the
    slide gets a crisp image at any size instead of a 200 dpi bitmap."""
    import fitz
    src = os.path.join(os.path.dirname(OUT), "arch_preempt.pdf")
    for name in ("arch_preempt", "arch_attention"):
        pdf = os.path.join(os.path.dirname(OUT), name + ".pdf")
        d = fitz.open(pdf)
        d[0].get_pixmap(dpi=300).save(os.path.join(OUT, "slide_" + name + ".png"))
        print(f"  slide_{name:22s} rasterised at 300 dpi")


if __name__ == "__main__":
    slide_style()
    print("slide figures (300 dpi, 15-16 pt type):")
    headline()
    signflip()
    tail()
    preempt()
    repro()
    engines()
    density()
    schematics()
