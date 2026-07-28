"""Figure — the two conditions that decide whether attention sparsity pays.

One algorithm, one threshold, two ways of not working. Both panels report the
same quantity, percentage change from adding C2, so the colour convention holds
across them: a bar below zero is an improvement, above zero a regression.

  (a) It depends on the decode kernel. The bf16 unified kernel serves grouped
      queries -- four query heads share one loaded block and several query
      positions are packed per tile -- so a block is skipped only if the whole
      tile agrees, realized sparsity collapses from 42 % to about 18.5 %, and the
      guard costs more than the skipping saves. The per-head kernel decodes one
      query for one head, the test is scalar, and a skipped block also skips the
      value dequantization that quantization introduced.

  (b) It depends on compute density, not on context length. At a fixed arrival
      rate a longer context means a larger per-request footprint, so the
      memory-equal pool admits fewer concurrent requests; the batch falls from 40
      to 6 and decode turns bandwidth-bound, where removing arithmetic cannot
      help because the key loads still happen.

Percentages are computed from the committed means, never transcribed.
Data: results/summaries/{b1_triton,c2_bf16,c1_tq4,c1c2_tq4}.json (a),
      results/summaries/lctq4{nc,blastnc}_{512,2048,4096,7168}.json (b)
Run:  python scripts/fig4_c2_regimes.py
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _style import SUMMARIES, apply_style, layout, load, save

RATE = 64
# Semantic colours for "improved" and "regressed". Deliberately NOT the STYLE
# hex codes: those are #1b7a4b for +C1+C2 and #9a3324 for B0, and panel (a)'s
# right-hand group IS the +C1 to +C1+C2 comparison, so reusing them would let a
# reader infer a per-configuration rule that every other figure contradicts.
GOOD, BAD = "#146b8c", "#b4531f"

KERNELS = [
    ("bf16\nunified (GQA)", "b1_triton", "c2_bf16"),
    ("TQ4\nper-head",       "c1_tq4",    "c1c2_tq4"),
]
METRICS = [
    ("ttft_ms", "mean", "TTFT mean", ""),
    ("tpot_ms", "mean", "TPOT mean", "//"),
    ("tpot_ms", "p99",  "TPOT p99",  "xx"),
]
CONTEXTS = [512, 2048, 4096, 7168]


def pct_change(dense: str, sparse: str, metric: str, stat: str) -> float:
    a = load(dense)[RATE][metric][stat]
    b = load(sparse)[RATE][metric][stat]
    return 100.0 * (b / a - 1.0)


def longctx():
    """(ctx, delta TPOT %, peak batch) from the chunked-prefill-disabled sweep."""
    out = []
    for ctx in CONTEXTS:
        dense = json.load(open(os.path.join(SUMMARIES, f"lctq4nc_{ctx}.json")))
        blast = json.load(open(os.path.join(SUMMARIES, f"lctq4blastnc_{ctx}.json")))
        d = dense["rates"][0]
        b = blast["rates"][0]
        out.append((ctx,
                    100.0 * (b["tpot_ms"]["mean"] / d["tpot_ms"]["mean"] - 1.0),
                    d["batch"]["peak_size"]))
    return out


def main() -> None:
    apply_style()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.45))

    # --- (a) the kernel decides the sign ---------------------------------
    width = 0.24
    x = np.arange(len(KERNELS))
    for i, (metric, stat, _, hatch) in enumerate(METRICS):
        deltas = [pct_change(dense, sparse, metric, stat)
                  for _, dense, sparse in KERNELS]
        bars = ax.bar(x + (i - 1) * width, deltas, width,
                      color=[GOOD if d < 0 else BAD for d in deltas],
                      edgecolor="white", linewidth=0.6)
        for bar in bars:
            bar.set_hatch(hatch)
        for xi, d in zip(x + (i - 1) * width, deltas):
            ax.annotate(f"{d:+.1f}", (xi, d), ha="center",
                        va="bottom" if d > 0 else "top", fontsize=10,
                        xytext=(0, 2 if d > 0 else -2), textcoords="offset points")
    ax.axhline(0, color="#111417", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([k for k, _, _ in KERNELS])
    ax.set_ylabel("Change from C2 (%)")
    ax.set_ylim(-62, 30)
    ax.set_title("(a) the decode kernel decides the sign", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(facecolor="#b9c0c4", hatch=h, edgecolor="white",
                             label=lbl) for _, _, lbl, h in METRICS],
              loc="lower left", fontsize=10, handlelength=1.6,
              labelspacing=0.25, handletextpad=0.5)

    # --- (b) the batch decides the magnitude ------------------------------
    rows = longctx()
    xs = np.arange(len(rows))
    deltas = [d for _, d, _ in rows]
    bars = ax2.bar(xs, deltas, 0.6,
                   color=[GOOD if d < 0 else BAD for d in deltas],
                   edgecolor="white", linewidth=0.6)
    for xi, d in zip(xs, deltas):
        ax2.annotate(f"{d:+.1f}", (xi, d), ha="center",
                     va="bottom" if d > 0 else "top", fontsize=10,
                     xytext=(0, 2 if d > 0 else -2), textcoords="offset points")
    ax2.axhline(0, color="#111417", linewidth=0.9)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(c) for c in CONTEXTS])
    ax2.set_xlabel("Context length (tokens), fixed 6 req/s")
    ax2.set_ylabel("Change in TPOT (%)")
    ax2.set_ylim(-30, 14)
    ax2.set_title("(b) the batch decides the magnitude", fontsize=10)
    ax2.grid(axis="x", visible=False)

    # peak batch as an annotation rather than a second axis: a right-hand scale
    # here rendered its ticks as minus signs beside a left axis that has real
    # negatives, which was genuinely ambiguous
    for xi, (_, _, batch) in zip(xs, rows):
        ax2.annotate(f"batch {batch:.0f}", (xi, 12), ha="center", va="top",
                     fontsize=10, color="#4a545c")

    layout(fig, pad=0.3, h_pad=0.9)
    save(fig, "fig4_c2_regimes")


if __name__ == "__main__":
    main()
