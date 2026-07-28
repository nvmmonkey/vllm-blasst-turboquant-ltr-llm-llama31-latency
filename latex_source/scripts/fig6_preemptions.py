"""Figure 6 — preemption is the mechanism, and swap and compression substitute for each other.

Left panel: preemptions against arrival rate on vLLM 0.25, all five configurations.
Scheduling trims them; compression removes most of them, because a cache that fits
is rarely evicted. This is the causal link behind the latency numbers.

Right panel: the substitution claim needs a different quantity. The engine's
`preemptions` counter records RECOMPUTE events only, so on the 0.4.1 fork, which
recovers by swapping blocks out to host memory, it reads zero at every rate even
with the pool at 99.8 % occupancy -- it measures the absence of one recovery
mechanism, not the absence of pressure. The right panel therefore plots peak KV
pool occupancy, which is instrumented identically on both engines: the
uncompressed arms run the pool to saturation and must recover somehow, by
swapping on 0.4.1 and by recomputing on 0.8.5, while compression keeps occupancy
off the ceiling so neither recovery path is invoked.

Data: results/summaries/{b0,b1,c2_bf16,c1,c1c2}_*.json (left),
      {b0,b1a,c1}_04p.json + {b0,b1,c1}_v1p.json (right)
Run:  python scripts/fig6_preemptions.py
"""
import json
import os

import matplotlib.pyplot as plt

from _style import DOUBLE, STYLE, apply_style, layout, load, save

RATES = [4, 8, 16, 32, 64]

# Left: the same five configurations, and the same colours, as fig1 and fig2.
ARMS = [
    ("B0",     "b0_triton"),
    ("B1",     "b1_triton"),
    ("+C2",    "c2_bf16"),
    ("+C1",    "c1_tq4"),
    ("+C1+C2", "c1c2_tq4"),
]

# Right: uncompressed against compressed on each of the two older engines. Only
# two things vary, so colour carries compressed-or-not and the dash pattern
# carries the engine.
#
# These deliberately do NOT reuse the per-configuration STYLE colours: panel (a)
# uses those for the configurations they belong to, so two neutral hues keep the
# two panels from contradicting each other.
UNCOMP, COMP = "#8a6a1a", "#0f5f8f"
SUBS = [
    ("0.4.1 swap",      "b1a_04p", UNCOMP, "s", (0, (1, 1.6))),
    ("0.8.5 recompute", "b1_v1p",  UNCOMP, "o", (0, (4, 2.2))),
    ("0.4.1 swap",      "c1_04p",  COMP,   "s", (0, (1, 1.6))),
    ("0.8.5 recompute", "c1_v1p",  COMP,   "o", "-"),
]


def engine_of(config: str) -> str:
    """Read the version out of the summary so a label cannot drift from its data."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results", "summaries", f"{config}.json")) as fh:
        return json.load(fh).get("vllm_version", "?")


def main() -> None:
    apply_style()
    # Two panels of five points each: DOUBLE would be generous, and the section
    # is over budget, so the canvas is the short one.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(3.5, 3.75))

    for label, config in ARMS:
        data = load(config)
        ax.plot(RATES, [data[r]["preemptions"] for r in RATES],
                label=label, **STYLE[label])
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES)
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_ylabel("Preemptions/run")
    ax.set_ylim(-10, 200)                    # headroom so the legend clears the data
    # Short titles: at full width these two panels sit close enough that a long
    # title on (a) collides with the y-axis label of (b).
    ax.set_title("(a) preemptions, vLLM 0.25")


    for label, config, colour, marker, dash in SUBS:
        data = load(config)
        ax2.plot(RATES, [100 * data[r]["kv"]["peak_usage_frac"] for r in RATES],
                 label=label, color=colour, marker=marker, linestyle=dash)
    ax2.axhline(100, color="#9a3324", linewidth=0.9, linestyle=(0, (2, 2)), zorder=0)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(RATES)
    ax2.set_xticklabels([str(r) for r in RATES])
    ax2.set_xlabel("Request rate (req/s)")
    ax2.set_ylabel("KV occupancy (%)")
    ax2.set_ylim(28, 132)
    ax2.set_title("(b) peak cache occupancy")

    # The four arms form two tight bands, so a legend box large enough to name
    # them all covered most of the panel. Label each band directly instead, ABOVE
    # the curves it names -- placed below, the label for the upper band ended up
    # abutting the lower one.
    ax2.text(4.2, 101, "pool full", fontsize=10, color="#9a3324", va="bottom")
    ax2.annotate("uncompressed", xy=(16, 100), xytext=(0, 15),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=10, color=UNCOMP, fontweight="bold")
    ax2.annotate("compressed", xy=(16, 78), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom",
                 fontsize=10, color=COMP, fontweight="bold")
    # Which dash is which engine belongs in the caption, not inside the axes:
    # at 10 pt it is a long line, and every position for it either sat on a data
    # point or ran into the y-axis label.

    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=10, frameon=False, columnspacing=1.0, handletextpad=0.4,
               handlelength=1.5)
    layout(fig, pad=0.35, h_pad=0.9, rect=(0, 0.155, 1, 1))
    save(fig, "fig6_preemptions")


if __name__ == "__main__":
    main()
