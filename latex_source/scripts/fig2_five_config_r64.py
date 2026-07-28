"""Figure 2 — the five-configuration comparison at the heaviest load (64 req/s).

This is the headline figure: it puts C1 and C2 side by side on the same engine
and shows that the two levers move DIFFERENT metrics.

  B0 / B1 / +C2  run on bf16 KV (no compression)
  +C1 / +C1+C2   run on TurboQuant-4bit KV

so C1 is exactly the bf16 -> TQ4 step. Read absolute values within a dtype
group; read across the groups as "what did compression buy, and what did it
cost". TTFT is on a log axis because B0 is two orders of magnitude above the
best arm; TPOT and e2e are linear.

Data: results/summaries/{b0_triton,b1_triton,c2_bf16,c1_tq4,c1c2_tq4}.json
Run:  python scripts/fig2_five_config_r64.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from _style import DOUBLE, STYLE, apply_style, layout, load, save

RATE = 64
ARMS = [
    ("B0",     "b0_triton",  "bf16"),
    ("B1",     "b1_triton",  "bf16"),
    ("+C2",    "c2_bf16",    "bf16"),
    ("+C1",    "c1_tq4",     "TQ4"),
    ("+C1+C2", "c1c2_tq4",   "TQ4"),
]
# All three panels are linear. TTFT was on a log axis to fit B0's two-order span,
# but a bar on a log axis encodes log(v) - log(baseline), not v: the 1747 ms bar
# rendered at about 2 % the height of the 12275 ms bar for a 7.0x ratio, which
# reads as "nearly zero" instead of "one seventh". Bar length must be
# proportional to the value it stands for, so the log axis has to go.
METRICS = [
    ("ttft_ms", "Mean TTFT (ms)",       False),
    ("tpot_ms", "Mean TPOT (ms/token)", False),
    ("e2e_ms",  "Mean end-to-end (ms)", False),
]


def main() -> None:
    apply_style()
    values = {cfg: load(cfg)[RATE] for _, cfg, _ in ARMS}
    labels = [label for label, _, _ in ARMS]
    colors = [STYLE[label]["color"] for label in labels]
    # hatch the compressed (TQ4) arms so the two dtype groups separate in print
    hatches = ["" if dtype == "bf16" else "//" for _, _, dtype in ARMS]

    # Rotated tick labels and a value on every bar need vertical room, but the
    # Evaluation section is over its page budget, so the canvas is trimmed to the
    # point where the annotations still clear the bars at 10 pt.
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE[0], 2.25))
    x = np.arange(len(ARMS))

    for ax, (metric, ylabel, log) in zip(axes, METRICS):
        heights = [values[cfg][metric]["mean"] for _, cfg, _ in ARMS]
        bars = ax.bar(x, heights, color=colors, width=0.68,
                      edgecolor="white", linewidth=0.6)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(top=max(heights) * 4.5)
            # a decade apart is too coarse here: add the 2x/5x ticks and print
            # them as plain numbers so the axis can be read without mental exp()
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0),
                                                  numticks=9))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(top=max(heights) * 1.26)
        # Adjacent bars of similar height would collide, so lift every second
        # label of such a pair instead of shrinking the type below 10 pt.
        offsets = [2.0] * len(heights)
        for i in range(1, len(heights)):
            lo, hi = sorted((heights[i - 1], heights[i]))
            if hi <= lo * 1.18 and offsets[i - 1] == 2.0:
                offsets[i] = 12.0
        for xi, h, dy in zip(x, heights, offsets):
            ax.annotate(f"{h:.0f}", (xi, h), ha="center", va="bottom",
                        fontsize=10, xytext=(0, dy), textcoords="offset points")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="x", visible=False)

    layout(fig, pad=0.4, w_pad=1.0)
    save(fig, "fig2_five_config_r64")


if __name__ == "__main__":
    main()
