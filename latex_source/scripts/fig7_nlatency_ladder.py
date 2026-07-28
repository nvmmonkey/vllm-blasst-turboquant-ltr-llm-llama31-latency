"""Figure 7 — reproducing the prior work's result in its own metric and on its own stack.

The prior learning-to-rank study reports normalised latency (milliseconds per
generated token) on the vLLM 0.4.1 fork with swap preemption. We rebuilt that
stack, ran the released ShareGPT trace through it, and swept the arrival rate
past the point the original table stopped at.

Two things the figure shows that a single "2.1x" number cannot:

  * At light load all arms coincide, matching the original observation that the
    methods are indistinguishable below saturation.
  * The benefit is load-dependent. Ordering pays off around the saturation
    knee; once the machine is deeply overloaded the curves flatten and
    compression (C1) carries the improvement instead.

Data: results/summaries/reference_ladder.json
Run:  python scripts/fig7_nlatency_ladder.py
"""
import json
import os

import matplotlib.pyplot as plt

from _style import DOUBLE, STYLE, SUMMARIES, apply_style, layout, save

ARMS = [("B0", "b0"), ("B1", "b1a"), ("+C1", "c1")]
# Two-line y labels, and the same label on both panels: stacked, each panel is
# about 1.5 in tall, and a rotated label longer than that pokes past the figure
# edge, where matplotlib's tight bbox clips it rather than growing for it. The
# titles already say which statistic each panel shows, so repeating "Mean" and
# "P99" in the labels bought nothing and cost the line that got cut.
YLABEL = "Normalized latency\n(ms/token)"
PANELS = [("nlat_mean", YLABEL, "(a) mean"),
          ("nlat_p99",  YLABEL, "(b) P99 tail")]


def main() -> None:
    apply_style()
    with open(os.path.join(SUMMARIES, "reference_ladder.json")) as fh:
        doc = json.load(fh)

    # Stacked in ONE column rather than side by side across two. Three arms over
    # six rates does not need full width, and the Evaluation section has a hard
    # three-page budget: side by side this figure cost 6.8 column-inches, stacked
    # it costs about half that, with the same data and a shared x axis that makes
    # the mean/tail comparison easier to read vertically.
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.15), sharex=True)
    rates = sorted({e["rate"] for e in doc["arms"]["b0"]})

    for ax, (field, ylabel, title) in zip(axes, PANELS):
        for label, arm in ARMS:
            entries = sorted(doc["arms"][arm], key=lambda e: e["rate"])
            ax.plot([e["rate"] for e in entries], [e[field] for e in entries],
                    label=label, **STYLE[label])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(rates)
        ax.set_xticklabels([str(r) for r in rates])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    axes[-1].set_xlabel("Request rate (req/s)")
    # Below the axes, like every other multi-arm figure here. In panel (a) the
    # three curves converge into the bottom-left corner at rate 2, which is the
    # one place an in-panel legend fits -- and an opaque box there does not dim
    # the curves under it, it removes them.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               handlelength=1.8, columnspacing=1.2, handletextpad=0.4,
               borderaxespad=0.1, frameon=False)

    layout(fig, pad=0.3, h_pad=0.7, rect=(0, 0.075, 1, 1))
    save(fig, "fig7_nlatency_ladder")


if __name__ == "__main__":
    main()
