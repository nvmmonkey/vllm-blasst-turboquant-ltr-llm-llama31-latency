"""Figure — the full TTFT distribution at 64 req/s (p25 through p99).

Mean and p99 alone hide the shape. Three things only the percentiles show:

  1. p25 is nearly configuration-independent: the fastest quarter of requests
     gets its first token quickly no matter what we do.
  2. B0 is bimodal -- a modest p50 next to an enormous p75, i.e. half the
     requests sail through and half are stuck behind head-of-line blocking.
  3. Scheduling wins the head and loses the tail: B1's p25 is the best of any
     arm while its p90/p99 explode, because shortest-job-first defers the long
     requests the engine then has to recompute. Compression flattens the WHOLE
     distribution instead of redistributing it.

Arms are the same five as the headline comparison and come from the same run
family, so a configuration means the same thing in both figures and the
distributions can be read against the means. The bf16 capacity control is an
attribution arm rather than a configuration, so it is quoted in the prose instead
of being plotted here.

Data: results/summaries/{b0_triton,b1_triton,c2_bf16,c1_tq4,c1c2_tq4}.json
Run:  python scripts/fig3_ttft_percentiles.py
"""
import matplotlib.pyplot as plt

from _style import SINGLE_TALL, STYLE, apply_style, layout, load, save

RATE = 64
PCTS = ["p25", "p50", "p75", "p90", "p99"]
ARMS = [
    ("B0",     "b0_triton"),
    ("B1",     "b1_triton"),
    ("+C2",    "c2_bf16"),
    ("+C1",    "c1_tq4"),
    ("+C1+C2", "c1c2_tq4"),
]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=SINGLE_TALL)
    x = range(len(PCTS))

    for label, config in ARMS:
        entry = load(config)[RATE]
        ax.plot(x, [entry["ttft_ms"][p] for p in PCTS], label=label,
                **STYLE[label])

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(PCTS)
    ax.set_xlabel("TTFT percentile at 64 req/s")
    ax.set_ylabel("TTFT (ms)")
    # Below the axes, not inside them: every curve rises left to right, so an
    # opaque in-panel box has to sit somewhere a curve will reach, and an opaque
    # box over a curve deletes it rather than merely obscuring it.
    fig.legend(loc="lower center", ncol=3, handlelength=1.5, fontsize=10,
               labelspacing=0.25, columnspacing=1.0, handletextpad=0.4,
               borderaxespad=0.1, frameon=False)
    layout(fig, pad=0.4, rect=(0, 0.135, 1, 1))
    save(fig, "fig3_ttft_percentiles")


if __name__ == "__main__":
    main()
