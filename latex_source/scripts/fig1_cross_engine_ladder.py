"""Figure 1 — B0 -> B1 -> +C1 TTFT ladder on all three engine stacks.

The point of the figure: the ladder has the SAME SHAPE on every engine, so
"add capacity, avoid saturation" is an engine-independent lever rather than an
artefact of one vLLM version. The v0.25 panel additionally carries C2, which
exists only there (the 0.4.1 / 0.8.5 kernels were never patched).

Each panel keeps its own log axis: absolute TTFT is NOT comparable across
stacks (different preemption modes, different KV caps, and C1 is fp8 on the two
older stacks but real TurboQuant-4bit on v0.25). Shape is what transfers.

Data: results/summaries/{b0,b1a,c1}_04p.json, {b0,b1,c1}_v1p.json,
      {b0_triton,b1_triton,c2_bf16,c1_tq4,c1c2_tq4}.json
Run:  python scripts/fig1_cross_engine_ladder.py
"""
import matplotlib.pyplot as plt

from _style import DOUBLE, STYLE, apply_style, layout, save, series

RATES = [4, 8, 16, 32, 64]

PANELS = [
    ("vLLM 0.4.1: swap", [
        ("B0",  "b0_04p"),
        ("B1",  "b1a_04p"),
        ("+C1", "c1_04p"),
    ]),
    ("vLLM 0.8.5: recompute", [
        ("B0",  "b0_v1p"),
        ("B1",  "b1_v1p"),
        ("+C1", "c1_v1p"),
    ]),
    ("vLLM 0.25: TurboQuant", [
        ("B0",     "b0_triton"),
        ("B1",     "b1_triton"),
        ("+C2",    "c2_bf16"),
        ("+C1",    "c1_tq4"),
        ("+C1+C2", "c1c2_tq4"),
    ]),
]


def main() -> None:
    apply_style()
    # sharey keeps the three panels on one scale so the shapes are comparable;
    # the tick labels are repeated on every panel so no panel can be read
    # without an axis of its own.
    # DOUBLE, not DOUBLE_TALL: three panels of five points each do not need the
    # extra height, and the Evaluation section is over its page budget.
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE[0], 2.25), sharey=True)
    for ax in axes:
        ax.tick_params(axis="y", labelleft=True)

    handles = {}
    for ax, (title, arms) in zip(axes, PANELS):
        for label, config in arms:
            line, = ax.plot(RATES, series(config, "ttft_ms", "mean", RATES),
                            label=label, **STYLE[label])
            handles.setdefault(label, line)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(RATES)
        ax.set_xticklabels([str(r) for r in RATES])
        ax.set_xlabel("Request rate (req/s)")

    axes[0].set_ylabel("Mean TTFT (ms)")

    # One legend below the axes rather than three inside them. In-panel legends
    # had to be opaque to stop curves running through the text, and an opaque box
    # over a rising curve does not obscure the line, it deletes it: the third
    # panel lost +C1 and +C1+C2 across the whole 16-32 interval. Outside the axes
    # nothing can be hidden, and the three panels share one configuration set
    # anyway, so a single key is also less to read.
    order = ["B0", "B1", "+C2", "+C1", "+C1+C2"]
    fig.legend([handles[k] for k in order if k in handles],
               [k for k in order if k in handles],
               loc="lower center", ncol=5, fontsize=10, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, borderaxespad=0.1,
               frameon=False)
    layout(fig, pad=0.4, w_pad=0.8, rect=(0, 0.085, 1, 1))
    save(fig, "fig1_cross_engine_ladder")


if __name__ == "__main__":
    main()
