"""The two Background schematics, drawn with matplotlib.

Both are system-architectural diagrams: labelled components and stores joined by
arrows that carry data. Every box is a component rather than a step, there are no
decision nodes, and every edge label names what travels on the edge ("victim",
"blocks", "append K,V") rather than a condition that triggers it.

The canvas is exactly one IEEE column wide, so \\includegraphics does not rescale
the drawing and its 10 pt type stays 10 pt on the page.

Run:  python scripts/arch_diagrams.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import apply_style, layout, save

COL = 3.5                      # one IEEE column, in inches -- never exceed
INK, MUTED = "#111417", "#4a545c"
FILL_STORE, FILL_PLAIN = "#e9edee", "#ffffff"


def box(ax, xy, w, h, text, *, fill=FILL_PLAIN, lw=1.0):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=INK, facecolor=fill,
                                zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=INK,
            zorder=3, linespacing=1.2)


def label(ax, xy, text, *, color=INK, width=None):
    ax.text(*xy, text, ha="center", va="center", fontsize=10, color=color,
            zorder=3, linespacing=1.2)


def arrow(ax, a, b, *, dashed=False, dotted=False, rad=0.0):
    style = "-"
    if dashed:
        style = (0, (4, 2.2))
    elif dotted:
        style = (0, (1, 2))
    SEGMENTS.append((a, b))
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.15, linestyle=style, color=INK,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2, zorder=1))


# Every segment drawn and every edge label placed, so check_labels() below can
# verify that no label sits on the line it names.
SEGMENTS: list = []
LABELS: list = []


def edge_label(ax, xy, text):
    """Opaque so an edge does not run through the words -- but placed beside the
    line, never on it, so the line is never visually severed."""
    LABELS.append(ax.text(*xy, text, ha="center", va="center", fontsize=10,
                          color=MUTED, zorder=4,
                          bbox=dict(boxstyle="round,pad=0.14", fc="white",
                                    ec="none", alpha=0.95)))


def _hits(p0, p1, box):
    """Liang-Barsky: does the segment p0->p1 cross the axis-aligned box?"""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - box.x0), (dx, box.x1 - p0[0]),
                 (-dy, p0[1] - box.y0), (dy, box.y1 - p0[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def check_labels(fig, ax, name):
    """An opaque label sitting on the line it names cuts that line in two, which
    reads as two unconnected stubs rather than one edge."""
    fig.canvas.draw()
    for text in LABELS:
        box = text.get_window_extent()
        for a, b in SEGMENTS:
            if _hits(ax.transData.transform(a), ax.transData.transform(b), box):
                print(f"  !! {name}: label {text.get_text()!r} sits on an edge")


def canvas(height):
    SEGMENTS.clear(); LABELS.clear()
    fig, ax = plt.subplots(figsize=(COL, height))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- Figure A
def elbow(ax, pts, *, dashed=False, dotted=False):
    """Route an edge along axis-parallel segments, arrowhead on the last one.

    Orthogonal routing keeps returning edges in their own channel instead of
    cutting diagonally across the boxes."""
    style = "-"
    if dashed:
        style = (0, (4, 2.2))
    elif dotted:
        style = (0, (1, 2))
    for a, b in zip(pts, pts[1:-1]):
        SEGMENTS.append((a, b))
        ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=style, linewidth=1.15,
                color=INK, zorder=1, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], dashed=dashed, dotted=dotted)


def preempt():
    fig, ax = canvas(3.2)
    W, H = 30, 9
    CX = 30                      # main column
    LANE_R, LANE_L = 90, 4       # return channels, clear of every box

    box(ax, (CX, 94), W, H, "Requests")
    box(ax, (CX, 79), W, H, "Scheduler\nFCFS or LTR")
    box(ax, (CX, 62), W, H, "Block allocator")
    box(ax, (CX, 45), W, H, "GPU KV cache\npaged blocks", fill=FILL_STORE)
    # A component that owns the two recovery paths, not an event that happens:
    # an event box with a condition on its inbound edge would be flowchart
    # grammar, and every box here has to be a thing that exists.
    box(ax, (CX, 26), W, H, "Eviction\nhandler")
    box(ax, (15, 8), 28, H, "Host RAM\nswapped", fill=FILL_STORE)
    box(ax, (61, 8), 30, H, "Prefill\nrecompute")
    box(ax, (72, 45), 30, H, "Decode step\nreads past KV")

    # allocation path, straight down the column
    arrow(ax, (CX, 89.5), (CX, 83.5))
    arrow(ax, (CX, 74.5), (CX, 66.5)); edge_label(ax, (46, 70.5), "next request")
    arrow(ax, (CX, 57.5), (CX, 49.5)); edge_label(ax, (42, 53.5), "blocks")

    # decode reads the pool; new K,V go back through the allocator via the right lane
    arrow(ax, (45, 45), (57, 45))
    elbow(ax, [(72, 49.5), (72, 62), (45, 62)], dashed=True)
    edge_label(ax, (60, 65.5), "append K,V")

    # the two recovery routes and their returns; every label names what travels
    # on the edge rather than a condition that triggers it
    arrow(ax, (CX, 40.5), (CX, 30.5)); edge_label(ax, (43, 35.5), "victim")
    arrow(ax, (24, 21.5), (17, 12.5))
    arrow(ax, (38, 21.5), (54, 12.5))
    elbow(ax, [(LANE_L + 2, 8), (LANE_L, 8), (LANE_L, 45), (15, 45)], dashed=True)
    edge_label(ax, (14, 35), "blocks")
    elbow(ax, [(74, 8), (LANE_R, 8), (LANE_R, 79), (45, 79)], dashed=True)
    edge_label(ax, (81, 30), "request")

    layout(fig, pad=0.12)
    check_labels(fig, ax, "arch_preempt")
    save(fig, "arch_preempt")


# ---------------------------------------------------------------- Figure B
def attention():
    fig, ax = canvas(3.2)
    CX = 40                      # the score/accumulate column

    box(ax, (CX, 88), 34, 9, "scores  q$\\cdot$K$_n$")
    box(ax, (CX, 70), 34, 9, "block maximum")
    box(ax, (CX, 30), 34, 11, "weight and\naccumulate")

    label(ax, (CX, 97.5), "query q")
    label(ax, (9, 88), "key K$_n$")
    label(ax, (9, 30), "value V$_n$")
    label(ax, (CX, 8), "attention output")

    # running state on the left, the threshold on the right of the value path
    box(ax, (17, 51), 30, 11, "running max\nand denom.", fill=FILL_STORE)
    box(ax, (82, 51), 28, 11, "sparsity\nthreshold", lw=1.9)

    arrow(ax, (CX, 94), (CX, 93))            # query in
    arrow(ax, (15, 88), (22.5, 88))          # key in
    arrow(ax, (CX, 83.5), (CX, 74.5))        # scores -> block maximum
    # block maximum feeds the running state (left) and the threshold (right),
    # each along its own channel rather than a diagonal. The left edge carries the
    # same value into a store already named "running max", so it needs no label;
    # the right one is where that value is used as a score, so that is where the
    # name goes.
    elbow(ax, [(23, 70), (17, 70), (17, 57)])
    elbow(ax, [(57, 70), (82, 70), (82, 57)])
    edge_label(ax, (70, 74.5), "block score")
    # The threshold compares against the running maximum, so this is the edge
    # that carries it.
    arrow(ax, (32, 51), (68, 51), dotted=True)
    edge_label(ax, (50, 55.5), "running max")
    elbow(ax, [(82, 45.5), (82, 30), (57.5, 30)])
    edge_label(ax, (68, 34), "kept block")
    arrow(ax, (15, 30), (22.5, 30))          # value in
    # Unlabelled on purpose: the left margin is too narrow for a label that does
    # not sit on this edge, and the store it leaves is already called "denom.",
    # so the only thing it can be carrying is named at both ends already.
    elbow(ax, [(17, 45.5), (17, 36), (23, 36)], dotted=True)
    arrow(ax, (CX, 24.5), (CX, 12))          # accumulate -> output

    ax.text(82, 17, "a block outside $\\tau$\nnever loads its values",
            ha="center", va="center", fontsize=10, color=MUTED, linespacing=1.2)

    layout(fig, pad=0.12)
    check_labels(fig, ax, "arch_attention")
    save(fig, "arch_attention")


if __name__ == "__main__":
    apply_style()
    plt.rcParams["font.size"] = 10
    preempt()
    attention()
