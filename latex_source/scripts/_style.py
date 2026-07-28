"""Shared plot style and data loaders for the final-report figures.

Every figure script in this directory imports from here so the whole figure set
reads as one system: same fonts, same colour per configuration, same sizing.

Style notes tied to the report format:
  * IEEE two-column: SINGLE = one column wide, DOUBLE = full page width.
    Each PDF is written at its natural size, so \\includegraphics does not
    rescale it and the 10 pt type in the figure stays 10 pt on the page.
  * All type is >= 10 pt.
  * Each configuration keeps one colour, marker and linestyle across every
    figure; the marker/linestyle pairing also works in greyscale.
"""
from __future__ import annotations

import json
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")           # headless: write files, never open a window
import matplotlib.pyplot as plt

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SUMMARIES = os.path.join(ROOT, "results", "summaries")
OUTDIR = os.path.join(ROOT, "results", "figures")

# --- canvas sizes (inches) for the IEEE conference template ----------------
# Kept short so the Evaluation section stays inside its page budget.
SINGLE = (3.5, 2.25)            # one column
SINGLE_TALL = (3.5, 2.6)
DOUBLE = (7.16, 2.55)           # full width (figure* environment)
DOUBLE_TALL = (7.16, 3.05)

# --- one colour/marker/linestyle per configuration -------------------------
STYLE = {
    "B0":        dict(color="#9a3324", marker="o", linestyle="--"),
    "B1":        dict(color="#b8860b", marker="s", linestyle="-"),
    "+C1":       dict(color="#0f766e", marker="^", linestyle="-"),
    "+C2":       dict(color="#6a5acd", marker="v", linestyle=":"),
    "+C1+C2":    dict(color="#1b7a4b", marker="D", linestyle="-"),
    "ctrl":      dict(color="#555555", marker="x", linestyle="-."),
}


def apply_style() -> None:
    """Set the rcParams every figure shares. Call once at the top of a script."""
    plt.rcParams.update({
        "font.family": "serif",              # matches the IEEE body text
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,                     # report format: nothing below 10 pt
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        # Opaque white with a hairline border: an unframed legend lets curves
        # run through the labels, and a handle on a curve reads as a data point.
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": "#c8cfd4",
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.3,
        "legend.handletextpad": 0.5,
        "figure.autolayout": False,
    })


# LaTeX targets. A figure included at exactly its natural width is not rescaled,
# which is the only way to be sure 10 pt type stays 10 pt on the page.
COLUMN_IN = 3.5000                          # \columnwidth, IEEEtran conference
TEXT_IN = 7.0 + 12 / 72                     # \textwidth + \columnsep


def layout(fig, **kw) -> None:
    """`fig.tight_layout(**kw)`, remembering the arguments on the figure.

    tight_layout stores its result as subplot fractions, not inches, so when
    save() shrinks the canvas those fractions still hold: the margins shrink but
    the type does not, and labels can run off the edge. Keeping the arguments
    here lets save() re-solve the layout at the size that actually ships.
    """
    fig._tight_kw = kw
    fig.tight_layout(**kw)


def _pdf_width_in(path: str) -> float:
    with open(path, "rb") as fh:
        blob = fh.read()
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blob)
    x0, _, x1, _ = (float(v) for v in box.groups())
    return (x1 - x0) / 72.0


def _clipped_text(fig):
    """Names of any text artists that stick out past the figure edge.

    A rotated axis label longer than its panel runs off the canvas, and the tight
    bbox crops it instead of growing for it, so the label silently loses a word.
    """
    fig.canvas.draw()
    page = fig.bbox
    out = []
    for ax in fig.axes:
        for artist, what in ((ax.yaxis.label, "ylabel"), (ax.xaxis.label, "xlabel"),
                             (ax.title, "title")):
            if not artist.get_text():
                continue
            box = artist.get_window_extent()
            if (box.x0 < page.x0 - 1 or box.y0 < page.y0 - 1
                    or box.x1 > page.x1 + 1 or box.y1 > page.y1 + 1):
                out.append(f"{what} {artist.get_text()!r}")
    return out


def save(fig, name: str) -> str:
    """Write <name>.pdf (vector, for LaTeX) and <name>.png (for quick viewing).

    `savefig.bbox="tight"` crops to the drawn extent, so the width on disk is
    whatever the labels needed, often a hair over one column; \\includegraphics
    would then scale the figure down and take the 10 pt type below the floor.
    Save once, measure, and if it overshoots, shrink the canvas by that ratio and
    save again, so the final file is at or just under its target width.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    pdf = os.path.join(OUTDIR, name + ".pdf")
    fig.savefig(pdf)

    target = TEXT_IN if fig.get_size_inches()[0] > 5 else COLUMN_IN
    ceiling = target * 0.99          # 1 % margin, so rounding never lands above
    actual = _pdf_width_in(pdf)
    for _ in range(3):               # re-solving changes the extent, so iterate
        if actual <= ceiling:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * ceiling / actual, h * ceiling / actual)
        # Re-solve: the old fractions were computed for the old canvas.
        fig.tight_layout(**getattr(fig, "_tight_kw", {}))
        fig.savefig(pdf)
        actual = _pdf_width_in(pdf)

    fig.savefig(os.path.join(OUTDIR, name + ".png"))
    clipped = _clipped_text(fig)
    plt.close(fig)

    # Mirror into latex_source so the report never compiles a stale figure.
    latex_dir = os.path.join(ROOT, "latex_source", "figures")
    if os.path.isdir(latex_dir):
        shutil.copyfile(pdf, os.path.join(latex_dir, name + ".pdf"))

    flag = "" if actual <= target + 1e-4 else f"  !! {actual:.3f}in > {target:.3f}in"
    for item in clipped:
        flag += f"\n  !! runs off the canvas: {item}"
    print(f"wrote {os.path.relpath(pdf, ROOT)}  ({actual:.3f} in){flag}")
    return pdf


# --- data loading ----------------------------------------------------------
def load(config: str) -> dict:
    """Load one committed sweep summary, keyed by request rate.

    Returns {rate: entry}, where entry holds ttft_ms / tpot_ms / e2e_ms
    (each with mean + p25/p50/p75/p90/p99), throughput, kv, batch, preemptions.
    """
    path = os.path.join(SUMMARIES, config + ".json")
    with open(path) as fh:
        doc = json.load(fh)
    out = {}
    for entry in doc.get("rates", []):
        out[int(float(entry["request_rate"]))] = entry
    return out


def series(config: str, metric: str, stat: str = "mean", rates=(4, 8, 16, 32, 64)):
    """Pull one metric across rates, e.g. series('c1_tq4', 'ttft_ms', 'p99')."""
    data = load(config)
    return [data[r][metric][stat] for r in rates]


def scalar(config: str, field: str, rate: int = 64):
    """Pull a non-nested field at one rate, e.g. scalar('c1_tq4', 'preemptions')."""
    return load(config)[rate][field]
