"""Shared figure style.

Colours come from a fixed, colour-vision-deficiency validated categorical
order (blue, orange, aqua, yellow, magenta, green, violet, red) and are
assigned by *meaning*, never by rank:

* cooling route: furnace cooling (FC) = blue, air cooling (AC) = orange,
  as-rolled (AR) = aqua, constant-rate dilatometry (CR) = violet
* constituents: ferrite = blue, pearlite = orange, bainite = aqua,
  martensite = yellow, proeutectoid cementite = magenta (stack order)

Marks are thin, grids are solid hairlines, every multi-series figure has a
legend, and no figure uses two y-axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SLOTS
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#b9b8b2"   # de-emphasised marks
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#f0efec", "#e34948")

ROUTE_COLORS = {"FC": BLUE, "AC": ORANGE, "AR": AQUA, "CR": VIOLET}
ROUTE_LABELS = {"FC": "Furnace cooled (annealed)", "AC": "Air cooled (normalized)",
                "AR": "As-rolled (air cooled)", "CR": "Constant-rate dilatometry"}
PHASE_COLORS = {"ferrite": BLUE, "pearlite": ORANGE, "bainite": AQUA, "martensite": YELLOW,
                "cementite": MAGENTA, "austenite": VIOLET}
PHASE_ORDER = ["ferrite", "pearlite", "cementite", "bainite", "martensite", "austenite"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def savefig(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def label_end(ax, x, y, text, color=INK2, dx=4, dy=0, ha="left", fontsize=9):
    """Direct label at a line end, in ink (not series) colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                color=color, fontsize=fontsize)


apply_style()
