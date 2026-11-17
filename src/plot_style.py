"""Matplotlib style for the notebooks, using the dashboard's light-mode colours."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = SERIES[0]

# sequential, light to dark
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

# shading for infection-favourable hours
WASH = (236 / 255, 131 / 255, 90 / 255, 0.18)


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "figure.figsize": (10, 4),
        "figure.constrained_layout.use": True,
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
    })


def subtitle(ax, text: str) -> None:
    """Secondary line under a left-aligned axes title (call after set_title)."""
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=22)
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def threshold(ax, y=None, x=None, label: str | None = None, **kw) -> None:
    """Dashed reference line at a threshold."""
    style = dict(color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1) | kw
    if y is not None:
        ax.axhline(y, **style)
        if label:
            ax.annotate(label, (1, y), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
    if x is not None:
        ax.axvline(x, **style)
        if label:
            ax.annotate(label, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                        textcoords="offset points", va="top", fontsize=8, color=MUTED)


def shade_runs(ax, x, mask, color=WASH) -> None:
    """Shade consecutive True runs of `mask` along datetime/float axis `x`."""
    m = np.asarray(mask, bool)
    xs = list(x)
    if len(xs) < 2:
        return
    step = xs[1] - xs[0]
    i = 0
    while i < len(m):
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(m) and m[j + 1]:
            j += 1
        ax.axvspan(xs[i] - step / 2, xs[j] + step / 2, color=color, lw=0, zorder=0)
        i = j + 1


plt.style.use("default")
apply()
