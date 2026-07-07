"""Shared publication style for ESWA/ACES figures.

One place to set typography, palette, and export so every figure looks
consistent and journal-grade (vector PDF + 600-dpi PNG). Import and call
`apply()` at the top of a figure script; build axes; finish with `save()`.

    from src.visualization import paperstyle as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ...
    ps.save(fig, "results/.../figures/Fig6")     # writes .pdf and .png

Palette is fixed across the paper so a method keeps its colour everywhere:
    raw = neutral grey, split = blue, aci = accent red.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fixed method palette (carried over from make_paper_figures.py)
PALETTE = {"raw": "#9aa0a6", "split": "#4c72b0", "aci": "#c44e52"}
LABEL = {"raw": "Raw (Gaussian)", "split": "Split-conformal",
         "aci": "Online ACI (ACES)"}
ACTUAL_C = "#222222"


def apply():
    """Set global rcParams for a clean, consistent journal style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",          # no CJK needed (sites romanised)
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.6,
    })


def save(fig, path_noext):
    """Write both vector (.pdf) and raster (.png) versions."""
    fig.savefig(f"{path_noext}.pdf")
    fig.savefig(f"{path_noext}.png")
    plt.close(fig)
