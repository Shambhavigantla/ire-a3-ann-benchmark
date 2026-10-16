"""Shared matplotlib style for every report figure.

Figures are static (PNG for viewing, PDF for the LaTeX report), light background.
Each entity keeps one fixed colour in every figure, so HNSW is always blue, etc.
The hues are the first slots of a colour-blind-validated categorical palette, in
fixed order; exact search is neutral grey because it is the reference, not a
competitor.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

# Categorical slots, in order. Never cycle; never reorder per figure.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Fixed entity -> colour (used across Parts 1-4).
COLORS = {
    "HNSW": SLOTS[0],
    "IVFFlat": SLOTS[1],
    "IVFPQ": SLOTS[2],
    "IVFPQ+rerank": SLOTS[3],
    "Flat": INK_SECONDARY,
}
# Part 1 timing modes.
MODE_COLORS = {
    "single, 1 thread": SLOTS[0],
    "batch, 1 thread": SLOTS[1],
    "batch, all threads": SLOTS[2],
}
MARKERS = {"HNSW": "o", "IVFFlat": "s", "IVFPQ": "^", "IVFPQ+rerank": "D", "Flat": "*"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "text.color": INK,
    })


def save_figure(fig, name: str, fig_dir: str) -> list[str]:
    """Save as <fig_dir>/<name>.png and .pdf; returns the paths."""
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
