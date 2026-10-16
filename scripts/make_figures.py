#!/usr/bin/env python
"""Build every report figure from results/results.csv.

Writes results/figures/<name>.png and .pdf. Parts whose results are missing are skipped.

  python scripts/make_figures.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ann_bench.plotting import INK_SECONDARY, MODE_COLORS, apply_style, save_figure  # noqa: E402
from ann_bench.tables import read_results  # noqa: E402

DATASET_NAMES = {"sift": "SIFT-128", "glove": "GloVe-100"}


def fig_part1(res_dir, fig_dir):
    """Fig 1: exact-search QPS, single query vs batch, per dataset."""
    rows = read_results(os.path.join(res_dir, "results.csv"), "part1_exact")
    if not rows:
        return []
    datasets = [d for d in DATASET_NAMES if any(r["dataset"] == d for r in rows)]
    n_base = {r["dataset"]: int(r["n_base"]) for r in rows}

    def mode_of(r):
        if r["mode"] == "single":
            return "single, 1 thread"
        return "batch, all threads" if int(r["threads"]) > 1 else "batch, 1 thread"

    modes = list(MODE_COLORS)
    qps = {(r["dataset"], mode_of(r)): float(r["qps"]) for r in rows}

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    width = 0.26
    x = np.arange(len(datasets))
    for i, mode in enumerate(modes):
        vals = [qps.get((d, mode), np.nan) for d in datasets]
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=MODE_COLORS[mode], label=mode)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:,.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(x, [f"{DATASET_NAMES[d]}\n({n_base[d]:,} vectors)" for d in datasets])
    ax.set_ylabel("Queries per second")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("Exact search: one query at a time vs one batch", loc="left")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return save_figure(fig, "fig1_exact_single_vs_batch", fig_dir)


PARTS = {1: fig_part1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--parts", nargs="+", type=int, default=sorted(PARTS))
    args = p.parse_args()
    apply_style()
    fig_dir = os.path.join(args.results_dir, "figures")
    for part in args.parts:
        for path in PARTS[part](args.results_dir, fig_dir):
            print("wrote", path)


if __name__ == "__main__":
    main()
