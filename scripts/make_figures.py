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

from ann_bench.frontier import group_configs, pareto_frontier  # noqa: E402
from ann_bench.plotting import (COLORS, GRID, INK_SECONDARY, MARKERS, MODE_COLORS,  # noqa: E402
                                apply_style, save_figure)
from ann_bench.tables import read_results  # noqa: E402

DATASET_NAMES = {"sift": "SIFT-128", "glove": "GloVe-100"}
INDEX_ORDER = ["HNSW", "IVFFlat", "IVFPQ"]
INDEX_NAMES = {"HNSW": "HNSW", "IVFFlat": "IVF-Flat", "IVFPQ": "IVF-PQ"}


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


def _pareto_panel(ax, groups, ds, exact_qps):
    for kind in INDEX_ORDER:
        cfgs = groups.get((ds, kind), [])
        if not cfgs:
            continue
        runs = [{"recall": c["test"]["recall"], "qps": c["test"]["qps"]} for c in cfgs]
        col = COLORS[kind]
        # every configuration, faint: shows how much the build setting matters
        ax.scatter([r["recall"] for r in runs], [r["qps"] for r in runs], s=9, color=col,
                   alpha=0.25, linewidths=0, zorder=2)
        front = pareto_frontier(runs)
        ax.plot([r["recall"] for r in front], [r["qps"] for r in front], color=col,
                marker=MARKERS[kind], markersize=4, markeredgecolor="white", markeredgewidth=0.6,
                label=f"{INDEX_NAMES[kind]} (best trade-off)", zorder=3)
    if exact_qps:
        ax.axhline(exact_qps, color=COLORS["Flat"], linestyle="--", linewidth=1, zorder=1)
        ax.scatter([1.0], [exact_qps], marker=MARKERS["Flat"], s=70, color=COLORS["Flat"],
                   edgecolors="white", linewidths=0.6, zorder=4, label="Exact search")
        ax.annotate(f"exact: {exact_qps:,.0f} QPS", (0.0, exact_qps), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    for t in (0.90, 0.95):
        ax.axvline(t, color=GRID, linewidth=1, linestyle=":", zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10 (test queries)")
    ax.set_ylabel("Queries per second (1 thread, log)")
    ax.set_title(DATASET_NAMES[ds], loc="left")
    ax.set_xlim(right=1.005)


def fig_part2(res_dir, fig_dir):
    """Fig 2: recall@10 vs QPS, best trade-off curve per index + exact baseline, per dataset."""
    rows = read_results(os.path.join(res_dir, "results.csv"))
    groups = group_configs(rows)
    if not groups:
        return []
    exact = {r["dataset"]: float(r["qps"]) for r in rows
             if r["experiment"] == "part1_exact" and r["mode"] == "single"}
    datasets = [d for d in DATASET_NAMES if any(k[0] == d for k in groups)]
    paths = []
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        _pareto_panel(ax, groups, ds, exact.get(ds))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        paths += save_figure(fig, f"fig2_pareto_{ds}", fig_dir)
    if len(datasets) > 1:   # side-by-side version for the report
        fig, axes = plt.subplots(1, len(datasets), figsize=(8.6, 3.4), sharey=True)
        for i, (ax, ds) in enumerate(zip(axes, datasets)):
            _pareto_panel(ax, groups, ds, exact.get(ds))
            if i:
                ax.set_ylabel("")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=len(labels))
        paths += save_figure(fig, "fig2_pareto", fig_dir)
    return paths


PARTS = {1: fig_part1, 2: fig_part2}


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
