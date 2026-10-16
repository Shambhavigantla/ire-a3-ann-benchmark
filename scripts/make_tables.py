#!/usr/bin/env python
"""Build every report table from results/ (results.csv + JSON summaries).

Writes results/tables/<name>.md and .tex. Parts whose results are missing are skipped,
so this can run after any subset of the experiments.

  python scripts/make_tables.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ann_bench.frontier import group_configs, select_config  # noqa: E402
from ann_bench.tables import fmt, read_results, write_table  # noqa: E402

DATASET_NAMES = {"sift": "SIFT-128", "glove": "GloVe-100"}
INDEX_ORDER = ["HNSW", "IVFFlat", "IVFPQ"]
INDEX_NAMES = {"HNSW": "HNSW", "IVFFlat": "IVF-Flat", "IVFPQ": "IVF-PQ", "Flat": "Exact (Flat)"}
TARGETS = (0.90, 0.95)
MODE_NAMES = {("single", False): "single, 1 thread", ("batch", False): "batch, 1 thread",
              ("batch", True): "batch, all threads"}


def _cpu_range(cpus):
    if not cpus:
        return "none (not a hybrid CPU, or unknown)"
    return f"logical CPUs {cpus[0]}–{cpus[-1]}" if cpus == list(range(cpus[0], cpus[-1] + 1)) else str(cpus)


def table_machine(res_dir, out_dir):
    path = os.path.join(res_dir, "machine_info.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        m = json.load(f)
    rows = [
        ["CPU", m.get("processor", "")],
        ["Cores", f"{m.get('physical_cpus', '?')} physical / {m.get('logical_cpus', '?')} logical"],
        ["RAM", f"{m.get('ram_gb', '?')} GB"],
        ["OS", m.get("platform", "")],
        ["P-cores used for 1-thread timing", _cpu_range(m.get("performance_cores"))],
        ["Python", m.get("python", "")],
        ["Libraries", ", ".join(f"{k} {m[k]}" for k in ("faiss", "numpy", "h5py", "threadpoolctl", "matplotlib") if m.get(k))],
    ]
    return write_table(rows, ["Item", "Value"], "t0_machine", out_dir,
                       caption="Machine and library versions used for every run.", align="ll")


def table_part1(res_dir, out_dir):
    rows = read_results(os.path.join(res_dir, "results.csv"), "part1_exact")
    summary_path = os.path.join(res_dir, "part1_summary.json")
    if not rows or not os.path.exists(summary_path):
        return []
    with open(summary_path) as f:
        summary = {s["dataset"]: s for s in json.load(f)}
    written = []

    # T1a: exact index cost and agreement with the file's ground truth.
    t1a = []
    for ds, s in summary.items():
        c = s["check_vs_file"]
        top_key = next((k for k in c if k.endswith("_ids_identical_in_order_frac")), None)
        t1a.append([
            DATASET_NAMES.get(ds, ds),
            fmt(s["n_base"]),
            fmt(c.get("n_queries_checked")),
            fmt(c.get("recall_tie_aware_mean"), 4),
            fmt(c.get("recall_strict_mean"), 4) if "recall_strict_mean" in c else "–",
            fmt(c.get("queries_where_only_ties_differ")) if "queries_where_only_ties_differ" in c else "–",
            f"{100 * c[top_key]:.2f}%" if top_key else "–",
            f"{c['distance_max_rel_err']:.1e}" if "distance_max_rel_err" in c else "–",
            fmt(s["build_time_s"], 3),
            fmt(s["index_bytes"] / 2**20, 1),
        ])
    written += write_table(
        t1a,
        ["Dataset", "Base vectors", "Queries checked", "Recall@10 (ties)", "Recall@10 (strict)",
         "Differ only by ties", "Top-100 identical", "Max rel. dist. err.", "Build (s)", "Index (MB)"],
        "t1a_exact_check", out_dir,
        caption="Exact search (IndexFlatL2) against the ground truth in the data file.")

    # T1b: single-query vs batch timing on the test queries.
    single_qps = {r["dataset"]: float(r["qps"]) for r in rows if r["mode"] == "single"}
    t1b = []
    for r in rows:
        multi = r["mode"] == "batch" and int(r["threads"]) > 1
        t1b.append([
            DATASET_NAMES.get(r["dataset"], r["dataset"]),
            MODE_NAMES[(r["mode"], multi)],
            fmt(int(r["threads"])),
            fmt(int(r["n_queries"])),
            f"{float(r['qps']):,.1f}",
            fmt(float(r["latency_ms"]), 3),
            f"{float(r['qps']) / single_qps[r['dataset']]:.1f}×",
            fmt(float(r["recall_at_k"]), 4),
            fmt(float(r["cpu_per_wall"]), 2) if r.get("cpu_per_wall") else "–",
        ])
    written += write_table(
        t1b, ["Dataset", "Mode", "Threads", "Queries", "QPS", "ms / query", "Speed-up", "Recall@10", "CPU / wall"],
        "t1b_exact_timing", out_dir,
        caption="Exact search speed, median of 3 repeats, test queries. Speed-up is relative to "
                "single-query. 1-thread runs are pinned to P-cores with OpenMP and BLAS capped at 1 thread; "
                "CPU / wall ≈ 1 confirms a single busy thread.")
    return written


def _exact_single(res_dir):
    """{dataset: (qps, build_s, index_bytes, n_base)} from Part 1."""
    out = {}
    summ = {}
    path = os.path.join(res_dir, "part1_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            summ = {s["dataset"]: s for s in json.load(f)}
    for r in read_results(os.path.join(res_dir, "results.csv"), "part1_exact"):
        if r["mode"] == "single":
            s = summ.get(r["dataset"], {})
            out[r["dataset"]] = (float(r["qps"]), s.get("build_time_s"), s.get("index_bytes"), int(r["n_base"]))
    return out


def table_part2(res_dir, out_dir):
    groups = group_configs(read_results(os.path.join(res_dir, "results.csv"), "part2"))
    if not groups:
        return []
    exact = _exact_single(res_dir)
    datasets = [d for d in DATASET_NAMES if any(k[0] == d for k in groups)]
    written = []

    # T2a: QPS at recall targets (chosen on tune, reported on test).
    t2a = []
    for ds in datasets:
        ex_qps = exact.get(ds, (None,))[0]
        for kind in INDEX_ORDER:
            cfgs = groups.get((ds, kind))
            if not cfgs:
                continue
            row = [DATASET_NAMES[ds], INDEX_NAMES[kind]]
            for t in TARGETS:
                c = select_config(cfgs, t)
                if c is None:
                    best = max(c2["test"]["recall"] for c2 in cfgs)
                    row += [f"not reached (max {best:.3f})", "–", "–"]
                else:
                    speed = f" ({c['test']['qps'] / ex_qps:,.0f}×)" if ex_qps else ""
                    row += [f"{c['test']['qps']:,.0f}{speed}", f"{c['test']['recall']:.4f}", c["label"]]
            t2a.append(row)
        if ds in exact:
            t2a.append([DATASET_NAMES[ds], INDEX_NAMES["Flat"], f"{exact[ds][0]:,.1f} (1×)", "1.0000", "–",
                        f"{exact[ds][0]:,.1f} (1×)", "1.0000", "–"])
    written += write_table(
        t2a, ["Dataset", "Index", "QPS @ 0.90", "Test recall", "Settings @ 0.90",
              "QPS @ 0.95", "Test recall", "Settings @ 0.95"],
        "t2a_qps_at_recall", out_dir,
        caption="Single-thread, one-query-at-a-time QPS at recall@10 targets. Settings are the fastest "
                "reaching the target on the tune queries; QPS and recall are on the test queries "
                "(× = speed-up over exact search).")

    # T2b: cost of the configuration chosen at 0.95.
    t2b = []
    for ds in datasets:
        for kind in INDEX_ORDER:
            cfgs = groups.get((ds, kind))
            if not cfgs:
                continue
            c = select_config(cfgs, 0.95)
            note = ""
            if c is None:   # fall back to the most accurate config, flagged
                c = max(cfgs, key=lambda c2: c2["tune"]["recall"])
                note = f" (0.95 not reached; best recall {c['test']['recall']:.3f})"
            n = c["notes"]
            raw = None
            if ds in exact and exact[ds][2]:
                raw = exact[ds][2] / exact[ds][3]
            t2b.append([DATASET_NAMES[ds], INDEX_NAMES[kind], c["label"] + note,
                        fmt(c["build_time_s"], 1), fmt(n.get("train_s"), 1) if n.get("train_s") else "–",
                        fmt(c["index_bytes"] / 2**20, 1), fmt(c["bytes_per_vector"], 1),
                        f"{c['bytes_per_vector'] / raw:.2f}×" if raw else "–"])
        if ds in exact and exact[ds][1] is not None:
            qps, b, size, nb = exact[ds]
            t2b.append([DATASET_NAMES[ds], INDEX_NAMES["Flat"], "–", fmt(b, 2), "–",
                        fmt(size / 2**20, 1), fmt(size / nb, 1), "1.00×"])
    written += write_table(
        t2b, ["Dataset", "Index", "Settings", "Build (s)", "of which train (s)", "Index (MB)",
              "Bytes / vector", "vs raw float32"],
        "t2b_cost_at_095", out_dir,
        caption="Build time (all threads), serialised index size and bytes per vector for the "
                "configuration chosen at recall 0.95.")

    # T2c: best recall each build configuration reaches (explains failures to reach 0.95).
    t2c = []
    for ds in datasets:
        for kind in INDEX_ORDER:
            by_build = {}
            for c in groups.get((ds, kind), []):
                by_build.setdefault(json.dumps(c["build"], sort_keys=True), []).append(c)
            for _, cs in sorted(by_build.items(), key=lambda kv: kv[1][0]["bytes_per_vector"]):
                best = max(cs, key=lambda c: c["test"]["recall"])
                t2c.append([DATASET_NAMES[ds], INDEX_NAMES[kind], best["label"],
                            f"{best['test']['recall']:.4f}", f"{best['test']['qps']:,.0f}",
                            fmt(best["build_time_s"], 1), fmt(best["bytes_per_vector"], 1)])
    written += write_table(
        t2c, ["Dataset", "Index", "Most accurate setting", "Test recall", "QPS", "Build (s)", "Bytes / vector"],
        "t2c_max_recall", out_dir,
        caption="Highest test recall@10 reached by each build configuration in the sweep.")
    return written


PARTS = {0: table_machine, 1: table_part1, 2: table_part2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--parts", nargs="+", type=int, default=sorted(PARTS))
    args = p.parse_args()
    out_dir = os.path.join(args.results_dir, "tables")
    for part in args.parts:
        for path in PARTS[part](args.results_dir, out_dir):
            print("wrote", path)


if __name__ == "__main__":
    main()
