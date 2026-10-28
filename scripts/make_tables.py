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

from ann_bench.drift_report import INDEX_ORDER as P4_ORDER, at_knob, knob_for_target, load_part4, shifts  # noqa: E402
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


P3_METHODS = [("IVFFlat", "IVF-Flat (reference)"), ("IVFPQ", "IVF-PQ (control)"),
              ("IVFPQ+rerank", "IVF-PQ + re-rank")]
P3_NPROBE = (4, 16, 64, 256)


def _p3_by_nprobe(cfgs):
    return {c["search"]["nprobe"]: c for c in cfgs}


def table_part3(res_dir, out_dir):
    groups = group_configs(read_results(os.path.join(res_dir, "results.csv"), "part3"), experiment="part3")
    if not groups:
        return []
    ds = next(iter(groups))[0]
    exact = _exact_single(res_dir)
    raw = exact[ds][2] / exact[ds][3] if ds in exact and exact[ds][2] else None
    ex_qps = exact.get(ds, (None,))[0]

    def builds(kind):
        out = {}
        for c in groups.get((ds, kind), []):
            out.setdefault(c["build"].get("m"), []).append(c)
        return sorted(out.items(), key=lambda kv: kv[0] or 0)

    # T3a: one row per (method, m): memory, build, best recall, QPS at the targets.
    t3a = []
    for kind, name in P3_METHODS:
        for m, cfgs in builds(kind):
            c0 = cfgs[0]
            best = max(cfgs, key=lambda c: c["test"]["recall"])
            row = [name, str(m) if m else "–", fmt(c0["bytes_per_vector"], 1),
                   f"{c0['bytes_per_vector'] / raw:.2f}×" if raw else "–", fmt(c0["build_time_s"], 1),
                   f"{best['test']['recall']:.4f} (nprobe={best['search']['nprobe']})"]
            for t in TARGETS:
                c = select_config(cfgs, t)
                if c is None:
                    row.append("not reached")
                else:
                    speed = f", {c['test']['qps'] / ex_qps:,.0f}×" if ex_qps else ""
                    row.append(f"{c['test']['qps']:,.0f} (nprobe={c['search']['nprobe']}{speed})")
            t3a.append(row)
    nlist = next(iter(groups.values()))[0]["build"]["nlist"]
    written = write_table(
        t3a, ["Method", "m", "Bytes / vector", "vs raw float32", "Build (s)", "Max test recall",
              "QPS @ 0.90", "QPS @ 0.95"],
        "t3a_compression_summary", out_dir,
        caption=f"Part 3, {DATASET_NAMES[ds]}, nlist={nlist}. Memory is the serialised index; the "
                "re-rank rows include the float32 original vectors held in RAM. QPS at a target: the "
                "fastest nprobe reaching it on the tune queries, reported on the test queries "
                "(× = speed-up over exact search).")

    # T3b: the same nprobe for every method -> gain from re-ranking, gap to IVF-Flat, QPS cost.
    ref = _p3_by_nprobe(groups.get((ds, "IVFFlat"), []))
    plain = dict(builds("IVFPQ"))
    rer = dict(builds("IVFPQ+rerank"))
    t3b = []
    for m in sorted(rer):
        p_by, r_by = _p3_by_nprobe(plain.get(m, [])), _p3_by_nprobe(rer[m])
        for npb in P3_NPROBE:
            if npb not in r_by:
                continue
            r, p, f = r_by[npb], p_by.get(npb), ref.get(npb)
            rr, cov = r["test"]["recall"], r["test"]["notes"].get("coverage")
            t3b.append([str(m), str(npb),
                        f"{p['test']['recall']:.4f}" if p else "–",
                        f"{rr:.4f}", f"{cov:.4f}" if cov is not None else "–",
                        f"{f['test']['recall']:.4f}" if f else "–",
                        f"{rr - p['test']['recall']:+.4f}" if p else "–",
                        f"{rr - f['test']['recall']:+.4f}" if f else "–",
                        f"{p['test']['qps']:,.0f}" if p else "–", f"{r['test']['qps']:,.0f}",
                        f"{r['test']['qps'] / p['test']['qps']:.2f}" if p else "–"])
    written += write_table(
        t3b, ["m", "nprobe", "IVF-PQ recall", "+ re-rank recall", "Coverage of top 100",
              "IVF-Flat recall", "Gain from re-rank", "Re-rank − IVF-Flat", "IVF-PQ QPS",
              "+ re-rank QPS", "QPS ratio"],
        "t3b_rerank_fixed_nprobe", out_dir, align="rr" + "r" * 9,
        caption="Part 3 at equal nprobe (same clusters scanned by all three methods; test queries). "
                "Coverage = share of the true top 10 anywhere in IVF-PQ's top 100 candidates; "
                "'–' = that method's sweep had stopped (recall plateau or >= 0.999).")
    return written


P4_KNOB = {"HNSW": "ef", "IVFFlat": "nprobe", "IVFPQ": "nprobe"}


def _p4_state_rows(states, summary):
    """Yield (kind, label, shift, state_name, st) in report order."""
    for kind in P4_ORDER:
        if (kind, "baseline", None) not in states:
            continue
        yield kind, "baseline (D)", None, "baseline", states[(kind, "baseline", None)]
        for s in shifts(summary):
            for name in ("incremental", "rebuilt"):
                st = states.get((kind, name, s))
                if st:
                    yield kind, name, s, name, st


def table_part4(res_dir, out_dir):
    states, summary = load_part4(res_dir)
    if not states:
        return []
    written = []
    shift_txt = lambda s: "–" if s is None else f"{s:g}σ"   # noqa: E731

    # T4a: everything at the fixed (baseline-chosen) knob.
    t4a = []
    for kind, label, s, _, st in _p4_state_rows(states, summary):
        fixed = summary["indexes"][kind]["fixed_knob"]
        o, n = at_knob(st, fixed, "old"), at_knob(st, fixed, "new")
        t4a.append([INDEX_NAMES[kind], f"{P4_KNOB[kind]}={fixed}", shift_txt(s), label,
                    f"{o['recall']:.4f}" if o else "–", f"{n['recall']:.4f}" if n else "–",
                    f"{o['qps']:,.0f}" if o else "–", f"{n['qps']:,.0f}" if n else "–",
                    f"{n['ndis']:,.0f}" if n else "–", fmt(st["maintenance_s"], 2),
                    fmt(st["size"] / 2**20, 1)])
    n_db, n_new = summary["n_db"], summary["n_new"]
    written += write_table(
        t4a, ["Index", "Fixed setting", "Shift s", "State", "Recall old", "Recall new", "QPS old",
              "QPS new", "Dist./query new", "Maintenance (s)", "Index (MB)"],
        "t4a_drift_fixed_setting", out_dir, align="llll" + "r" * 7,
        caption=f"Part 4: D = {n_db:,} SIFT vectors; a new cluster of {n_new:,} SIFT vectors moved by "
                f"s·σ (σ = {summary['sigma']:.0f}, RMS distance of D from its mean) is inserted (incremental: "
                "no retraining) or the index is rebuilt on D′. Each index keeps one search setting, chosen on "
                f"the baseline as the smallest reaching {summary['fixed_fraction']:.2f}× its best baseline "
                "recall. Old / new = original-region / new-region test queries; recall against exact search "
                "on the database searched. Maintenance = insertion time (incremental) or full build (rebuilt "
                "and baseline), all threads.")

    # T4b: mechanism statistics.
    t4b = []
    for kind, label, s, _, st in _p4_state_rows(states, summary):
        m = st["mechanism"]
        if kind == "HNSW":
            detail = (f"mean degree {m['mean_degree']:.1f}" if "new_links_to_new" not in m else
                      f"new→new links {m['new_links_to_new']:.1%}; old nodes linking new "
                      f"{m['old_nodes_linking_new']:.1%}; old→new links {m['old_to_new_links']:,}")
        else:
            detail = f"imbalance {m['imbalance']:.2f}; largest list {m['max_list']:,} (mean {m['mean_list']:.0f})"
            if m.get("n_new"):
                detail += (f"; 90% of new vectors in {m['lists_holding_90pct_new']:,} of {m['nlist']:,} lists; "
                           f"largest list {m['new_share_of_largest_list']:.0%} new")
            if kind == "IVFPQ":
                detail += (f"; PQ error new/old {m['pq_mse_ratio']:.2f}×" if "pq_mse_ratio" in m
                           else f"; PQ error (old) {m['pq_mse_old']:,.0f}")
        t4b.append([INDEX_NAMES[kind], shift_txt(s), label, detail])
    written += write_table(
        t4b, ["Index", "Shift s", "State", "Mechanism statistics"], "t4b_drift_mechanism", out_dir,
        align="lllp{10cm}",
        caption="Part 4 mechanism statistics. IVF imbalance = nlist·Σ size² / N² (1 = equal lists; ≈ expected "
                "vectors scanned per probe relative to equal lists). PQ error = mean squared reconstruction "
                "error of 5,000 new vectors divided by that of 5,000 old vectors. HNSW: level-0 links.")

    # T4c: cost of bringing new-region recall back to the target, by re-tuning.
    t4c = []
    for kind, label, s, name, st in _p4_state_rows(states, summary):
        target = summary["indexes"][kind]["target_recall"]
        k = knob_for_target(st, "new", target)
        cell = at_knob(st, k, "new") if k is not None else None
        best = max(c["new-test"]["recall"] for c in st["knobs"].values() if "new-test" in c)
        t4c.append([INDEX_NAMES[kind], f"{target:.3f}", shift_txt(s), label,
                    f"{P4_KNOB[kind]}={k}" if cell else f"not reached (best {best:.3f})",
                    f"{cell['recall']:.4f}" if cell else "–", f"{cell['qps']:,.0f}" if cell else "–",
                    f"{cell['ndis']:,.0f}" if cell else "–"])
    written += write_table(
        t4c, ["Index", "Target", "Shift s", "State", "Setting needed (new region)", "Test recall", "QPS",
              "Dist./query"], "t4c_drift_retune", out_dir, align="llll" + "r" * 4,
        caption="Part 4: the smallest search setting at which the new-region tune queries reach the index's "
                "target recall (the same target that fixed its setting on the baseline), and the new-region "
                "test recall and QPS there.")
    return written


PARTS = {0: table_machine, 1: table_part1, 2: table_part2, 3: table_part3, 4: table_part4}


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
