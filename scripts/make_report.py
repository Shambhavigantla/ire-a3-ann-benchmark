#!/usr/bin/env python
"""Build the report (report/report.html -> report.pdf) from results/.

Every number in the report is computed here from results.csv and the JSON summaries
and substituted into report/report_template.html (string.Template, ${name}), so
`bash run_all.sh` regenerates the report together with the tables and figures.
The PDF is printed with a headless Chromium browser (Edge or Chrome) if one is found;
otherwise only the HTML is written.

  python scripts/make_report.py [--results-dir results] [--out-dir report]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ann_bench.drift_report import at_knob, knob_for_target, load_part4  # noqa: E402
from ann_bench.frontier import group_configs, select_config  # noqa: E402
from ann_bench.tables import read_results  # noqa: E402

DS = {"sift": "SIFT", "glove": "GloVe"}
IDX = {"HNSW": "HNSW", "IVFFlat": "IVF-Flat", "IVFPQ": "IVF-PQ"}
P4_KNOB = {"HNSW": "ef", "IVFFlat": "nprobe", "IVFPQ": "nprobe"}


# ---------------------------------------------------------------- formatting
def n0(x):
    return f"{x:,.0f}"


def n1(x):
    return f"{x:,.1f}"


def r3(x):
    return f"{x:.3f}"


def r4(x):
    return f"{x:.4f}"


def signed(x, d=3):
    s = f"{x:+.{d}f}"
    return "0." + "0" * d if float(s) == 0 else s


def span(lo, hi, d):
    a, b = f"{lo:.{d}f}", f"{hi:.{d}f}"
    return a if a == b else f"{a}–{b}"


def times(x, d=1):
    return f"{x:.{d}f}×"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------- sections
def machine(res, V):
    m = json.load(open(res / "machine_info.json"))
    pc = m.get("performance_cores") or []
    V.update(cpu=m["processor"], phys=m["physical_cpus"], logical=m["logical_cpus"], ram=m["ram_gb"],
             os=m["platform"].split("-")[0] + " " + m["platform"].split("-")[1], py=m["python"],
             faiss=m["faiss"], numpy=m["numpy"], h5py=m["h5py"], threadpoolctl=m["threadpoolctl"],
             matplotlib=m["matplotlib"], pcores=f"{pc[0]}–{pc[-1]}" if pc else "all")


def part1(res, V):
    summ = {s["dataset"]: s for s in json.load(open(res / "part1_summary.json"))}
    rows = []
    for ds, s in summ.items():
        t = {(x["mode"], x["threads"] > 1): x for x in s["timings"]}
        single, b1, ball = t[("single", False)], t[("batch", False)], t[("batch", True)]
        chk = s["check_vs_file"]
        V[f"{ds}_n"] = n0(s["n_base"])
        V[f"{ds}_rec_tie"] = r4(chk["recall_tie_aware_mean"])
        # a --quick subset is only checked against queries whose file neighbours are all in the subset
        if "recall_strict_mean" in chk:
            V[f"{ds}_rec_strict"] = f"{chk['recall_strict_mean']:.5f}"
            V[f"{ds}_ties"] = n0(chk["queries_where_only_ties_differ"])
            V[f"{ds}_relerr"] = f"{chk['distance_max_rel_err']:.1e}"
        V[f"{ds}_exact_qps"] = n1(single["qps"])
        V[f"{ds}_exact_ms"] = n1(single["latency_ms"])
        V[f"{ds}_b1_speed"] = times(b1["qps"] / single["qps"])
        V[f"{ds}_ball_over_b1"] = times(ball["qps"] / b1["qps"])
        rows.append([f"{DS[ds]} ({n0(s['n_base'])} × {s['dim']})",
                     f"{n1(single['qps'])} ({single['latency_ms']:.1f} ms)",
                     f"{n0(b1['qps'])} ({b1['latency_ms']:.2f} ms)", f"{b1['qps'] / single['qps']:.1f}×",
                     f"{n0(ball['qps'])} ({ball['threads']} thr.)", f"{ball['qps'] / b1['qps']:.1f}×"])
        if ds == "sift":
            gb = s["index_bytes"] / (single["latency_ms"] / 1e3) / 1e9
            flop = 2.0 * s["n_base"] * s["dim"]
            V["sift_bw"] = f"{gb:.0f}"
            V["sift_mb"] = n0(s["index_bytes"] / 2**20)
            V["sift_gflops"] = f"{flop / (b1['latency_ms'] / 1e3) / 1e9:.0f}"
            V["sift_mflop"] = f"{flop / 1e6:.0f}"
    V["T1"] = table(["Dataset", "Single query, 1 thread", "Batch of 5,000, 1 thread", "Speed-up",
                     "Batch, all threads", "vs 1 thread"], rows)


def part2(res, V, rows_all):
    groups = group_configs([r for r in rows_all if r["experiment"] == "part2"])
    summ = {s["dataset"]: s for s in json.load(open(res / "part2_summary.json"))}
    p2rows = [r for r in rows_all if r["experiment"] == "part2"]
    V["p2_rows"] = n0(len(p2rows))
    V["p2_settings"] = n0(len({(r["dataset"], r["index_type"], r["build_params"], r["search_params"]) for r in p2rows}))
    cw = [float(r["cpu_per_wall"]) for r in p2rows]
    V["p2_cw"] = f"{min(cw):.2f}–{max(cw):.2f}"
    V["overhead_us"] = f"{1e6 / statistics.mean(s['call_overhead_qps'] for s in summ.values()):.1f}"
    T2 = []
    for ds in ("sift", "glove"):
        ex = summ[ds]["exact_single_qps"]
        best = {}
        for kind in ("HNSW", "IVFFlat", "IVFPQ"):
            cfgs = groups.get((ds, kind), [])
            c90, c95 = select_config(cfgs, 0.90), select_config(cfgs, 0.95)
            mx = max(cfgs, key=lambda c: c["test"]["recall"])
            V[f"{ds}_{kind}_max"] = r3(mx["test"]["recall"])
            V[f"{ds}_{kind}_max_label"] = mx["label"]
            if c95:
                best[kind] = c95
                V[f"{ds}_{kind}_qps95"] = n0(c95["test"]["qps"])
                V[f"{ds}_{kind}_x95"] = n0(c95["test"]["qps"] / ex)
                V[f"{ds}_{kind}_set95"] = c95["label"]
            cost = c95 or mx
            q90 = f"{n0(c90['test']['qps'])} ({r3(c90['test']['recall'])})" if c90 else "–"
            q95 = f"{n0(c95['test']['qps'])} ({r3(c95['test']['recall'])})" if c95 else f"not reached (max {r3(mx['test']['recall'])})"
            T2.append([DS[ds], IDX[kind], q90, c90["label"] if c90 else "–", q95,
                       cost["label"] + ("" if c95 else " *"), n1(cost["build_time_s"]),
                       n0(cost["index_bytes"] / 2**20), n0(cost["bytes_per_vector"])])
        raw = 4 * {"sift": 128, "glove": 100}[ds]
        T2.append([DS[ds], "Exact", f"{n1(ex)} (1.000)", "–", f"{n1(ex)} (1.000)", "–",
                   n1(float(summ_p1(res)[ds]["build_time_s"])), n0(summ_p1(res)[ds]["index_bytes"] / 2**20), n0(raw)])
        win = max(best, key=lambda k: best[k]["test"]["qps"])
        V[f"{ds}_win95"] = IDX[win]
        if "IVFFlat" in best and "HNSW" in best:
            V[f"{ds}_hnsw_over_ivf95"] = times(best["HNSW"]["test"]["qps"] / best["IVFFlat"]["test"]["qps"])
        # memory per vector of chosen configs
        for kind in ("HNSW", "IVFFlat"):
            if kind in best:
                V[f"{ds}_{kind}_bpv95"] = n0(best[kind]["bytes_per_vector"])
                V[f"{ds}_{kind}_build95"] = n0(best[kind]["build_time_s"])
        V[f"{ds}_HNSW_ef95"] = str(best["HNSW"]["search"]["efSearch"]) if "HNSW" in best else "–"
        V[f"{ds}_HNSW_ms95"] = f"{1e3 / best['HNSW']['test']['qps']:.2f}" if "HNSW" in best else "–"
        V[f"{ds}_IVF_np95"] = str(best["IVFFlat"]["search"]["nprobe"]) if "IVFFlat" in best else "–"
    V["T2"] = table(["Data", "Index", "QPS @0.90 (test recall)", "Setting @0.90", "QPS @0.95 (test recall)",
                     "Setting @0.95", "Build (s)", "Size (MB)", "B / vec"], T2, "small")
    # bytes per vector by family on SIFT (any config)
    b = {}
    for kind in ("HNSW", "IVFFlat", "IVFPQ"):
        for c in groups.get(("sift", kind), []):
            b.setdefault((kind, json.dumps(c["build"], sort_keys=True)), (c["build"], c["bytes_per_vector"], c["build_time_s"]))
    hn = {v[0]["M"]: v[1] for k, v in b.items() if k[0] == "HNSW"}
    pq = {v[0]["m"]: v[1] for k, v in b.items() if k[0] == "IVFPQ" and v[0]["nlist"] == 4096}
    V["hnsw16_bpv"], V["hnsw32_bpv"] = n0(hn.get(16, 0)), n0(hn.get(32, 0))
    V["pq64_bpv"] = n0(pq.get(64, 0))
    V["pq_saving"] = pct(1 - pq.get(64, 0) / 512) if pq.get(64) else "–"
    V["pq_1e8_gb"] = n1(1e8 * pq.get(64, 0) / 1e9) if pq.get(64) else "–"
    ratios = []
    for ds, raw in (("sift", 512), ("glove", 400)):
        for c in groups.get((ds, "HNSW"), []):
            if c["build"]["M"] >= 16:
                ratios.append(c["bytes_per_vector"] / raw)
    V["hnsw_raw_x"] = f"{min(ratios):.1f}–{max(ratios):.1f}×" if ratios else "–"
    ivf_bt = {v[0]["nlist"]: v[2] for k, v in b.items() if k[0] == "IVFFlat"}
    V["ivf_bt_small"] = f"{min(ivf_bt.get(1024, 0), ivf_bt.get(4096, 0)):.0f}–{max(ivf_bt.get(1024, 0), ivf_bt.get(4096, 0)):.0f}"
    V["ivf_bt_16k"] = n0(ivf_bt.get(16384, 0))
    hbt = [v[2] for k, v in b.items() if k[0] == "HNSW"]
    pbt = [v[2] for k, v in b.items() if k[0] == "IVFPQ"]
    V["hnsw_bt"] = f"{min(hbt):.0f}–{max(hbt):.0f}"
    V["pq_bt"] = f"{min(pbt):.0f}–{max(pbt):.0f}"
    pqmax = {c["build"]["m"]: c["test"]["recall"] for c in groups.get(("sift", "IVFPQ"), []) if c["build"]["nlist"] == 4096}
    V["sift_pq_m16_max"] = r3(max(r for m, r in pqmax.items() if m == 16)) if 16 in pqmax else "–"
    return groups


_P1 = {}


def summ_p1(res):
    if not _P1:
        _P1.update({s["dataset"]: s for s in json.load(open(res / "part1_summary.json"))})
    return _P1


def part3(res, V, rows_all):
    groups = group_configs([r for r in rows_all if r["experiment"] == "part3"], experiment="part3")
    s3 = json.load(open(res / "part3_summary.json"))
    ds = s3["dataset"]
    V["p3_nlist"], V["p3_ms"] = n0(s3["nlist"]), ", ".join(map(str, s3["m"]))
    ref = {c["search"]["nprobe"]: c for c in groups[(ds, "IVFFlat")]}
    by = {}
    for kind in ("IVFPQ", "IVFPQ+rerank"):
        for c in groups[(ds, kind)]:
            by.setdefault((kind, c["build"]["m"]), {})[c["search"]["nprobe"]] = c
    ex = s3["exact_single_qps"]
    rows, gains, ratios, gaps, eq = [], {}, [], {}, 0
    npb = 64 if all(64 in by[("IVFPQ+rerank", m)] for m in s3["m"]) else 16
    V["p3_np"] = str(npb)
    for m in s3["m"]:
        p, rr = by[("IVFPQ", m)], by[("IVFPQ+rerank", m)]
        pmax = max(c["test"]["recall"] for c in p.values())
        rmax = max(c["test"]["recall"] for c in rr.values())
        for v, c in rr.items():
            eq += abs(c["test"]["notes"].get("coverage", -1) - c["test"]["recall"]) < 1e-9
            if v in p:
                ratios.append(c["test"]["qps"] / p[v]["test"]["qps"])
            if v in ref:
                gaps.setdefault(m, []).append(c["test"]["recall"] - ref[v]["test"]["recall"])
        g = rr[npb]["test"]["recall"] - p[npb]["test"]["recall"] if npb in p else float("nan")
        gains[m] = g
        c95 = select_config(list(rr.values()), 0.95)
        V[f"p3_m{m}_pmax"], V[f"p3_m{m}_rmax"], V[f"p3_m{m}_gain"] = r3(pmax), r3(rmax), f"+{g:.3f}"
        V[f"p3_m{m}_bpv"] = n0(next(iter(p.values()))["bytes_per_vector"])
        V[f"p3_m{m}_rbpv"] = n0(next(iter(rr.values()))["bytes_per_vector"])
        V[f"p3_m{m}_gapmin"] = signed(min(gaps[m]))
        V[f"p3_m{m}_gapmax_abs"] = f"{max(abs(x) for x in gaps[m]):.4f}"
        if c95:
            V[f"p3_m{m}_qps95"] = n0(c95["test"]["qps"])
        rows.append([str(m), n0(next(iter(p.values()))["bytes_per_vector"]),
                     n0(next(iter(rr.values()))["bytes_per_vector"]), r3(pmax), r3(rmax),
                     f"+{g:.3f}", signed(min(gaps[m])),
                     f"{n0(c95['test']['qps'])} (np={c95['search']['nprobe']})" if c95 else "not reached"])
    refmax = max(c["test"]["recall"] for c in ref.values())
    rc95 = select_config(list(ref.values()), 0.95)
    rows.append(["IVF-Flat", n0(next(iter(ref.values()))["bytes_per_vector"]), "–", r3(refmax), "–", "–", "0",
                 f"{n0(rc95['test']['qps'])} (np={rc95['search']['nprobe']})" if rc95 else "not reached"])
    V["p3_ivf_qps95"] = n0(rc95["test"]["qps"]) if rc95 else "–"
    V["p3_ivf_bpv"] = n0(next(iter(ref.values()))["bytes_per_vector"])
    hi = [next(iter(by[("IVFPQ+rerank", m)].values()))["bytes_per_vector"] / 512 for m in s3["m"]
          if max(c["test"]["recall"] for c in by[("IVFPQ+rerank", m)].values()) >= 0.999]
    V["p3_rr_raw_x"] = f"{min(hi):.2f}–{max(hi):.2f}×" if hi else "–"
    V["p3_ms_txt"] = " / ".join(map(str, s3["m"]))
    V["p3_pmax_txt"] = " / ".join(V[f"p3_m{m}_pmax"] for m in s3["m"])
    V["T3"] = table(["m", "B/vec plain", "B/vec + re-rank", "Best recall plain", "Best recall re-rank",
                     f"Gain @ nprobe {npb}", "Re-rank − IVF-Flat (worst)", "QPS @0.95, re-rank"], rows, "small")
    V["p3_eq"] = f"{eq} of {sum(len(by[('IVFPQ+rerank', m)]) for m in s3['m'])}"
    V["p3_ratio"] = f"{min(ratios):.2f}–{max(ratios):.2f}"
    V["p3_same_centroids"] = "identical" if s3["same_centroids_as_ivfflat"] else "NOT identical"
    m_lo, m_hi = min(s3["m"]), max(s3["m"])
    V["p3_mlo"], V["p3_mhi"] = str(m_lo), str(m_hi)
    # the largest m whose re-ranked recall stays within 0.001 of IVF-Flat everywhere
    tight = [m for m in s3["m"] if max(abs(x) for x in gaps[m]) <= 0.001]
    V["p3_tight_m"] = str(min(tight)) if tight else "none"
    # m=16 re-rank vs IVF-Flat speed at 0.95
    if 16 in s3["m"] and rc95 and "p3_m16_qps95" in V:
        q16 = select_config(list(by[("IVFPQ+rerank", 16)].values()), 0.95)["test"]["qps"]
        V["p3_16_vs_ivf"] = f"{100 * (q16 / rc95['test']['qps'] - 1):+.0f}%"
    # m=64 plain vs IVF-Flat at nprobe 64: per-cluster scan cost
    if 64 in s3["m"] and 64 in by[("IVFPQ", 64)] and 64 in ref:
        V["p3_pq64_np64"] = n0(by[("IVFPQ", 64)][64]["test"]["qps"])
        V["p3_ivf_np64"] = n0(ref[64]["test"]["qps"])
    V["p3_list"] = n0(1_000_000 / s3["nlist"]) if s3["n_base"] == 1_000_000 else n0(s3["n_base"] / s3["nlist"])
    return groups, by, ref


def part4(res, V):
    states, s4 = load_part4(str(res))
    shifts = [float(s) for s in s4["shifts"]]
    V.update(p4_ndb=n0(s4["n_db"]), p4_nnew=n0(s4["n_new"]), p4_sigma=n0(s4["sigma"]),
             p4_shifts=", ".join(f"{s:g}" for s in shifts[1:]), p4_total=n0(s4["n_db"] + s4["n_new"]),
             p4_share_ctrl=pct(s4["true_nn_new_share"]["0.0"]["new"], 1),
             p4_share_small=pct(s4["true_nn_new_share"][str(shifts[1])]["new"], 1),
             p4_newpct=pct(s4["n_new"] / (s4["n_db"] + s4["n_new"])))
    info = s4["indexes"]
    for k in info:
        V[f"p4_{k}_fixed"] = f"{P4_KNOB[k]}={info[k]['fixed_knob']}"
        V[f"p4_{k}_target"] = r3(info[k]["target_recall"])
    spec = info["IVFPQ"]["spec"]
    V["p4_nlist"], V["p4_m"], V["p4_M"] = n0(spec["nlist"]), str(spec["m"]), str(info["HNSW"]["spec"]["M"])

    def cell(kind, state, s, group="new"):
        st = states.get((kind, state, s))
        return at_knob(st, info[kind]["fixed_knob"], group) if st else None

    hdr = ["New-region queries, fixed setting"] + [("0 (control)" if s == 0 else f"{s:g}σ") for s in shifts] + ["rebuilt, " + f"{shifts[-1]:g}σ"]
    rows = []
    lines = [("IVF-PQ recall", "IVFPQ", "recall", r3), ("IVF-PQ: PQ error new ÷ old", "IVFPQ", "pq", None),
             ("IVF-Flat recall", "IVFFlat", "recall", r3), ("IVF-Flat distances / query", "IVFFlat", "ndis", n0),
             ("IVF-Flat QPS", "IVFFlat", "qps", n0), ("IVF-Flat: lists holding 90% of new vectors", "IVFFlat", "l90", None),
             ("HNSW recall", "HNSW", "recall", r3), ("HNSW: new nodes' links to new nodes", "HNSW", "links", None)]
    for name, kind, what, f in lines:
        r = [name]
        for state, s in [("incremental", s) for s in shifts] + [("rebuilt", shifts[-1])]:
            st = states[(kind, state, s)]
            mech = st["mechanism"]
            if what == "pq":
                r.append(f"{mech['pq_mse_ratio']:.2f}×" if mech["pq_mse_ratio"] < 10 else f"{mech['pq_mse_ratio']:.0f}×")
            elif what == "l90":
                r.append(f"{n0(mech['lists_holding_90pct_new'])} / {n0(mech['nlist'])}")
            elif what == "links":
                r.append(pct(mech["new_links_to_new"], 1 if mech["new_links_to_new"] > 0.99 else 0))
            else:
                r.append(f(cell(kind, state, s)[what]))
        rows.append(r)
    V["T4"] = table(hdr, rows, "small")

    big, small, mid = shifts[-1], shifts[1], shifts[2]
    for kind in ("HNSW", "IVFFlat", "IVFPQ"):
        for s in shifts:
            c = cell(kind, "incremental", s)
            tag = {0.0: "c", small: "s", mid: "m", big: "l"}[s]
            V[f"p4_{kind}_rec_{tag}"] = r3(c["recall"])
            V[f"p4_{kind}_qps_{tag}"] = n0(c["qps"])
            V[f"p4_{kind}_nd_{tag}"] = n0(c["ndis"])
            rb = cell(kind, "rebuilt", s)
            V[f"p4_{kind}_rrec_{tag}"] = r3(rb["recall"])
            V[f"p4_{kind}_rqps_{tag}"] = n0(rb["qps"])
            V[f"p4_{kind}_rnd_{tag}"] = n0(rb["ndis"])
        # best recall any knob for new region (incremental), and maintenance
        ins = [states[(kind, "incremental", s)]["maintenance_s"] for s in shifts]
        reb = [states[(kind, "rebuilt", s)]["maintenance_s"] for s in shifts]
        V[f"p4_{kind}_ins"] = span(min(ins), max(ins), 1)
        V[f"p4_{kind}_reb"] = span(min(reb), max(reb), 0)
        V[f"p4_{kind}_rebx"] = f"{statistics.mean(reb) / statistics.mean(ins):.0f}×"
        # old-region worst deviation from control
        dev = [abs(cell(kind, st, s, "old")["recall"] - cell(kind, "incremental", 0.0, "old")["recall"])
               for st in ("incremental", "rebuilt") for s in shifts]
        V[f"p4_{kind}_olddev"] = f"{max(dev):.3f}"
    V["p4_olddev"] = f"{max(float(V[f'p4_{k}_olddev']) for k in info):.2f}"
    # IVF-PQ best new-region recall at any knob, incremental
    bests = []
    for s in shifts[1:]:
        st = states[("IVFPQ", "incremental", s)]
        bests.append(max(c["new-test"]["recall"] for c in st["knobs"].values() if "new-test" in c))
    V["p4_pq_best_any"] = " / ".join(r3(b) for b in bests)
    kr = knob_for_target(states[("IVFPQ", "rebuilt", big)], "new", info["IVFPQ"]["target_recall"])
    V["p4_pq_reb_knob"] = str(kr)
    mech = {s: states[("IVFPQ", "incremental", s)]["mechanism"] for s in shifts}
    V["p4_pqerr_s"], V["p4_pqerr_m"], V["p4_pqerr_l"] = (f"{mech[s]['pq_mse_ratio']:.1f}×" for s in (small, mid, big))
    V["p4_pqerr_reb"] = f"{states[('IVFPQ', 'rebuilt', big)]['mechanism']['pq_mse_ratio']:.2f}×"
    V["p4_pqerr_abs_l"] = n0(mech[big]["pq_mse_new"])
    mi = {s: states[("IVFFlat", "incremental", s)]["mechanism"] for s in shifts}
    V["p4_imb_c"], V["p4_imb_l"] = f"{mi[0.0]['imbalance']:.2f}", f"{mi[big]['imbalance']:.2f}"
    V["p4_maxl_c"], V["p4_maxl_l"] = n0(mi[0.0]["max_list"]), n0(mi[big]["max_list"])
    V["p4_l90_l"] = n0(mi[big]["lists_holding_90pct_new"])
    V["p4_newshare_l"] = pct(mi[big]["new_share_of_largest_list"])
    V["p4_l90_reb"] = n0(states[("IVFFlat", "rebuilt", big)]["mechanism"]["lists_holding_90pct_new"])
    c, l = cell("IVFFlat", "incremental", 0.0), cell("IVFFlat", "incremental", big)
    V["p4_ivf_nd_ratio"] = times(l["ndis"] / c["ndis"])
    V["p4_ivf_qps_ratio"] = times(l["qps"] / c["qps"], 2)
    rb = cell("IVFFlat", "rebuilt", big)
    V["p4_ivf_reb_speed"] = times(rb["qps"] / l["qps"])
    mh = {s: states[("HNSW", "incremental", s)]["mechanism"] for s in shifts}
    V["p4_hn_nn_m"] = pct(mh[mid]["new_links_to_new"], 1)
    V["p4_hn_o2n_m"] = n0(mh[mid]["old_to_new_links"])
    V["p4_hn_o2n_l"] = n0(mh[big]["old_to_new_links"])
    V["p4_hn_o2n_rl"] = n0(states[("HNSW", "rebuilt", big)]["mechanism"]["old_to_new_links"])
    hr = [cell("HNSW", "incremental", s)["recall"] for s in shifts]
    V["p4_hn_range"] = f"{min(hr):.3f}–{max(hr):.3f}"
    hgain = max(cell("HNSW", "rebuilt", s)["recall"] - cell("HNSW", "incremental", s)["recall"] for s in shifts)
    V["p4_hn_rebgain"] = f"{hgain:.3f}"
    V["p4_exact_D"] = n0(s4["exact_single_qps"]["D"])
    V["p4_exact_D2"] = n0(s4["exact_single_qps"]["D'"])
    # IVF-Flat re-tune grid artifact: control knobs
    kc = knob_for_target(states[("IVFFlat", "incremental", 0.0)], "new", info["IVFFlat"]["target_recall"])
    kcr = knob_for_target(states[("IVFFlat", "rebuilt", 0.0)], "new", info["IVFFlat"]["target_recall"])
    V["p4_grid_note"] = f"nprobe {kc} vs {kcr}"
    return states, s4


def q4(res, V, s4):
    """Exact single-query latency vs N, from three sizes measured in the same pipeline."""
    p1 = summ_p1(res)["sift"]
    t1 = next(t for t in p1["timings"] if t["mode"] == "single")["latency_ms"]
    pts = [(s4["n_db"], 1e3 / s4["exact_single_qps"]["D"]),
           (s4["n_db"] + s4["n_new"], 1e3 / s4["exact_single_qps"]["D'"]), (p1["n_base"], t1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    V["q4_ns_per_vec"] = f"{b * 1e6:.1f}"
    V["q4_150k"] = n1(a + b * 150_000)
    V["q4_10m"] = n0(a + b * 10_000_000)
    V["q4_fit_pts"] = ", ".join(f"{n0(x / 1000)}K: {y:.1f} ms" for x, y in pts)
    V["q4_150k_mb"] = n0(150_000 * 512 / 2**20)
    V["q4_10m_raw_gb"] = n1(10_000_000 * 512 / 1e9)
    V["q4_10m_h16"] = n1(10_000_000 * float(V["hnsw16_bpv"].replace(",", "")) / 1e9)
    V["q4_10m_h32"] = n1(10_000_000 * float(V["hnsw32_bpv"].replace(",", "")) / 1e9)
    ball = next(t for t in p1["timings"] if t["mode"] == "batch" and t["threads"] > 1)["latency_ms"]
    V["q4_batch_10m"] = n0(ball * 10)
    V["q4_budget_x"] = n0(100 / (a + b * 150_000))


def q5(V, rows_all):
    """Same (index, build, search) measured in Part 2 and again in Part 3 of the same pipeline."""
    key = lambda r: (r["dataset"], r["n_base"], r["index_type"], r["build_params"], r["search_params"])  # noqa: E731
    p2 = {key(r): r for r in rows_all if r["experiment"] == "part2" and r["query_set"] == "test"}
    ratios, rec_same, n = [], 0, 0
    for r in rows_all:
        if r["experiment"] == "part3" and r["query_set"] == "test" and key(r) in p2:
            o = p2[key(r)]
            ratios.append(float(r["qps"]) / float(o["qps"]))
            rec_same += abs(float(r["recall_at_k"]) - float(o["recall_at_k"])) < 1e-12
            n += 1
    if ratios:
        V["q5_n"] = n0(n)
        V["q5_med"] = f"{statistics.median(ratios):.2f}"
        V["q5_rng"] = f"{min(ratios):.2f}–{max(ratios):.2f}"
        V["q5_rec_same"] = f"{rec_same} of {n}"
    else:
        V.update(q5_n="0", q5_med="–", q5_rng="–", q5_rec_same="–")


def git_hash(pattern):
    try:
        out = subprocess.run(["git", "log", "--format=%h %s", "--", "hypothesis.md"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if re.search(pattern, line, re.I):
                return line.split()[0]
    except Exception:
        pass
    return "(see git log)"


def find_browser():
    env = os.environ.get("REPORT_BROWSER")
    cands = [env] if env else []
    cands += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    cands += [shutil.which(n) for n in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome")]
    return next((c for c in cands if c and os.path.exists(c)), None)


def print_pdf(html: Path, pdf: Path):
    browser = find_browser()
    if not browser:
        print("no Edge/Chrome found: wrote HTML only (set REPORT_BROWSER to print the PDF)")
        return None
    import tempfile
    import time
    if pdf.exists():
        pdf.unlink()
    profile = tempfile.mkdtemp(prefix="report_browser_")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
           f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", html.resolve().as_uri()]
    subprocess.run(cmd, capture_output=True, timeout=180)
    size, t0 = -1, time.time()
    while time.time() - t0 < 60:          # wait until the file exists and stops growing
        if pdf.exists() and pdf.stat().st_size == size and size > 0:
            break
        size = pdf.stat().st_size if pdf.exists() else -1
        time.sleep(1.0)
    shutil.rmtree(profile, ignore_errors=True)
    if not pdf.exists():
        print("PDF printing failed; HTML is at", html)
        return None
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print(f"wrote {pdf} ({pages} pages)" + ("  WARNING: over the 5-page limit" if pages > 5 else ""))
    return pages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default=None, help="default: report/ (results/quick/report/ for results/quick)")
    args = p.parse_args()
    res = (ROOT / args.results_dir).resolve()
    out = Path(args.out_dir) if args.out_dir else (ROOT / "report" if res == (ROOT / "results").resolve()
                                                   else res / "report")
    out = (ROOT / out).resolve() if not out.is_absolute() else out
    out.mkdir(parents=True, exist_ok=True)
    rows_all = read_results(str(res / "results.csv"))

    V = {}
    machine(res, V)
    part1(res, V)
    part2(res, V, rows_all)
    part3(res, V, rows_all)
    _, s4 = part4(res, V)
    q4(res, V, s4)
    q5(V, rows_all)
    V["h3"], V["h4"] = git_hash(r"part.?3"), git_hash(r"part.?4")
    V["fig"] = os.path.relpath(res / "figures", out).replace(os.sep, "/")

    tpl = Template((ROOT / "report" / "report_template.html").read_text(encoding="utf-8"))
    missing = sorted({m.group("named") or m.group("braced") for m in tpl.pattern.finditer(tpl.template)
                      if (m.group("named") or m.group("braced")) and (m.group("named") or m.group("braced")) not in V})
    if missing:
        print(f"WARNING: {len(missing)} report values not available from these results, shown as '–': "
              + ", ".join(missing))
        V.update({k: "–" for k in missing})
    html = out / "report.html"
    html.write_text(tpl.substitute(V), encoding="utf-8")
    print("wrote", html)
    print_pdf(html, out / "report.pdf")


if __name__ == "__main__":
    main()
