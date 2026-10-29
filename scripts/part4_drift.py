#!/usr/bin/env python
"""Part 4 (drift): a new, shifted cluster appears. Incremental insertion vs rebuild.

Hypothesis: hypothesis.md, committed before this script was written or run.

Data (ann_bench/synthetic.py, seeded): D = the first N_DB SIFT vectors; the new
cluster = the next N_NEW SIFT vectors translated by s*sigma*u (u a seeded unit
direction, sigma the RMS distance of D from its mean); D'_s = D + new cluster.
s = 0 is the control (more SIFT data, unmoved); s in SHIFTS[1:] are small, medium
and large shifts. Everything else is fixed: N_DB, N_NEW, u, build settings, seeds.

For each index (HNSW, IVF-Flat, IVF-PQ):
  1. baseline     build on D (IVF centroids / PQ codebooks trained on D)
  for each s:
  2-3. incremental  a copy of the baseline index, the new vectors added: no retraining
  4.   rebuilt      a fresh index built and trained on D'_s
Each state is swept over the search knob. Queries keep the tune/test halves and split
each into "old" (original SIFT queries) and "new" (other SIFT queries moved by the
same translation). Ground truth is exact search on the database version searched
(D or D'_s); recall uses the tie-aware recall@10 of Part 1.

Per (state, knob, group): recall on all its tune and test queries; on test queries
QPS (one query at a time, 1 thread, P-cores, median of 3 repeats >= MIN_REPEAT_S, as
in Part 2) and distance computations per query. Per state: maintenance time
(insertion or full rebuild), index size and mechanism statistics (IVF list balance
and where new vectors land, PQ error of old vs new vectors, HNSW links between old and
new nodes).

Comparable settings: the FIXED knob of each index is chosen once, on the baseline
tune "old" queries, as the smallest knob reaching 0.95 x that index's best baseline
recall (= about 0.95 for HNSW/IVF-Flat; IVF-PQ plateaus lower). The tables also give
the knob needed to bring each group back to that target.

  python scripts/part4_drift.py           # 400K + 100K SIFT vectors
  python scripts/part4_drift.py --quick   # 80K + 20K (the first 100K vectors)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from ann_bench.data import load_dataset  # noqa: E402
from ann_bench.diagnostics import (copy_index, distances_per_query, hnsw_link_stats,  # noqa: E402
                                   ivf_list_stats, pq_error)
from ann_bench.exact import build_flat, faiss_search_fn, index_size_bytes  # noqa: E402
from ann_bench.indexes import IndexSpec, build_index, set_search_param  # noqa: E402
from ann_bench.metrics import recall_at_k, strict_recall_at_k  # noqa: E402
from ann_bench.results import drop_rows, replace_rows, write_machine_info  # noqa: E402
from ann_bench.synthetic import make_drift, query_groups  # noqa: E402
from ann_bench.timing import limit_threads, performance_cores, pinned_to, time_search  # noqa: E402
from part2_benchmark import (MIN_REPEAT_S, PLATEAU_GAIN, SEED, STOP_RECALL,  # noqa: E402
                             SWEEPS, SWEEPS_QUICK)

K = 10
SHIFTS = [0.0, 0.5, 1.0, 2.0]      # x sigma: control, small (overlapping), medium, large (separate)
FIXED_FRACTION = 0.95              # fixed knob reaches this share of the best baseline recall
DIRECTION_SEED = 1234
N_ERR_SAMPLE = 5000                # vectors per group for the PQ-error estimate
GRID = {
    "full": {"n_db": 400_000, "n_new": 100_000,
             "HNSW": {"M": 16, "efConstruction": 200}, "IVFFlat": {"nlist": 2048},
             "IVFPQ": {"nlist": 2048, "m": 32}},
    "quick": {"n_db": 80_000, "n_new": 20_000,
              "HNSW": {"M": 16, "efConstruction": 200}, "IVFFlat": {"nlist": 512},
              "IVFPQ": {"nlist": 512, "m": 8}},     # PQ training ~0.9 s per sub-quantizer: keep quick small
}
GROUPS = ("old", "new")
KEY = ("experiment", "dataset", "n_base", "index_type", "build_params")


def exact_gt(db, queries):
    with limit_threads(os.cpu_count()):
        return build_flat(db).search(queries, K)[1]


def exact_single_qps(db, queries, pcores, args):
    idx = build_flat(db)
    with pinned_to(pcores), limit_threads(1):
        return time_search(faiss_search_fn(idx), queries, K, "single", args.repeats).qps


def mechanism(index, kind, n_old, drift, shift, rng_idx):
    """Per-state statistics that explain the recall/speed numbers."""
    out = {}
    if kind in ("IVFFlat", "IVFPQ"):
        out.update(ivf_list_stats(index, n_old))
    if kind == "IVFPQ":
        old = drift.db[rng_idx["old"]]
        out["pq_mse_old"] = pq_error(index, old)
        if shift is not None:
            out["pq_mse_new"] = pq_error(index, drift.new_vectors(shift)[rng_idx["new"]])
            out["pq_mse_ratio"] = out["pq_mse_new"] / out["pq_mse_old"]
    if kind == "HNSW":
        out.update(hnsw_link_stats(index, n_old))
    return out


def sweep(index, spec, db, qsets, gts, ctx, args, state, min_knob=0):
    """Sweep the knob; returns rows. qsets/gts: {(group, split): array}."""
    rows, hist = [], {g: [] for g in GROUPS}
    knobs = (SWEEPS_QUICK if args.quick else SWEEPS)[spec.kind]
    for value in knobs:
        if spec.kind != "HNSW" and value > spec.params["nlist"]:
            break
        set_search_param(index, spec, value)
        rec, ndis, timing = {}, {}, {}
        with limit_threads(ctx["all_threads"]):
            for key, q in qsets.items():
                ids = index.search(q, K)[1]
                rec[key] = (float(recall_at_k(ids, db, q, gts[key], k=K).mean()),
                            float(strict_recall_at_k(ids, gts[key], k=K).mean()))
            for g in GROUPS:
                ndis[g] = distances_per_query(index, spec.kind, qsets[(g, "test")], K)
        with pinned_to(ctx["pcores"]), limit_threads(1):
            for i, g in enumerate(GROUPS):
                # settle only before the first timing after the multi-threaded step (D30)
                timing[g] = time_search(lambda x, k: index.search(x, k)[1],
                                        qsets[(g, "test")][:args.time_queries], K, "single", args.repeats,
                                        min_repeat_s=args.min_repeat_s, settle_s=args.settle_s if i == 0 else 0.0)
        for (g, split), (r, rs) in rec.items():
            tr = timing[g] if split == "test" else None
            if tr is not None and tr.cpu_per_wall > 1.3 and tr.median_s >= 0.05:
                print(f"    WARNING: 1-thread timing used {tr.cpu_per_wall:.1f} cores", flush=True)
            notes = {**state["notes"], "group": g, "recall_queries": len(qsets[(g, split)])}
            if tr is not None:
                notes.update(distances_per_query=ndis[g], timing_loops=tr.loops, timing=ctx["pin_note"])
            rows.append({
                "experiment": "part4", "dataset": "sift-drift", "n_base": db.shape[0],
                "index_type": spec.kind, "build_params": state["build_params"],
                "search_params": json.dumps({spec.search_param: value}),
                "query_set": f"{g}-{split}", "n_queries": tr.n_queries if tr else len(qsets[(g, split)]),
                "mode": "single" if tr else "batch", "threads": 1 if tr else ctx["all_threads"], "k": K,
                "recall_at_k": r, "recall_strict": rs,
                "qps": tr.qps if tr else "", "latency_ms": tr.latency_ms if tr else "",
                "time_median_s": tr.median_s if tr else "",
                "time_runs_s": json.dumps([round(t, 6) for t in tr.times_s]) if tr else "",
                "cpu_per_wall": tr.cpu_per_wall if tr else "",
                "build_time_s": state["maintenance_s"], "index_bytes": state["size"],
                "bytes_per_vector": state["size"] / db.shape[0], "notes": json.dumps(notes),
            })
        print(f"    {spec.search_param}={value:<5d} " + " | ".join(
            f"{g}: recall {rec[(g, 'test')][0]:.4f} {timing[g].qps:8.0f} QPS {ndis[g]:9.0f} dist/q"
            for g in GROUPS), flush=True)

        # Part 2 stop rules (tune recall >= 0.999 or plateau), applied once every group is
        # done, never before min_knob (the fixed knob must be measured in every state).
        done = True
        for g in GROUPS:
            t = rec[(g, "tune")][0]
            hist[g].append(t)
            back = 1 if args.quick else 2
            plateau = len(hist[g]) > back and hist[g][-1] - hist[g][-1 - back] < PLATEAU_GAIN
            if not (t >= STOP_RECALL or plateau):
                done = False
        slow = ctx["exact_qps"] and max(timing[g].qps for g in GROUPS) < ctx["exact_qps"]
        if value >= min_knob and (done or slow):
            break
    return rows


def run_index(kind, drift, splits_all, ctx, args, grid):
    spec = IndexSpec(kind, grid[kind])
    n_old = drift.db.shape[0]
    rng = np.random.default_rng(SEED)
    err_idx = {"old": rng.choice(n_old, N_ERR_SAMPLE, replace=False),
               "new": rng.choice(drift.reserved.shape[0], N_ERR_SAMPLE, replace=False)}
    summary = {"spec": spec.params, "states": []}

    def record(state_name, shift, index, db, maint_s, extra):
        size = index_size_bytes(index)
        mech = mechanism(index, kind, n_old, drift, shift, err_idx)
        bp = {**spec.params, "state": state_name, "shift": shift}
        state = {"build_params": json.dumps(bp, sort_keys=True), "maintenance_s": maint_s, "size": size,
                 "notes": {"state": state_name, "shift": shift, "maintenance_s": round(maint_s, 3),
                           "mechanism": mech, **extra}}
        print(f"\n  {kind} {state_name} s={shift}: maintenance {maint_s:.1f}s, "
              f"{size / 2**20:.1f} MB, {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in mech.items()})}",
              flush=True)
        return state

    # 1. baseline on D: both query groups unshifted
    with limit_threads(ctx["all_threads"]):
        base, info = build_index(spec, drift.db, seed=SEED)
    qsets = {(g, s): drift.group_queries(g, s, 0.0) for g in GROUPS for s in ("tune", "test")}
    gts = {key: splits_all["gt_D"][key] for key in qsets}
    st = record("baseline", None, base, drift.db, info["build_s"],
                {"train_s": round(info["train_s"], 3), "train_n": info["train_n"]})
    rows = sweep(base, spec, drift.db, qsets, gts, ctx, args, st)
    replace_rows(args.results, rows, key_columns=KEY)

    # Fixed knob: smallest reaching FIXED_FRACTION of the best baseline tune recall (old queries).
    tune_old = [(json.loads(r["search_params"])[spec.search_param], r["recall_at_k"])
                for r in rows if r["query_set"] == "old-tune"]
    target = FIXED_FRACTION * max(r for _, r in tune_old)
    fixed = min(v for v, r in tune_old if r >= target)
    print(f"  {kind}: fixed {spec.search_param}={fixed} (baseline tune recall target {target:.4f})", flush=True)
    summary.update(fixed_knob=fixed, target_recall=target, baseline_build_s=info["build_s"])

    for shift in SHIFTS:
        db2 = drift.updated_db(shift)
        qsets = {(g, s): drift.group_queries(g, s, shift) for g in GROUPS for s in ("tune", "test")}
        gts = splits_all["gt_D2"][shift]

        # 2-3. incremental: copy of the baseline index + the new vectors, no retraining
        inc = copy_index(base)
        new = drift.new_vectors(shift)
        with limit_threads(ctx["all_threads"]):
            t0 = time.perf_counter()
            inc.add(new)
            add_s = time.perf_counter() - t0
        st = record("incremental", shift, inc, db2, add_s, {"insert_s": round(add_s, 3)})
        rows = sweep(inc, spec, db2, qsets, gts, ctx, args, st, min_knob=fixed)
        replace_rows(args.results, rows, key_columns=KEY)
        del inc

        # 4. rebuild on D'_s (retrains IVF centroids and PQ codebooks)
        with limit_threads(ctx["all_threads"]):
            reb, info2 = build_index(spec, db2, seed=SEED)
        st = record("rebuilt", shift, reb, db2, info2["build_s"],
                    {"train_s": round(info2["train_s"], 3), "train_n": info2["train_n"]})
        rows = sweep(reb, spec, db2, qsets, gts, ctx, args, st, min_knob=fixed)
        replace_rows(args.results, rows, key_columns=KEY)
        del reb, db2
    del base
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="80K + 20K vectors (the first 100K) and a smaller grid")
    p.add_argument("--indexes", nargs="+", default=["HNSW", "IVFFlat", "IVFPQ"],
                   choices=["HNSW", "IVFFlat", "IVFPQ"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default=None, help="default: results/ (results/quick/ with --quick)")
    p.add_argument("--time-queries", type=int, default=None, help="test queries timed per group; default 500 (quick 100)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-repeat-s", type=float, default=None, help="default 0.2 (quick: 0.02)")
    p.add_argument("--threads", type=int, default=None, help="threads for building and recall")
    p.add_argument("--no-pin", action="store_true")
    args = p.parse_args()
    args.results_dir = args.results_dir or ("results/quick" if args.quick else "results")
    args.time_queries = args.time_queries or (100 if args.quick else 500)
    if args.min_repeat_s is None:
        args.min_repeat_s = 0.02 if args.quick else MIN_REPEAT_S
    args.settle_s = 0.1 if args.quick else 0.25    # pause before a timing block (D24, D30)
    args.results = os.path.join(args.results_dir, "results.csv")
    grid = GRID["quick" if args.quick else "full"]
    t_start = time.perf_counter()

    write_machine_info(os.path.join(args.results_dir, "machine_info.json"))
    ds = load_dataset("sift", args.data_dir, grid["n_db"] + grid["n_new"])
    drift = make_drift(ds.base, ds.queries, grid["n_db"], grid["n_new"], seed=DIRECTION_SEED)
    print(f"=== Part 4 drift: D {drift.db.shape}, new cluster {drift.reserved.shape}, sigma={drift.sigma:.1f}, "
          f"shifts {SHIFTS} x sigma ===", flush=True)
    pcores = None if args.no_pin else performance_cores()
    ctx = {"pcores": pcores, "all_threads": args.threads or os.cpu_count(),
           "pin_note": f"1 thread, pinned to P-cores {pcores[0]}-{pcores[-1]}" if pcores else "1 thread, not pinned"}

    # Exact ground truth for every (database version, group, split).
    t0 = time.perf_counter()
    gt_D = {(g, s): exact_gt(drift.db, drift.group_queries(g, s, 0.0)) for g in GROUPS for s in ("tune", "test")}
    gt_D2 = {}
    nn_new_share = {}
    for shift in SHIFTS:
        db2 = drift.updated_db(shift)
        gt_D2[shift] = {(g, s): exact_gt(db2, drift.group_queries(g, s, shift)) for g in GROUPS for s in ("tune", "test")}
        # how many true neighbours of each group are new vectors (describes the geometry)
        nn_new_share[shift] = {g: float((gt_D2[shift][(g, "test")] >= drift.db.shape[0]).mean()) for g in GROUPS}
    print(f"ground truth in {time.perf_counter() - t0:.1f}s; share of true top-10 that are new vectors: "
          f"{json.dumps(nn_new_share)}", flush=True)

    q_ref = drift.group_queries("old", "test", 0.0)[:200]
    exact_qps = {"D": exact_single_qps(drift.db, q_ref, pcores, args),
                 "D'": exact_single_qps(drift.updated_db(0.0), q_ref, pcores, args)}
    ctx["exact_qps"] = exact_qps["D'"]
    print(f"exact single-query QPS: {exact_qps}", flush=True)

    drop_rows(args.results, experiment="part4")
    summary = {"n_db": grid["n_db"], "n_new": grid["n_new"], "sigma": drift.sigma,
               "shifts": SHIFTS, "direction_seed": DIRECTION_SEED, "fixed_fraction": FIXED_FRACTION,
               "query_groups": {f"{g}-{s}": [sl.start, sl.stop] for (g, s), sl in query_groups(ds.queries).items()},
               "true_nn_new_share": {str(k): v for k, v in nn_new_share.items()},
               "exact_single_qps": exact_qps, "time_queries_per_group": args.time_queries,
               "repeats": args.repeats, "min_repeat_s": args.min_repeat_s, "pinning": ctx["pin_note"],
               "build_threads": ctx["all_threads"], "indexes": {}}
    for kind in args.indexes:
        print(f"\n=== {kind} ===", flush=True)
        summary["indexes"][kind] = run_index(kind, drift, {"gt_D": gt_D, "gt_D2": gt_D2}, ctx, args, grid)
        with open(os.path.join(args.results_dir, "part4_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    print(f"\nPart 4 done in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
