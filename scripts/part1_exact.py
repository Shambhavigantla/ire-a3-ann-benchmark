#!/usr/bin/env python
"""Part 1: exact FAISS baseline.

For each dataset:
  1. build IndexFlatL2 (exact brute force) and record build time / size;
  2. check our exact results against the ground truth stored in the file;
  3. time exact search on the TEST half of the queries:
       - one query at a time, 1 thread            (the Part 2 protocol)
       - all queries in one batch, 1 thread       (isolates the BLAS/cache effect)
       - all queries in one batch, all threads    (adds parallelism)
  4. write rows to results.csv and a JSON summary.

Usage:
  python scripts/part1_exact.py                      # full data
  python scripts/part1_exact.py --n-base 100000      # quick subset
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from ann_bench.data import N_TUNE, load_dataset, split_queries  # noqa: E402
from ann_bench.exact import (build_flat, faiss_search_fn, ground_truth_for,  # noqa: E402
                             index_size_bytes)
from ann_bench.metrics import recall_at_k, strict_recall_at_k  # noqa: E402
from ann_bench.results import replace_rows, write_machine_info  # noqa: E402
from ann_bench.timing import limit_threads, performance_cores, pinned_to, time_search  # noqa: E402

K = 10


def check_against_file_full(ds, exact_ids, exact_d):
    """Full base set: our exact top-k must match the file's neighbours (up to ties)."""
    gt = ds.file_neighbors
    rec = recall_at_k(exact_ids[:, :K], ds.base, ds.queries, gt, k=K)
    strict = strict_recall_at_k(exact_ids[:, :K], gt, k=K)
    n_top = exact_ids.shape[1]

    # FAISS returns squared L2. Map the file's distances into the same units.
    ours = exact_d[:, :K].astype(np.float64)
    fd = ds.file_distances[:, :K].astype(np.float64)
    if ds.file_metric == "euclidean":
        candidates = {"d^2": fd ** 2}
    else:  # angular: for unit vectors ||a-b||^2 = 2*(1-cos); the file may store 1-cos or a Euclidean value
        candidates = {"2*d (cosine distance)": 2 * fd, "d^2": fd ** 2}
    errs = {name: float(np.max(np.abs(ours - c))) for name, c in candidates.items()}
    mapping = min(errs, key=errs.get)
    rel = float(np.max(np.abs(ours - candidates[mapping]) / np.maximum(np.abs(ours), 1e-12)))

    return {
        "mode": "full",
        "n_queries_checked": int(len(rec)),
        "recall_tie_aware_mean": float(rec.mean()),
        "recall_tie_aware_min": float(rec.min()),
        "queries_with_tie_aware_recall_below_1": int((rec < 1).sum()),
        "recall_strict_mean": float(strict.mean()),
        "queries_where_only_ties_differ": int(((strict < 1) & (rec == 1)).sum()),
        f"top{n_top}_ids_identical_in_order_frac": float((exact_ids == gt[:, :n_top]).all(axis=1).mean()),
        "distance_mapping": mapping,
        "distance_max_abs_err": errs[mapping],
        "distance_max_rel_err": rel,
    }


def check_against_file_subset(ds, exact_ids):
    """Subset: the file's top-100 restricted to ids < n_base, when >= K of them remain,
    are exactly the subset's top-K (any other subset vector is beyond the full 100th
    neighbour). So they are a valid ground truth for those queries."""
    usable, derived = [], []
    for i, row in enumerate(ds.file_neighbors):
        inside = row[row < ds.n_base]
        if len(inside) >= K:
            usable.append(i)
            derived.append(inside[:K])
    if not usable:
        return {"mode": "subset", "n_queries_checked": 0}
    usable = np.array(usable)
    rec = recall_at_k(exact_ids[usable, :K], ds.base, ds.queries[usable], np.stack(derived), k=K)
    return {
        "mode": "subset",
        "n_queries_checked": int(len(usable)),
        "recall_tie_aware_mean": float(rec.mean()),
        "recall_tie_aware_min": float(rec.min()),
        "queries_with_tie_aware_recall_below_1": int((rec < 1).sum()),
    }


def run_dataset(name, args):
    print(f"\n=== {name} ===", flush=True)
    ds = load_dataset(name, args.data_dir, args.n_base)
    print(f"base {ds.base.shape}, queries {ds.queries.shape}, subset={ds.is_subset}", flush=True)

    with limit_threads(1):
        t0 = time.perf_counter()
        index = build_flat(ds.base)
        build_time = time.perf_counter() - t0
    size = index_size_bytes(index)

    # Correctness check on all 10k queries (multi-threaded batch; speed irrelevant here).
    n_check = 100
    exact_d, exact_ids = index.search(ds.queries, n_check)
    if ds.is_subset:
        check = check_against_file_subset(ds, exact_ids)
    else:
        check = check_against_file_full(ds, exact_ids, exact_d)
    print("check vs file:", json.dumps(check, indent=2), flush=True)

    gt = ground_truth_for(ds, k=100, cache_dir=os.path.join(args.data_dir, "cache"))
    _, q_test = split_queries(ds.queries)
    _, gt_test = split_queries(gt)

    search = faiss_search_fn(index)
    n_single = min(args.single_queries, len(q_test))
    all_threads = args.threads or os.cpu_count()
    # 1-thread runs: cap OpenMP + OpenBLAS at 1 thread and keep them on P-cores (hybrid CPUs).
    pcores = None if args.no_pin else performance_cores()
    pin_note = f"pinned to P-cores {pcores[0]}-{pcores[-1]}" if pcores else "not pinned"
    runs = []
    with pinned_to(pcores), limit_threads(1):
        runs.append(("single", 1, pin_note,
                     time_search(search, q_test[:n_single], K, "single", args.repeats)))
        runs.append(("batch", 1, pin_note, time_search(search, q_test, K, "batch", args.repeats)))
    with limit_threads(all_threads):
        runs.append(("batch", all_threads, "all cores",
                     time_search(search, q_test, K, "batch", args.repeats)))

    rows, timings = [], []
    for mode, threads, note, tr in runs:
        g = gt_test[:tr.n_queries]
        q = q_test[:tr.n_queries]
        rec = float(recall_at_k(tr.ids, ds.base, q, g, k=K).mean())
        strict = float(strict_recall_at_k(tr.ids, g, k=K).mean())
        print(f"{mode:6s} threads={threads:<3d} nq={tr.n_queries:<5d} "
              f"qps={tr.qps:10.1f}  latency={tr.latency_ms:8.3f} ms  "
              f"recall@10={rec:.4f} (strict {strict:.4f})  cpu/wall={tr.cpu_per_wall:4.1f}  "
              f"runs={['%.3f' % t for t in tr.times_s]}  [{note}]", flush=True)
        if threads == 1 and tr.cpu_per_wall > 1.3:
            print(f"  WARNING: '1-thread' run used {tr.cpu_per_wall:.1f} cores of CPU time", flush=True)
        timings.append({"mode": mode, "threads": threads, "n_queries": tr.n_queries,
                        "qps": tr.qps, "latency_ms": tr.latency_ms, "recall": rec,
                        "cpu_per_wall": tr.cpu_per_wall, "times_s": tr.times_s, "note": note})
        rows.append({
            "experiment": "part1_exact",
            "dataset": name,
            "n_base": ds.n_base,
            "index_type": "Flat",
            "build_params": "{}",
            "search_params": "{}",
            "query_set": f"test[{N_TUNE}:{N_TUNE + tr.n_queries}]",
            "n_queries": tr.n_queries,
            "mode": mode,
            "threads": threads,
            "k": K,
            "recall_at_k": rec,
            "recall_strict": strict,
            "qps": tr.qps,
            "latency_ms": tr.latency_ms,
            "time_median_s": tr.median_s,
            "time_runs_s": json.dumps([round(t, 6) for t in tr.times_s]),
            "cpu_per_wall": tr.cpu_per_wall,
            "build_time_s": build_time,
            "index_bytes": size,
            "bytes_per_vector": size / ds.n_base,
            "notes": note,
        })

    single_qps = timings[0]["qps"]
    for t in timings[1:]:
        print(f"  batch/{t['threads']}T speed-up over single-query: {t['qps'] / single_qps:.1f}x")

    replace_rows(args.results, rows)
    return {"dataset": name, "n_base": ds.n_base, "dim": ds.dim, "build_time_s": build_time,
            "index_bytes": size, "check_vs_file": check, "timings": timings}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n-base", type=int, default=None, help="use only the first N base vectors")
    p.add_argument("--single-queries", type=int, default=1000,
                   help="test queries used for the one-at-a-time timing (QPS is a rate, so a "
                        "subsample suffices; full exact single-query on 5000 x 1M is slow)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=None, help="threads for the multi-threaded batch run")
    p.add_argument("--no-pin", action="store_true", help="do not pin 1-thread runs to P-cores")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    args.results = os.path.join(args.results_dir, "results.csv")

    info = write_machine_info(os.path.join(args.results_dir, "machine_info.json"))
    print("machine:", json.dumps(info))

    summary = [run_dataset(name, args) for name in args.datasets]
    with open(os.path.join(args.results_dir, "part1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
