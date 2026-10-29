#!/usr/bin/env python
"""Part 3 (compression): IVF-PQ code size m, with and without exact re-ranking.

Hypothesis: hypothesis.md, committed before this script was written or run.

On SIFT, nlist is fixed and m (PQ sub-vectors = bytes per code) is varied. Each
IVF-PQ index is built once and searched two ways at every nprobe:
  IVFPQ          plain: top 10 by PQ (approximate) distance          (control)
  IVFPQ+rerank   top RERANK_DEPTH by PQ distance, re-ranked by exact L2 on the
                 original vectors (FAISS IndexRefineFlat, k_factor = depth / 10)
IVF-Flat with the same nlist is the reference. It is trained with the same seed on
the same sample, so its centroids are identical (checked below): at equal nprobe it
scans exactly the clusters IVF-PQ scans, and so bounds IVF-PQ + re-rank per query.

Also recorded on the re-rank rows (notes.coverage): candidate coverage, the share of
the true top 10 anywhere in PQ's top RERANK_DEPTH. Re-ranked recall must equal it.

Same protocol as Part 2: recall on all tune/test queries; QPS one query at a time,
1 thread, pinned to P-cores, median of 3 repeats of >= MIN_REPEAT_S; the same nprobe
grid; and per method the same stop rules (tune recall >= 0.999, plateau, slower than
exact). Memory: serialised index size; the re-rank index includes its float32 copy
of the base vectors, i.e. the original vectors are counted as held in RAM.

  python scripts/part3_compression.py           # full SIFT
  python scripts/part3_compression.py --quick   # 100K subset, smaller grid
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

import numpy as np  # noqa: E402

from ann_bench.data import N_TUNE, load_dataset, split_queries  # noqa: E402
from ann_bench.exact import ground_truth_for, index_size_bytes  # noqa: E402
from ann_bench.indexes import (IndexSpec, add_exact_rerank, build_index,  # noqa: E402
                               coarse_centroids, set_search_param)
from ann_bench.metrics import coverage_at_k, recall_at_k, strict_recall_at_k  # noqa: E402
from ann_bench.results import drop_rows, replace_rows, write_machine_info  # noqa: E402
from ann_bench.timing import limit_threads, performance_cores, pinned_to, time_search  # noqa: E402
from part2_benchmark import (MIN_REPEAT_S, PLATEAU_GAIN, SEED, STOP_RECALL,  # noqa: E402
                             SWEEPS, SWEEPS_QUICK, call_overhead_qps, exact_single_qps)

K = 10
DATASET = "sift"
RERANK_DEPTH = 100
GRID = {"full": {"nlist": 4096, "m": [8, 16, 32, 64]},       # m divides d=128; 8-64 bytes/code
        "quick": {"nlist": 1024, "m": [8, 16, 32]}}
KEY = ("experiment", "dataset", "n_base", "index_type", "build_params")


class Method:
    """One searchable variant, with its own Part 2 stop rules."""

    def __init__(self, kind, build_params, index, size, build_s, info, extra_notes=None):
        self.kind, self.build_params, self.index = kind, build_params, index
        self.size, self.build_s, self.info = size, build_s, info
        self.extra_notes = extra_notes or {}
        self.active, self.tune_hist, self.rows = True, [], []

    def search(self, x, k):
        return self.index.search(x, k)[1]


def sweep(methods, ds, splits, ctx, args):
    """Run every active method at each nprobe; a method drops out on its own stop rule."""
    (q_tune, gt_tune), (q_test, gt_test) = splits
    nlist = methods[0].build_params["nlist"]
    spec = IndexSpec("IVFFlat", {"nlist": nlist})          # only used to set nprobe
    for value in (SWEEPS_QUICK if args.quick else SWEEPS)["IVFPQ"]:
        live = [m for m in methods if m.active]
        if value > nlist or not live:
            break
        for m in live:
            set_search_param(m.index, spec, value)
        msg = []
        for m in live:
            recall, coverage = {}, {}
            with limit_threads(ctx["all_threads"]):
                for name, q, gt in (("tune", q_tune, gt_tune), ("test", q_test, gt_test)):
                    ids = m.search(q, K)
                    recall[name] = (float(recall_at_k(ids, ds.base, q, gt, k=K).mean()),
                                    float(strict_recall_at_k(ids, gt, k=K).mean()))
                    if m.kind == "IVFPQ+rerank":       # what the wrapped PQ index hands over
                        cand = m.index.base_index.search(q, RERANK_DEPTH)[1]
                        coverage[name] = float(coverage_at_k(cand, ds.base, q, gt, k=K).mean())
            timing = {}
            with pinned_to(ctx["pcores"]), limit_threads(1):
                for i, (name, q) in enumerate((("tune", q_tune[:args.tune_time_queries]),
                                               ("test", q_test[:args.test_time_queries]))):
                    # settle only before the first timing after the multi-threaded step (D30)
                    timing[name] = time_search(m.search, q, K, "single", args.repeats,
                                               min_repeat_s=args.min_repeat_s, settle_s=args.settle_s if i == 0 else 0.0)
            for name in ("tune", "test"):
                tr = timing[name]
                if tr.cpu_per_wall > 1.3 and tr.median_s >= 0.05:
                    print(f"    WARNING: 1-thread timing used {tr.cpu_per_wall:.1f} cores", flush=True)
                notes = {"train_s": round(m.info["train_s"], 3), "add_s": round(m.info["add_s"], 3),
                         "train_n": m.info["train_n"], "build_threads": ctx["all_threads"],
                         "timing_loops": tr.loops, "timing": ctx["pin_note"],
                         "recall_queries": len(q_tune if name == "tune" else q_test), **m.extra_notes}
                if name in coverage:
                    notes["coverage"] = coverage[name]
                m.rows.append({
                    "experiment": "part3", "dataset": ds.name, "n_base": ds.n_base,
                    "index_type": m.kind, "build_params": json.dumps(m.build_params, sort_keys=True),
                    "search_params": json.dumps({"nprobe": value}),
                    "query_set": name, "n_queries": tr.n_queries, "mode": "single", "threads": 1, "k": K,
                    "recall_at_k": recall[name][0], "recall_strict": recall[name][1],
                    "qps": tr.qps, "latency_ms": tr.latency_ms, "time_median_s": tr.median_s,
                    "time_runs_s": json.dumps([round(t, 6) for t in tr.times_s]),
                    "cpu_per_wall": tr.cpu_per_wall, "build_time_s": m.build_s,
                    "index_bytes": m.size, "bytes_per_vector": m.size / ds.n_base,
                    "notes": json.dumps(notes),
                })
            cov = f" cov {coverage['test']:.4f}" if coverage else ""
            msg.append(f"{m.kind:<12s} {recall['test'][0]:.4f}{cov} {timing['test'].qps:8.0f} QPS")

            # Part 2 stop rules, per method
            if recall["tune"][0] >= STOP_RECALL:
                m.active = False
            m.tune_hist.append(recall["tune"][0])
            back = 1 if args.quick else 2
            if len(m.tune_hist) > back and m.tune_hist[-1] - m.tune_hist[-1 - back] < PLATEAU_GAIN:
                m.active = False
            if ctx["exact_qps"] and timing["tune"].qps < ctx["exact_qps"]:
                m.active = False
        print(f"    nprobe={value:<5d} " + " | ".join(msg), flush=True)


def save(methods, args):
    rows = [r for m in methods for r in m.rows]
    replace_rows(args.results, rows, key_columns=KEY)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="first 100K vectors and the quick grid")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default=None, help="default: results/ (results/quick/ with --quick)")
    p.add_argument("--tune-time-queries", type=int, default=None, help="default 500 (quick: 100)")
    p.add_argument("--test-time-queries", type=int, default=None, help="default 2000 (quick: 200)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-repeat-s", type=float, default=None, help="default 0.2 (quick: 0.05)")
    p.add_argument("--threads", type=int, default=None, help="threads for building and recall")
    p.add_argument("--no-pin", action="store_true")
    args = p.parse_args()
    args.results_dir = args.results_dir or ("results/quick" if args.quick else "results")
    args.tune_time_queries = args.tune_time_queries or (100 if args.quick else 500)
    args.test_time_queries = args.test_time_queries or (200 if args.quick else 2000)
    if args.min_repeat_s is None:
        args.min_repeat_s = 0.05 if args.quick else MIN_REPEAT_S
    args.settle_s = 0.1 if args.quick else 0.25    # pause before a timing block (D24, D30)
    args.results = os.path.join(args.results_dir, "results.csv")
    grid = GRID["quick" if args.quick else "full"]
    t_start = time.perf_counter()

    write_machine_info(os.path.join(args.results_dir, "machine_info.json"))
    ds = load_dataset(DATASET, args.data_dir, 100_000 if args.quick else None)
    print(f"=== Part 3 compression: {DATASET} base {ds.base.shape}, nlist={grid['nlist']}, "
          f"m={grid['m']}, re-rank depth {RERANK_DEPTH} ===", flush=True)
    gt = ground_truth_for(ds, k=100, cache_dir=os.path.join(args.data_dir, "cache"))
    q_tune, q_test = split_queries(ds.queries)
    gt_tune, gt_test = split_queries(gt)
    splits = ((q_tune, gt_tune), (q_test, gt_test))
    pcores = None if args.no_pin else performance_cores()
    ctx = {"pcores": pcores,
           "pin_note": f"1 thread, pinned to P-cores {pcores[0]}-{pcores[-1]}" if pcores else "1 thread, not pinned",
           "all_threads": args.threads or os.cpu_count(),
           "exact_qps": exact_single_qps(args.results, DATASET, ds.n_base)}
    overhead = call_overhead_qps(ds.dim, 2000, pcores, args.repeats)
    print(f"exact single-query QPS (Part 1): {ctx['exact_qps']}; call-overhead ceiling: {overhead:,.0f} QPS")
    drop_rows(args.results, experiment="part3", dataset=DATASET, n_base=ds.n_base)
    raw_bytes = ds.base.nbytes
    nlist = grid["nlist"]

    # Reference: IVF-Flat, same nlist / seed / training sample as every IVF-PQ below.
    with limit_threads(ctx["all_threads"]):
        ivf, info = build_index(IndexSpec("IVFFlat", {"nlist": nlist}), ds.base, seed=SEED)
    ref_centroids = coarse_centroids(ivf)
    ref = Method("IVFFlat", {"nlist": nlist}, ivf, index_size_bytes(ivf), info["build_s"], info)
    print(f"\n  IVF-Flat nlist={nlist}: build {info['build_s']:.1f}s, {ref.size / ds.n_base:.1f} B/vec", flush=True)
    sweep([ref], ds, splits, ctx, args)
    save([ref], args)
    del ivf, ref

    same_centroids = []
    for m in grid["m"]:
        spec = IndexSpec("IVFPQ", {"nlist": nlist, "m": m})
        with limit_threads(ctx["all_threads"]):
            pq, info = build_index(spec, ds.base, seed=SEED)
            t0 = time.perf_counter()
            rr = add_exact_rerank(pq, ds.base, depth=RERANK_DEPTH, k=K)
            copy_s = time.perf_counter() - t0
        same = bool(np.array_equal(coarse_centroids(pq), ref_centroids))
        same_centroids.append(same)
        pq_size = index_size_bytes(pq)
        rr_size = pq_size + raw_bytes   # = serialised IndexRefineFlat up to a few header bytes
        extra = {"same_centroids_as_ivfflat": same}
        plain = Method("IVFPQ", {"nlist": nlist, "m": m}, pq, pq_size, info["build_s"], info, extra)
        rerank = Method("IVFPQ+rerank", {"nlist": nlist, "m": m, "rerank": RERANK_DEPTH}, rr, rr_size,
                        info["build_s"] + copy_s, info,
                        {**extra, "rerank_depth": RERANK_DEPTH, "raw_vector_bytes": raw_bytes,
                         "pq_index_bytes": pq_size, "copy_raw_s": round(copy_s, 3)})
        print(f"\n  IVF-PQ m={m}: build {info['build_s']:.1f}s (train {info['train_s']:.1f}s), "
              f"{pq_size / ds.n_base:.1f} B/vec plain, {rr_size / ds.n_base:.1f} B/vec with raw vectors; "
              f"centroids identical to IVF-Flat: {same}", flush=True)
        sweep([plain, rerank], ds, splits, ctx, args)
        save([plain, rerank], args)
        del pq, rr, plain, rerank

    summary = {"dataset": DATASET, "n_base": ds.n_base, "nlist": nlist, "m": grid["m"],
               "rerank_depth": RERANK_DEPTH, "same_centroids_as_ivfflat": all(same_centroids),
               "exact_single_qps": ctx["exact_qps"], "call_overhead_qps": overhead,
               "pinning": ctx["pin_note"], "build_threads": ctx["all_threads"],
               "sweep": (SWEEPS_QUICK if args.quick else SWEEPS)["IVFPQ"],
               "stop_recall": STOP_RECALL, "plateau_gain": PLATEAU_GAIN,
               "tune_time_queries": args.tune_time_queries, "test_time_queries": args.test_time_queries,
               "repeats": args.repeats, "min_repeat_s": args.min_repeat_s,
               "memory_accounting": "serialised index; re-rank includes float32 base vectors (held in RAM)",
               "query_split": f"tune [0:{N_TUNE}], test [{N_TUNE}:10000]"}
    with open(os.path.join(args.results_dir, "part3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPart 3 done in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
