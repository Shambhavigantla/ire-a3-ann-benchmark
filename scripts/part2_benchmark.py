#!/usr/bin/env python
"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on both datasets.

For every build configuration in GRIDS, build once (all threads, timed), then sweep
the search knob (efSearch / nprobe) upwards. For each value:
  * recall@10 on ALL tune queries and ALL test queries (batch search, all threads -
    results do not depend on batching, so this is just faster);
  * QPS one query at a time, 1 thread (OpenMP + BLAS capped), pinned to P-cores,
    median of --repeats, on the first --tune-time-queries tune queries and the
    first --test-time-queries test queries.
A sweep stops early once tune recall >= STOP_RECALL, once doubling-ish the knob
gains < PLATEAU_GAIN tune recall (further points would only be dominated), or once
the index is slower than exact single-query search (no reason to use it past that).

One results.csv row per (build config, search value, query set). Tables pick, on
the TUNE rows, the fastest config reaching each recall target and report its
TEST row (scripts/make_tables.py).

  python scripts/part2_benchmark.py                 # full data
  python scripts/part2_benchmark.py --quick         # 100K subset, smaller grids
  python scripts/part2_benchmark.py --datasets sift --indexes HNSW --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from ann_bench.data import N_TUNE, load_dataset, split_queries  # noqa: E402
from ann_bench.exact import ground_truth_for, index_size_bytes  # noqa: E402
from ann_bench.indexes import IndexSpec, build_index, set_search_param  # noqa: E402
from ann_bench.metrics import recall_at_k, strict_recall_at_k  # noqa: E402
from ann_bench.results import drop_rows, replace_rows, write_machine_info  # noqa: E402
from ann_bench.tables import read_results  # noqa: E402
from ann_bench.timing import limit_threads, performance_cores, pinned_to, time_search  # noqa: E402

K = 10
SEED = 0
STOP_RECALL = 0.999
PLATEAU_GAIN = 0.001
PART2_KEY = ("experiment", "dataset", "n_base", "index_type", "build_params")

# Roughly x sqrt(2) steps: fine enough that the first value reaching a recall target
# is not far past it (x2 steps jumped IVF-Flat on SIFT from 0.90 to 0.966).
_GEOM = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SWEEPS = {
    "HNSW": [10] + [v for v in _GEOM if v > 10],          # efSearch < k=10 is meaningless
    "IVFFlat": [v for v in _GEOM if v <= 1024],          # never above nlist (checked in the loop)
    "IVFPQ": [v for v in _GEOM if v <= 1024],
}
# --quick only checks that the pipeline reproduces, so x2 steps are enough there.
SWEEPS_QUICK = {kind: [v for v in vals if v == 10 or (v & (v - 1)) == 0] for kind, vals in SWEEPS.items()}
MIN_REPEAT_S = 0.2       # each timing repeat lasts >= this (fast indexes loop over the queries)


def _hnsw(*ms):
    return [{"M": m, "efConstruction": 200} for m in ms]


# Build settings. One knob varied per family; IVF-PQ code sizes m = d/8.., d/4, d/2 bytes.
GRIDS = {
    "full": {
        "sift": {"HNSW": _hnsw(8, 16, 32),
                 "IVFFlat": [{"nlist": 1024}, {"nlist": 4096}, {"nlist": 16384}],
                 "IVFPQ": [{"nlist": 1024, "m": 32}, {"nlist": 4096, "m": 16},
                           {"nlist": 4096, "m": 32}, {"nlist": 4096, "m": 64}]},
        "glove": {"HNSW": _hnsw(8, 16, 32),
                  "IVFFlat": [{"nlist": 1024}, {"nlist": 4096}, {"nlist": 16384}],
                  "IVFPQ": [{"nlist": 1024, "m": 25}, {"nlist": 4096, "m": 20},
                            {"nlist": 4096, "m": 25}, {"nlist": 4096, "m": 50}]},
    },
    # Quick: PQ training costs ~0.9 s per sub-quantizer regardless of n (it is capped at
    # 65,536 training points), so the quick grid keeps only two small IVF-PQ codes.
    "quick": {
        "sift": {"HNSW": _hnsw(8, 16, 32),
                 "IVFFlat": [{"nlist": 256}, {"nlist": 1024}],
                 "IVFPQ": [{"nlist": 1024, "m": 16}, {"nlist": 1024, "m": 32}]},
        "glove": {"HNSW": _hnsw(8, 16, 32),
                  "IVFFlat": [{"nlist": 256}, {"nlist": 1024}],
                  "IVFPQ": [{"nlist": 1024, "m": 20}, {"nlist": 1024, "m": 25}]},
    },
}


def exact_single_qps(results_path, dataset, n_base):
    for r in read_results(results_path, "part1_exact"):
        if r["dataset"] == dataset and int(r["n_base"]) == n_base and r["mode"] == "single":
            return float(r["qps"])
    return None


def call_overhead_qps(d, n_queries, pcores, repeats):
    """QPS ceiling from Python + FAISS call overhead alone: search a 1-vector index."""
    idx = faiss.IndexFlatL2(d)
    idx.add(np.zeros((1, d), dtype=np.float32))
    q = np.zeros((n_queries, d), dtype=np.float32)
    with pinned_to(pcores), limit_threads(1):
        tr = time_search(lambda x, k: idx.search(x, 1)[1], q, 1, "single", repeats)
    return tr.qps


def existing_build_keys(results_path, dataset, n_base):
    return {(r["index_type"], r["build_params"]) for r in read_results(results_path, "part2")
            if r["dataset"] == dataset and int(r["n_base"]) == n_base}


def run_build(ds, spec, splits, ctx, args):
    (q_tune, gt_tune), (q_test, gt_test) = splits
    with limit_threads(ctx["all_threads"]):
        index, info = build_index(spec, ds.base, seed=SEED)
    size = index_size_bytes(index)
    print(f"\n  {spec.label}: build {info['build_s']:.1f}s (train {info['train_s']:.1f}s on "
          f"{info['train_n']:,}), {size / 2**20:.1f} MB, {size / ds.n_base:.1f} B/vec", flush=True)

    def search(x, k):
        return index.search(x, k)[1]

    rows, tune_hist = [], []
    sweep = SWEEPS_QUICK if args.quick else SWEEPS
    for value in sweep[spec.kind]:
        if spec.kind != "HNSW" and value > spec.params["nlist"]:
            break
        set_search_param(index, spec, value)
        recall = {}
        with limit_threads(ctx["all_threads"]):
            for name, q, gt in (("tune", q_tune, gt_tune), ("test", q_test, gt_test)):
                ids = search(q, K)
                recall[name] = (float(recall_at_k(ids, ds.base, q, gt, k=K).mean()),
                                float(strict_recall_at_k(ids, gt, k=K).mean()))
        timing = {}
        with pinned_to(ctx["pcores"]), limit_threads(1):
            for name, q in (("tune", q_tune[:args.tune_time_queries]), ("test", q_test[:args.test_time_queries])):
                timing[name] = time_search(search, q, K, "single", args.repeats, min_repeat_s=args.min_repeat_s)

        for name in ("tune", "test"):
            tr = timing[name]
            if tr.cpu_per_wall > 1.3 and tr.median_s >= 0.05:   # CPU clock ticks ~15 ms on Windows
                print(f"    WARNING: 1-thread timing used {tr.cpu_per_wall:.1f} cores", flush=True)
            rows.append({
                "experiment": "part2", "dataset": ds.name, "n_base": ds.n_base,
                "index_type": spec.kind, "build_params": spec.params_json,
                "search_params": json.dumps({spec.search_param: value}),
                "query_set": name, "n_queries": tr.n_queries, "mode": "single", "threads": 1, "k": K,
                "recall_at_k": recall[name][0], "recall_strict": recall[name][1],
                "qps": tr.qps, "latency_ms": tr.latency_ms, "time_median_s": tr.median_s,
                "time_runs_s": json.dumps([round(t, 6) for t in tr.times_s]),
                "cpu_per_wall": tr.cpu_per_wall, "build_time_s": info["build_s"],
                "index_bytes": size, "bytes_per_vector": size / ds.n_base,
                "notes": json.dumps({"train_s": round(info["train_s"], 3), "add_s": round(info["add_s"], 3),
                                     "train_n": info["train_n"], "build_threads": ctx["all_threads"],
                                     "timing_loops": tr.loops,
                                     "recall_queries": len(q_tune if name == "tune" else q_test),
                                     "timing": ctx["pin_note"]}),
            })
        t, s = timing["test"], timing["tune"]
        print(f"    {spec.search_param}={value:<5d} recall tune {recall['tune'][0]:.4f} test "
              f"{recall['test'][0]:.4f} | QPS tune {s.qps:9.1f} test {t.qps:9.1f} "
              f"({t.latency_ms:.3f} ms, cpu/wall {t.cpu_per_wall:.2f})", flush=True)

        if recall["tune"][0] >= STOP_RECALL:
            break
        tune_hist.append(recall["tune"][0])
        back = 1 if args.quick else 2          # compare against ~half the current knob value
        if len(tune_hist) > back and tune_hist[-1] - tune_hist[-1 - back] < PLATEAU_GAIN:
            print(f"    stop: recall plateau (doubling gained < {PLATEAU_GAIN})", flush=True)
            break
        if ctx["exact_qps"] and s.qps < ctx["exact_qps"]:
            print(f"    stop: slower than exact search ({ctx['exact_qps']:.1f} QPS)", flush=True)
            break

    del index
    replace_rows(args.results, rows, key_columns=PART2_KEY)
    return rows


def run_dataset(name, args):
    ds = load_dataset(name, args.data_dir, args.n_base)
    print(f"\n=== {name}: base {ds.base.shape}, subset={ds.is_subset} ===", flush=True)
    gt = ground_truth_for(ds, k=100, cache_dir=os.path.join(args.data_dir, "cache"))
    q_tune, q_test = split_queries(ds.queries)
    gt_tune, gt_test = split_queries(gt)

    pcores = None if args.no_pin else performance_cores()
    ctx = {
        "pcores": pcores,
        "pin_note": f"1 thread, pinned to P-cores {pcores[0]}-{pcores[-1]}" if pcores else "1 thread, not pinned",
        "all_threads": args.threads or os.cpu_count(),
        "exact_qps": exact_single_qps(args.results, name, ds.n_base),
    }
    overhead = call_overhead_qps(ds.dim, 2000, pcores, args.repeats)
    print(f"exact single-query QPS (Part 1): {ctx['exact_qps']}; call-overhead ceiling: {overhead:,.0f} QPS")

    if args.resume:
        done = existing_build_keys(args.results, name, ds.n_base)
    else:   # fresh run: drop stale rows, e.g. from configs no longer in the grid
        done = set()
        for kind in args.indexes:
            drop_rows(args.results, experiment="part2", dataset=name, n_base=ds.n_base, index_type=kind)
    grid = GRIDS["quick" if args.quick else "full"][name]
    for kind in args.indexes:
        for params in grid[kind]:
            spec = IndexSpec(kind, params)
            if (kind, spec.params_json) in done:
                print(f"  skip {spec.label} (already in results.csv)")
                continue
            run_build(ds, spec, ((q_tune, gt_tune), (q_test, gt_test)), ctx, args)

    return {"dataset": name, "n_base": ds.n_base, "exact_single_qps": ctx["exact_qps"],
            "call_overhead_qps": overhead, "pinning": ctx["pin_note"],
            "build_threads": ctx["all_threads"], "grid": grid, "sweeps": SWEEPS,
            "stop_recall": STOP_RECALL, "plateau_gain": PLATEAU_GAIN, "tune_time_queries": args.tune_time_queries,
            "test_time_queries": args.test_time_queries, "repeats": args.repeats,
            "min_repeat_s": args.min_repeat_s, "sweeps_used": SWEEPS_QUICK if args.quick else SWEEPS,
            "query_split": f"tune [0:{N_TUNE}], test [{N_TUNE}:10000]"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["sift", "glove"])
    p.add_argument("--indexes", nargs="+", default=["HNSW", "IVFFlat", "IVFPQ"],
                   choices=["HNSW", "IVFFlat", "IVFPQ"])
    p.add_argument("--quick", action="store_true", help="first 100K vectors and the quick grids")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default=None, help="default: results/ (results/quick/ with --quick)")
    p.add_argument("--tune-time-queries", type=int, default=None, help="default 500 (quick: 300)")
    p.add_argument("--test-time-queries", type=int, default=None, help="default 2000 (quick: 500)")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--min-repeat-s", type=float, default=None, help="default 0.2 (quick: 0.1)")
    p.add_argument("--threads", type=int, default=None, help="threads for building and recall")
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip build configs already in results.csv")
    args = p.parse_args()
    args.n_base = 100_000 if args.quick else None
    args.results_dir = args.results_dir or ("results/quick" if args.quick else "results")
    args.tune_time_queries = args.tune_time_queries or (200 if args.quick else 500)
    args.test_time_queries = args.test_time_queries or (300 if args.quick else 2000)
    if args.min_repeat_s is None:
        args.min_repeat_s = 0.1 if args.quick else MIN_REPEAT_S
    args.results = os.path.join(args.results_dir, "results.csv")

    write_machine_info(os.path.join(args.results_dir, "machine_info.json"))
    t0 = time.perf_counter()
    summary = [run_dataset(name, args) for name in args.datasets]
    path = os.path.join(args.results_dir, "part2_summary.json")
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = {s["dataset"]: s for s in json.load(f)}
    old.update({s["dataset"]: s for s in summary})
    with open(path, "w") as f:
        json.dump(list(old.values()), f, indent=2)
    print(f"\nPart 2 done in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
