# Design notes

This is a running record of design decisions: what we chose, why, and what we rejected. Add
entries as the experiments progress. Each decision has an id (D*n*) so the report can cite it.
The plain-language walkthrough of the steps is in [PROCESS_LOG.md](PROCESS_LOG.md).

---

## Required outputs (submission checklist)

Status: ✅ done · 🟡 code done, waiting on the full run · ⬜ not started.

| # | Output | Part | File(s) in `results/` | Status |
|---|---|---|---|---|
| T0 | Machine specs and library versions | all | `tables/t0_machine.*`, `machine_info.json` | ✅ |
| T1a | Exact search agrees with the file's ground truth; build time and size | 1 | `tables/t1a_exact_check.*` | ✅ |
| T1b | Exact search QPS: single query vs batch (1 thread and all threads) | 1 | `tables/t1b_exact_timing.*` | ✅ |
| F1 | Bar chart of exact search, single vs batch | 1 | `figures/fig1_exact_single_vs_batch.*` | ✅ |
| — | Unit tests for recall@10 | 1 | `tests/test_metrics.py` | ✅ |
| F2 | Recall@10 vs QPS, best trade-off curve per index plus the exact baseline, one plot per dataset (plus a side-by-side version) | 2 | `figures/fig2_pareto_{sift,glove}.*`, `fig2_pareto.*` | ✅ (final run pending, D27) |
| T2a | QPS of each index at recall 0.90 and 0.95 | 2 | `tables/t2a_qps_at_recall.*` | ✅ (final run pending, D27) |
| T2b | Build time, index size and bytes per vector for each index at recall 0.95 | 2 | `tables/t2b_cost_at_095.*` | ✅ (final run pending, D27) |
| T2c | Highest recall each build setting reaches (supports the "why an index fails" text) | 2 | `tables/t2c_max_recall.*` | ✅ (final run pending, D27) |
| — | Why an index fails to reach high recall (text) | 2 | report | ⬜ |
| F3 | Part 3 main figure | 3 | `figures/fig3_*` | ⬜ |
| — | Part 3 hypothesis committed before running | 3 | `hypothesis.md` | ⬜ |
| F4 | Part 4 main figure (one figure, at most one page) | 4 | `figures/fig4_*` | ⬜ |
| — | Part 4 hypothesis committed before running; seeded generator | 4 | `hypothesis.md`, `ann_bench/synthetic.py` | ⬜ |
| — | `results.csv`: one row per run behind every reported number | all | `results.csv` | 🟡 |
| — | Report ≤ 5 pages answering Q1–Q5 | 5 | `report/` → `report.pdf` | ⬜ |
| — | `ai_log/` and `ai_statement.md` (outside `code/`, at the top level of the zip) | — | `../ai_log/` | 🟡 |

---

## General

**D1. FAISS for all three indexes (`IndexHNSWFlat`, `IndexIVFFlat`, `IndexIVFPQ`).**
hnswlib has no Windows wheel and would need a C++ compiler to install. A single library also means
the same distance kernels, the same threading and the same Python call overhead for every index. That
makes comparing QPS fairer. *Rejected:* hnswlib for HNSW.

**D2. Squared L2 everywhere; GloVe L2-normalised at load time.**
FAISS's `IndexFlatL2` returns squared distances. Squaring keeps the order of neighbours, so rankings
and recall are the same as with plain L2. For unit vectors, ‖a−b‖² = 2−2cos, so after normalising,
Euclidean ranking equals the angular ranking the GloVe file was built with. Normalisation is done
once, in `load_dataset`, for both base and queries, so no code path can see un-normalised GloVe.

**D3. Fixed query split.** Queries 0–4999 are for tuning, 5000–9999 for reported numbers.
Parameters that reach a recall target are chosen on the tune half. All reported QPS and recall come
from the test half. This keeps the reported numbers from overfitting the tuning.

**D4. One results table for everything.** Each benchmark run writes one row to
`results/results.csv` with a fixed set of columns (`ann_bench/results.py`). A re-run replaces its
own rows, keyed by (experiment, dataset, n_base), so the file never collects stale duplicates. Every
table and figure is regenerated from this CSV plus small JSON summaries, never typed in by hand.

**D5. Quick mode = first 100K base vectors, written to `results/quick/`.** The file's ground truth is
invalid for a subset, so it is recomputed by brute force and cached in `data/cache/`. Quick results
go in their own folder so they can never overwrite the full results.

**D6. Figures.** matplotlib, static PNG (for viewing) and PDF (for the report), white background.
Each index has one colour in every figure: HNSW blue, IVF-Flat orange, IVF-PQ aqua, IVF-PQ+rerank
yellow, exact grey (`ann_bench/plotting.py`). These are the first slots of a palette checked for
colour-blindness, used in a fixed order. Axes start at zero for bars, and bars are never drawn on a
log scale, because that distorts their lengths. That was caught and fixed in F1. Recall vs QPS
curves can use a log QPS axis, because those are lines and points, not bars.

## Part 1: exact baseline

**D7. Tie-aware recall@10 compares distances, not ids** (`ann_bench/metrics.py`).
- The threshold d₁₀ is the largest exact distance from the query to its 10 ground-truth ids.
- A returned id is a hit if its exact distance is ≤ d₁₀·(1+10⁻⁶).
- This is the same as "id is in the ground truth, or is tied with the 10th neighbour", because every
  ground-truth id is within d₁₀.
- Duplicate ids count once. `-1` padding and missing results count as misses.

*Why:* SIFT has integer components, so exact ties at the 10th neighbour happen. The ground truth can
only list one of the tied vectors.
*Rejected:* plain id intersection. It is still reported as `recall_strict`, so the effect of ties
is visible.

**D8. Distances for recall are recomputed in float64 from the raw vectors, never taken from the
index.** IVF-PQ returns approximate distances that could put a far vector under the threshold. In
float64, true ties among float32 inputs come out bit-identical, so the 10⁻⁶ slack only absorbs
rounding.

**D9. Threshold = the largest of the 10 recomputed ground-truth distances, not the 10th entry as
listed.** This keeps the threshold safe even if the file's order is slightly off.

**D10. Checking the subset against the file.** For a 100K prefix, if at least 10 of a query's 100
file neighbours have id < n, those are exactly the subset's top 10. Any other subset vector lies
beyond the full set's 100th neighbour. This gives an independent check even in quick mode: 5,078
SIFT and 3,376 GloVe queries qualify.

**D11. Timing protocol.**
- `time.perf_counter`, a 50-query warm-up, 3 repeats, median reported.
- The Python loop is included in single-query time, the same for every index.
- Three modes: single query with 1 thread (the Part 2 protocol), batch with 1 thread, batch with
  all threads. The 1-thread batch separates the effect of batching (BLAS, cache reuse) from the
  effect of parallelism.

**D12. The single-query exact timing uses the first 1,000 test queries (300 in quick mode).**
QPS is a rate, and all 5,000 queries on 1M vectors at about 40 ms each would take over 3 minutes per
repeat. Recall on those 1,000 is still computed.

**D13. "1 thread" has to cap BLAS too (bug found in the first full run).**
FAISS's batch search calls OpenBLAS (`faiss_cpu.libs/libopenblas.dll`), which has its own thread pool
and ignores `faiss.omp_set_num_threads`. A diagnostic showed a "1-thread" batch using 3.3 cores of CPU
time. `limit_threads(n)` now caps OpenMP and BLAS together, via threadpoolctl. Single-query search
was never affected: with fewer than 20 queries FAISS doesn't use BLAS, and CPU/wall was 1.0.
*Guard:* every run records `cpu_per_wall` (process CPU time ÷ wall time) in `results.csv`. A 1-thread
run above 1.3 prints a warning.

**D14. Pin 1-thread timings to P-cores.** The i5-13500 is a hybrid CPU: logical CPUs 0–11 are P-cores
and 12–19 are E-cores. A pinned microbenchmark measured 7.3 ms per query on P-cores and 10.5–11.6 ms
on E-cores. Windows moved the single-threaded process between the two types. In the first full run,
the three GloVe single-query repeats took 50.4 s, 50.2 s and 28.0 s, so the median reflected the
E-cores. `performance_cores()` reads each CPU's EfficiencyClass from Windows
(`GetSystemCpuSetInformation`; on Linux, `/sys/devices/cpu_core/cpus`). All 1-thread runs are then
pinned to the P-cores. The multi-threaded run uses all cores. The pinning is recorded in the `notes`
column. `--no-pin` turns it off.
*Implication for Part 2:* every ANN timing must use the same `pinned_to(performance_cores())` and
`limit_threads(1)`. Otherwise the indexes are timed on different core types.

**Part 1 results (full data, after D13/D14). Use these in the report.**

| | SIFT 1M × 128 | GloVe 1.18M × 100 |
|---|---|---|
| Single query, 1 thread | 41.3 QPS (24.2 ms) | 43.2 QPS (23.2 ms) |
| Batch of 5000, 1 thread | 379.5 QPS (9.2×) | 339.9 QPS (7.9×) |
| Batch of 5000, 20 threads | 737.4 QPS (17.9×) | 499.8 QPS (11.6×) |
| Agreement with the file (tie-aware) | 1.0 on 10,000 queries | 1.0 on 10,000 queries |
| Queries that differ only by ties | 65 (strict recall 0.99935) | 0 |
| Top-100 identical in order | 44.0% | 98.8% |

- *Ties:* SIFT's integer components produce exact ties, so 65 queries would be wrongly penalised by
  plain id matching, and the top-100 order differs often. GloVe's normalised float vectors almost
  never tie exactly.
- *Distances match the file:* SIFT stores L2, max relative error 1.2e-7. GloVe stores cosine
  distance (1−cos) and our squared L2 = 2 × that, max relative error 2.6e-5, which is float32
  rounding.
- *Single-query speed scales with the data size:* GloVe has 118M floats and SIFT 128M. The ratio
  0.92 roughly matches the measured 23.2 / 24.2 = 0.96. That is the signature of a scan limited by
  memory bandwidth.
- *Multi-threaded batch scales poorly* (GloVe used only 5.8 cores of CPU time). It isn't part of the
  Part 2 protocol and is reported only as context. Possible causes are OpenBLAS's threading thresholds
  for d=100 and the serial top-k merge. Not investigated further.
- *Stability:* the three single-query repeats are within 1% (SIFT) and 5% (GloVe). Before D14 they
  differed by up to 1.8×.

**Superseded observation (first runs, before D13/D14).** The "1-thread" batch was actually using about
3 cores, and unpinned single-query runs landed partly on E-cores. Examples: SIFT quick single 266 QPS
before vs 392 after; GloVe full single 19.9 before vs 43.2 after. None of those numbers should be
used.

## Part 2: three indexes

**D15. What's measured for each (build, search) setting** (`scripts/part2_benchmark.py`).
- **Recall:** on all 5,000 tune and all 5,000 test queries, using a multi-threaded batch search.
  An index returns the same neighbours whether queries come one at a time or in a batch, so batching
  only saves time. Recall uses D7.
- **QPS:** one query at a time, `limit_threads(1)`, pinned to the P-cores (D13/D14), median of 3
  repeats. It's timed on the first 500 tune queries and the first 2,000 test queries; quick mode uses
  300 and 500. QPS is a rate, so a subsample is enough, and this keeps the full run at roughly an hour.
- **Rows:** each setting writes one row for the tune half and one for the test half.

**D16. Choosing settings for the tables (tune/test protocol).** For each recall target (0.90, 0.95)
and each index, the chosen setting is the fastest one whose **tune** recall reaches the target, across
all build settings. The tables report that setting's **test** recall and QPS. Test recall can land
slightly under the target. That is reported as it is, not hidden: it is the honest generalisation gap.
*Rejected:* interpolating along the test curve. That picks settings by looking at the test results.

**D17. Best trade-off curve** (`ann_bench/frontier.py`). A run is on the curve if no other run of the
same index family, across all build settings, is both at least as accurate and faster. The figure
shows the curve as a line, all runs as faint dots, and exact search as a star with a dashed line. The
QPS axis is log scale, which is fine for lines and points (D6).

**D18. Build settings: one knob per family.**
- **HNSW:** M ∈ {8, 16, 32}, with efConstruction fixed at 200.
- **IVF-Flat:** nlist ∈ {1024, 4096, 16384}, roughly √N, 4√N and 16√N. Quick mode uses {256, 1024}.
- **IVF-PQ:** nlist ∈ {1024, 4096}, code size m ∈ {d/8 or d/5, d/4, d/2} bytes. That is SIFT
  m ∈ {16, 32, 64} and GloVe m ∈ {20, 25, 50}, because m must divide d. 8 bits per sub-code, with
  residual encoding.

**D19. Search sweeps.**
- Values go up in steps of about ×√2: 1, 2, 3, 4, 6, 8, 12, 16, 24, …
- efSearch from 10 up to 1024; nprobe from 1 up to 512, but never above nlist.
- With ×2 steps, the first value reaching a target could be far past it. On SIFT, IVF-Flat jumped
  from tune recall just under 0.90 to 0.966, so "QPS at 0.90" was really QPS at 0.966.
- Quick mode keeps ×2 steps, because it only checks that the pipeline runs.
- A sweep stops early on any of three rules:
  - tune recall ≥ 0.999;
  - doubling the knob (two ×√2 steps) gains < 0.001 tune recall (a plateau, so further points
    could only be dominated);
  - the index is slower than exact single-query search.
- The plateau rule was added after the first quick run. There, IVF-PQ kept sweeping at a constant
  recall (SIFT m=16 sat at 0.646 from nprobe 64 to 512).

**D20. Building.**
- Builds use all threads, and the build time is wall time with all threads. This is noted in the
  tables.
- IVF k-means trains on a seeded sample of min(n, max(100K, 50·nlist)) vectors. FAISS warns below
  39·nlist.
- PQ codebooks use FAISS's own cap of 65,536 training points.
- IVF builds are bit-reproducible, and a test checks this. HNSW's multi-threaded insertion is not
  exactly reproducible.

**D21. Turned off FAISS's polysemous training for IVF-PQ.** `index_factory` turns it on by default. It
only helps polysemous filtering (`polysemous_ht > 0`), which we never use. It cost 10–25% of IVF-PQ
training time: m=64 took 75.6 s with it and 57.0 s without on 100K vectors. It doesn't change the
encoding or search results.

**D22. IVF-PQ build time is dominated by PQ training, and the cost doesn't shrink with n.** Each
sub-quantizer is a 256-centroid k-means on up to 65,536 points, about 0.9 s each here, so training
time ≈ 0.9·m seconds. The quick grid therefore keeps only the two smallest codes. *Report point:* PQ
saves memory but costs more build time than IVF-Flat or HNSW at this scale.

**D23. Python call overhead is measured, not assumed.** Searching a 1-vector index gives the ceiling
from overhead alone: about 394K QPS (2.5 µs per call). The fastest HNSW setting takes about 16 µs
per query, so overhead is at most about 15% of the fastest runs and negligible for the rest. It is
recorded in `part2_summary.json`.

**D24. Let OpenMP threads go idle before timing (bug found in the quick run).** After a
multi-threaded FAISS call (a build, or the recall search), FAISS's idle worker threads keep spinning
for about 0.1–0.2 s. They burn CPU on the same P-cores as the timed thread. In a controlled test, a
1-thread HNSW timing started immediately afterwards showed CPU/wall about 9–10 and **24% lower QPS**
(13.6K vs 17.9K). With a 0.2 s pause it was back to about 1.0 and full speed. `time_search` now sleeps
`settle_s = 0.25` s first. Part 1 wasn't affected: its 50-query warm-up took over a second.

**D25. Each timing repeat lasts at least 0.2 s.** Fast settings (around 20 µs per query) finished a
repeat over 300 queries in about 6 ms. That's below Windows' CPU-clock tick (about 15.6 ms) and too
short to measure reliably: tune and test QPS for the same setting differed by up to 2×. The warm-up
now estimates the time for one pass over the queries, and each repeat loops over the query set until
it reaches `min_repeat_s` (0.2 s; 0.1 s in quick mode). The number of passes is stored as
`timing_loops`. After this fix, CPU/wall stayed at 0.99–1.20 and the median tune/test QPS ratio was
1.00 (10th–90th percentile 0.88–1.15; the two halves are different query sets).
*Rule for Parts 3/4:* use `time_search(..., min_repeat_s=0.2)` for every 1-thread QPS.

**D26. The IVF nprobe cap was raised from 512 to 1024 after the first full run.** On GloVe, IVF-Flat
with nlist=16384 was still improving at the cap: recall 0.927 at 712 QPS for nprobe=512. Its likely
crossing of 0.95 (about 450 QPS by extrapolation) would beat the 399 QPS chosen in T2a. So the T2a
entry was a limit of the sweep, not of the index. Several GloVe IVF-PQ sweeps also ended at the cap
without plateauing. The affected builds were rerun with the higher cap: GloVe IVF-Flat and IVF-PQ,
and SIFT IVF-Flat. SIFT IVF-PQ had already plateaued below 512. The first full run's CSV is kept as
`results/results_before_ivf1024.csv.bak` (not submitted).
*HNSW caps that were left alone:* GloVe HNSW with M=8 (0.865) and M=16 (0.952) still end at
efSearch=1024. Both are dominated by M=32 at every recall they reach, so a higher cap would only add
slow, dominated points. The report must not present "M=8 reaches 0.865" as that build's ceiling.

**D27. Speed is reproducible within a run but varies ±30–40% between runs; recall is exactly
reproducible.** The D26 rerun repeated 175 (build, search) settings from the first full run:

| Rerun builds | Recall change | QPS rerun ÷ first run, median (range) |
|---|---|---|
| SIFT IVF-Flat (49 settings) | 0 | 0.99 (0.93–1.05) |
| GloVe IVF-Flat (54) | 0 | 0.92 (0.69–1.21) |
| GloVe IVF-PQ (72) | 0 | 1.20 (0.86–1.43) |

The same GloVe nlist=16384 build took 221 s in the first run and 396 s in the rerun. The machine was
busier during part of one run: it was locked overnight, so Windows maintenance tasks are a likely
cause. Within a run, the 3 repeats agree within a few percent and CPU/wall ≤ 1.06, so the noise is
between runs, not within one.
*Consequences:*
1. Only compare QPS from the same run. The GloVe IVF-Flat choice at 0.95 swapped between
   nlist=16384/nprobe=1024 (357 QPS) and nlist=4096/nprobe=384 (356 QPS). Those are identical
   within noise, so the report should say "IVF-Flat ≈ 360 QPS", not argue over which nlist wins.
2. Conclusions that rest on gaps much larger than 40% are safe: HNSW 3.5× faster than IVF-Flat on
   SIFT at 0.95, 2.5× on GloVe; IVF-PQ never reaching 0.90; exact search 20–200× slower.
3. **Final numbers must come from one uninterrupted `bash run_all.sh` on an idle machine:** other
   apps closed, plugged in, Windows Update paused. The current `results.csv` mixes two runs (HNSW and
   SIFT IVF-PQ from the first; the IVF reruns from the second).
4. This is a ready answer to report Q5 ("ways the benchmark could mislead"), and it was tested.
5. Build times are single measurements, so they are just as noisy. Treat them as rough sizes.

## Part 3

*(choice, hypothesis, design)*

## Part 4

*(question, generator, control, settings)*
