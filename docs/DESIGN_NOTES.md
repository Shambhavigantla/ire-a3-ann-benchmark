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
| F2 | Recall@10 vs QPS, best trade-off curve per index plus the exact baseline, one plot per dataset | 2 | `figures/fig2_pareto_{sift,glove}.*` | ⬜ |
| T2a | QPS of each index at recall 0.90 and 0.95 | 2 | `tables/t2a_qps_at_recall.*` | ⬜ |
| T2b | Build time, index size and bytes per vector for each index at recall 0.95 | 2 | `tables/t2b_cost_at_095.*` | ⬜ |
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

*(to be filled in)*

## Part 3

*(choice, hypothesis, design)*

## Part 4

*(question, generator, control, settings)*
