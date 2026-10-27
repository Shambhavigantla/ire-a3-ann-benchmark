# Report notes: what to write, with numbers and mechanisms

Raw material for `report.pdf` (at most 5 pages), arranged the way the assignment asks for it.
The numbers come from the results as of Part 2. **Before submitting, regenerate them from one clean
`bash run_all.sh`** (DESIGN_NOTES D27): recall will come out identical, and QPS may move a little.
Table and figure ids match `results/tables/` and `results/figures/`.

---

## Setup paragraph (put the machine and versions in the report, Table T0)

- Intel Core i5-13500 (6 P-cores + 8 E-cores, 20 logical CPUs), 15.7 GB RAM, Windows 11.
- Python 3.12, faiss-cpu 1.15.1, numpy 2.5.4.
- Protocol:
  - tune on queries 0–4999, report on 5000–9999;
  - QPS one query at a time, 1 thread (OpenMP and OpenBLAS both capped), pinned to the P-cores;
  - median of 3 repeats, each at least 0.2 s;
  - builds use all 20 threads.

## Part 1: exact baseline

**Exact search.** `IndexFlatL2`. GloVe is L2-normalised, so ‖a−b‖² = 2 − 2cos and Euclidean ranking
equals angular ranking.

**Recall@10 with ties.**
- A returned id is a hit if its exact distance to the query is ≤ d₁₀, the distance of the 10th
  ground-truth neighbour.
- Distances are recomputed in float64 from the raw vectors, never taken from the index, because
  IVF-PQ only returns approximate ones.
- Duplicates count once; `-1` and missing results are misses.
- Unit tests cover: a tie at the 10th neighbour, a point just past it, an integer grid with many ties,
  duplicates, `-1` padding, too few results, wrongly ordered ground truth, and agreement with plain id
  overlap when there are no ties.

**Agreement with the file (T1a).** Tie-aware recall is 1.0 on all 10,000 queries for both datasets.
- **SIFT:** 65 queries differ from the file only by ties (plain id-overlap recall 0.99935). SIFT
  components are integers, so exact distance ties occur. **This shows the tie rule matters.**
- **GloVe:** 0 ties (continuous values).
- **Distances:** SIFT matches the file within 1.2e−7 relative error. GloVe's file stores cosine
  distance, and ours = 2 × that, within 2.6e−5.

**Single query vs batch (T1b, F1).**

| | SIFT | GloVe |
|---|---|---|
| One query at a time, 1 thread | 41 QPS (24.2 ms) | 43 QPS (23.2 ms) |
| One batch of 5000, 1 thread | 380 QPS (8–9× faster) | 340 QPS |
| One batch, 20 threads | 737 QPS | 500 QPS |

*Mechanism:*
- **A single query is limited by memory bandwidth.** It streams all 488 MB of SIFT vectors in
  24.2 ms, about 20 GB/s, roughly what one core can pull from DRAM. It does only about 2 flops per
  byte loaded.
- **A batch is limited by compute.** From 20 queries upwards, FAISS computes ‖q‖² + ‖x‖² − 2q·x with
  one sgemm. Each block of base vectors is loaded into cache once and reused for many queries. That
  reaches about 97 GFLOPS on one core (256 MFLOP per query in 2.6 ms).
- Using all 20 threads adds only 1.5–2×. It's context only, not part of the Part 2 protocol.

## Part 2: benchmark

**Required items:** F2 (`fig2_pareto`, plus one file per dataset), T2a (QPS at 0.90 and 0.95), T2b
(build time, size and bytes per vector at 0.95). T2c (best recall per build) supports the failure
sentence.

**Protocol to state in one or two lines.**
- Build settings: HNSW M ∈ {8, 16, 32} (efC=200); IVF-Flat nlist ∈ {1024, 4096, 16384}; IVF-PQ
  nlist ∈ {1024, 4096} × m ∈ {d/8 or d/5, d/4, d/2}.
- efSearch / nprobe go up in ×√2 steps, nprobe ≤ 1024.
- The chosen setting is the fastest that reaches the target on **tune**; T2a reports its **test**
  recall and QPS.

**Why IVF-PQ never reaches high recall (the required sentence).**
- Best recall is 0.884 on SIFT (m=64) and 0.869 on GloVe (m=50); with m=16 it tops out at 0.59.
- PQ ranks candidates by an approximate distance computed from m-byte codes. When that error is
  larger than the gap between the 10th and 11th true distances, true neighbours drop out of the top
  10.
- Raising nprobe only adds candidates, all ranked with the same error, so the curve goes vertical in
  F2.
- More bytes raise the ceiling. Re-ranking candidates with exact distances removes it.

## Q1. Fastest at 0.95 on each dataset? Does it change?

| at recall 0.95 | SIFT | GloVe |
|---|---|---|
| HNSW | **8,246 QPS** (200× exact), M=32, ef=48 | **889 QPS** (21×), M=32, ef=384 |
| IVF-Flat | 2,331 (56×), nlist=16384, nprobe=96 | about 360 (8×), two settings tie (D27) |
| IVF-PQ | doesn't reach it | doesn't reach it |

- **The winner doesn't change: HNSW on both.** The size of the advantage does: 200× on SIFT, 21× on
  GloVe.
- **Why HNSW beats IVF-Flat:**
  - The graph walk moves greedily towards the query, so it computes distances mostly to vectors that
    really are close.
  - IVF-Flat at 0.95 on SIFT compares the query with all 16,384 centroids, then scans 96 whole
    clusters (about 96 × 61 ≈ 5,900 vectors), most of them not neighbours.
  - *(Optional, to strengthen this: count HNSW distance computations with `faiss.cvar.hnsw_stats`.)*
- **Why GloVe is about 10× harder for ANN.** Exact search costs the same on both datasets, but GloVe
  needs much more search effort (ef 384 vs 48, nprobe about 1,000 vs 96) for the same recall.
  - Word vectors are less clustered, and a query's nearest neighbours are only slightly closer than
    everything else (low relative contrast).
  - So true neighbours are spread over many IVF cells, and greedy graph steps are less decisive.
  - A Part 3 "query difficulty" run would test this directly.

## Q2. Build time and memory. What does PQ save and give up?

**Bytes per vector follow from the data structures (T2b):**
- **IVF-Flat:** 520 B = 512 (raw float32) + 8 (int64 id), so 1.03× raw.
- **HNSW M=32:** 784 B = 512 + 2M × 4 = 256 bytes of level-0 links + upper levels. M=16: 656 B.
  So 1.5–1.7× raw.
- **IVF-PQ m=64:** 74 B = 64 (code) + 8 (id) + centroids, so 0.15× raw (7× smaller).

**Build time (all threads, single measurement, so rough):**
- **IVF-Flat:** 4–20 s for nlist ≤ 4096, but about 4–7 minutes for nlist=16384. k-means cost grows
  with training size × nlist × d.
- **HNSW:** 40–120 s. Every insertion is itself a graph search, and the cost grows with M and n.
- **IVF-PQ:** 50–80 s, dominated by training m separate 256-centroid k-means. That's about 0.9 s
  per sub-quantizer, so roughly proportional to m.

**PQ saves** about 85% of memory, which matters most at 10M+ vectors or on small machines. **It gives
up** recall above about 0.88 unless candidates are re-ranked with exact distances, and re-ranking
needs the raw vectors again, in RAM or on disk.

## Q3. What Parts 3 and 4 show, and when the headline numbers could mislead

*(after Parts 3 and 4)*

## Q4. 150K or 10M vectors, 100 ms per query, 16 GB RAM

Extrapolated from the 1M timings; say so in the report.
- **150K vectors:** exact search takes about 24.2 ms × 0.15 ≈ 3.6 ms per query (the cost is linear in
  N; the quick run measured 2.6 ms at 100K) and needs about 77 MB. **Don't use ANN**: exact search is
  about 28× under budget, has recall 1.0, and needs no tuning or rebuilding.
- **10M vectors:** exact search takes about 240 ms per query, over budget. The raw vectors are
  5.1 GB (SIFT-like).
  - **Use HNSW**, M=16–32, with efSearch tuned on held-out queries. At 1M it takes 0.1–1 ms per
    query, so there is room to raise ef for recall above 0.99, even if ef has to grow at 10M.
    Memory is about 6.6 GB (M=16) to 7.8 GB (M=32), which fits in 16 GB with headroom.
  - **Alternative:** IVF-Flat with nlist ≈ 4√N ≈ 12K. Cheaper to build, about 5.2 GB.
  - **Nuance:** if queries arrive in batches, exact search with 20 threads (1.4 ms per query at 1M,
    so about 14 ms at 10M) meets the budget for *throughput*, but not for a single query's latency.
  - *(If Part 3 is "scaling", use its measured growth instead of linear extrapolation.)*

## Q5. Two ways our benchmark could mislead (all of these were tested)

1. **Speed varies between runs by ±30–40%; recall doesn't change.** 175 settings were repeated in a
   second run. Recall was identical; QPS ratio medians were 0.92–1.20 and ranges 0.69–1.43. The same
   build took 221 s in one run and 396 s in the other. Within one run, the 3 repeats agree within a
   few percent. *Consequence:* compare only within a run, and don't read differences of 10–20%
   (D27).
2. **Hidden threading and CPU effects.** We found and fixed four:
   - OpenBLAS ignored the FAISS thread limit (a "1-thread" batch used 3.3 cores);
   - Windows moved runs between P-cores and E-cores (1.8× spread between repeats);
   - spin-waiting OpenMP workers made the following 1-thread timing 24% slower;
   - repeats of a few milliseconds were too short to time.

   Every row stores CPU/wall as evidence (D13, D14, D24, D25).
3. **Sweep limits can change the conclusion.** With nprobe capped at 512, GloVe IVF-Flat's result at
   0.95 was a limit of our sweep, not of the index. Raising the cap changed the table (D26).
4. **Python call overhead** is about 2.5 µs per call. That's at most about 15% of the fastest HNSW
   timings, so it slightly compresses the differences at the high-QPS end (D23).
5. **Not tested:** k ≠ 10; other hardware; batched or multi-threaded serving; inserts and deletes;
   more than two datasets.
