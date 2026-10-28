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

## Part 3: compression (F3, T3a, T3b)

**Set-up, in one or two lines.**
- SIFT 1M, IVF nlist = 4096; m ∈ {8, 16, 32, 64}.
- Each IVF-PQ index is searched plain (control) and with exact re-ranking of its top 100 (FAISS
  `IndexRefineFlat`). Reference: IVF-Flat with bit-identical centroids, so at equal nprobe all three
  scan the same clusters.
- Memory counts the original vectors as in RAM for re-ranking.
- *Prior evidence:* Part 2's plain IVF-PQ at m = 16/32/64 informed prediction 1. State it.

**Verdict on each prediction (test queries):**

| Prediction | Result | Verdict |
|---|---|---|
| 1. Plain recall rises with m, so does memory | best recall 0.409 / 0.588 / 0.742 / 0.884; 18 / 26 / 42 / 74 B per vector | supported |
| 2. Re-ranking helps at every m, more at small m | gain at nprobe 64: +0.460 / +0.378 / +0.239 / +0.103; positive at every nprobe | supported |
| 3. Re-ranked recall stays below IVF-Flat (can't recover missing candidates) | never above it, and recall = coverage on all rows. m ≥ 32: *equal* to IVF-Flat (≤ 0.0003). m = 8: gap grows to −0.122 at nprobe 256 | supported on the bound; "below" holds only for m ≤ 16 |
| 4. Re-ranking lowers QPS and needs the raw vectors | QPS 0.73–1.00× plain (one tie); +512 B per vector (26 → 538 B at m=16) | supported; the time cost is small, the memory cost large |

**Mechanisms to write:**
- **Why m matters.** With m sub-vectors, each 8-bit codebook quantises d/m dimensions with 256
  centroids. m = 8 means 16 dimensions per 256 centroids: large error, which misorders neighbours
  whose distances differ by less than that error.
- **Why re-ranked recall equals coverage.** Exact re-ranking returns the true 10 closest of the 100
  candidates, so every true neighbour among them is found and none outside can be.
- **Why the coverage gap grows with nprobe at m = 8.** More clusters bring more candidates competing
  for 100 slots. If the PQ error is large, true neighbours fall below rank 100. At m ≥ 32 the error is
  small enough that the true top 10 are always in the top 100, so re-ranking reaches the IVF-Flat
  limit, and further gains can only come from nprobe (cluster selection), not from m.
- **Why re-ranking costs little time.** 100 exact distances (about 12.8K multiplies plus 100 random
  51 KB reads) against scanning nprobe × ~244 codes, plus per-cluster PQ lookup tables.
- **Why plain PQ isn't faster per cluster than IVF-Flat here.** With nlist = 4096, a list holds only
  ~244 vectors, so per-cluster PQ set-up (lookup tables of 256 × m entries, residual terms) costs about
  as much as scanning the raw vectors. m = 64 at nprobe 64: 855 QPS vs IVF-Flat 1,369. *(Likely
  explanation, not separately measured: say so.)*

**Observation beyond the hypothesis.**
- At recall 0.95, re-ranked m = 16 took 2,298 QPS vs IVF-Flat 1,926 (+19%, same run) at 1.05× vs
  1.02× raw memory.
- Small codes are cheap to scan and good enough to put the true neighbours in the top 100. The exact
  step then fixes the order.
- The 19% is within the cross-run noise of D27, but it was measured within one run. Call it "about
  equal or slightly faster", not a clear win.

**For Q3 ("when the headline numbers mislead").**
- "IVF-PQ uses 0.15× memory" and "IVF-PQ + re-rank reaches 0.999" are each true, but not at the same
  time: the re-rank version needs the raw vectors (1.15×).
- Plain-PQ recall at a single m also hides that re-ranking changes the best m: with re-ranking,
  m = 16 is enough for 0.95, while without it no m reaches 0.90.

**Not tested:** re-rank depth ≠ 100; nlist ≠ 4096; GloVe; raw vectors on SSD; OPQ.

## Part 4: drift. A shifted cluster appears (F4, T4a, T4b, T4c). Fits in one page.

**What we changed.** One property: the shift s of a new 100K-vector cluster, inserted into indexes
built on D (400K SIFT).
- The cluster is real SIFT vectors [400K:500K] translated by s·σ·u: σ = 385 (RMS spread of D), u a
  seeded random direction, s = 0 (control), 0.5, 1, 2.
- A translation keeps the cluster's internal geometry, so only its position relative to the old
  data, and to what each index learned, varies.
- Fixed: N, the direction, the build settings (HNSW M=16; IVF nlist=2048; PQ m=32) and the seeds.
- Query groups: old-region and new-region (moved with the cluster), from disjoint query halves.
- Each index keeps one search setting chosen on the baseline, at 0.95 × its best baseline recall:
  HNSW ef=48, IVF-Flat nprobe=48, IVF-PQ nprobe=24 (target 0.705).
- *Geometry check:* even at s = 0.5, 98.6% of new-region true neighbours are new vectors (22.6% in
  the control).

**What happened** (new-region test queries, at the fixed setting):

| | control s=0 | 0.5σ | 1σ | 2σ | rebuilt (1σ / 2σ) |
|---|---|---|---|---|---|
| IVF-PQ recall | 0.718 | 0.649 | 0.522 | **0.303** | 0.753 / 0.753 |
| IVF-PQ PQ error new ÷ old | 0.99× | 2.6× | 8.9× | **55×** | 1.05× |
| IVF-Flat recall | 0.978 | 0.956 | 0.965 | 0.992 | 0.997 / 0.997 |
| IVF-Flat dist./query | 13,474 | 13,718 | 15,638 | **25,100** | 12,829 |
| IVF-Flat QPS | 2,456 | 2,181 | 1,956 | **1,326** | 2,467 / 2,790 |
| HNSW recall | 0.961 | 0.972 | 0.951 | 0.977 | 0.964 / 0.982 |
| HNSW new→new level-0 links | 27% | 93% | 99.9% | 100% | 100% |

- Old-region recall is within ±0.01 of the control in every state, for all three indexes.
- **Re-tuning (T4c):** incremental IVF-PQ cannot reach 0.705 on the new region at *any* nprobe (best
  0.680 / 0.536 / 0.304). Rebuilt, it reaches it at nprobe 12. IVF-Flat and HNSW reach the target
  everywhere.
- **Maintenance** (all threads, inserting 100K vs rebuilding 500K):

| Index | Insert | Rebuild | Ratio |
|---|---|---|---|
| IVF-Flat | 0.4–0.7 s | 9.2–9.4 s | ≈ 15× |
| IVF-PQ | 1.2 s | 37–39 s | ≈ 32× |
| HNSW | 4.6–9.5 s | 38–41 s | ≈ 4–8× |

**Mechanisms:**
- **IVF-PQ: stale codebooks.** Encoding stores the residual x − centroid with codebooks learned on
  D's residuals. A shifted vector's residual from its (distant) nearest old centroid contains most of
  the shift, which is far larger than anything the codebooks saw. The quantisation error swamps the
  neighbour distance gaps: 55× the error at 2σ, and a mean error of about 4,055 × 55 ≈ 223K in
  squared units, against a squared 10-NN distance of about 57K.
  - Every candidate in the new region is mis-ranked, and probing more lists only adds more
    mis-ranked candidates, so re-tuning can't help.
  - Rebuilding retrains both centroids and codebooks on D′, and the error ratio returns to 1.05×.
- **IVF-Flat: stale centroids cost speed, not exactness.**
  - At 2σ, every new vector goes to its nearest *old* centroid, which is on the old data's edge.
    90% land in 240 of 2,048 lists; the largest list grows to 2,897 (93% new); imbalance rises from
    1.24 to 1.84.
  - A new-region query probes those same overloaded lists, so it scans 1.9× more vectors.
  - Distances are exact and the neighbours sit together in those lists, so recall stays high, even
    0.99.
  - At 0.5σ, recall dips a little (0.956 vs 0.978). The cluster then sits among old centroids that
    cut it along boundaries learned for the old data, so neighbours spread across lists.
  - Rebuilding gives the cluster its own ~316 lists: 12.8K dist./query and recall 0.997.
- **HNSW: nothing learned, nothing stale.** Each insertion runs a graph search to find the new node's
  neighbours, so a far cluster builds its own well-connected subgraph (99.9–100% of new nodes' links
  are new→new).
  - The few remaining bridges to the old graph (1,906 links at 1σ incremental, 5 at 2σ rebuilt, at
    level 0) are enough for the greedy search to enter the cluster.
  - Upper layers also contain new nodes. We measured only level-0 links; say so.
  - Recall changes by at most ±0.02 with no trend. The new region is even slightly cheaper (800–860
    vs 950 dist./query), because the cluster is a smaller graph.

**Verdict on the hypothesis:**

| Hypothesis claim | Result |
|---|---|
| "Shift reduces recall for at least one incrementally updated index, particularly for new-region queries" | **Supported**, by IVF-PQ: −0.42 at 2σ, new region only |
| IVF-Flat: new vectors crowd into few lists, imbalance and candidates scanned rise, rebuild restores efficiency | **Supported** (all three measured). Recall did *not* fall at large shift |
| IVF-PQ more sensitive than IVF-Flat, residuals poorly represented | **Supported**, strongly (PQ error 55×; re-tuning cannot recover) |
| HNSW "depends on how new vectors connect"; rebuild "not guaranteed" to help | **Supported as stated**: almost all links stay within the cluster, but search still works; rebuild gains ≤ 0.012 recall |
| Rebuilding costs more maintenance time | **Supported** (4–32×). It pays off for IVF-PQ (the only fix), for IVF-Flat at large shift (+0.005 recall, 2.1× QPS), not for HNSW |

**Caveats to mention:**
- Panel (c): IVF-Flat's controls re-tune to nprobe 32 (incremental) vs 48 (rebuilt). That is one √2
  grid step, worth about 1.6× QPS. It is why the rebuilt control sits at 0.61, a grid-step artifact
  and not a drift effect.
- HNSW insertion order is not bit-reproducible with multi-threaded builds (D-notes), so HNSW recall
  differences of about 0.01 are noise.

**Not tested:**
- rotations or several new clusters;
- a growing share of new vectors (fixed at 20%);
- deletions (FAISS HNSW can't delete);
- other nlist, M and m;
- GloVe;
- drift in queries only;
- partial retraining.

**For Q3 ("when the headline numbers mislead"):**
- A benchmark that builds and queries on one static distribution (Parts 1–2) predicts nothing about
  IVF-PQ after drift. Its recall halved with no warning in the old-region numbers, and averaging both
  query groups would hide half of the drop.
- The cheap incremental insert (32× cheaper than rebuilding) is the trap.

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

*(Part 3 and Part 4 points are in their sections above, under "For Q3".)*

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
