# Process log, in plain words

A step-by-step diary of what we did, why, and what came out, written so that anyone (or a new AI
session) can pick up from here. Newest step at the bottom. Design reasoning is in
[DESIGN_NOTES.md](DESIGN_NOTES.md).

---

## The big picture

We have millions of vectors (points in a 100- or 128-dimensional space). Given a query vector, we
want the 10 stored vectors closest to it.

- **Exact search** checks every stored vector. It is always right, but slow.
- **ANN indexes** (HNSW, IVF-Flat, IVF-PQ) look at only part of the data. They are much faster, but
  sometimes miss a true neighbour.

The assignment is to measure that trade-off carefully: how fast each index is at a given accuracy,
what it costs in memory and build time, and when the numbers can mislead.

- **Accuracy** = recall@10: the fraction of the true 10 nearest neighbours that the index found.
- **Speed** = queries per second (QPS), one query at a time, on 1 thread.

## Datasets

| Name | Vectors | Dim | Distance | Our handling |
|---|---|---|---|---|
| SIFT | 1,000,000 | 128 | Euclidean | used as is |
| GloVe-100 | 1,183,514 | 100 | angular (cosine) | each vector scaled to length 1, then Euclidean |

Each file also has 10,000 queries and the true 100 nearest neighbours of each query. We tune on
queries 0–4999 and report on queries 5000–9999.

---

## Step 1: setting up (done)

- Installed Python 3.12 and created a venv at `A3/.venv`.
- `pip install -r requirements.txt` (includes `threadpoolctl`, see Step 2). We dropped hnswlib: it has no Windows build, and FAISS has
  HNSW anyway.
- Downloaded both datasets into `code/data/` (about 1 GB, not committed to git).
- Run scripts from Git Bash, or from PowerShell with `& "C:\Program Files\Git\bin\bash.exe" run_all.sh`.

## Step 2: Part 1, the exact baseline (done)

**What we built**
- `IndexFlatL2`: FAISS brute force, our "always right" reference.
- Our own recall@10 that handles **ties**. Suppose the 10th true neighbour is 100 units away and
  another vector is also exactly 100 away. The ground-truth file had to pick one of them. If an index
  returns the other one, that is not a mistake. So we count a returned vector as correct if it is no
  farther than the 10th true neighbour. We always recompute that distance ourselves from the raw
  vectors.
- 28 unit tests: the recall function, the timing and data helpers, and a FAISS check. All pass.
- A check that our exact search agrees with the neighbours stored in the file.
- Timing in three ways: one query at a time on 1 thread, all queries in one call on 1 thread, and
  all queries in one call on all 20 threads.

**A problem we found and fixed: timings that weren't what they claimed**
The first full run gave odd numbers. One GloVe repeat was almost twice as fast as the other two, and
GloVe's "1-thread" batch was as fast as its 20-thread batch. There were two causes:
1. *The batch search was secretly using several threads.* FAISS hands batch maths to a library called
   OpenBLAS, which runs its own threads and ignores FAISS's thread setting. Fix: we now limit both
   (with the `threadpoolctl` package).
2. *The CPU has two kinds of cores.* The i5-13500 has 6 fast "P-cores" (logical CPUs 0–11) and 8
   slower "E-cores" (12–19), about 1.5× slower. Windows moved our program between them during a run.
   Fix: 1-thread timings are now locked to the P-cores.
Every run now also records **CPU / wall**: how many cores were actually busy. For a 1-thread run it
must be about 1, and it now is. Any timing from before this fix should not be used.

**What we saw (full data)**
- Exact search agreed with the file on all 10,000 queries for both datasets (tie-aware recall 1.0).
- SIFT: 65 queries would have looked "wrong" with plain id matching, but they only differ by ties.
  This shows the tie handling matters. GloVe has no exact ties, since its numbers are not whole
  numbers.
- About 41–43 queries per second one at a time (about 24 ms each). About 340–380 per second as one
  batch on 1 thread, which is 8–9× faster. About 500–740 per second on all threads.
- *Why batching wins:* one query has to read the whole dataset (about 0.5 GB) from memory just to do
  a tiny amount of maths with each vector, so memory speed is the limit. With a batch, FAISS uses
  one big matrix multiplication. Each chunk of data is loaded into the fast CPU cache once and reused
  for thousands of queries.
- *Why exact recall is exactly 1:* exact search compares against every vector, so it can't miss
  anything. The "check vs file" block is the real test, because it compares against ground truth
  computed by someone else's code.

**Outputs:** `results/tables/t0_machine`, `t1a_exact_check`, `t1b_exact_timing`, and
`results/figures/fig1_exact_single_vs_batch`. The quick-run versions are in `results/quick/`.

**Rule for every later part:** time with `pinned_to(performance_cores())` and `limit_threads(1)`.

## Step 2b: putting the code on GitHub

- The repo is `code/` (it has its own `.git`). `data/` and `results/quick/` are git-ignored.
  `ai_log/` sits outside `code/`, so it isn't in the repo.
- Keep the GitHub repo **private**: this is individual coursework.
- `.gitattributes` keeps `*.sh` with Unix line endings, so `bash run_all.sh` works for a Linux or Mac
  grader.
- Workflow from now on: commit after each step. **Commit `hypothesis.md` on its own, before running
  that experiment.** The commit order is the proof the assignment asks for.

```bash
git add -A && git commit -m "<what changed>"
git push
```

## Step 3: Part 2, benchmarking the three indexes (code done; quick run done; full run in progress)

**The three indexes, in plain words**
- **HNSW:** a graph where each vector links to about M neighbours. A search walks the graph towards
  the query. `efSearch` = how many candidates it keeps while walking: more is slower but more
  accurate.
- **IVF-Flat:** first groups the vectors into `nlist` clusters (k-means). A search only scans the
  `nprobe` clusters closest to the query: more clusters is slower but more accurate.
- **IVF-PQ:** like IVF-Flat, but each vector is compressed to `m` bytes (product quantization)
  instead of 400–512 bytes. Much smaller, but distances become approximate, so some true neighbours
  get ranked out of the top 10.

**What the script does** (`scripts/part2_benchmark.py`)
1. For each build setting (e.g. HNSW with M=16), build the index once and record build time and size.
2. Turn the search knob up step by step. At each step:
   - measure recall on all 5,000 tune and 5,000 test queries;
   - measure speed one query at a time on 1 thread (P-cores, median of 3).
3. Stop the sweep when recall reaches 0.999, stops improving, or the index becomes slower than exact
   search.
4. The tables pick the fastest setting that reaches recall 0.90 or 0.95 on the **tune** queries, and
   report its **test** numbers.

**Problems found and fixed while testing (details in DESIGN_NOTES D21–D25)**
- IVF-PQ builds were slow: FAISS's default "polysemous training" step was on but useless here.
  Turned off.
- Timings straight after a multi-threaded step were 24% too slow: idle threads keep spinning for a
  moment. Fix: a 0.25 s pause before every timing.
- Very fast settings finished in a few milliseconds, which is too short to time reliably. Fix: each
  repeat loops over the queries for at least 0.2 s.
- Sweep steps were too coarse (×2) for the "QPS at 0.90/0.95" table. Now ×√2.

**First look (quick run, 100K vectors; not for the report)**
- HNSW is fastest on SIFT. On GloVe, HNSW and IVF-Flat are close.
- IVF-PQ never reaches 0.90 (max 0.78 on SIFT, 0.63 on GloVe). Compression loses too much precision
  to rank the top 10 correctly. Re-ranking with exact distances would fix this (Part 3 "compression"
  option).
- GloVe is much harder than SIFT: every index needs far more work for the same recall.

## Step 3 (cont.): full run results (done)

- The full run took 68 minutes. We then raised the IVF sweep limit (nprobe up to 1024), because
  GloVe IVF-Flat was still improving at 512, and reran those builds (about 60 more minutes).
- 672 result rows. CPU/wall stayed between 0.94 and 1.06, so every timing really used 1 thread.

**Headline numbers (speed at recall 0.95, one query at a time, 1 thread)**

| | SIFT (1M) | GloVe (1.18M) |
|---|---|---|
| HNSW | **8,246 QPS** (200× exact), M=32, ef=48 | **889 QPS** (21×), M=32, ef=384 |
| IVF-Flat | 2,331 QPS (56×), nlist=16384, nprobe=96 | about 360 QPS (8×); two settings tie |
| IVF-PQ | never reaches 0.95 (best 0.884) | never reaches 0.95 (best 0.869) |
| Exact | 41 QPS | 43 QPS |

**What it means, in plain words**
- **HNSW is fastest on both datasets** at 0.90 and at 0.95. It walks a graph straight towards the
  answer, while IVF scans whole clusters.
- **GloVe is about 10× harder than SIFT** for the same recall. Word vectors don't form tight clusters,
  and many points are almost equally far from a query, so every index has to look at many more
  candidates.
- **IVF-PQ can't reach high recall.** Each vector is squeezed to 26–74 bytes instead of 400–512, so
  its distances are approximate. Several true neighbours end up ranked just outside the top 10, and
  probing more clusters doesn't help (the curve goes vertical). Fix: re-rank the top candidates with
  exact distances.
- **Memory:** HNSW uses 1.5–1.7× the raw vectors (it stores the graph links), IVF-Flat 1.03×, and
  IVF-PQ only 0.15×.
- **Build time:** IVF-Flat is quickest (4–20 s) unless nlist is huge (16384 needs 4–7 minutes of
  k-means). HNSW takes 40–120 s; IVF-PQ 50–80 s, mostly codebook training.

**A measurement lesson.** Recall came out identical in both runs. Speed for the same setting changed
by up to ±30–40% between the two runs, probably from background Windows activity while the PC was
locked. Within a run the repeats agree within a few percent. So:
(1) only compare speeds from the same run;
(2) small differences (for example GloVe IVF-Flat 357 vs 356 QPS) mean nothing;
(3) the final report numbers must come from one clean `bash run_all.sh` on an idle machine.
This also answers report Q5 ("how could the benchmark mislead?"). See DESIGN_NOTES D27.

## Step 4: Part 3, compression (done)

**What we asked.** IVF-PQ squeezes each vector into m bytes. Does a bigger m help, and can exact
re-ranking fix PQ's mistakes? The hypothesis was written and committed first (`0b6391d`).

**What we did** (`scripts/part3_compression.py`, design in DESIGN_NOTES D28)
- SIFT 1M, nlist = 4096 fixed, m = 8, 16, 32, 64 bytes per vector.
- Each IVF-PQ index was built once and searched two ways at every nprobe:
  - **plain:** the top 10 by PQ's approximate distance;
  - **re-rank:** take PQ's top 100, recompute their exact distances from the original vectors, and
    keep the closest 10.
- Reference: IVF-Flat with the *same* clusters (same seed, so the centroids are bit-identical,
  checked). At equal nprobe all three look in exactly the same clusters.
- We also measured **coverage**: how many of the true top 10 are anywhere in PQ's top 100.
- Same timing rules as Part 2. Took 27 minutes. 300 rows; CPU/wall 0.92–1.02.

**What we saw** (tables t3a, t3b; figure fig3_compression)
- **More bytes, better plain PQ:** best recall 0.41 (m=8) → 0.59 → 0.74 → 0.88 (m=64). Memory
  18 → 74 bytes per vector. Plain IVF-PQ never reaches 0.90.
- **Re-ranking helps a lot, and most when m is small.** At nprobe = 64 the gain is +0.46 (m=8),
  +0.38, +0.24, +0.10 (m=64).
- **Re-ranking can only reorder, never add.** Re-ranked recall equalled coverage on every row.
- **IVF-Flat is the ceiling.** Re-ranked recall never beat IVF-Flat. For m ≥ 32 it *matched* it
  (within 0.0003), meaning PQ's top 100 already held every true neighbour from the scanned
  clusters. For m = 8 it fell further behind as nprobe grew (−0.12 at nprobe 256). With more
  candidates in play, PQ's rough ranking pushes true neighbours out of the 100.
- **Re-ranking is cheap in time** (QPS 0.73–1.00× of plain; 100 exact distances are few next to
  scanning the clusters) **but expensive in memory**: the original vectors add 512 bytes per vector.
  So re-ranked IVF-PQ needs about as much RAM as IVF-Flat, unless the vectors live on disk.
- **Surprise:** at recall 0.95, re-ranked m = 16 was a little faster than IVF-Flat (2,298 vs
  1,926 QPS, same run). Plain PQ with large m scans clusters *slower* than IVF-Flat (m=64 at
  nprobe 64: 855 vs 1,369 QPS). So small codes plus re-ranking was the best mix here.

**Also fixed:** the timing helper now estimates how many loops it needs from a warm pass, not the
slow first call (D29).

## Step 5: Part 4, drift on our own data (done)

**What we asked.** A new group of vectors appears somewhere the index never saw. If we just insert it,
how does each index cope, and does rebuilding help? Hypothesis committed first (`81de567`).

**The data** (`ann_bench/synthetic.py`, seeded)
- D = the first 400K SIFT vectors.
- New cluster = the next 100K SIFT vectors, all moved by the same arrow: a random direction (seed
  1234) times s × 385. 385 is how spread out D is: the RMS distance of D from its mean.
- s = 0 is the control (more ordinary SIFT data); then 0.5, 1 and 2. 500K vectors in total.
- Moving all the vectors by the same arrow keeps the cluster's internal shape, so it is exactly as
  hard to search as normal SIFT. Only *where* it is changes.
- Queries: half the queries stay as they are ("old region"); the other half move with the cluster
  ("new region").

**What we did** (`scripts/part4_drift.py`; design in DESIGN_NOTES D31). For HNSW, IVF-Flat and
IVF-PQ:
1. Build on D (baseline).
2. For each s:
   - copy the baseline index and *insert* the new vectors, with no retraining (incremental);
   - build a fresh index on everything (rebuilt).
3. Each index keeps one search setting, chosen on the baseline.
4. Measure recall, speed and work per query for both query groups, plus "why" statistics: how full
   the IVF lists get, how badly PQ encodes the new vectors, and how HNSW links them.

33 minutes; CPU/wall 0.90–1.04 on all 784 timed rows.

**What we saw** (tables t4a–c, figure fig4_drift)
- **Even the smallest shift separates the neighbours.** At 0.5σ, 98.6% of a new-region query's
  true neighbours are new vectors. The query's neighbours moved with it.
- **IVF-PQ breaks.** New-region recall at its fixed setting goes 0.72 (control) → 0.65 → 0.52 →
  0.30. Its codebooks were learned on old data and can't describe the new vectors: their encoding
  error is 2.6×, 9× and 55× that of old vectors. Turning nprobe up doesn't fix it (best 0.68, 0.54,
  0.30). Rebuilding fixes it completely (0.75).
- **IVF-Flat keeps its recall, but slows down.** At 2σ, 90% of the new vectors pile into 240 of
  2,048 lists near the old data's edge (the largest list grows from 1,117 to 2,897). A query there
  scans 1.9× more vectors and runs at 0.54× the speed. Rebuilding gives the cluster its own lists:
  recall 0.997 at 2.1× the speed.
- **HNSW barely notices** (recall 0.95–0.98 at every shift). New nodes link almost only to each other
  (99.9% at 1σ). Only a few hundred to a few thousand links join them to the old graph, yet searches
  still find their way. Rebuilding changes little.
- **Old-region queries are not affected** (within ±0.01 everywhere).
- **Maintenance:** inserting 100K vectors takes 0.4–1.2 s (IVF) or 5–9 s (HNSW). Rebuilding takes
  9 s (IVF-Flat), 37–39 s (IVF-PQ) or 38–41 s (HNSW).
- **Verdict:** rebuilding is essential for IVF-PQ, worth it for IVF-Flat at large shifts, and not
  needed for HNSW.

**Also done for the 15-minute `--quick` limit:** one pause per timing block instead of two, and a
faster strict-recall function (same results; D30).

## Step 6: report (not started)
