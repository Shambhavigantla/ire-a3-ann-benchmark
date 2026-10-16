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

## Step 3: Part 2, benchmarking the three indexes (not started)

## Step 4: Part 3 experiment (not started)

Choose one of: scaling / query difficulty / compression. Write the hypothesis in `hypothesis.md`
and **commit it before running**.

## Step 5: Part 4 experiment on our own data (not started)

## Step 6: report (not started)
