# How to form the Part 3 and Part 4 hypotheses

The hypotheses in `hypothesis.md` must be **your own** reasoning, committed to git **before** the
experiment runs. This guide covers what to understand first and how to turn that into a testable
prediction. It deliberately doesn't give the predictions. A wrong hypothesis is fine; the marks are
for the reasoning and the care of the experiment.

---

## 1. What a good hypothesis looks like

It has four parts:

1. **Change:** the one property you vary, its settings (at least 3 in Part 4), and the control.
2. **Prediction:** what happens to a named metric for each named index, with a direction and a rough
   size. For example "falls", "roughly doubles", or "X degrades more than Y".
3. **Mechanism:** *why*, in terms of how the index works. This is the part that earns the marks.
4. **Test:** what result would support it and what would refute it, decided in advance.

Template:

> When **[property]** goes from **[a]** to **[b] to [c]**, with **[everything else]** fixed,
> **[metric]** of **[index A]** will **[direction / size]**, and **[index B]** will **[...]**,
> because **[mechanism]**.
> Supported if **[observable result]**; refuted if **[observable result]**.

**Two different kinds of metric. Pick deliberately:**
- **At a fixed setting** (efSearch / nprobe fixed): "what happens to recall?" This shows the failure
  directly.
- **At a fixed recall** (re-tuned for each setting, e.g. QPS at 0.95): "what does it cost to keep
  quality?" This is the practical view.

They can disagree. An index can lose recall at a fixed setting and still be cheap to fix by
re-tuning.

**Be honest about prior evidence.** You've already seen Part 2's results (and quick-run results at
100K). If those influenced your prediction, say so in one line. That's good practice, not a weakness.

## 2. What you need to understand about each index

For each index, know: what work it does per query, what makes it miss a true neighbour, and what
its knob changes.

| Index | Work per query (roughly) | How it misses a true neighbour | Knob |
|---|---|---|---|
| Exact (Flat) | N × d (every vector) | never | none |
| IVF-Flat | nlist centroid distances + nprobe × (N / nlist) vectors | the neighbour is in a cluster that wasn't probed; common when the query lies near cluster **boundaries** or the data isn't well clustered | nprobe (more clusters scanned) |
| HNSW | about efSearch × M distance computations (grows slowly with N) | the greedy walk gets stuck in a **local region**; the graph doesn't link the query's area to its true neighbours | efSearch (wider search) |
| IVF-PQ | like IVF, but each candidate costs m table lookups instead of d multiplies | as IVF-Flat, **plus** approximate distances rank true neighbours below false ones when the **distance gap** between neighbours is smaller than the PQ error | nprobe; and m sets the error |

**Ideas that come up in almost every hypothesis:**
- **Relative contrast (RC)** of a query = (average distance to the data) ÷ (distance to its k-th
  nearest neighbour).
  - RC close to 1 means the neighbours are barely closer than everything else, so the query is
    *hard*.
  - It drops as the dimension grows (distances "concentrate") and when the data isn't clustered.
- **Distance gap:** how much closer the 10th neighbour is than the 11th, 20th, and so on. Small gaps
  hurt anything approximate (PQ especially), because small errors reorder the list.
- **Cluster structure:** IVF assumes the data falls into clusters. Tight, well-separated clusters
  make it excellent. Uniform or overlapping data pushes true neighbours across cluster borders.
- **Scaling with N:** exact is linear in N. IVF with nlist fixed scans N / nlist per cluster, so it's
  also linear. IVF with nlist ∝ √N grows roughly like √N. HNSW grows roughly like log N per query, but
  the efSearch needed for the same recall can creep up.
- **Memory:**
  - raw vectors: N × d × 4 bytes;
  - HNSW adds about 2M × 4 bytes per vector;
  - IVF adds 8 bytes per vector (the stored id);
  - PQ replaces the vector with m bytes;
  - re-ranking needs the raw vectors back.

---

## 3. Part 3: pick one option. What to understand for each

### A. Scaling (SIFT at 100K, 300K, 1M; HNSW, IVF-Flat, exact; cost to reach recall 0.95)
Understand:
- how each index's work per query grows with N (table above);
- **one design decision you must make and state:** keep nlist fixed, or scale it (for example
  nlist ≈ 4√N)? The prediction for IVF depends on this choice;
- whether the efSearch / nprobe needed for 0.95 stays the same or grows with N, and why. Think about
  what it means for a graph walk or a cluster scan when there are 10× more points in the same space;
- how to show it: QPS at 0.95 (or ms per query) against N on log-log axes. **The slope is the
  hypothesis**: slope 1 is linear, 0.5 is like √N, near 0 is like log N.

Prior evidence to declare: Part 2 has HNSW and IVF-Flat at 100K (quick run) and 1M.
Cost: moderate. One build per index per size; the 1M builds already exist in Part 2.

### B. Query difficulty (GloVe; RC per query; 4 bins; recall per index per bin)
Understand:
- how to compute RC for each query from data you already have:
  - the distance to the 10th neighbour comes from the ground truth;
  - the average distance can be estimated against a fixed random sample of base vectors (seeded),
    for example 10K;
- binning: quartiles of RC, so each bin has the same number of queries;
- **fix the operating point first.** Use each index at the setting Part 2 chose for overall recall
  0.95 (tune half). Then compare recall per bin on the test half. Otherwise the bins aren't
  comparable;
- for each index, which mechanism makes a low-RC query hard, and so **which index should lose most
  in the hardest bin**: the IVF boundary effect, HNSW's local regions, PQ's small distance gaps;
- that an average of 0.95 overall can hide much lower recall on the hardest queries. That links
  directly to Q3 ("when the headline numbers mislead").

Prior evidence: none yet. Cost: low. It reuses the Part 2 settings; no sweeps needed, only one
search per index.

### C. Compression (SIFT; vary m; IVF-PQ vs IVF-PQ + exact re-ranking of the top 100)
Understand:
- how PQ error depends on m (bytes per vector) and the sub-vector size d/m;
- **what re-ranking can and can't fix.** Re-ranking the top 100 candidates with exact distances can
  only recover true neighbours that are *somewhere in those 100*. So the recall after re-ranking is
  limited by how good PQ's top 100 is. Think about how that limit changes with m and nprobe;
- the **memory trap.** Re-ranking needs the raw vectors (512 B per vector), so "IVF-PQ + re-rank"
  isn't 74 B per vector unless the raw vectors live on disk. Decide how you'll count memory and state
  it;
- the cost of re-ranking: 100 exact distance computations per query. Compare that with scanning
  nprobe clusters.

Prior evidence to declare: Part 2 has plain IVF-PQ at m = 16, 32, 64 on SIFT (best recall 0.59,
0.74, 0.88). Cost: moderate. Several IVF-PQ builds (about 1 minute each) plus the re-rank sweeps.

---

## 4. Part 4: your own data, your own failure mode

Rules from the assignment:
- a different question from Part 3;
- at least 2 of the 3 indexes;
- data you **generate** (seeded), at most 500K vectors;
- change **one thing**, with at least **3 settings + an unchanged control**;
- one figure and one page in the report.

How to choose:
1. Pick a property that a mechanism from section 2 says should matter: cluster structure, dimension,
   drift, updates, or memory.
2. Make sure you can **generate it in a controlled way**: one parameter that moves only that property.
3. Decide the control (the "nothing changed" setting) and the 3+ settings, for example a mild, a
   medium and a severe shift.
4. Decide the metric: recall at a fixed setting, or QPS at fixed recall (section 1).
5. Write what you *didn't* test. The report asks for that.

Ideas, and what to understand for each:
- **Drifting data** (train or build on distribution A, query or insert from a shifted B):
  - IVF's centroids are learned once, so new data that falls between or outside them makes lists
    unbalanced and probes less useful. Rebuilding or retraining should restore it.
  - HNSW has no training step; it adapts as you insert.
  - Variable: the size of the mean shift, the rotation angle, or the fraction of new clusters.
- **Update cost** (insert or delete a growing fraction):
  - FAISS `IndexHNSWFlat` **can't delete** (`remove_ids` isn't supported), but IVF can. So a deletion
    study only works for IVF; inserts work for both.
  - Measure insertion time, recall and QPS against the fraction inserted, and find the point where
    rebuilding is cheaper.
- **Memory limits** (fix a RAM budget and find the best recall and speed inside it):
  - Understand the bytes-per-vector formulas in section 2. The budget picks which settings are even
    allowed.
  - Variable: the budget (at least 3 levels).
- **Hard geometry:**
  - *Dimension* with uniform random data: as d grows, RC → 1 (distance concentration). IVF clusters
    become meaningless and the graph can't navigate well. Keep N fixed and vary d (e.g. 8, 32, 128,
    512).
  - *Cluster tightness:* Gaussian clusters with a varying spread σ. Tight clusters suit IVF; once
    they overlap, boundary misses rise.
  - *Unbalanced clusters:* a few huge clusters make some IVF lists enormous, so the cost per probe
    varies a lot.
  - *Near-duplicates:* many vectors within ε of each other. Lots of near-ties at the 10th neighbour
    (this tests the tie rule) and lots of redundant graph links.

**Guardrails:**
- Seed everything (the generator, sampling, IVF training).
- Use the same timing rules as Part 2: P-cores, 1 thread, `min_repeat_s`.
- Compute ground truth with exact search on the generated data.
- Keep runs small enough for `--quick` (the whole `--quick` pipeline must finish in 15 minutes).

---

## 5. Workflow (the order matters for grading)

1. Read the relevant section above and Part 2's results (`docs/REPORT_NOTES.md`).
2. Write the hypothesis in `hypothesis.md`: change, prediction, mechanism, test.
3. **Commit `hypothesis.md` on its own:** `git add hypothesis.md`, then
   `git commit -m "Part 3 hypothesis"`, then `git push`.
4. Only then build and run the experiment code.
5. In the report, say whether the result supported the hypothesis, and why or why not.
