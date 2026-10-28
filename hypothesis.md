# Hypotheses

Each hypothesis is written and committed to git **before** the experiment it describes is run.
The commit history is the evidence. Results and verdicts go in the report, not here.

## Part 3: experiment on the standard data

*Choice:* compression

*Hypothesis:* 
Change: On SIFT, vary the number of PQ subvectors (m) across at least three values while holding nlist fixed. For each configuration, evaluate plain IVF-PQ and IVF-PQ followed by exact reranking of the top 100 candidates using the original vectors. Use plain IVF-PQ as the control and IVF-Flat with the same nlist and nprobe as the reference.

Predictions:

Increasing (m) will generally improve plain IVF-PQ recall@10 while increasing index memory.

Exact reranking will improve recall by correcting ranking errors among retrieved candidates. Its gain may be larger at smaller (m), where quantization is less precise.

Reranked recall will likely remain below IVF-Flat because reranking cannot recover true neighbours absent from the candidate set.

Reranking will reduce query throughput and require access to the original vectors; their storage will be included in the memory accounting.

Mechanism: Increasing (m) reduces the number of dimensions represented by each subvector and generally improves the precision of the PQ approximation, at the cost of larger codes. Exact reranking corrects ordering errors among the retrieved candidates but cannot recover neighbours excluded during cluster selection or candidate generation. Consequently, post-reranking recall is bounded by candidate coverage.

## Part 4: experiment on our own data

*Question:* When a new cluster of vectors appears in a database, how do IVF-Flat, IVF-PQ, and HNSW differ in their ability to maintain approximate nearest-neighbour (ANN) search quality after incremental insertion, and how much performance can be recovered by rebuilding the indexes?

*Property varied (one at a time), its settings (at least 3), and the control:*
Use a fixed subset of 400,000 SIFT vectors as the initial database (D). Reserve a disjoint subset of 100,000 SIFT vectors to construct the new cluster. Build IVF-Flat, IVF-PQ, and HNSW on (D).

Generate a seeded random translation direction and translate the reserved vectors by a controlled shift magnitude (s). Add these 100,000 vectors to the initial database to form the updated database (D'). Keep the number of inserted vectors constant across conditions.

Test four conditions: no shift ((s=0)), small shift, medium shift, and large shift. Define shift magnitudes relative to a documented measure of the original data's spread, such as the root-mean-square distance from the initial database mean. Use the same translation direction across shift levels so that only the magnitude changes.

For each condition, compare:

Incremental update: insert the new vectors into the existing index without retraining IVF centroids or IVF-PQ codebooks. Insert new vectors into the existing HNSW graph.

Rebuild: construct a fresh index on the complete updated database (D'), retraining IVF centroids and IVF-PQ codebooks as appropriate.

Run two query groups: original-region queries from the original SIFT query set, and new-region queries formed by applying the same translation to a disjoint set of SIFT queries. Compute exact top-10 ground truth against the corresponding database version for each query group.

*Hypothesis:* (what we expect to break, and why)
At fixed search parameters, increasing shift magnitude will reduce recall for at least one incrementally updated index, particularly for queries from the new region.

IVF-Flat may assign many new vectors to a small number of existing inverted lists when the new cluster lies far from the original centroids. This may increase list imbalance and the number of candidates scanned. Rebuilding may improve search efficiency by learning centroids that better represent the updated database.

IVF-PQ may be more sensitive to drift than IVF-Flat because its coarse centroids and product-quantization codebooks are both learned from the original data. Large shifts may produce residual vectors that are poorly represented by the existing codebooks. Retraining may improve performance, although the magnitude of the improvement is empirical.

HNSW may handle incremental insertion differently because it does not learn IVF-style centroids or PQ codebooks. Its performance will depend on how new vectors connect to the existing graph. Rebuilding may improve search quality or efficiency, but this is not guaranteed.

Rebuilding will incur additional maintenance cost compared with incremental insertion. The experiment will quantify whether any search-performance recovery justifies that cost.

