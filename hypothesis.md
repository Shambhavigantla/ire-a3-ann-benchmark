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

*Question:*

*Property varied (one at a time), its settings (at least 3), and the control:*

*Hypothesis:* (what we expect to break, and why)

*Committed on:*
