**Part 4 mechanism statistics. IVF imbalance = nlist·Σ size² / N² (1 = equal lists; ≈ expected vectors scanned per probe relative to equal lists). PQ error = mean squared reconstruction error of 5,000 new vectors divided by that of 5,000 old vectors. HNSW: level-0 links.**

| Index | Shift s | State | Mechanism statistics |
|---|---|---|---|
| HNSW | – | baseline (D) | mean degree 21.8 |
| HNSW | 0σ | incremental | new→new links 27.1%; old nodes linking new 89.7%; old→new links 1,259,499 |
| HNSW | 0σ | rebuilt | new→new links 27.5%; old nodes linking new 96.2%; old→new links 1,588,837 |
| HNSW | 0.5σ | incremental | new→new links 93.2%; old nodes linking new 23.8%; old→new links 144,935 |
| HNSW | 0.5σ | rebuilt | new→new links 93.5%; old nodes linking new 21.4%; old→new links 114,868 |
| HNSW | 1σ | incremental | new→new links 99.9%; old nodes linking new 0.4%; old→new links 1,906 |
| HNSW | 1σ | rebuilt | new→new links 100.0%; old nodes linking new 0.1%; old→new links 334 |
| HNSW | 2σ | incremental | new→new links 100.0%; old nodes linking new 0.1%; old→new links 247 |
| HNSW | 2σ | rebuilt | new→new links 100.0%; old nodes linking new 0.0%; old→new links 5 |
| IVF-Flat | – | baseline (D) | imbalance 1.21; largest list 737 (mean 195) |
| IVF-Flat | 0σ | incremental | imbalance 1.24; largest list 1,117 (mean 244); 90% of new vectors in 1,511 of 2,048 lists; largest list 39% new |
| IVF-Flat | 0σ | rebuilt | imbalance 1.23; largest list 1,142 (mean 244); 90% of new vectors in 1,519 of 2,048 lists; largest list 52% new |
| IVF-Flat | 0.5σ | incremental | imbalance 1.24; largest list 1,208 (mean 244); 90% of new vectors in 1,336 of 2,048 lists; largest list 44% new |
| IVF-Flat | 0.5σ | rebuilt | imbalance 1.21; largest list 922 (mean 244); 90% of new vectors in 324 of 2,048 lists; largest list 0% new |
| IVF-Flat | 1σ | incremental | imbalance 1.31; largest list 1,406 (mean 244); 90% of new vectors in 907 of 2,048 lists; largest list 81% new |
| IVF-Flat | 1σ | rebuilt | imbalance 1.21; largest list 917 (mean 244); 90% of new vectors in 316 of 2,048 lists; largest list 0% new |
| IVF-Flat | 2σ | incremental | imbalance 1.83; largest list 2,896 (mean 244); 90% of new vectors in 240 of 2,048 lists; largest list 93% new |
| IVF-Flat | 2σ | rebuilt | imbalance 1.21; largest list 917 (mean 244); 90% of new vectors in 316 of 2,048 lists; largest list 0% new |
| IVF-PQ | – | baseline (D) | imbalance 1.21; largest list 737 (mean 195); PQ error (old) 4,054 |
| IVF-PQ | 0σ | incremental | imbalance 1.24; largest list 1,117 (mean 244); 90% of new vectors in 1,511 of 2,048 lists; largest list 39% new; PQ error new/old 0.99× |
| IVF-PQ | 0σ | rebuilt | imbalance 1.23; largest list 1,142 (mean 244); 90% of new vectors in 1,519 of 2,048 lists; largest list 52% new; PQ error new/old 0.94× |
| IVF-PQ | 0.5σ | incremental | imbalance 1.24; largest list 1,208 (mean 244); 90% of new vectors in 1,336 of 2,048 lists; largest list 44% new; PQ error new/old 2.60× |
| IVF-PQ | 0.5σ | rebuilt | imbalance 1.21; largest list 922 (mean 244); 90% of new vectors in 324 of 2,048 lists; largest list 0% new; PQ error new/old 1.05× |
| IVF-PQ | 1σ | incremental | imbalance 1.31; largest list 1,406 (mean 244); 90% of new vectors in 907 of 2,048 lists; largest list 81% new; PQ error new/old 8.96× |
| IVF-PQ | 1σ | rebuilt | imbalance 1.21; largest list 917 (mean 244); 90% of new vectors in 316 of 2,048 lists; largest list 0% new; PQ error new/old 1.05× |
| IVF-PQ | 2σ | incremental | imbalance 1.83; largest list 2,896 (mean 244); 90% of new vectors in 240 of 2,048 lists; largest list 93% new; PQ error new/old 55.47× |
| IVF-PQ | 2σ | rebuilt | imbalance 1.21; largest list 917 (mean 244); 90% of new vectors in 316 of 2,048 lists; largest list 0% new; PQ error new/old 1.05× |
