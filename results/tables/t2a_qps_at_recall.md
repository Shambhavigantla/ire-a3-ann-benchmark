**Single-thread, one-query-at-a-time QPS at recall@10 targets. Settings are the fastest reaching the target on the tune queries; QPS and recall are on the test queries (× = speed-up over exact search).**

| Dataset | Index | QPS @ 0.90 | Test recall | Settings @ 0.90 | QPS @ 0.95 | Test recall | Settings @ 0.95 |
|---|---|---|---|---|---|---|---|
| SIFT-128 | HNSW | 14,023 (340×) | 0.9182 | M=16, ef=32 | 8,246 (200×) | 0.9711 | M=32, ef=48 |
| SIFT-128 | IVF-Flat | 3,131 (76×) | 0.9263 | nlist=4096, nprobe=32 | 2,326 (56×) | 0.9590 | nlist=16384, nprobe=96 |
| SIFT-128 | IVF-PQ | not reached (max 0.884) | – | – | not reached (max 0.884) | – | – |
| SIFT-128 | Exact (Flat) | 41.3 (1×) | 1.0000 | – | 41.3 (1×) | 1.0000 | – |
| GloVe-100 | HNSW | 1,760 (41×) | 0.9179 | M=32, ef=192 | 889 (21×) | 0.9546 | M=32, ef=384 |
| GloVe-100 | IVF-Flat | 882 (20×) | 0.9105 | nlist=16384, nprobe=384 | 399 (9×) | 0.9544 | nlist=4096, nprobe=384 |
| GloVe-100 | IVF-PQ | not reached (max 0.858) | – | – | not reached (max 0.858) | – | – |
| GloVe-100 | Exact (Flat) | 43.2 (1×) | 1.0000 | – | 43.2 (1×) | 1.0000 | – |
