**Single-thread, one-query-at-a-time QPS at recall@10 targets. Settings are the fastest reaching the target on the tune queries; QPS and recall are on the test queries (× = speed-up over exact search).**

| Dataset | Index | QPS @ 0.90 | Test recall | Settings @ 0.90 | QPS @ 0.95 | Test recall | Settings @ 0.95 |
|---|---|---|---|---|---|---|---|
| SIFT-128 | HNSW | 9,983 (335×) | 0.9182 | M=16, ef=32 | 5,497 (185×) | 0.9712 | M=16, ef=64 |
| SIFT-128 | IVF-Flat | 2,042 (69×) | 0.9265 | nlist=4096, nprobe=32 | 1,481 (50×) | 0.9580 | nlist=4096, nprobe=48 |
| SIFT-128 | IVF-PQ | not reached (max 0.884) | – | – | not reached (max 0.884) | – | – |
| SIFT-128 | Exact (Flat) | 29.8 (1×) | 1.0000 | – | 29.8 (1×) | 1.0000 | – |
| GloVe-100 | HNSW | 1,250 (40×) | 0.9179 | M=32, ef=192 | 635 (20×) | 0.9546 | M=32, ef=384 |
| GloVe-100 | IVF-Flat | 603 (19×) | 0.9104 | nlist=16384, nprobe=384 | 257 (8×) | 0.9590 | nlist=16384, nprobe=1024 |
| GloVe-100 | IVF-PQ | not reached (max 0.869) | – | – | not reached (max 0.869) | – | – |
| GloVe-100 | Exact (Flat) | 31.2 (1×) | 1.0000 | – | 31.2 (1×) | 1.0000 | – |
