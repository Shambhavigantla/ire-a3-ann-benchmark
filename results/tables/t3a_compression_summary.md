**Part 3, SIFT-128, nlist=4096. Memory is the serialised index; the re-rank rows include the float32 original vectors held in RAM. QPS at a target: the fastest nprobe reaching it on the tune queries, reported on the test queries (× = speed-up over exact search).**

| Method | m | Bytes / vector | vs raw float32 | Build (s) | Max test recall | QPS @ 0.90 | QPS @ 0.95 |
|---|---|---|---|---|---|---|---|
| IVF-Flat (reference) | – | 522.1 | 1.02× | 35.7 | 0.9998 (nprobe=384) | 2,424 (nprobe=32, 59×) | 1,926 (nprobe=48, 47×) |
| IVF-PQ (control) | 8 | 18.3 | 0.04× | 46.6 | 0.4087 (nprobe=128) | not reached | not reached |
| IVF-PQ (control) | 16 | 26.3 | 0.05× | 34.5 | 0.5881 (nprobe=192) | not reached | not reached |
| IVF-PQ (control) | 32 | 42.3 | 0.08× | 57.8 | 0.7423 (nprobe=384) | not reached | not reached |
| IVF-PQ (control) | 64 | 74.3 | 0.15× | 99.9 | 0.8839 (nprobe=512) | not reached | not reached |
| IVF-PQ + re-rank | 8 | 530.3 | 1.04× | 46.8 | 0.8770 (nprobe=384) | not reached | not reached |
| IVF-PQ + re-rank | 16 | 538.3 | 1.05× | 34.6 | 0.9840 (nprobe=512) | 3,430 (nprobe=32, 83×) | 2,298 (nprobe=64, 56×) |
| IVF-PQ + re-rank | 32 | 554.3 | 1.08× | 57.9 | 0.9995 (nprobe=384) | 2,490 (nprobe=32, 60×) | 1,810 (nprobe=48, 44×) |
| IVF-PQ + re-rank | 64 | 586.3 | 1.15× | 100.1 | 0.9998 (nprobe=384) | 1,416 (nprobe=32, 34×) | 1,074 (nprobe=48, 26×) |
