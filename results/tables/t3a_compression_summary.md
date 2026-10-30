**Part 3, SIFT-128, nlist=4096. Memory is the serialised index; the re-rank rows include the float32 original vectors held in RAM. QPS at a target: the fastest nprobe reaching it on the tune queries, reported on the test queries (× = speed-up over exact search).**

| Method | m | Bytes / vector | vs raw float32 | Build (s) | Max test recall | QPS @ 0.90 | QPS @ 0.95 |
|---|---|---|---|---|---|---|---|
| IVF-Flat (reference) | – | 522.1 | 1.02× | 34.8 | 0.9998 (nprobe=384) | 1,590 (nprobe=32, 53×) | 1,221 (nprobe=48, 41×) |
| IVF-PQ (control) | 8 | 18.3 | 0.04× | 50.3 | 0.4073 (nprobe=128) | not reached | not reached |
| IVF-PQ (control) | 16 | 26.3 | 0.05× | 59.2 | 0.5878 (nprobe=192) | not reached | not reached |
| IVF-PQ (control) | 32 | 42.3 | 0.08× | 83.9 | 0.7423 (nprobe=384) | not reached | not reached |
| IVF-PQ (control) | 64 | 74.3 | 0.15× | 128.4 | 0.8843 (nprobe=512) | not reached | not reached |
| IVF-PQ + re-rank | 8 | 530.3 | 1.04× | 50.5 | 0.8774 (nprobe=384) | not reached | not reached |
| IVF-PQ + re-rank | 16 | 538.3 | 1.05× | 59.4 | 0.9846 (nprobe=512) | 2,904 (nprobe=32, 98×) | 1,920 (nprobe=64, 65×) |
| IVF-PQ + re-rank | 32 | 554.3 | 1.08× | 84.1 | 0.9995 (nprobe=384) | 2,111 (nprobe=32, 71×) | 1,526 (nprobe=48, 51×) |
| IVF-PQ + re-rank | 64 | 586.3 | 1.15× | 128.6 | 0.9998 (nprobe=384) | 1,275 (nprobe=32, 43×) | 925 (nprobe=48, 31×) |
