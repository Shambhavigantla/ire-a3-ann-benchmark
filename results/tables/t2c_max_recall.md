**Highest test recall@10 reached by each build configuration in the sweep.**

| Dataset | Index | Most accurate setting | Test recall | QPS | Build (s) | Bytes / vector |
|---|---|---|---|---|---|---|
| SIFT-128 | HNSW | M=8, ef=1024 | 0.9972 | 484 | 39.2 | 592.6 |
| SIFT-128 | HNSW | M=16, ef=384 | 0.9995 | 1,433 | 56.2 | 656.3 |
| SIFT-128 | HNSW | M=32, ef=256 | 0.9994 | 1,826 | 69.5 | 784.1 |
| SIFT-128 | IVF-Flat | nlist=1024, nprobe=128 | 0.9997 | 300 | 3.7 | 520.5 |
| SIFT-128 | IVF-Flat | nlist=4096, nprobe=384 | 0.9998 | 347 | 20.5 | 522.1 |
| SIFT-128 | IVF-Flat | nlist=16384, nprobe=512 | 0.9983 | 650 | 234.7 | 528.5 |
| SIFT-128 | IVF-PQ | m=16, nlist=4096, nprobe=192 | 0.5881 | 1,319 | 33.9 | 26.3 |
| SIFT-128 | IVF-PQ | m=32, nlist=1024, nprobe=128 | 0.7323 | 425 | 32.5 | 40.7 |
| SIFT-128 | IVF-PQ | m=32, nlist=4096, nprobe=384 | 0.7423 | 402 | 48.8 | 42.3 |
| SIFT-128 | IVF-PQ | m=64, nlist=4096, nprobe=512 | 0.8839 | 163 | 77.7 | 74.3 |
| GloVe-100 | HNSW | M=8, ef=1024 | 0.8652 | 484 | 42.8 | 480.6 |
| GloVe-100 | HNSW | M=16, ef=1024 | 0.9516 | 385 | 76.3 | 544.3 |
| GloVe-100 | HNSW | M=32, ef=1024 | 0.9845 | 292 | 119.0 | 672.2 |
| GloVe-100 | IVF-Flat | nlist=1024, nprobe=512 | 0.9972 | 90 | 4.0 | 408.4 |
| GloVe-100 | IVF-Flat | nlist=4096, nprobe=512 | 0.9672 | 306 | 19.9 | 409.4 |
| GloVe-100 | IVF-Flat | nlist=16384, nprobe=512 | 0.9266 | 712 | 221.4 | 413.6 |
| GloVe-100 | IVF-PQ | m=20, nlist=4096, nprobe=512 | 0.5318 | 343 | 58.6 | 29.5 |
| GloVe-100 | IVF-PQ | m=25, nlist=1024, nprobe=512 | 0.6153 | 104 | 32.5 | 33.4 |
| GloVe-100 | IVF-PQ | m=25, nlist=4096, nprobe=512 | 0.6210 | 349 | 63.2 | 34.5 |
| GloVe-100 | IVF-PQ | m=50, nlist=4096, nprobe=512 | 0.8580 | 186 | 74.4 | 59.5 |
