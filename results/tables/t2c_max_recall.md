**Highest test recall@10 reached by each build configuration in the sweep.**

| Dataset | Index | Most accurate setting | Test recall | QPS | Build (s) | Bytes / vector |
|---|---|---|---|---|---|---|
| SIFT-128 | HNSW | M=8, ef=1024 | 0.9972 | 387 | 70.3 | 592.6 |
| SIFT-128 | HNSW | M=16, ef=384 | 0.9995 | 1,011 | 93.7 | 656.3 |
| SIFT-128 | HNSW | M=32, ef=256 | 0.9994 | 1,168 | 114.3 | 784.1 |
| SIFT-128 | IVF-Flat | nlist=1024, nprobe=128 | 0.9997 | 198 | 6.8 | 520.5 |
| SIFT-128 | IVF-Flat | nlist=4096, nprobe=384 | 0.9998 | 235 | 34.2 | 522.1 |
| SIFT-128 | IVF-Flat | nlist=16384, nprobe=768 | 0.9994 | 311 | 396.1 | 528.5 |
| SIFT-128 | IVF-PQ | m=16, nlist=4096, nprobe=192 | 0.5878 | 875 | 56.9 | 26.3 |
| SIFT-128 | IVF-PQ | m=32, nlist=1024, nprobe=128 | 0.7323 | 297 | 51.9 | 40.7 |
| SIFT-128 | IVF-PQ | m=32, nlist=4096, nprobe=384 | 0.7423 | 326 | 81.2 | 42.3 |
| SIFT-128 | IVF-PQ | m=64, nlist=4096, nprobe=512 | 0.8843 | 119 | 104.7 | 74.3 |
| GloVe-100 | HNSW | M=8, ef=1024 | 0.8652 | 378 | 79.2 | 480.6 |
| GloVe-100 | HNSW | M=16, ef=1024 | 0.9516 | 288 | 129.1 | 544.3 |
| GloVe-100 | HNSW | M=32, ef=1024 | 0.9845 | 218 | 195.4 | 672.2 |
| GloVe-100 | IVF-Flat | nlist=1024, nprobe=768 | 0.9998 | 37 | 6.3 | 408.4 |
| GloVe-100 | IVF-Flat | nlist=4096, nprobe=1024 | 0.9892 | 97 | 35.9 | 409.4 |
| GloVe-100 | IVF-Flat | nlist=16384, nprobe=1024 | 0.9590 | 257 | 389.0 | 413.6 |
| GloVe-100 | IVF-PQ | m=20, nlist=4096, nprobe=1024 | 0.5346 | 147 | 65.2 | 29.5 |
| GloVe-100 | IVF-PQ | m=25, nlist=1024, nprobe=1024 | 0.6158 | 44 | 45.0 | 33.4 |
| GloVe-100 | IVF-PQ | m=25, nlist=4096, nprobe=1024 | 0.6247 | 126 | 71.9 | 34.5 |
| GloVe-100 | IVF-PQ | m=50, nlist=4096, nprobe=1024 | 0.8685 | 68 | 106.3 | 59.5 |
