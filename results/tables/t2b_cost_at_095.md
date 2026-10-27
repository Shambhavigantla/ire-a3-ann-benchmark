**Build time (all threads), serialised index size and bytes per vector for the configuration chosen at recall 0.95.**

| Dataset | Index | Settings | Build (s) | of which train (s) | Index (MB) | Bytes / vector | vs raw float32 |
|---|---|---|---|---|---|---|---|
| SIFT-128 | HNSW | M=32, ef=48 | 69.5 | – | 747.8 | 784.1 | 1.53× |
| SIFT-128 | IVF-Flat | nlist=16384, nprobe=96 | 244.8 | 218.1 | 504.0 | 528.5 | 1.03× |
| SIFT-128 | IVF-PQ | m=64, nlist=4096, nprobe=512 (0.95 not reached; best recall 0.884) | 77.7 | 65.5 | 70.8 | 74.3 | 0.15× |
| SIFT-128 | Exact (Flat) | – | 0.12 | – | 488.3 | 512.0 | 1.00× |
| GloVe-100 | HNSW | M=32, ef=384 | 119.0 | – | 758.6 | 672.2 | 1.68× |
| GloVe-100 | IVF-Flat | nlist=16384, nprobe=1024 | 396.0 | 341.4 | 466.9 | 413.6 | 1.03× |
| GloVe-100 | IVF-PQ | m=50, nlist=4096, nprobe=1024 (0.95 not reached; best recall 0.869) | 66.4 | 53.7 | 67.2 | 59.5 | 0.15× |
| GloVe-100 | Exact (Flat) | – | 0.11 | – | 451.5 | 400.0 | 1.00× |
