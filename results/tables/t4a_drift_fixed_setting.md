**Part 4: D = 400,000 SIFT vectors; a new cluster of 100,000 SIFT vectors moved by s·σ (σ = 385, RMS distance of D from its mean) is inserted (incremental: no retraining) or the index is rebuilt on D′. Each index keeps one search setting, chosen on the baseline as the smallest reaching 0.95× its best baseline recall. Old / new = original-region / new-region test queries; recall against exact search on the database searched. Maintenance = insertion time (incremental) or full build (rebuilt and baseline), all threads.**

| Index | Fixed setting | Shift s | State | Recall old | Recall new | QPS old | QPS new | Dist./query new | Maintenance (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| HNSW | ef=48 | – | baseline (D) | 0.9601 | 0.9630 | 9,763 | 9,785 | 930 | 32.20 | 250.3 |
| HNSW | ef=48 | 0σ | incremental | 0.9568 | 0.9612 | 9,515 | 9,381 | 953 | 9.45 | 312.9 |
| HNSW | ef=48 | 0σ | rebuilt | 0.9570 | 0.9622 | 9,185 | 9,682 | 947 | 41.12 | 312.9 |
| HNSW | ef=48 | 0.5σ | incremental | 0.9592 | 0.9720 | 9,593 | 11,458 | 855 | 7.25 | 312.9 |
| HNSW | ef=48 | 0.5σ | rebuilt | 0.9625 | 0.9772 | 9,682 | 10,374 | 867 | 39.31 | 312.9 |
| HNSW | ef=48 | 1σ | incremental | 0.9600 | 0.9513 | 9,306 | 11,473 | 858 | 5.03 | 312.9 |
| HNSW | ef=48 | 1σ | rebuilt | 0.9626 | 0.9635 | 9,865 | 12,924 | 818 | 37.83 | 312.9 |
| HNSW | ef=48 | 2σ | incremental | 0.9600 | 0.9766 | 9,545 | 12,565 | 800 | 4.59 | 312.9 |
| HNSW | ef=48 | 2σ | rebuilt | 0.9624 | 0.9821 | 10,108 | 13,009 | 812 | 37.99 | 312.9 |
| IVF-Flat | nprobe=48 | – | baseline (D) | 0.9688 | 0.9773 | 2,495 | 2,557 | 10,758 | 8.68 | 199.4 |
| IVF-Flat | nprobe=48 | 0σ | incremental | 0.9713 | 0.9777 | 2,360 | 2,456 | 13,474 | 0.38 | 249.0 |
| IVF-Flat | nprobe=48 | 0σ | rebuilt | 0.9732 | 0.9758 | 2,111 | 2,126 | 13,553 | 9.44 | 249.0 |
| IVF-Flat | nprobe=48 | 0.5σ | incremental | 0.9688 | 0.9560 | 2,026 | 2,181 | 13,718 | 0.68 | 249.0 |
| IVF-Flat | nprobe=48 | 0.5σ | rebuilt | 0.9763 | 0.9928 | 2,251 | 2,365 | 12,999 | 9.21 | 249.0 |
| IVF-Flat | nprobe=48 | 1σ | incremental | 0.9688 | 0.9650 | 2,242 | 1,956 | 15,638 | 0.66 | 249.0 |
| IVF-Flat | nprobe=48 | 1σ | rebuilt | 0.9773 | 0.9971 | 2,160 | 2,467 | 12,829 | 9.39 | 249.0 |
| IVF-Flat | nprobe=48 | 2σ | incremental | 0.9688 | 0.9916 | 2,256 | 1,326 | 25,100 | 0.62 | 249.0 |
| IVF-Flat | nprobe=48 | 2σ | rebuilt | 0.9773 | 0.9971 | 2,356 | 2,790 | 12,829 | 9.28 | 249.0 |
| IVF-PQ | nprobe=24 | – | baseline (D) | 0.7232 | 0.7212 | 4,259 | 4,119 | 5,511 | 37.81 | 16.4 |
| IVF-PQ | nprobe=24 | 0σ | incremental | 0.7168 | 0.7180 | 3,996 | 3,808 | 6,929 | 1.21 | 20.2 |
| IVF-PQ | nprobe=24 | 0σ | rebuilt | 0.7176 | 0.7192 | 3,833 | 3,923 | 6,942 | 39.19 | 20.2 |
| IVF-PQ | nprobe=24 | 0.5σ | incremental | 0.7226 | 0.6489 | 3,796 | 3,693 | 7,066 | 1.19 | 20.2 |
| IVF-PQ | nprobe=24 | 0.5σ | rebuilt | 0.7240 | 0.7514 | 3,819 | 4,293 | 6,611 | 38.72 | 20.2 |
| IVF-PQ | nprobe=24 | 1σ | incremental | 0.7232 | 0.5223 | 3,952 | 3,787 | 8,250 | 1.19 | 20.2 |
| IVF-PQ | nprobe=24 | 1σ | rebuilt | 0.7263 | 0.7534 | 3,825 | 4,157 | 6,609 | 38.98 | 20.2 |
| IVF-PQ | nprobe=24 | 2σ | incremental | 0.7232 | 0.3034 | 3,964 | 2,446 | 14,732 | 1.18 | 20.2 |
| IVF-PQ | nprobe=24 | 2σ | rebuilt | 0.7259 | 0.7532 | 4,639 | 5,335 | 6,609 | 36.99 | 20.2 |
