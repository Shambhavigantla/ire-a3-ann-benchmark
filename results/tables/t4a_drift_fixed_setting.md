**Part 4: D = 400,000 SIFT vectors; a new cluster of 100,000 SIFT vectors moved by s·σ (σ = 385, RMS distance of D from its mean) is inserted (incremental: no retraining) or the index is rebuilt on D′. Each index keeps one search setting, chosen on the baseline as the smallest reaching 0.95× its best baseline recall. Old / new = original-region / new-region test queries; recall against exact search on the database searched. Maintenance = insertion time (incremental) or full build (rebuilt and baseline), all threads.**

| Index | Fixed setting | Shift s | State | Recall old | Recall new | QPS old | QPS new | Dist./query new | Maintenance (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| HNSW | ef=48 | – | baseline (D) | 0.9601 | 0.9630 | 8,121 | 8,450 | 930 | 34.94 | 250.3 |
| HNSW | ef=48 | 0σ | incremental | 0.9568 | 0.9612 | 7,773 | 8,085 | 953 | 10.10 | 312.9 |
| HNSW | ef=48 | 0σ | rebuilt | 0.9570 | 0.9622 | 7,655 | 7,913 | 947 | 46.91 | 312.9 |
| HNSW | ef=48 | 0.5σ | incremental | 0.9592 | 0.9720 | 7,958 | 8,904 | 855 | 8.15 | 312.9 |
| HNSW | ef=48 | 0.5σ | rebuilt | 0.9625 | 0.9772 | 7,453 | 9,004 | 867 | 45.14 | 312.9 |
| HNSW | ef=48 | 1σ | incremental | 0.9600 | 0.9513 | 8,046 | 9,289 | 858 | 6.50 | 312.9 |
| HNSW | ef=48 | 1σ | rebuilt | 0.9626 | 0.9635 | 6,027 | 7,091 | 818 | 43.92 | 312.9 |
| HNSW | ef=48 | 2σ | incremental | 0.9600 | 0.9766 | 8,495 | 10,128 | 800 | 6.08 | 312.9 |
| HNSW | ef=48 | 2σ | rebuilt | 0.9624 | 0.9821 | 7,632 | 9,943 | 812 | 50.40 | 312.9 |
| IVF-Flat | nprobe=48 | – | baseline (D) | 0.9687 | 0.9773 | 1,774 | 1,833 | 10,759 | 9.25 | 199.4 |
| IVF-Flat | nprobe=48 | 0σ | incremental | 0.9712 | 0.9776 | 1,546 | 1,644 | 13,475 | 0.75 | 249.0 |
| IVF-Flat | nprobe=48 | 0σ | rebuilt | 0.9732 | 0.9758 | 1,521 | 1,617 | 13,553 | 9.82 | 249.0 |
| IVF-Flat | nprobe=48 | 0.5σ | incremental | 0.9687 | 0.9558 | 1,538 | 1,642 | 13,718 | 0.82 | 249.0 |
| IVF-Flat | nprobe=48 | 0.5σ | rebuilt | 0.9763 | 0.9928 | 1,369 | 1,626 | 12,999 | 9.82 | 249.0 |
| IVF-Flat | nprobe=48 | 1σ | incremental | 0.9687 | 0.9651 | 1,616 | 1,450 | 15,637 | 0.73 | 249.0 |
| IVF-Flat | nprobe=48 | 1σ | rebuilt | 0.9773 | 0.9971 | 1,555 | 1,640 | 12,829 | 9.69 | 249.0 |
| IVF-Flat | nprobe=48 | 2σ | incremental | 0.9687 | 0.9916 | 1,616 | 897 | 25,098 | 0.76 | 249.0 |
| IVF-Flat | nprobe=48 | 2σ | rebuilt | 0.9773 | 0.9971 | 1,511 | 1,628 | 12,829 | 9.84 | 249.0 |
| IVF-PQ | nprobe=24 | – | baseline (D) | 0.7199 | 0.7224 | 3,578 | 3,236 | 5,510 | 54.38 | 16.4 |
| IVF-PQ | nprobe=24 | 0σ | incremental | 0.7158 | 0.7184 | 2,775 | 3,280 | 6,929 | 1.60 | 20.2 |
| IVF-PQ | nprobe=24 | 0σ | rebuilt | 0.7176 | 0.7192 | 2,967 | 3,241 | 6,942 | 55.29 | 20.2 |
| IVF-PQ | nprobe=24 | 0.5σ | incremental | 0.7194 | 0.6506 | 3,160 | 2,959 | 7,066 | 1.56 | 20.2 |
| IVF-PQ | nprobe=24 | 0.5σ | rebuilt | 0.7240 | 0.7514 | 3,085 | 3,446 | 6,611 | 55.01 | 20.2 |
| IVF-PQ | nprobe=24 | 1σ | incremental | 0.7199 | 0.5240 | 3,083 | 2,869 | 8,249 | 1.58 | 20.2 |
| IVF-PQ | nprobe=24 | 1σ | rebuilt | 0.7263 | 0.7534 | 3,193 | 3,240 | 6,609 | 55.18 | 20.2 |
| IVF-PQ | nprobe=24 | 2σ | incremental | 0.7199 | 0.2988 | 3,203 | 1,964 | 14,728 | 1.60 | 20.2 |
| IVF-PQ | nprobe=24 | 2σ | rebuilt | 0.7259 | 0.7532 | 3,195 | 3,225 | 6,609 | 54.30 | 20.2 |
