**Part 4: the smallest search setting at which the new-region tune queries reach the index's target recall (the same target that fixed its setting on the baseline), and the new-region test recall and QPS there.**

| Index | Target | Shift s | State | Setting needed (new region) | Test recall | QPS | Dist./query |
|---|---|---|---|---|---|---|---|
| HNSW | 0.949 | – | baseline (D) | ef=48 | 0.9630 | 8,450 | 930 |
| HNSW | 0.949 | 0σ | incremental | ef=48 | 0.9612 | 8,085 | 953 |
| HNSW | 0.949 | 0σ | rebuilt | ef=48 | 0.9622 | 7,913 | 947 |
| HNSW | 0.949 | 0.5σ | incremental | ef=48 | 0.9720 | 8,904 | 855 |
| HNSW | 0.949 | 0.5σ | rebuilt | ef=48 | 0.9772 | 9,004 | 867 |
| HNSW | 0.949 | 1σ | incremental | ef=48 | 0.9513 | 9,289 | 858 |
| HNSW | 0.949 | 1σ | rebuilt | ef=48 | 0.9635 | 7,091 | 818 |
| HNSW | 0.949 | 2σ | incremental | ef=32 | 0.9511 | 13,708 | 605 |
| HNSW | 0.949 | 2σ | rebuilt | ef=32 | 0.9597 | 13,148 | 616 |
| IVF-Flat | 0.949 | – | baseline (D) | nprobe=48 | 0.9773 | 1,833 | 10,759 |
| IVF-Flat | 0.949 | 0σ | incremental | nprobe=32 | 0.9543 | 2,301 | 9,133 |
| IVF-Flat | 0.949 | 0σ | rebuilt | nprobe=48 | 0.9758 | 1,617 | 13,553 |
| IVF-Flat | 0.949 | 0.5σ | incremental | nprobe=48 | 0.9558 | 1,642 | 13,718 |
| IVF-Flat | 0.949 | 0.5σ | rebuilt | nprobe=24 | 0.9711 | 3,094 | 6,611 |
| IVF-Flat | 0.949 | 1σ | incremental | nprobe=48 | 0.9651 | 1,450 | 15,637 |
| IVF-Flat | 0.949 | 1σ | rebuilt | nprobe=24 | 0.9806 | 2,917 | 6,609 |
| IVF-Flat | 0.949 | 2σ | incremental | nprobe=24 | 0.9652 | 1,578 | 14,728 |
| IVF-Flat | 0.949 | 2σ | rebuilt | nprobe=24 | 0.9806 | 2,979 | 6,609 |
| IVF-PQ | 0.709 | – | baseline (D) | nprobe=24 | 0.7224 | 3,236 | 5,510 |
| IVF-PQ | 0.709 | 0σ | incremental | nprobe=24 | 0.7184 | 3,280 | 6,929 |
| IVF-PQ | 0.709 | 0σ | rebuilt | nprobe=24 | 0.7192 | 3,241 | 6,942 |
| IVF-PQ | 0.709 | 0.5σ | incremental | not reached (best 0.683) | – | – | – |
| IVF-PQ | 0.709 | 0.5σ | rebuilt | nprobe=12 | 0.7287 | 5,138 | 3,352 |
| IVF-PQ | 0.709 | 1σ | incremental | not reached (best 0.537) | – | – | – |
| IVF-PQ | 0.709 | 1σ | rebuilt | nprobe=12 | 0.7313 | 5,360 | 3,376 |
| IVF-PQ | 0.709 | 2σ | incremental | not reached (best 0.299) | – | – | – |
| IVF-PQ | 0.709 | 2σ | rebuilt | nprobe=12 | 0.7313 | 5,365 | 3,376 |
