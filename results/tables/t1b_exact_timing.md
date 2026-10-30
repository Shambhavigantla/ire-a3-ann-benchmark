**Exact search speed, median of 3 repeats, test queries. Speed-up is relative to single-query. 1-thread runs are pinned to P-cores with OpenMP and BLAS capped at 1 thread; CPU / wall ≈ 1 confirms a single busy thread.**

| Dataset | Mode | Threads | Queries | QPS | ms / query | Speed-up | Recall@10 | CPU / wall |
|---|---|---|---|---|---|---|---|---|
| SIFT-128 | single, 1 thread | 1 | 1,000 | 29.8 | 33.593 | 1.0× | 1.0000 | 0.97 |
| SIFT-128 | batch, 1 thread | 1 | 5,000 | 285.6 | 3.501 | 9.6× | 1.0000 | 0.98 |
| SIFT-128 | batch, all threads | 12 | 5,000 | 503.2 | 1.987 | 16.9× | 1.0000 | 10.00 |
| GloVe-100 | single, 1 thread | 1 | 1,000 | 31.2 | 32.018 | 1.0× | 1.0000 | 0.99 |
| GloVe-100 | batch, 1 thread | 1 | 5,000 | 232.5 | 4.301 | 7.4× | 1.0000 | 0.99 |
| GloVe-100 | batch, all threads | 12 | 5,000 | 394.7 | 2.534 | 12.6× | 1.0000 | 8.85 |
