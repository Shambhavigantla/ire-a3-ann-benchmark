**Exact search (IndexFlatL2) against the ground truth in the data file.**

| Dataset | Base vectors | Queries checked | Recall@10 (ties) | Recall@10 (strict) | Differ only by ties | Top-100 identical | Max rel. dist. err. | Build (s) | Index (MB) |
|---|---|---|---|---|---|---|---|---|---|
| SIFT-128 | 1,000,000 | 10,000 | 1.0000 | 0.9993 | 65 | 44.01% | 1.2e-07 | 0.198 | 488.3 |
| GloVe-100 | 1,183,514 | 10,000 | 1.0000 | 1.0000 | 0 | 98.80% | 2.6e-05 | 0.158 | 451.5 |
