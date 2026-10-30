# CS4.406 Assignment 3: Benchmarking ANN search

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
bash download_data.sh        # ~1 GB: SIFT-128 and GloVe-100 from ann-benchmarks into data/
```

## Running

| Command | What it does |
|---|---|
| `bash run_all.sh` | Full datasets. Writes `results/results.csv`, `results/figures/`, and the summaries. |
| `bash run_all.sh --quick` | Same pipeline on the first 100,000 base vectors. Writes to `results/quick/`. Takes under 15 minutes. |
| `python -m pytest -q` | Unit tests only. |
| `python scripts/make_report.py` | Rebuild `report/report.html` and `report/report.pdf` from `results/` (PDF needs Edge or Chrome; set `REPORT_BROWSER` to its path if it isn't found). |
| `python scripts/part1_exact.py --datasets sift` | Part 1 on a single dataset. |

Set `PYTHON=python3` if your interpreter has a different name. `results/machine_info.json` records the
CPU, RAM, OS and library versions.

## Layout

✅ = exists, ⬜ = planned.

```
code/
├── run_all.sh              ✅ one command: tests → experiments → tables → figures (--quick = 100K subset)
├── download_data.sh        ✅ fetches the two HDF5 files into data/
├── requirements.txt  pytest.ini  .gitignore
├── hypothesis.md           ✅ Part 3 and 4 hypotheses; each committed BEFORE its experiment
├── ann_bench/              shared library, reused by every part
│   ├── data.py             ✅ HDF5 loading, GloVe normalisation, tune/test query split
│   ├── metrics.py          ✅ tie-aware recall@k (numpy only)
│   ├── timing.py           ✅ single-query and batch timing, repeats, median; FAISS thread control
│   ├── exact.py            ✅ IndexFlatL2, ground truth for subsets, index size
│   ├── results.py          ✅ results.csv writer, machine info
│   ├── tables.py           ✅ Markdown and LaTeX table writer
│   ├── plotting.py         ✅ shared figure style and fixed per-index colours
│   ├── indexes.py          ✅ P2: build HNSW / IVF-Flat / IVF-PQ (seeded), set efSearch / nprobe
│   ├── frontier.py         ✅ P2: best trade-off curve, tune/test selection at recall 0.90 / 0.95
│   ├── synthetic.py        ✅ P4: seeded drift generator (SIFT vectors translated by s·σ)
│   ├── diagnostics.py      ✅ P4: IVF list balance, PQ error, HNSW links, distances per query
│   └── drift_report.py     ✅ P4: reads Part 4 rows back for tables and figures
├── scripts/
│   ├── part1_exact.py      ✅ exact baseline, check against the file, single vs batch timing
│   ├── part2_benchmark.py  ✅ build × search sweeps on both datasets (--quick, --resume, --indexes)
│   ├── part3_compression.py ✅ IVF-PQ m, plain vs exact re-ranking of the top 100, vs IVF-Flat
│   ├── part4_drift.py      ✅ new shifted cluster: inserted vs rebuilt, HNSW / IVF-Flat / IVF-PQ
│   ├── make_tables.py      ✅ results/ → results/tables/*.md and *.tex
│   ├── make_report.py      ✅ results/ → report/report.html and report.pdf (every number filled in)
│   └── make_figures.py     ✅ results/ → results/figures/*.png and *.pdf
├── tests/                  ✅ unit tests (recall, timing, data, frontier, Part 3 coverage, Part 4 generator) + FAISS tests
├── docs/
│   ├── DESIGN_NOTES.md     ✅ decisions and rationale, plus the checklist of required outputs
│   ├── PROCESS_LOG.md      ✅ plain-language log of each step
│   ├── REPORT_NOTES.md     ✅ observations, numbers and mechanisms for the report
│   └── HYPOTHESIS_GUIDE.md ✅ how the Part 3/4 hypotheses were approached
├── report/                 ✅ report_template.html (text, ${…} placeholders) → report.html, report.pdf (≤ 5 pages)
├── results/                full-run outputs (submitted)
│   ├── results.csv         one row per benchmark run
│   ├── machine_info.json  part*_summary.json
│   ├── tables/  figures/
│   └── quick/              --quick outputs, same layout (not submitted)
└── data/                   datasets and cached ground truth (git-ignored, not submitted)
```

The submission zip's top level holds `report.pdf`, `code/` and `ai_log/`. `ai_log/` lives next to
`code/`, not inside it.

## Part 1: what is measured and how

**Exact baseline.** `faiss.IndexFlatL2` compares each query with every base vector. It is the
reference for all later parts. GloVe vectors and queries are L2-normalised first. For unit vectors,
‖a−b‖² = 2 − 2cos(a,b), so Euclidean ranking is the same as angular ranking.

**Recall@10 with ties** (`ann_bench/metrics.py`). For each query:

1. Recompute in float64 the exact squared distances from the query to its first 10 ground-truth ids.
   The threshold d₁₀ is the largest of these.
2. Recompute in float64 the exact distances from the query to the first 10 ids the index returned.
   The distances the index reports are never used, because IVF-PQ only reports approximate ones.
3. A returned id is a hit if its distance is ≤ d₁₀·(1 + 10⁻⁶). Duplicate ids count once. `-1` padding
   and missing results count as misses.
4. recall@10 = hits / 10. The dataset value is the mean over queries.

A vector exactly as far away as the 10th true neighbour is just as correct as the one the ground
truth happened to list. Plain id intersection would mark it wrong. SIFT has integer components, so
exact distance ties do happen. The script reports both metrics (`recall_at_k` and `recall_strict`)
and counts the queries where only ties make the difference.

**Agreement with the data file.** On the full sets, all 10,000 queries are searched exactly with
k=100. The script reports:
- tie-aware recall against the file's `neighbors`, which should be 1.0;
- how many queries differ only by ties;
- whether the top-100 lists are identical in order;
- the largest difference between FAISS distances and the file's `distances`, after converting units.
  FAISS returns squared L2. SIFT stores L2. For GloVe the script tries both possible conversions and
  reports the one that fits.

On a `--quick` subset the file's ground truth refers to vectors that were not loaded, so the script
recomputes it (and caches it in `data/cache/`). It can still check against the file. If at least 10
of a query's 100 file neighbours have id < n, those are exactly the subset's top 10. Every other
subset vector lies beyond the full set's 100th neighbour.

**Timing** (`ann_bench/timing.py`). Every run gets a short warm-up, then 3 repeats, and the median is
reported. All timing uses the test half of the queries. There are three modes:

| mode | threads | purpose |
|---|---|---|
| single | 1 | One `search()` call per query. This is the protocol Part 2 uses. By default it uses the first 1000 test queries: QPS is a rate, and 5000 single exact queries on 1M vectors take several minutes. |
| batch | 1 | All 5000 test queries in one call. Separates the effect of batching from the effect of parallelism. |
| batch | all | Same, using every core. |

**Why batch is much faster.** One exact query reads the whole base set (512 MB for SIFT) from RAM.
It does only about 2 flops per byte loaded, so it is limited by memory bandwidth. It also pays
Python and FAISS call overhead each time. When FAISS gets 20 or more queries at once
(`distance_compute_blas_threshold`), it computes ‖q‖² + ‖x‖² − 2·q·x with a single BLAS matrix
multiply over blocks of queries and base vectors. Each block of base vectors is loaded into cache
once and reused for many queries, so the work becomes compute-bound and runs on SIMD-optimised
GEMM. The multi-threaded batch then also spreads that work across cores. The single-thread batch row
shows how much of the speed-up comes from batching alone.
