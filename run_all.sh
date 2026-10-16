#!/usr/bin/env bash
# Regenerate every number, table and figure in the report.
#   bash run_all.sh           full datasets  -> results/
#   bash run_all.sh --quick   first 100K base vectors -> results/quick/  (< 15 min on a laptop)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
QUICK=0
if [[ "${1:-}" == "--quick" ]]; then QUICK=1; fi

bash download_data.sh

echo "== unit tests =="
"$PY" -m pytest -q

if [[ $QUICK -eq 1 ]]; then
  RES=results/quick
  COMMON=(--n-base 100000 --results-dir "$RES")
  P1=(--single-queries 300 --repeats 3)
else
  RES=results
  COMMON=(--results-dir "$RES")
  P1=(--single-queries 1000 --repeats 3)
fi
mkdir -p "$RES/figures"
rm -f "$RES/results.csv"

echo "== Part 1: exact baseline =="
"$PY" scripts/part1_exact.py "${COMMON[@]}" "${P1[@]}"

echo "== Part 2: HNSW / IVF-Flat / IVF-PQ benchmark =="
if [[ $QUICK -eq 1 ]]; then
  "$PY" scripts/part2_benchmark.py --quick --results-dir "$RES"
else
  "$PY" scripts/part2_benchmark.py --results-dir "$RES"
fi

# Parts 3-4 are added here as they are implemented: scripts/part3_*.py, scripts/part4_*.py

echo "== Tables and figures =="
"$PY" scripts/make_tables.py --results-dir "$RES"
"$PY" scripts/make_figures.py --results-dir "$RES"

echo "Done. Results in $RES/ (tables in $RES/tables, figures in $RES/figures)"
