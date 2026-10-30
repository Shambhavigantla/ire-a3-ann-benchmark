#!/usr/bin/env bash
# Regenerate every number, table and figure in the report.
#   bash run_all.sh           full datasets  -> results/
#   bash run_all.sh --quick   first 100K base vectors -> results/quick/  (< 15 min on a laptop)
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
else
  PY=""
  for candidate in \
    "${VIRTUAL_ENV:-}/Scripts/python.exe" \
    "${VIRTUAL_ENV:-}/bin/python" \
    ".venv-1/Scripts/python.exe" \
    ".venv/Scripts/python.exe" \
    ".venv-1/bin/python" \
    ".venv/bin/python"; do
    if [[ -x "$candidate" ]] && "$candidate" -c "import pytest" >/dev/null 2>&1; then
      PY="$candidate"
      break
    fi
  done
  PY="${PY:-python}"
fi
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

echo "== Part 3: compression (IVF-PQ m, with vs without exact re-ranking) =="
if [[ $QUICK -eq 1 ]]; then
  "$PY" scripts/part3_compression.py --quick --results-dir "$RES"
else
  "$PY" scripts/part3_compression.py --results-dir "$RES"
fi

echo "== Part 4: drift (a shifted cluster inserted vs rebuilt; generated, seeded data) =="
if [[ $QUICK -eq 1 ]]; then
  "$PY" scripts/part4_drift.py --quick --results-dir "$RES"
else
  "$PY" scripts/part4_drift.py --results-dir "$RES"
fi

echo "== Tables and figures =="
"$PY" scripts/make_tables.py --results-dir "$RES"
"$PY" scripts/make_figures.py --results-dir "$RES"

echo "Done. Results in $RES/ (tables in $RES/tables, figures in $RES/figures)"
