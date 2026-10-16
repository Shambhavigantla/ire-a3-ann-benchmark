#!/usr/bin/env bash
# Download the two ann-benchmarks datasets into data/ (skipped if already present).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

for f in sift-128-euclidean.hdf5 glove-100-angular.hdf5; do
  if [[ ! -s "data/$f" ]]; then
    echo "downloading $f ..."
    curl -L --fail -o "data/$f.part" "http://ann-benchmarks.com/$f"
    mv "data/$f.part" "data/$f"
  fi
done
