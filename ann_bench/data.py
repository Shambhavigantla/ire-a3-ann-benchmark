"""Loading the ann-benchmarks HDF5 files and the fixed tune/test query split.

Each file holds:
    train      (n, d)    vectors to index
    test       (10000, d) queries
    neighbors  (10000, 100) ids of the true 100 nearest neighbours over the FULL train set
    distances  (10000, 100) their distances, in the file's own metric

GloVe is angular; we L2-normalise base and queries so that squared Euclidean
distance = 2 - 2*cos, which ranks neighbours exactly like cosine similarity.
Everything downstream then uses Euclidean (IndexFlatL2-style) distance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATASETS = {
    "sift": {
        "filename": "sift-128-euclidean.hdf5",
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "normalise": False,
        "file_metric": "euclidean",
    },
    "glove": {
        "filename": "glove-100-angular.hdf5",
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "normalise": True,
        "file_metric": "angular",
    },
}

# First N_TUNE queries are for tuning, the rest for reported numbers.
N_TUNE = 5000


@dataclass
class Dataset:
    name: str
    base: np.ndarray            # (n, d) float32, C-contiguous
    queries: np.ndarray         # (10000, d) float32, C-contiguous
    file_neighbors: np.ndarray  # (10000, 100) int64, w.r.t. the full base set
    file_distances: np.ndarray  # (10000, 100) float32, in file_metric
    n_base_total: int
    file_metric: str

    @property
    def n_base(self) -> int:
        return self.base.shape[0]

    @property
    def dim(self) -> int:
        return self.base.shape[1]

    @property
    def is_subset(self) -> bool:
        """True when only the first n_base vectors were loaded (file ground truth no longer valid)."""
        return self.n_base < self.n_base_total


def normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; zero rows stay zero instead of becoming NaN."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, 1e-12), dtype=np.float32)


def split_queries(arr: np.ndarray, n_tune: int = N_TUNE):
    """Return (tune, test) halves of any per-query array."""
    return arr[:n_tune], arr[n_tune:]


def dataset_path(name: str, data_dir: str = "data") -> str:
    return os.path.join(data_dir, DATASETS[name]["filename"])


def load_dataset(name: str, data_dir: str = "data", n_base: int | None = None) -> Dataset:
    """Load a dataset; n_base keeps only the first n_base train vectors (for --quick runs)."""
    import h5py

    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[name]
    path = dataset_path(name, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `bash download_data.sh` first")

    with h5py.File(path, "r") as f:
        n_total = f["train"].shape[0]
        n = n_total if n_base is None else min(int(n_base), n_total)
        base = f["train"][:n]
        queries = f["test"][:]
        neighbors = f["neighbors"][:].astype(np.int64)
        distances = f["distances"][:]

    if spec["normalise"]:
        base, queries = normalise(base), normalise(queries)
    else:
        base = np.ascontiguousarray(base, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)

    return Dataset(
        name=name,
        base=base,
        queries=queries,
        file_neighbors=neighbors,
        file_distances=np.asarray(distances, dtype=np.float32),
        n_base_total=n_total,
        file_metric=spec["file_metric"],
    )
