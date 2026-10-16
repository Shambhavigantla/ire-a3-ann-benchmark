"""Exact (brute-force) search with FAISS and ground truth for subsets."""
from __future__ import annotations

import os

import faiss
import numpy as np

from .data import Dataset


def build_flat(base: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    return index


def faiss_search_fn(index):
    """Adapt a FAISS index to the (queries, k) -> ids signature used by timing.py."""
    def search(q: np.ndarray, k: int) -> np.ndarray:
        return index.search(q, k)[1]
    return search


def index_size_bytes(index) -> int:
    """Size of the serialised index, i.e. what it costs to store/load it."""
    return int(faiss.serialize_index(index).nbytes)


def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100,
                         cache_path: str | None = None) -> np.ndarray:
    """Exact top-k ids by brute force (all threads). Cached to .npy if a path is given."""
    if cache_path and os.path.exists(cache_path):
        gt = np.load(cache_path)
        if gt.shape == (queries.shape[0], k):
            return gt
    _, ids = build_flat(base).search(np.ascontiguousarray(queries, dtype=np.float32), k)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, ids)
    return ids


def ground_truth_for(ds: Dataset, k: int = 100, cache_dir: str = "data/cache") -> np.ndarray:
    """Ground-truth ids valid for ds.base.

    On the full set this is the file's `neighbors`; on a subset (e.g. --quick) the
    file's neighbours refer to vectors we did not load, so we recompute them.
    """
    if not ds.is_subset:
        return ds.file_neighbors[:, :k]
    cache = os.path.join(cache_dir, f"{ds.name}_n{ds.n_base}_gt{k}.npy")
    return compute_ground_truth(ds.base, ds.queries, k=k, cache_path=cache)
