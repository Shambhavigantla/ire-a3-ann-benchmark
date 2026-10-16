"""Integration test: FAISS exact search on small random data has recall 1 under our metric."""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from ann_bench.exact import build_flat  # noqa: E402
from ann_bench.metrics import recall_at_k  # noqa: E402


def test_flat_index_matches_numpy_brute_force():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((2000, 16)).astype(np.float32)
    q = rng.standard_normal((50, 16)).astype(np.float32)
    d = ((q[:, None, :].astype(np.float64) - base[None, :, :]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :100]
    _, ids = build_flat(base).search(q, 10)
    assert recall_at_k(ids, base, q, gt, k=10).mean() == 1.0


def test_flat_index_with_duplicate_vectors_is_perfect_despite_ties():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 4, size=(3000, 4)).astype(np.float32)   # tiny integer grid -> massive ties
    q = rng.integers(0, 4, size=(20, 4)).astype(np.float32)
    d = ((q[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1, kind="stable")[:, :100]
    _, ids = build_flat(base).search(q, 10)
    assert recall_at_k(ids, base, q, gt, k=10).mean() == 1.0
