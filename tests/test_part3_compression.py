"""Tests for candidate coverage and exact re-ranking (Part 3)."""
import numpy as np
import pytest

from ann_bench.metrics import coverage_at_k, recall_at_k


def _toy():
    base = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]], dtype=np.float32)
    q = np.array([[0.0]], dtype=np.float32)
    gt = np.array([[0, 1, 2]])          # true top 3
    return base, q, gt


def test_coverage_counts_true_neighbours_anywhere_in_candidates():
    base, q, gt = _toy()
    # true neighbours 0 and 2 are in the list but not in the first k positions
    cand = np.array([[4, 3, 0, 2]])
    assert coverage_at_k(cand, base, q, gt, k=3)[0] == pytest.approx(2 / 3)
    assert recall_at_k(cand, base, q, gt, k=3)[0] == pytest.approx(1 / 3)   # only id 0 in first 3


def test_coverage_capped_at_k_and_ignores_duplicates_and_padding():
    base, q, gt = _toy()
    assert coverage_at_k(np.array([[0, 0, -1, 1, 2]]), base, q, gt, k=3)[0] == 1.0
    assert coverage_at_k(np.array([[0, 0, 0, -1]]), base, q, gt, k=3)[0] == pytest.approx(1 / 3)


def test_coverage_counts_ties_at_kth_distance():
    base = np.array([[0.0], [1.0], [-1.0], [5.0]], dtype=np.float32)
    q = np.array([[0.0]], dtype=np.float32)
    gt = np.array([[0, 1]])             # id 2 ties with id 1
    assert coverage_at_k(np.array([[3, 2, 0]]), base, q, gt, k=2)[0] == 1.0


def test_coverage_never_below_recall_of_same_list():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((300, 8)).astype(np.float32)
    q = rng.standard_normal((20, 8)).astype(np.float32)
    d = ((q[:, None] - base[None]) ** 2).sum(-1)
    gt = np.argsort(d, 1)[:, :10]
    cand = rng.integers(0, 300, (20, 50))
    assert (coverage_at_k(cand, base, q, gt) >= recall_at_k(cand, base, q, gt)).all()


faiss = pytest.importorskip("faiss")
from ann_bench.indexes import (IndexSpec, add_exact_rerank, build_index,  # noqa: E402
                               coarse_centroids, set_search_param)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((4000, 16)).astype(np.float32)
    q = rng.standard_normal((100, 16)).astype(np.float32)
    flat = faiss.IndexFlatL2(16)
    flat.add(base)
    gt = flat.search(q, 10)[1]
    pq, _ = build_index(IndexSpec("IVFPQ", {"nlist": 32, "m": 4}), base, seed=0)
    ivf, _ = build_index(IndexSpec("IVFFlat", {"nlist": 32}), base, seed=0)
    return base, q, gt, pq, ivf


def test_rerank_recall_equals_candidate_coverage(setup):
    base, q, gt, pq, _ = setup
    rr = add_exact_rerank(pq, base, depth=50, k=10)
    for nprobe in (1, 4, 32):
        set_search_param(rr, IndexSpec("IVFPQ", {"nlist": 32, "m": 4}), nprobe)
        assert faiss.extract_index_ivf(pq).nprobe == nprobe      # knob reaches the wrapped index
        rec = recall_at_k(rr.search(q, 10)[1], base, q, gt)
        cov = coverage_at_k(pq.search(q, 50)[1], base, q, gt)
        np.testing.assert_allclose(rec, cov)


def test_same_seed_gives_same_centroids_so_ivfflat_bounds_rerank(setup):
    base, q, gt, pq, ivf = setup
    np.testing.assert_array_equal(coarse_centroids(pq), coarse_centroids(ivf))
    rr = add_exact_rerank(pq, base, depth=100, k=10)
    for nprobe in (1, 2, 8):
        faiss.extract_index_ivf(pq).nprobe = nprobe
        faiss.extract_index_ivf(ivf).nprobe = nprobe
        r_rr = recall_at_k(rr.search(q, 10)[1], base, q, gt)
        r_ivf = recall_at_k(ivf.search(q, 10)[1], base, q, gt)
        assert (r_rr <= r_ivf + 1e-12).all()      # per query: same clusters scanned


def test_rerank_depth_must_be_multiple_of_k(setup):
    base, _, _, pq, _ = setup
    with pytest.raises(ValueError):
        add_exact_rerank(pq, base, depth=55, k=10)
