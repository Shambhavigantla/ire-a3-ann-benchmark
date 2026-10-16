"""Unit tests for the tie-aware recall@k (numpy only, no FAISS needed)."""
import numpy as np
import pytest

from ann_bench.metrics import (first_occurrence_mask, recall_at_k, recall_from_distances,
                               squared_l2_to_ids, strict_recall_at_k)


def line_data(n=20):
    """1-D points 1, 2, ..., n with the query at 0: distance to id i is (i+1)^2, no ties."""
    base = np.arange(1, n + 1, dtype=np.float32).reshape(-1, 1)
    query = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(n, dtype=np.int64)[None, :]
    return base, query, gt


def test_perfect_retrieval_is_one():
    base, q, gt = line_data()
    assert recall_at_k(gt[:, :10], base, q, gt, k=10)[0] == 1.0


def test_nothing_correct_is_zero():
    base, q, gt = line_data()
    assert recall_at_k(np.arange(10, 20)[None, :], base, q, gt, k=10)[0] == 0.0


def test_partial_recall():
    base, q, gt = line_data()
    retrieved = np.array([[0, 1, 2, 3, 4, 5, 6, 15, 16, 17]])  # 7 of the true top-10
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == pytest.approx(0.7)


def test_order_does_not_matter():
    base, q, gt = line_data()
    retrieved = gt[:, :10][:, ::-1]
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == 1.0


def test_tie_at_kth_neighbour_counts_as_correct():
    # Points at +1..+10 and one extra at -10: id 10 is exactly as far as the 10th neighbour (id 9).
    base = np.array([[x] for x in range(1, 11)] + [[-10]], dtype=np.float32)
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])          # file chose id 9 as 10th
    retrieved = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])      # index returned the tied id 10
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == 1.0
    assert strict_recall_at_k(retrieved, gt, k=10)[0] == pytest.approx(0.9)  # naive metric is wrong


def test_point_just_beyond_kth_is_not_a_tie():
    base = np.array([[x] for x in range(1, 11)] + [[-11]], dtype=np.float32)
    q = np.zeros((1, 1), dtype=np.float32)
    gt = np.arange(11)[None, :]
    retrieved = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 10]])      # id 10 is at distance 11 > 10
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == pytest.approx(0.9)


def test_many_ties_integer_data_like_sift():
    # 2-D integer grid: many points share the same distance to the query, as in SIFT.
    xs, ys = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    base = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    q = np.zeros((1, 2), dtype=np.float32)
    d = (base ** 2).sum(1)
    order = np.argsort(d, kind="stable")
    gt = order[None, :]
    # Squared distances 0 (x1), 1 (x4), 2 (x4) give 9 points; the 10th is one of 4 points at distance 4.
    kth = d[order[9]]
    assert kth == 4 and d[order[8]] < kth
    tied_not_chosen = [t for t in np.where(d == kth)[0] if t != order[9]]
    for t in tied_not_chosen:
        retrieved = np.array([list(order[:9]) + [t]])
        assert recall_at_k(retrieved, base, q, gt, k=10)[0] == 1.0
        assert strict_recall_at_k(retrieved, gt, k=10)[0] == pytest.approx(0.9)


def test_duplicate_ids_count_once():
    base, q, gt = line_data()
    retrieved = np.array([[0, 0, 0, 1, 2, 3, 4, 5, 6, 7]])        # 8 distinct correct ids
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == pytest.approx(0.8)


def test_minus_one_padding_is_a_miss():
    base, q, gt = line_data()
    retrieved = np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1]])   # FAISS pads with -1
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == pytest.approx(0.5)


def test_fewer_than_k_results_is_a_miss():
    base, q, gt = line_data()
    assert recall_at_k(gt[:, :5], base, q, gt, k=10)[0] == pytest.approx(0.5)


def test_only_first_k_retrieved_are_scored():
    base, q, gt = line_data()
    retrieved = np.concatenate([np.arange(10, 20), np.arange(10)])[None, :]  # correct ids at ranks 11-20
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == 0.0


def test_threshold_uses_max_of_gt_even_if_gt_misordered():
    base, q, gt = line_data()
    shuffled = gt.copy()
    shuffled[0, :10] = shuffled[0, :10][::-1]                     # 10th column is now the closest
    assert recall_at_k(gt[:, :10], base, q, shuffled, k=10)[0] == 1.0


def test_index_distances_are_ignored():
    # recall_at_k recomputes distances, so an index that "claims" a far point is close gains nothing.
    base, q, gt = line_data()
    retrieved = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 19]])
    assert recall_at_k(retrieved, base, q, gt, k=10)[0] == pytest.approx(0.9)


def test_tolerance_in_recall_from_distances():
    ids = np.arange(10)[None, :]
    kth = np.array([1.0])
    near = np.array([[0.1] * 9 + [1.0 + 1e-9]])
    far = np.array([[0.1] * 9 + [1.0 + 1e-3]])
    assert recall_from_distances(ids, near, kth, k=10)[0] == 1.0
    assert recall_from_distances(ids, far, kth, k=10)[0] == pytest.approx(0.9)


def test_per_query_values_and_mean():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((500, 8)).astype(np.float32)
    q = rng.standard_normal((4, 8)).astype(np.float32)
    d = ((q[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :100]
    retrieved = gt[:, :10].copy()
    retrieved[1, 5:] = gt[1, 50:55]          # query 1: 5 wrong
    retrieved[3, :] = gt[3, 20:30]           # query 3: all wrong
    rec = recall_at_k(retrieved, base, q, gt, k=10)
    np.testing.assert_allclose(rec, [1.0, 0.5, 1.0, 0.0])
    # Continuous random data has no ties, so tie-aware == strict here.
    np.testing.assert_allclose(rec, strict_recall_at_k(retrieved, gt, k=10))


def test_brute_force_matches_squared_l2():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((50, 5)).astype(np.float32)
    q = rng.standard_normal((3, 5)).astype(np.float32)
    ids = np.array([[0, 1, -1], [49, 2, 3], [7, 7, 50]])
    d = squared_l2_to_ids(base, q, ids)
    assert d[0, 2] == np.inf and d[2, 2] == np.inf               # -1 and out of range
    np.testing.assert_allclose(d[1, 0], ((base[49] - q[1]).astype(np.float64) ** 2).sum())


def test_first_occurrence_mask():
    ids = np.array([[3, 1, 3, 2, 1]])
    np.testing.assert_array_equal(first_occurrence_mask(ids), [[True, True, False, True, False]])


def test_gt_with_too_few_columns_raises():
    base, q, gt = line_data()
    with pytest.raises(ValueError):
        recall_at_k(gt[:, :10], base, q, gt[:, :5], k=10)
