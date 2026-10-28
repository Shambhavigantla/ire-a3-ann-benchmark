"""Tests for the Part 4 drift generator and mechanism diagnostics."""
import numpy as np
import pytest

from ann_bench.synthetic import data_spread, make_drift, query_groups, shift_direction


@pytest.fixture(scope="module")
def drift():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 50, (600, 16)).astype(np.float32)
    queries = rng.integers(0, 50, (40, 16)).astype(np.float32)
    return make_drift(base, queries, n_db=400, n_new=200, seed=7, n_tune=20), base, queries


def test_generator_is_seeded_and_direction_is_unit():
    assert np.array_equal(shift_direction(16, 3), shift_direction(16, 3))
    assert not np.array_equal(shift_direction(16, 3), shift_direction(16, 4))
    assert np.linalg.norm(shift_direction(16, 3)) == pytest.approx(1.0)


def test_spread_is_rms_distance_from_mean():
    x = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 3.0], [0.0, -3.0]])
    assert data_spread(x) == pytest.approx(np.sqrt((1 + 1 + 9 + 9) / 4))


def test_db_and_new_cluster_are_disjoint_real_vectors(drift):
    d, base, _ = drift
    assert np.array_equal(d.db, base[:400]) and np.array_equal(d.reserved, base[400:600])
    assert np.array_equal(d.new_vectors(0.0), base[400:600])              # control: unmoved
    db2 = d.updated_db(1.0)
    assert db2.shape == (600, 16) and np.array_equal(db2[:400], base[:400])   # old ids unchanged


def test_shift_moves_by_exactly_s_sigma_and_keeps_internal_distances(drift):
    d, _, _ = drift
    for s in (0.5, 2.0):
        moved = d.new_vectors(s).astype(np.float64)
        delta = moved - d.reserved
        np.testing.assert_allclose(np.linalg.norm(delta, axis=1), s * d.sigma, rtol=1e-4)
        np.testing.assert_allclose(delta / np.linalg.norm(delta, axis=1, keepdims=True),
                                   np.tile(d.direction, (200, 1)), atol=1e-4)   # same direction
        pd0 = np.linalg.norm(d.reserved[:50, None] - d.reserved[None, :50], axis=-1)
        pd1 = np.linalg.norm(moved[:50, None] - moved[None, :50], axis=-1)
        np.testing.assert_allclose(pd0, pd1, atol=1e-2)                     # translation only


def test_query_groups_disjoint_and_respect_tune_test_split():
    g = query_groups(np.zeros((10000, 2)))
    assert g[("old", "tune")] == slice(0, 2500) and g[("new", "tune")] == slice(2500, 5000)
    assert g[("old", "test")] == slice(5000, 7500) and g[("new", "test")] == slice(7500, 10000)


def test_new_region_queries_shifted_old_untouched(drift):
    d, _, q = drift
    sl = query_groups(q, n_tune=20)
    assert np.array_equal(d.group_queries("old", "test", 2.0), q[sl[("old", "test")]])
    shifted = d.group_queries("new", "test", 2.0)
    np.testing.assert_allclose(shifted - q[sl[("new", "test")]], np.tile(2.0 * d.sigma * d.direction, (len(shifted), 1)),
                               atol=1e-3)


faiss = pytest.importorskip("faiss")
from ann_bench.diagnostics import (copy_index, distances_per_query, hnsw_link_stats,  # noqa: E402
                                   ivf_list_stats, pq_error)
from ann_bench.indexes import IndexSpec, build_index  # noqa: E402


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    old = rng.standard_normal((3000, 16)).astype(np.float32)
    new = (rng.standard_normal((1000, 16)) + 50).astype(np.float32)   # far away cluster
    return old, new


def test_far_cluster_lands_in_few_ivf_lists_and_raises_imbalance(blobs):
    old, new = blobs
    idx, _ = build_index(IndexSpec("IVFFlat", {"nlist": 32}), old, seed=0)
    before = ivf_list_stats(idx, n_old=3000)
    inc = copy_index(idx)
    inc.add(new)
    after = ivf_list_stats(inc, n_old=3000)
    assert idx.ntotal == 3000 and inc.ntotal == 4000          # copy is independent
    assert before["n_new"] == 0 and after["n_new"] == 1000
    assert after["lists_holding_90pct_new"] <= 3               # all go to the nearest border centroid(s)
    assert after["imbalance"] > 2 * before["imbalance"]
    reb, _ = build_index(IndexSpec("IVFFlat", {"nlist": 32}), np.concatenate([old, new]), seed=0)
    assert ivf_list_stats(reb, n_old=3000)["lists_with_new"] > after["lists_with_new"]   # retraining spreads them


def test_pq_error_higher_for_vectors_unlike_training_data(blobs):
    old, new = blobs
    idx, _ = build_index(IndexSpec("IVFPQ", {"nlist": 8, "m": 4}), old, seed=0)
    assert pq_error(idx, new) > 5 * pq_error(idx, old[:1000])


def test_hnsw_link_stats_and_distance_counter(blobs):
    old, new = blobs
    idx, _ = build_index(IndexSpec("HNSW", {"M": 8, "efConstruction": 40}), old, seed=0)
    idx.add(new)
    st = hnsw_link_stats(idx, n_old=3000)
    assert st["new_links_to_new"] > 0.9            # a far cluster links mostly to itself
    assert 0 <= st["old_nodes_linking_new"] < 0.05
    idx.hnsw.efSearch = 32
    assert distances_per_query(idx, "HNSW", old[:20], 10) > 32
