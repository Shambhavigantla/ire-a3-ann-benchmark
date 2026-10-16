"""Tests for the timing harness and data helpers (no FAISS or dataset files needed)."""
import numpy as np
import pytest

from ann_bench.data import normalise, split_queries
from ann_bench.timing import time_search


def fake_search(q, k):
    return np.tile(np.arange(k), (q.shape[0], 1))


@pytest.mark.parametrize("mode", ["single", "batch"])
def test_time_search_shapes_and_repeats(mode):
    q = np.zeros((25, 4), dtype=np.float32)
    tr = time_search(fake_search, q, k=10, mode=mode, repeats=3)
    assert tr.ids.shape == (25, 10)
    assert len(tr.times_s) == 3
    assert tr.qps > 0 and tr.median_s == np.median(tr.times_s)


def test_single_mode_calls_one_query_at_a_time():
    seen = []

    def search(q, k):
        seen.append(q.shape[0])
        return fake_search(q, k)

    time_search(search, np.zeros((7, 2), dtype=np.float32), k=3, mode="single", repeats=1, n_warmup=0)
    assert seen == [1] * 7


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        time_search(fake_search, np.zeros((2, 2), dtype=np.float32), k=1, mode="nope")


def test_normalise_unit_norm_and_zero_row():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    y = normalise(x)
    np.testing.assert_allclose(y[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(y[1], [0.0, 0.0])
    assert y.dtype == np.float32 and y.flags["C_CONTIGUOUS"]


def test_split_is_first_and_last_5000():
    a = np.arange(10000)
    tune, test = split_queries(a)
    assert tune[0] == 0 and tune[-1] == 4999 and test[0] == 5000 and len(test) == 5000


def test_cpu_per_wall_near_one_for_single_thread_busy_loop():
    def busy(q, k):
        x = 0.0
        for i in range(150000):  # long enough for coarse Windows CPU clocks (~15 ms)
            x += i * 0.5
        return fake_search(q, k)

    tr = time_search(busy, np.zeros((20, 2), dtype=np.float32), k=2, mode="single", repeats=2)
    assert 0.5 < tr.cpu_per_wall < 1.5


def test_parse_cpu_list():
    from ann_bench.timing import _parse_cpu_list
    assert _parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
