"""Tests for the Pareto frontier / selection and a FAISS smoke test of each index type."""
import numpy as np
import pytest

from ann_bench.frontier import pareto_frontier, select_at_recall


def run(recall, qps, name=""):
    return {"recall": recall, "qps": qps, "name": name}


def test_frontier_drops_dominated_runs():
    runs = [run(0.80, 1000, "a"), run(0.90, 500, "b"), run(0.85, 400, "dominated"), run(0.99, 100, "c")]
    assert [r["name"] for r in pareto_frontier(runs)] == ["a", "b", "c"]


def test_frontier_equal_recall_keeps_faster():
    runs = [run(0.9, 100, "slow"), run(0.9, 200, "fast")]
    assert [r["name"] for r in pareto_frontier(runs)] == ["fast"]


def test_frontier_sorted_by_recall_and_qps_decreasing():
    rng = np.random.default_rng(0)
    runs = [run(float(r), float(q)) for r, q in zip(rng.random(50), rng.random(50) * 1000)]
    front = pareto_frontier(runs)
    rec = [r["recall"] for r in front]
    qps = [r["qps"] for r in front]
    assert rec == sorted(rec) and qps == sorted(qps, reverse=True)
    for r in runs:  # every run is dominated by or equal to some frontier point
        assert any(f["recall"] >= r["recall"] and f["qps"] >= r["qps"] for f in front)


def test_select_at_recall_picks_fastest_reaching_target():
    runs = [run(0.94, 2000, "miss"), run(0.95, 800, "ok"), run(0.97, 900, "best"), run(0.99, 100, "slow")]
    assert select_at_recall(runs, 0.95)["name"] == "best"
    assert select_at_recall(runs, 0.999) is None


faiss = pytest.importorskip("faiss")
from ann_bench.indexes import IndexSpec, build_index, set_search_param  # noqa: E402
from ann_bench.metrics import recall_at_k  # noqa: E402


@pytest.mark.parametrize("spec", [
    IndexSpec("HNSW", {"M": 16, "efConstruction": 64}),
    IndexSpec("IVFFlat", {"nlist": 16}),
    IndexSpec("IVFPQ", {"nlist": 16, "m": 8}),
])
def test_each_index_builds_and_recall_rises_with_search_param(spec):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5000, 32)).astype(np.float32)
    q = rng.standard_normal((100, 32)).astype(np.float32)
    d = ((q[:, None, :].astype(np.float64) - base[None]) ** 2).sum(-1)
    gt = np.argsort(d, axis=1)[:, :100]
    index, info = build_index(spec, base, seed=0)
    assert index.ntotal == 5000 and info["build_s"] >= 0
    lo, hi = (8, 256) if spec.kind == "HNSW" else (1, 16)
    recalls = []
    for v in (lo, hi):
        set_search_param(index, spec, v)
        recalls.append(recall_at_k(index.search(q, 10)[1], base, q, gt).mean())
    assert recalls[1] >= recalls[0]
    if spec.kind == "IVFFlat":          # nprobe = nlist scans everything -> exact
        assert recalls[1] == 1.0


def test_ivf_build_is_reproducible():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((3000, 16)).astype(np.float32)
    spec = IndexSpec("IVFPQ", {"nlist": 8, "m": 4})
    a, _ = build_index(spec, base, seed=0)
    b, _ = build_index(spec, base, seed=0)
    assert np.array_equal(faiss.serialize_index(a), faiss.serialize_index(b))
