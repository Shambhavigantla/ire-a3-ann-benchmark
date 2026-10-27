"""Building the three FAISS ANN indexes and setting their search-time knob.

    HNSW     IndexHNSWFlat(d, M), efConstruction      search knob: efSearch
    IVFFlat  IVF{nlist},Flat                          search knob: nprobe
    IVFPQ    IVF{nlist},PQ{m}x8 (residual encoding)   search knob: nprobe

IVF k-means and PQ codebooks are trained on a seeded random sample of the base
vectors, so builds are reproducible. (HNSW's multi-threaded insertion order is
not bit-reproducible, which shifts recall only in the 3rd-4th decimal.)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import faiss
import numpy as np

SEARCH_PARAM = {"HNSW": "efSearch", "IVFFlat": "nprobe", "IVFPQ": "nprobe"}


@dataclass(frozen=True)
class IndexSpec:
    kind: str                      # HNSW | IVFFlat | IVFPQ
    params: dict = field(hash=False)

    def __post_init__(self):
        if self.kind not in SEARCH_PARAM:
            raise ValueError(f"unknown index kind {self.kind!r}")

    @property
    def search_param(self) -> str:
        return SEARCH_PARAM[self.kind]

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    @property
    def label(self) -> str:
        return f"{self.kind}(" + ", ".join(f"{k}={v}" for k, v in sorted(self.params.items())) + ")"


def ivf_train_size(n: int, nlist: int) -> int:
    """Training sample for k-means: >= 50 points per centroid (FAISS warns below 39), >= 100K, <= n."""
    return int(min(n, max(100_000, 50 * nlist)))


def build_index(spec: IndexSpec, base: np.ndarray, seed: int = 0):
    """Build and fill the index. Returns (index, info) with train/add timings.

    Thread count is whatever the caller set (see timing.limit_threads).
    """
    d = base.shape[1]
    info = {"train_s": 0.0, "train_n": 0}
    if spec.kind == "HNSW":
        index = faiss.IndexHNSWFlat(d, int(spec.params["M"]))
        index.hnsw.efConstruction = int(spec.params["efConstruction"])
    else:
        nlist = int(spec.params["nlist"])
        if spec.kind == "IVFFlat":
            index = faiss.index_factory(d, f"IVF{nlist},Flat")
        else:
            m = int(spec.params["m"])
            if d % m:
                raise ValueError(f"PQ m={m} must divide d={d}")
            index = faiss.index_factory(d, f"IVF{nlist},PQ{m}x8")
        ivf = faiss.downcast_index(faiss.extract_index_ivf(index))
        ivf.cp.seed = seed
        if spec.kind == "IVFPQ":
            ivf.pq.cp.seed = seed
            # index_factory enables polysemous training, which only serves polysemous
            # filtering (polysemous_ht > 0) - never used here - and costs 10-25% of train time.
            ivf.do_polysemous_training = False
        n_train = ivf_train_size(base.shape[0], nlist)
        rng = np.random.default_rng(seed)
        sample = base[np.sort(rng.choice(base.shape[0], n_train, replace=False))]
        t0 = time.perf_counter()
        index.train(sample)
        info.update(train_s=time.perf_counter() - t0, train_n=n_train)

    t0 = time.perf_counter()
    index.add(base)
    info["add_s"] = time.perf_counter() - t0
    info["build_s"] = info["train_s"] + info["add_s"]
    return index, info


def set_search_param(index, spec: IndexSpec, value: int) -> None:
    if spec.kind == "HNSW":
        index.hnsw.efSearch = int(value)
    else:
        faiss.extract_index_ivf(index).nprobe = int(value)


def add_exact_rerank(index, base: np.ndarray, depth: int, k: int):
    """Wrap an index so each search re-ranks its top `depth` candidates by exact L2.

    IndexRefineFlat keeps its own float32 copy of `base` (that memory is part of the
    index's cost). A search for k results asks `index` for k * k_factor candidates
    and returns their k closest by exact distance. Search knobs (nprobe) are still
    set on the wrapped index, and set_search_param reaches it through the wrapper.
    """
    if depth % k:
        raise ValueError(f"re-rank depth {depth} must be a multiple of k={k}")
    refine = faiss.IndexRefineFlat(index, faiss.swig_ptr(np.ascontiguousarray(base, dtype=np.float32)))
    refine.k_factor = float(depth // k)
    return refine


def coarse_centroids(index) -> np.ndarray:
    """The IVF coarse centroids (nlist, d) of an IVF index, possibly wrapped."""
    ivf = faiss.extract_index_ivf(index)
    return ivf.quantizer.reconstruct_n(0, ivf.nlist)
