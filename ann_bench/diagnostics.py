"""Mechanism measurements for the drift experiment (Part 4).

They explain *why* recall or speed changes, rather than only showing that it does:
  * ivf_list_stats      how inverted lists fill up, and where the new vectors land
  * pq_error            PQ reconstruction error of a set of vectors
  * hnsw_link_stats     whether new HNSW nodes are linked to each other and to old nodes
  * distances_per_query distance computations per query (FAISS search statistics)
"""
from __future__ import annotations

import faiss
import numpy as np


def _list_ids(invlists, i: int) -> np.ndarray:
    i = int(i)
    n = invlists.list_size(i)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    return faiss.rev_swig_ptr(invlists.get_ids(i), n).copy()


def ivf_list_stats(index, n_old: int) -> dict:
    """List sizes and the placement of ids >= n_old (the new vectors).

    imbalance = nlist * sum(size^2) / ntotal^2: 1.0 for equal lists; it is the expected
    number of vectors scanned per probe relative to balanced lists, if queries probe
    lists in proportion to their size.
    """
    ivf = faiss.extract_index_ivf(index)
    inv = ivf.invlists
    sizes = np.array([inv.list_size(i) for i in range(ivf.nlist)], dtype=np.int64)
    new_per_list = np.zeros(ivf.nlist, dtype=np.int64)
    for i in np.flatnonzero(sizes):
        new_per_list[i] = int((_list_ids(inv, int(i)) >= n_old).sum())
    n_new = int(new_per_list.sum())
    out = {"nlist": int(ivf.nlist), "imbalance": float(inv.imbalance_factor()),
           "max_list": int(sizes.max()), "mean_list": float(sizes.mean()),
           "empty_lists": int((sizes == 0).sum()), "n_new": n_new}
    if n_new:
        srt = np.sort(new_per_list)[::-1]
        out["lists_with_new"] = int((new_per_list > 0).sum())
        out["lists_holding_90pct_new"] = int(np.searchsorted(np.cumsum(srt), 0.9 * n_new) + 1)
        out["new_share_of_largest_list"] = float(new_per_list[sizes.argmax()] / sizes.max())
    return out


def pq_error(index, x: np.ndarray) -> float:
    """Mean squared L2 error of encoding then decoding x (coarse centroid + PQ residual)."""
    rec = index.sa_decode(index.sa_encode(np.ascontiguousarray(x, dtype=np.float32)))
    return float(((rec.astype(np.float64) - x) ** 2).sum(axis=1).mean())


def hnsw_level0_links(index) -> np.ndarray:
    """(ntotal, 2M) level-0 neighbour ids, -1 = unused slot."""
    h = index.hnsw
    nb = faiss.vector_to_array(h.neighbors)
    off = faiss.vector_to_array(h.offsets).astype(np.int64)
    width = int(faiss.vector_to_array(h.cum_nneighbor_per_level)[1])
    return nb[off[:-1, None] + np.arange(width)[None, :]]


def hnsw_link_stats(index, n_old: int) -> dict:
    """Level-0 link structure between old (< n_old) and new (>= n_old) nodes."""
    links = hnsw_level0_links(index)
    valid = links >= 0
    is_new_target = valid & (links >= n_old)
    out = {"mean_degree": float(valid.sum(axis=1).mean())}
    n = links.shape[0]
    if n > n_old:
        new_rows = slice(n_old, n)
        out["new_links_to_new"] = float(is_new_target[new_rows].sum() / max(1, valid[new_rows].sum()))
        old_rows = slice(0, n_old)
        out["old_nodes_linking_new"] = float(is_new_target[old_rows].any(axis=1).mean())
        out["old_to_new_links"] = int(is_new_target[old_rows].sum())
        out["new_to_old_links"] = int((valid[new_rows] & ~is_new_target[new_rows]).sum())
    return out


def distances_per_query(index, kind: str, queries: np.ndarray, k: int) -> float:
    """Distance computations per query in one batch search (codes scanned for IVF-PQ)."""
    stats = faiss.cvar.hnsw_stats if kind == "HNSW" else faiss.cvar.indexIVF_stats
    stats.reset()
    index.search(queries, k)
    return float(stats.ndis) / queries.shape[0]


def copy_index(index):
    """Independent deep copy (serialise + deserialise works for every index type)."""
    return faiss.deserialize_index(faiss.serialize_index(index))
