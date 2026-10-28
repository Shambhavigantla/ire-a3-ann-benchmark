"""Tie-aware recall@k.

Definition used throughout the project
--------------------------------------
For query q let d_k be the (squared Euclidean) distance from q to its k-th true
nearest neighbour, taken from the ground-truth ids. A retrieved id counts as a
hit if its true distance to q is <= d_k (up to a tiny tolerance). Then

    recall@k(q) = (# distinct, valid retrieved ids among the first k that are hits) / k

Why distances and not id intersection: when several base vectors are exactly as
far from q as the k-th neighbour, the ground truth had to pick one of them
arbitrarily. An index that returns a different one of the tied vectors is not
wrong, but plain set intersection would score it as a miss. SIFT vectors are
integer valued, so exact ties are common there.

Two further details matter:
* Distances are recomputed from the raw vectors in float64, never taken from
  the index. Compressed indexes (IVF-PQ) return *approximate* distances, which
  would otherwise let a far-away vector sneak under the threshold.
* d_k is the max of the recomputed distances of the first k ground-truth ids,
  so a slightly mis-ordered ground-truth file cannot make the threshold too low.

This module only depends on numpy so the unit tests run without FAISS.
"""
from __future__ import annotations

import numpy as np

# Relative slack on the tie threshold. Distances are recomputed in float64 from
# float32 inputs, so genuine ties (e.g. duplicate vectors, integer SIFT data)
# come out bit-identical; the slack only absorbs last-ulp rounding noise.
DEFAULT_RTOL = 1e-6


def squared_l2_to_ids(base: np.ndarray, queries: np.ndarray, ids: np.ndarray,
                      max_chunk_elems: int = 1 << 23) -> np.ndarray:
    """Exact squared L2 distance from queries[i] to base[ids[i, j]], in float64.

    Ids that are negative (FAISS pads with -1) or out of range get +inf, so
    they can never count as hits.
    """
    base = np.asarray(base)
    queries = np.asarray(queries)
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[0] != queries.shape[0]:
        raise ValueError(f"ids must be (n_queries, m); got {ids.shape} for {queries.shape[0]} queries")

    nq, m = ids.shape
    out = np.full((nq, m), np.inf, dtype=np.float64)
    if m == 0:
        return out
    valid = (ids >= 0) & (ids < base.shape[0])

    # Gather in chunks so a (nq, 100, d) float64 tensor never gets too large.
    chunk = max(1, max_chunk_elems // (m * base.shape[1]))
    for s in range(0, nq, chunk):
        e = min(s + chunk, nq)
        v = valid[s:e]
        x = base[np.where(v, ids[s:e], 0)].astype(np.float64)        # (c, m, d)
        diff = x - queries[s:e, None, :].astype(np.float64)
        d = np.einsum("cmd,cmd->cm", diff, diff)
        out[s:e] = np.where(v, d, np.inf)
    return out


def first_occurrence_mask(ids: np.ndarray) -> np.ndarray:
    """Boolean mask that is False for repeated ids within a row (first copy stays True)."""
    ids = np.asarray(ids)
    order = np.argsort(ids, axis=1, kind="stable")
    s = np.take_along_axis(ids, order, axis=1)
    first_sorted = np.ones(s.shape, dtype=bool)
    first_sorted[:, 1:] = s[:, 1:] != s[:, :-1]
    mask = np.empty_like(first_sorted)
    np.put_along_axis(mask, order, first_sorted, axis=1)
    return mask


def recall_from_distances(retrieved_ids: np.ndarray, retrieved_dists: np.ndarray,
                          kth_dist: np.ndarray, k: int = 10,
                          rtol: float = DEFAULT_RTOL, atol: float = 0.0) -> np.ndarray:
    """Per-query tie-aware recall@k given TRUE distances of the retrieved ids.

    retrieved_ids   (nq, >=0) ids returned by the index (-1 = no result)
    retrieved_dists (nq, same) exact distances of those ids to the query
    kth_dist        (nq,) exact distance of the k-th true neighbour
    Only the first k columns are scored; fewer than k columns count as misses.
    """
    ids = np.asarray(retrieved_ids)[:, :k]
    d = np.asarray(retrieved_dists, dtype=np.float64)[:, :k]
    kth = np.asarray(kth_dist, dtype=np.float64)
    if ids.shape != d.shape or kth.shape != (ids.shape[0],):
        raise ValueError("shape mismatch between ids, distances and kth_dist")

    threshold = kth + rtol * np.abs(kth) + atol
    hits = (ids >= 0) & first_occurrence_mask(ids) & (d <= threshold[:, None])
    return hits.sum(axis=1) / float(k)


def recall_at_k(retrieved_ids: np.ndarray, base: np.ndarray, queries: np.ndarray,
                gt_ids: np.ndarray, k: int = 10,
                rtol: float = DEFAULT_RTOL, atol: float = 0.0) -> np.ndarray:
    """Per-query tie-aware recall@k (see module docstring). Take .mean() for the dataset value."""
    gt_ids = np.asarray(gt_ids)
    retrieved_ids = np.asarray(retrieved_ids)
    if gt_ids.shape[1] < k:
        raise ValueError(f"ground truth has only {gt_ids.shape[1]} neighbours, need k={k}")
    if retrieved_ids.shape[0] != gt_ids.shape[0]:
        raise ValueError("retrieved_ids and gt_ids have different numbers of queries")

    gt_d = squared_l2_to_ids(base, queries, gt_ids[:, :k])
    if not np.isfinite(gt_d).all():
        raise ValueError("ground truth contains invalid ids")
    kth = gt_d.max(axis=1)

    ret = retrieved_ids[:, :k]
    ret_d = squared_l2_to_ids(base, queries, ret)
    return recall_from_distances(ret, ret_d, kth, k=k, rtol=rtol, atol=atol)


def strict_recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, k: int = 10) -> np.ndarray:
    """Plain id-intersection recall@k with NO tie handling. Reported only for comparison."""
    ret = np.asarray(retrieved_ids)[:, :k]
    gt = np.asarray(gt_ids)[:, :k]
    # |set(valid retrieved) & set(gt)|: each distinct valid retrieved id counts once if in gt
    keep = (ret >= 0) & first_occurrence_mask(ret)
    in_gt = (ret[:, :, None] == gt[:, None, :]).any(axis=2)
    return (keep & in_gt).sum(axis=1) / float(k)


def coverage_at_k(candidate_ids: np.ndarray, base: np.ndarray, queries: np.ndarray,
                  gt_ids: np.ndarray, k: int = 10,
                  rtol: float = DEFAULT_RTOL, atol: float = 0.0) -> np.ndarray:
    """Per-query tie-aware share of the true top k found ANYWHERE in a candidate list.

    candidate_ids may have any number of columns (e.g. the 100 candidates that are
    re-ranked). Same hit rule as recall_at_k; hits are capped at k. Exact re-ranking
    of the candidates returns their k closest, so its recall@k equals this value -
    re-ranking can reorder candidates but never add one.
    """
    gt_ids = np.asarray(gt_ids)
    cand = np.asarray(candidate_ids)
    if cand.shape[0] != gt_ids.shape[0]:
        raise ValueError("candidate_ids and gt_ids have different numbers of queries")
    gt_d = squared_l2_to_ids(base, queries, gt_ids[:, :k])
    if not np.isfinite(gt_d).all():
        raise ValueError("ground truth contains invalid ids")
    kth = gt_d.max(axis=1)
    d = squared_l2_to_ids(base, queries, cand)
    threshold = kth + rtol * np.abs(kth) + atol
    hits = (cand >= 0) & first_occurrence_mask(cand) & (d <= threshold[:, None])
    return np.minimum(hits.sum(axis=1), k) / float(k)
