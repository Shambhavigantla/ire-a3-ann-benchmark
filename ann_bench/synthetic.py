"""Part 4 data generator: a new, translated cluster of SIFT vectors (drift).

    D      = base[:n_db]                          the original database
    R_s    = base[n_db:n_db + n_new] + s*sigma*u  the new cluster (disjoint real vectors, moved)
    D'_s   = D followed by R_s                    ids >= n_db are the new vectors

u is a seeded random unit direction, the same for every shift s, so only the
magnitude changes; sigma is the RMS distance of D from its mean. A translation
keeps every distance inside the new cluster unchanged: the cluster is exactly as
hard to search as real SIFT, and only its position relative to the old data (and to
anything an index learned from it) changes. s = 0 is the control: more SIFT data
from the same distribution.

Queries keep the assignment's tune/test halves and split each into two disjoint
groups: "old" (original-region, untranslated) and "new" (new-region, moved by the
same s*sigma*u as R_s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import N_TUNE


def shift_direction(d: int, seed: int) -> np.ndarray:
    """Seeded uniformly random unit vector in R^d (float64)."""
    u = np.random.default_rng(seed).standard_normal(d)
    return u / np.linalg.norm(u)


def data_spread(x: np.ndarray) -> float:
    """RMS distance of the rows of x from their mean."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))


def translate(x: np.ndarray, shift: float, sigma: float, u: np.ndarray) -> np.ndarray:
    """x + shift*sigma*u, as float32 (shift = 0 returns an unchanged copy)."""
    return (np.asarray(x, dtype=np.float64) + shift * sigma * u).astype(np.float32)


def query_groups(queries: np.ndarray, n_tune: int = N_TUNE) -> dict:
    """Disjoint index ranges {(group, split): slice}: each half split into old / new."""
    n = queries.shape[0]
    h_tune, h_test = n_tune // 2, n_tune + (n - n_tune) // 2
    return {("old", "tune"): slice(0, h_tune), ("new", "tune"): slice(h_tune, n_tune),
            ("old", "test"): slice(n_tune, h_test), ("new", "test"): slice(h_test, n)}


@dataclass
class DriftData:
    db: np.ndarray          # D, (n_db, d) float32
    reserved: np.ndarray    # untranslated vectors the new cluster is made from, (n_new, d)
    queries: np.ndarray     # all queries, untranslated
    sigma: float
    direction: np.ndarray   # u, (d,) float64
    n_tune: int = N_TUNE    # queries [0:n_tune] are the tune half

    def new_vectors(self, shift: float) -> np.ndarray:
        return translate(self.reserved, shift, self.sigma, self.direction)

    def updated_db(self, shift: float) -> np.ndarray:
        """D'_s = D then R_s (new vectors get ids n_db, n_db + 1, ...)."""
        return np.concatenate([self.db, self.new_vectors(shift)])

    def group_queries(self, group: str, split: str, shift: float) -> np.ndarray:
        q = self.queries[query_groups(self.queries, self.n_tune)[(group, split)]]
        return translate(q, shift, self.sigma, self.direction) if group == "new" else q.copy()


def make_drift(base: np.ndarray, queries: np.ndarray, n_db: int, n_new: int, seed: int,
               n_tune: int = N_TUNE) -> DriftData:
    if n_db + n_new > base.shape[0]:
        raise ValueError(f"need {n_db + n_new} base vectors, have {base.shape[0]}")
    db = np.ascontiguousarray(base[:n_db], dtype=np.float32)
    reserved = np.ascontiguousarray(base[n_db:n_db + n_new], dtype=np.float32)
    return DriftData(db=db, reserved=reserved, queries=np.asarray(queries, dtype=np.float32),
                     sigma=data_spread(db), direction=shift_direction(base.shape[1], seed), n_tune=n_tune)
