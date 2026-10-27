"""Query timing: one-query-at-a-time vs one batch, repeated, median taken.

`search` is any callable (queries[nq, d], k) -> ids[nq, k], so the same code
times any FAISS index.

Two machine effects that silently corrupt timings, both handled here:
* FAISS's batch path calls OpenBLAS, which has its own thread pool and ignores
  faiss.omp_set_num_threads. `limit_threads` caps OpenMP AND BLAS together.
* Hybrid CPUs (Intel 12th gen+) mix fast P-cores and slower E-cores; the OS can
  move a single-threaded run between them mid-measurement. `pinned_to` with
  `performance_cores()` keeps timed runs on P-cores.
* After a multi-threaded FAISS call, idle OpenMP workers keep spin-waiting for
  ~0.1-0.2 s. They burn CPU on the same cores and slowed a following 1-thread
  timing by ~24% in a test. `time_search` therefore sleeps `settle_s` first.
Each TimingResult records CPU time / wall time, so a run that used more
threads than intended is visible in results.csv.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np

SearchFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class TimingResult:
    mode: str                 # "single" or "batch"
    n_queries: int            # distinct queries
    times_s: list[float]      # wall time of each repeat (`loops` passes over the query set)
    cpu_s: list[float]        # process CPU time of each repeat (all threads)
    ids: np.ndarray           # results from the last pass, (n_queries, k)
    loops: int = 1            # passes per repeat, so each repeat lasts >= min_repeat_s

    @property
    def median_s(self) -> float:
        return float(np.median(self.times_s))

    @property
    def qps(self) -> float:
        return self.n_queries * self.loops / self.median_s

    @property
    def latency_ms(self) -> float:
        """Mean time per query implied by the median run."""
        return 1000.0 / self.qps

    @property
    def cpu_per_wall(self) -> float:
        """~1.0 for a truly single-threaded run; ~n for n busy threads."""
        return float(np.sum(self.cpu_s) / np.sum(self.times_s))


def _run_single(search: SearchFn, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for i in range(queries.shape[0]):
        out[i] = search(queries[i:i + 1], k)[0]
    return out


def _run_batch(search: SearchFn, queries: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(search(queries, k), dtype=np.int64)


def time_search(search: SearchFn, queries: np.ndarray, k: int, mode: str = "single",
                repeats: int = 3, n_warmup: int = 50, settle_s: float = 0.25,
                min_repeat_s: float = 0.0) -> TimingResult:
    """Time `search` over all `queries`, `repeats` times; report the median.

    A short warm-up run first pulls code and index pages into cache so the
    first repeat is not penalised. Its duration also estimates how many passes
    over the query set make one repeat last >= `min_repeat_s`: fast indexes
    otherwise finish a repeat in a few ms, below timer and CPU-clock resolution.
    """
    if mode == "single":
        runner = _run_single
    elif mode == "batch":
        runner = _run_batch
    else:
        raise ValueError(f"mode must be 'single' or 'batch', got {mode!r}")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if settle_s > 0:        # let spinning OpenMP workers from earlier parallel calls go idle
        time.sleep(settle_s)
    nq = queries.shape[0]
    loops = 1
    if n_warmup > 0:
        n_w = min(n_warmup, nq)
        runner(search, queries[:n_w], k)        # cold: first-call and cache-miss costs
        t0 = time.perf_counter()                # warm: estimates the per-pass time
        runner(search, queries[:n_w], k)
        est_pass = (time.perf_counter() - t0) * nq / n_w
        if min_repeat_s > 0 and est_pass > 0:
            loops = max(1, int(np.ceil(min_repeat_s / est_pass)))

    times, cpus, ids = [], [], None
    for _ in range(repeats):
        c0, t0 = time.process_time(), time.perf_counter()
        for _ in range(loops):
            ids = runner(search, queries, k)
        times.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return TimingResult(mode=mode, n_queries=nq, times_s=times, cpu_s=cpus, ids=ids, loops=loops)


@contextmanager
def limit_threads(n: int):
    """Cap FAISS's OpenMP threads AND the BLAS thread pool FAISS uses for batches."""
    import faiss
    from threadpoolctl import threadpool_limits

    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(int(n))
    try:
        with threadpool_limits(limits=int(n)):
            yield
    finally:
        faiss.omp_set_num_threads(old)


# Backwards-compatible name.
faiss_threads = limit_threads


def _parse_cpu_list(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _windows_cpu_efficiency() -> dict[int, int]:
    """Logical CPU index -> EfficiencyClass (higher = faster core), via GetSystemCpuSetInformation."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel32.GetSystemCpuSetInformation
    fn.argtypes = [ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                   wintypes.HANDLE, wintypes.ULONG]
    fn.restype = wintypes.BOOL
    needed = wintypes.ULONG(0)
    fn(None, 0, ctypes.byref(needed), None, 0)
    buf = (ctypes.c_ubyte * needed.value)()
    if not fn(buf, needed, ctypes.byref(needed), None, 0):
        return {}
    raw, out, off = bytes(buf), {}, 0
    while off < needed.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        if size == 0:
            break
        group = int.from_bytes(raw[off + 12:off + 14], "little")
        logical, eff = raw[off + 14], raw[off + 18]
        if group == 0:
            out[logical] = eff
        off += size
    return out


def performance_cores() -> list[int] | None:
    """Logical CPUs on the fastest core type, or None if the CPU is not hybrid / unknown."""
    try:
        if sys.platform == "win32":
            eff = _windows_cpu_efficiency()
            if eff and len(set(eff.values())) > 1:
                best = max(eff.values())
                return sorted(c for c, e in eff.items() if e == best)
        elif os.path.exists("/sys/devices/cpu_core/cpus"):     # Linux, Intel hybrid
            with open("/sys/devices/cpu_core/cpus") as f:
                return _parse_cpu_list(f.read())
    except Exception:
        pass
    return None


@contextmanager
def pinned_to(cpus: list[int] | None):
    """Restrict this process to the given logical CPUs (no-op for None)."""
    if not cpus:
        yield
        return
    import psutil

    proc = psutil.Process()
    old = proc.cpu_affinity()
    proc.cpu_affinity(list(cpus))
    try:
        yield
    finally:
        proc.cpu_affinity(old)
