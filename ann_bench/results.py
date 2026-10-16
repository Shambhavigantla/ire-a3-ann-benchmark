"""results/results.csv bookkeeping and machine description."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone

COLUMNS = [
    "experiment",        # e.g. part1_exact, part2, part3, part4
    "dataset",
    "n_base",
    "index_type",        # Flat, HNSW, IVFFlat, IVFPQ, ...
    "build_params",      # JSON string
    "search_params",     # JSON string
    "query_set",         # tune / test
    "n_queries",
    "mode",              # single / batch
    "threads",
    "k",
    "recall_at_k",       # tie-aware, mean over queries
    "recall_strict",     # id intersection, for comparison
    "qps",
    "latency_ms",
    "time_median_s",
    "time_runs_s",       # JSON list of each repeat
    "cpu_per_wall",      # process CPU time / wall time: ~1 = single-threaded
    "build_time_s",
    "index_bytes",
    "bytes_per_vector",
    "notes",
    "timestamp",
]
KEY_COLUMNS = ("experiment", "dataset", "n_base")


def replace_rows(path: str, rows: list[dict], key_columns=KEY_COLUMNS) -> None:
    """Write rows, first dropping existing rows with the same key, so re-runs don't duplicate."""
    for r in rows:
        unknown = set(r) - set(COLUMNS)
        if unknown:
            raise KeyError(f"unknown result columns: {sorted(unknown)}")

    def key(r):
        return tuple(str(r.get(c, "")) for c in key_columns)

    new_keys = {key(r) for r in rows}
    kept = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            kept = [r for r in csv.DictReader(f) if key(r) not in new_keys]

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in kept:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
        for r in rows:
            r = {"timestamp": stamp, **r}
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def drop_rows(path: str, **match) -> int:
    """Delete rows whose columns equal all of `match` (compared as strings). Returns count."""
    if not os.path.exists(path):
        return 0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if not all(str(r.get(k, "")) == str(v) for k, v in match.items())]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in keep:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return len(rows) - len(keep)


def machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                info["processor"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
    try:
        from .timing import performance_cores
        info["performance_cores"] = performance_cores()
    except Exception:
        info["performance_cores"] = None
    for mod in ("numpy", "faiss", "h5py", "threadpoolctl", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            info[mod] = None
    return info


def write_machine_info(path: str) -> dict:
    info = machine_info()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return info
