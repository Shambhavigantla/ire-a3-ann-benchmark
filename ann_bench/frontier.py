"""Best trade-off (Pareto) curves and recall-target selection.

A run is a dict with at least 'recall' and 'qps'. A run is on the frontier if
no other run is at least as accurate AND strictly faster.

Selection follows the tune/test protocol: choose on TUNE runs, report the
chosen configuration's TEST numbers (which may land slightly below target -
that is reported honestly, not hidden).
"""
from __future__ import annotations

import math


def pareto_frontier(runs: list[dict]) -> list[dict]:
    """Non-dominated runs, sorted by increasing recall (so QPS decreases along the curve)."""
    ordered = sorted(runs, key=lambda r: (-r["recall"], -r["qps"]))
    front, best_qps = [], -math.inf
    for r in ordered:
        if r["qps"] > best_qps:
            front.append(r)
            best_qps = r["qps"]
    return front[::-1]


def select_at_recall(runs: list[dict], target: float) -> dict | None:
    """Fastest run whose recall reaches `target`; None if no run does."""
    ok = [r for r in runs if r["recall"] >= target]
    return max(ok, key=lambda r: r["qps"]) if ok else None


SHORT = {"efSearch": "ef", "efConstruction": "efC"}


def settings_label(build: dict, search: dict) -> str:
    """'M=16, ef=128' / 'nlist=4096, m=32, nprobe=64' (efConstruction omitted: fixed at 200)."""
    parts = [f"{SHORT.get(k, k)}={v}" for k, v in sorted(build.items()) if k != "efConstruction"]
    parts += [f"{SHORT.get(k, k)}={v}" for k, v in search.items()]
    return ", ".join(parts)


def group_configs(rows: list[dict], experiment: str = "part2") -> dict:
    """results.csv rows -> {(dataset, index_type): [config, ...]}.

    A config is one (build params, search params) pair with its 'tune' and 'test'
    numbers side by side, plus build cost.
    """
    import json

    configs = {}
    for r in rows:
        if r["experiment"] != experiment:
            continue
        key = (r["dataset"], r["index_type"], r["build_params"], r["search_params"])
        c = configs.setdefault(key, {
            "dataset": r["dataset"], "index": r["index_type"],
            "build": json.loads(r["build_params"]), "search": json.loads(r["search_params"]),
            "build_time_s": float(r["build_time_s"]), "index_bytes": int(float(r["index_bytes"])),
            "bytes_per_vector": float(r["bytes_per_vector"]),
            "notes": json.loads(r["notes"]) if r.get("notes") else {},
        })
        c[r["query_set"]] = {"recall": float(r["recall_at_k"]), "qps": float(r["qps"]),
                             "latency_ms": float(r["latency_ms"])}
    out = {}
    for c in configs.values():
        if "tune" in c and "test" in c:
            c["label"] = settings_label(c["build"], c["search"])
            out.setdefault((c["dataset"], c["index"]), []).append(c)
    return out


def select_config(configs: list[dict], target: float) -> dict | None:
    """Tune/test protocol: fastest config whose TUNE recall reaches target."""
    runs = [{"recall": c["tune"]["recall"], "qps": c["tune"]["qps"], "cfg": c} for c in configs]
    best = select_at_recall(runs, target)
    return best["cfg"] if best else None
