"""Read the Part 4 (drift) rows back into one structure for tables and figures."""
from __future__ import annotations

import json
import os

from .tables import read_results

INDEX_ORDER = ["HNSW", "IVFFlat", "IVFPQ"]


def load_part4(res_dir: str):
    """-> (states, summary) or (None, None) if Part 4 has not run.

    states[(kind, state, shift)] = {"maintenance_s", "size", "n_base", "mechanism",
        "knobs": {knob: {"old-test": {"recall", "qps", "ndis"}, "old-tune": {"recall"}, ...}}}
    state is baseline (shift None) / incremental / rebuilt.
    """
    path = os.path.join(res_dir, "part4_summary.json")
    rows = read_results(os.path.join(res_dir, "results.csv"), "part4")
    if not rows or not os.path.exists(path):
        return None, None
    with open(path) as f:
        summary = json.load(f)
    states = {}
    for r in rows:
        bp = json.loads(r["build_params"])
        notes = json.loads(r["notes"])
        key = (r["index_type"], bp["state"], bp["shift"])
        st = states.setdefault(key, {"maintenance_s": float(r["build_time_s"]), "size": int(float(r["index_bytes"])),
                                     "n_base": int(r["n_base"]), "mechanism": notes.get("mechanism", {}),
                                     "knobs": {}})
        knob = next(iter(json.loads(r["search_params"]).values()))
        cell = {"recall": float(r["recall_at_k"])}
        if r["qps"] not in ("", None):
            cell.update(qps=float(r["qps"]), ndis=notes.get("distances_per_query"))
        st["knobs"].setdefault(knob, {})[r["query_set"]] = cell
    return states, summary


def at_knob(st: dict, knob: int, group: str):
    """Test numbers of `group` at `knob` (None if that knob was not reached)."""
    return st["knobs"].get(knob, {}).get(f"{group}-test")


def knob_for_target(st: dict, group: str, target: float):
    """Smallest knob whose TUNE recall for `group` reaches target (tune/test protocol)."""
    ok = [k for k, cells in st["knobs"].items()
          if f"{group}-tune" in cells and cells[f"{group}-tune"]["recall"] >= target]
    return min(ok) if ok else None


def shifts(summary) -> list[float]:
    return [float(s) for s in summary["shifts"]]
