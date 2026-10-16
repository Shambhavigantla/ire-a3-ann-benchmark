"""Write report tables as Markdown (for reading) and LaTeX (for the report)."""
from __future__ import annotations

import csv
import os


def read_results(path: str, experiment: str | None = None) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if experiment is None or r["experiment"] == experiment]


def fmt(x, digits: int = 3) -> str:
    """Format numbers compactly; pass strings through; None -> en dash."""
    if x is None or x == "":
        return "–"
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return f"{x:,}"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    return f"{x:.{digits}g}" if ax < 0.001 and ax > 0 else f"{x:.{digits}f}"


def _latex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("–", "--"), ("×", r"$\times$"), ("≥", r"$\geq$"), ("≤", r"$\leq$")):
        s = s.replace(a, b)
    return s


def write_table(rows: list[list], header: list[str], name: str, table_dir: str,
                caption: str = "", align: str | None = None) -> list[str]:
    """rows are lists of already-formatted strings. Writes <name>.md and <name>.tex."""
    os.makedirs(table_dir, exist_ok=True)
    rows = [[str(c) for c in r] for r in rows]

    md = []
    if caption:
        md += [f"**{caption}**", ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md_path = os.path.join(table_dir, f"{name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    align = align or ("l" * 2 + "r" * (len(header) - 2))
    tex = [r"\begin{table}[t]", r"\centering", r"\small"]
    if caption:
        tex.append(r"\caption{" + _latex_escape(caption) + "}")
    tex += [r"\label{tab:" + name + "}", r"\begin{tabular}{" + align + "}", r"\toprule",
            " & ".join(_latex_escape(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join(_latex_escape(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = os.path.join(table_dir, f"{name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    return [md_path, tex_path]
