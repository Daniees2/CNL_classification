#!/usr/bin/env python3
"""
Publication-style figure: hierarchical accuracy at depths 1–3 for each benchmark variant.

Reads ``summary.json`` from ``python -m eval.benchmark`` (``hierarchy_acc_by_depth``).

Example:

  python -m eval.plot_nature_benchmark --summary outputs/eval_benchmark/summary.json
  python -m eval.plot_nature_benchmark --summary outputs/eval_benchmark/summary.json --out outputs/eval_benchmark/fig_accuracy_depth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _nature_rc() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "axes.grid": False,
        }
    )


# Muted, print-safe palette with strong contrast in grayscale
_COLORS = ["#2B6CB0", "#2F855A", "#B7791F", "#6B46C1"]


def _short_legend_label(variant: Dict[str, Any]) -> str:
    key = str(variant.get("key", ""))
    return {
        "baseline": "Baseline",
        "finetune": "Fine-tuned (original)",
        "finetune_equal": "Fine-tuned + synthetic (uniform)",
        "finetune_balance": "Fine-tuned + synthetic (class-balanced)",
    }.get(key, str(variant.get("display_name", key)))


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_depth(display_path: str) -> int:
    return len([p for p in display_path.split(">") if p.strip()])


def _count_eligible_by_depth(summary: Dict[str, Any], out_base: Path, depths: tuple[int, ...]) -> Dict[int, int]:
    """
    Compute denominator n per depth from saved prediction rows (gold labels).
    Falls back to n_rows_evaluated when prediction files are unavailable.
    """
    pred_file = out_base.parent / "predictions_baseline.jsonl"
    if pred_file.is_file():
        lines = [ln for ln in pred_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = [json.loads(ln) for ln in lines]
        return {d: sum(1 for r in rows if _split_depth(str(r.get("gold", ""))) >= d) for d in depths}
    n = int(summary.get("n_rows_evaluated", 0) or 0)
    return {d: n for d in depths}


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        return p, p
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * float(np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return lo, hi


def plot_from_summary(
    summary: Dict[str, Any],
    out_base: Path,
    fig_width_mm: float = 89,
) -> None:
    variants: List[Dict[str, Any]] = [v for v in summary.get("variants", []) if not v.get("skipped")]

    depths = (1, 2, 3)
    n_groups = len(depths)
    n_series = len(variants)
    if n_series == 0:
        raise SystemExit("No non-skipped variants in summary.")

    values: List[List[float]] = []
    labels: List[str] = []
    for v in variants:
        labels.append(_short_legend_label(v))
        by_d = (v.get("metrics") or {}).get("hierarchy_acc_by_depth") or {}
        row = [float(by_d.get(str(d), 0.0)) for d in depths]
        values.append(row)

    values_arr = np.array(values, dtype=float)

    _nature_rc()
    fig_w_in = fig_width_mm / 25.4
    fig_h_in = fig_w_in * 0.64
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), layout="constrained")

    x = np.arange(n_groups, dtype=float)
    n_by_depth = _count_eligible_by_depth(summary, out_base, depths)
    markers = ["o", "s", "D", "^"]
    yerr_lo_all = np.zeros_like(values_arr)
    yerr_hi_all = np.zeros_like(values_arr)
    for i in range(n_series):
        y = values_arr[i]
        yerr_lo = []
        yerr_hi = []
        for j, d in enumerate(depths):
            lo, hi = _wilson_interval(float(y[j]), int(n_by_depth[d]))
            yerr_lo.append(float(y[j] - lo))
            yerr_hi.append(float(hi - y[j]))
            yerr_lo_all[i, j] = yerr_lo[-1]
            yerr_hi_all[i, j] = yerr_hi[-1]

        ax.plot(
            x,
            y,
            color=_COLORS[i % len(_COLORS)],
            linewidth=1.15,
            marker=markers[i % len(markers)],
            markersize=4.2,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=labels[i],
            zorder=4,
        )
        ax.errorbar(
            x,
            y,
            yerr=np.array([yerr_lo, yerr_hi]),
            fmt="none",
            ecolor=_COLORS[i % len(_COLORS)],
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("Hierarchy depth")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0.0, 0.71, 0.1))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.15,
        columnspacing=0.7,
        fontsize=6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="x", rotation=0)
    ax.set_axisbelow(False)

    out_base = out_base.resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    plot_dot_whisker(
        depths=depths,
        labels=labels,
        values_arr=values_arr,
        yerr_lo_all=yerr_lo_all,
        yerr_hi_all=yerr_hi_all,
        out_base=out_base.parent / f"{out_base.name}_dotwhisker",
        fig_width_mm=fig_width_mm,
    )


def plot_dot_whisker(
    depths: tuple[int, ...],
    labels: List[str],
    values_arr: np.ndarray,
    yerr_lo_all: np.ndarray,
    yerr_hi_all: np.ndarray,
    out_base: Path,
    fig_width_mm: float,
) -> None:
    """Dot-and-whisker variant: no connecting lines, only points + CIs."""
    _nature_rc()
    n_series = values_arr.shape[0]
    fig_w_in = fig_width_mm / 25.4
    fig_h_in = fig_w_in * 0.64
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), layout="constrained")

    x = np.arange(len(depths), dtype=float)
    offsets = np.linspace(-0.16, 0.16, n_series) if n_series > 1 else np.array([0.0])
    markers = ["o", "s", "D", "^"]

    for i in range(n_series):
        xi = x + offsets[i]
        yi = values_arr[i]
        ax.errorbar(
            xi,
            yi,
            yerr=np.array([yerr_lo_all[i], yerr_hi_all[i]]),
            fmt=markers[i % len(markers)],
            color=_COLORS[i % len(_COLORS)],
            ecolor=_COLORS[i % len(_COLORS)],
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            linestyle="none",
            label=labels[i],
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel("Hierarchy depth")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0.0, 0.71, 0.1))
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        borderaxespad=0,
        handlelength=1.0,
        handletextpad=0.35,
        labelspacing=0.15,
        columnspacing=0.7,
        fontsize=6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="x", rotation=0)
    ax.set_axisbelow(False)

    out_base = out_base.resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".png"), format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Nature-style depth accuracy plot from eval summary.json")
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/eval_benchmark/summary.json"),
        help="Path to summary.json from eval.benchmark",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path without suffix (writes .pdf and .png). Default: next to summary as fig_accuracy_hierarchy",
    )
    ap.add_argument(
        "--width-mm",
        type=float,
        default=89,
        help="Figure width in mm (Nature single column ~89 mm)",
    )
    args = ap.parse_args()

    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    out = args.out
    if out is None:
        out = summary_path.parent / "fig_accuracy_hierarchy"
    else:
        out = out.resolve()

    summary = load_summary(summary_path)
    plot_from_summary(summary, out, fig_width_mm=args.width_mm)
    print(f"Wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
