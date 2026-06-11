#!/usr/bin/env python3
"""
Alternative publication-grade category accuracy visualizations.

Outputs:
  - fig_category_accuracy_heatmap.(png|pdf)
  - fig_category_accuracy_faceted_forest.(png|pdf)
  - fig_category_accuracy_compact_compare.(png|pdf)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _rc() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "axes.grid": False,
        }
    )


VARIANTS: List[Tuple[str, str, str]] = [
    ("baseline", "Baseline", "#2B6CB0"),
    ("finetune", "Fine-tuned (original)", "#2F855A"),
    ("finetune_equal", "Fine-tuned + synthetic (uniform)", "#B7791F"),
    ("finetune_balance", "Fine-tuned + synthetic (class-balanced)", "#6B46C1"),
]


def load_predictions(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * float(np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return lo, hi


def aggregate(rows: List[dict]) -> Dict[str, dict]:
    d: Dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        g = str(r.get("gold", "")).strip()
        if not g:
            continue
        d[g]["n"] += 1
        d[g]["correct"] += int(bool(r.get("exact_match")))
    out: Dict[str, dict] = {}
    for g, x in d.items():
        n = int(x["n"])
        c = int(x["correct"])
        p = (c / n) if n else 0.0
        lo, hi = wilson(p, n)
        out[g] = {"n": n, "correct": c, "acc": p, "ci_lo": lo, "ci_hi": hi}
    return out


def parse_key(path: Path) -> List[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[dict] = []
    for ln in lines[1:]:
        cid, cat, n = ln.split("\t")
        out.append({"C": cid, "category": cat, "n": int(n)})
    return out


def load_all(pred_dir: Path) -> Tuple[List[dict], Dict[str, Dict[str, dict]]]:
    key_path = pred_dir / "category_id_key.tsv"
    if not key_path.is_file():
        raise SystemExit(f"Missing {key_path}. Run eval.plot_category_accuracy first.")
    key_rows = parse_key(key_path)

    stats_by_variant: Dict[str, Dict[str, dict]] = {}
    for key, _, _ in VARIANTS:
        p = pred_dir / f"predictions_{key}.jsonl"
        if p.is_file():
            stats_by_variant[key] = aggregate(load_predictions(p))
    return key_rows, stats_by_variant


def make_matrix(key_rows: List[dict], stats_by_variant: Dict[str, Dict[str, dict]]) -> np.ndarray:
    mat = np.zeros((len(key_rows), len(VARIANTS)), dtype=float)
    for i, r in enumerate(key_rows):
        cat = r["category"]
        for j, (k, _, _) in enumerate(VARIANTS):
            mat[i, j] = float(stats_by_variant.get(k, {}).get(cat, {"acc": 0.0})["acc"])
    return mat


def plot_heatmap(key_rows: List[dict], mat: np.ndarray, out_base: Path) -> None:
    _rc()
    fig_w_in = 170 / 25.4
    fig_h_in = max(4.2, 0.30 * len(key_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    im = ax.imshow(mat, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_yticks(np.arange(len(key_rows)))
    ax.set_yticklabels([f"{r['C']} (n={r['n']})" for r in key_rows])
    ax.set_xticks(np.arange(len(VARIANTS)))
    ax.set_xticklabels([lbl for _, lbl, _ in VARIANTS])
    ax.set_xlabel("Model variant")
    ax.set_ylabel("Category")

    # Subtle cell values
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt_color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6.5, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy")
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.94, top=0.97, bottom=0.11)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_faceted_forest(key_rows: List[dict], stats_by_variant: Dict[str, Dict[str, dict]], out_base: Path) -> None:
    _rc()
    fig_w_in = 230 / 25.4
    fig_h_in = max(4.4, 0.30 * len(key_rows) + 1.3)
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(fig_w_in, fig_h_in), sharey=True)

    y = np.arange(len(key_rows), dtype=float)
    for ax, (k, lbl, color) in zip(axes, VARIANTS):
        stats = stats_by_variant.get(k, {})
        acc = []
        lo = []
        hi = []
        for r in key_rows:
            rec = stats.get(r["category"], {"acc": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
            a = float(rec["acc"])
            acc.append(a)
            lo.append(a - float(rec["ci_lo"]))
            hi.append(float(rec["ci_hi"]) - a)

        ax.errorbar(
            acc,
            y,
            xerr=np.array([lo, hi]),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=3.6,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            linestyle="none",
            zorder=3,
        )
        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel(lbl)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r['C']} (n={r['n']})" for r in key_rows])
    axes[0].set_ylabel("Category")
    axes[0].invert_yaxis()

    fig.supxlabel("Category accuracy")
    fig.subplots_adjust(left=0.28, right=0.99, top=0.98, bottom=0.12, wspace=0.12)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_compact_compare(key_rows: List[dict], mat: np.ndarray, out_base: Path) -> None:
    """
    Single-panel comparison designed for easier cross-model reading than facets,
    while staying cleaner than full CI overlap plots.
    """
    _rc()
    fig_w_in = 320 / 25.4
    fig_h_in = 5.3
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    x = np.arange(len(key_rows), dtype=float)
    group_w = 0.80
    bar_w = group_w / len(VARIANTS)
    offsets = np.linspace(
        -(group_w - bar_w) / 2.0,
        (group_w - bar_w) / 2.0,
        len(VARIANTS),
    )

    # Keep model order exactly as declared in VARIANTS.
    for j, (_, lbl, color) in enumerate(VARIANTS):
        ax.bar(
            x + offsets[j],
            mat[:, j],
            width=bar_w * 0.92,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            label=lbl,
            zorder=3,
        )

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Category accuracy", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['C']} (n={r['n']})" for r in key_rows])
    ax.set_xlabel("Category", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fontsize=9,
        handlelength=1.1,
        handletextpad=0.4,
        labelspacing=0.2,
        columnspacing=0.8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.22)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Alternative category accuracy visualizations.")
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("outputs/eval_benchmark"),
        help="Directory containing category_id_key.tsv and predictions_*.jsonl",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: pred-dir)")
    args = ap.parse_args()

    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    key_rows, stats_by_variant = load_all(pred_dir)
    mat = make_matrix(key_rows, stats_by_variant)

    plot_heatmap(key_rows, mat, out_dir / "fig_category_accuracy_heatmap")
    plot_faceted_forest(key_rows, stats_by_variant, out_dir / "fig_category_accuracy_faceted_forest")
    plot_compact_compare(key_rows, mat, out_dir / "fig_category_accuracy_compact_compare")

    print(
        f"Wrote {out_dir / 'fig_category_accuracy_heatmap.png'}, "
        f"{out_dir / 'fig_category_accuracy_faceted_forest.png'}, and "
        f"{out_dir / 'fig_category_accuracy_compact_compare.png'}"
    )


if __name__ == "__main__":
    main()

