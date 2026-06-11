#!/usr/bin/env python3
"""
Compute and plot validation metrics at hierarchy levels:
  Domain (depth 1), Component (depth 2), Item (depth 3)

Metrics per level:
  - Accuracy
  - Macro Precision
  - Macro Recall
  - Macro F1

Uses predictions_*.jsonl produced by eval.benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
            "axes.grid": False,
        }
    )


VARIANTS: List[Tuple[str, str, str]] = [
    ("baseline", "Baseline", "#2B6CB0"),
    ("finetune", "Fine-tuned", "#2F855A"),
    ("finetune_equal", "FT+Synthetic (uniform)", "#B7791F"),
    ("finetune_balance", "FT+Synthetic (balanced)", "#6B46C1"),
]

DEPTHS: List[Tuple[int, str]] = [(1, "Domain"), (2, "Component"), (3, "Item")]
METRICS = ("accuracy", "precision", "recall", "f1")


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def split_display_path(path: str) -> Tuple[str, ...]:
    return tuple(p.strip() for p in path.split(">") if p.strip())


def label_at_depth(path: str, depth: int) -> str:
    parts = split_display_path(path)
    if len(parts) < depth:
        return ""
    return parts[depth - 1]


def metrics_for_depth(rows: List[dict], depth: int) -> Dict[str, float]:
    gold = []
    pred = []
    for r in rows:
        g = str(r.get("gold", "")).strip()
        p = str(r.get("pred_matched", "")).strip()
        gd = label_at_depth(g, depth)
        pd = label_at_depth(p, depth)
        if not gd:
            continue
        gold.append(gd)
        pred.append(pd)

    if not gold:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    acc = float(accuracy_score(gold, pred))
    pr, rc, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="macro",
        zero_division=0,
    )
    return {
        "n": len(gold),
        "accuracy": acc,
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
    }


def compute_all(pred_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, _, _ in VARIANTS:
        p = pred_dir / f"predictions_{key}.jsonl"
        if not p.is_file():
            continue
        rows = load_jsonl(p)
        out[key] = {}
        for depth, depth_name in DEPTHS:
            out[key][depth_name] = metrics_for_depth(rows, depth)
    return out


def plot_all_models(results: Dict[str, Dict[str, Dict[str, float]]], out_base: Path) -> None:
    _rc()
    fig, axes = plt.subplots(1, 4, figsize=(250 / 25.4, 72 / 25.4), sharex=True, sharey=True)
    x = np.arange(len(DEPTHS), dtype=float)
    depth_names = [d for _, d in DEPTHS]
    markers = ["o", "s", "D", "^"]

    for ax, metric in zip(axes, METRICS):
        for i, (key, label, color) in enumerate(VARIANTS):
            if key not in results:
                continue
            y = np.array([float(results[key][d][metric]) for d in depth_names], dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                marker=markers[i % len(markers)],
                markersize=4.2,
                linewidth=1.3,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=label,
                zorder=3,
            )
        ax.set_title(metric.capitalize(), fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(depth_names)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    axes[0].set_ylabel("Score")
    fig.supxlabel("Hierarchy level")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.78, bottom=0.19, wspace=0.18)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_ft_validation(results: Dict[str, Dict[str, Dict[str, float]]], out_base: Path) -> None:
    if "finetune" not in results:
        return
    _rc()
    fig, ax = plt.subplots(figsize=(145 / 25.4, 64 / 25.4))
    depth_names = [d for _, d in DEPTHS]
    x = np.arange(len(depth_names), dtype=float)
    w = 0.18
    colors = {
        "accuracy": "#2F855A",
        "precision": "#2B6CB0",
        "recall": "#B7791F",
        "f1": "#6B46C1",
    }
    offsets = {
        "accuracy": -1.5 * w,
        "precision": -0.5 * w,
        "recall": 0.5 * w,
        "f1": 1.5 * w,
    }

    for m in METRICS:
        y = np.array([float(results["finetune"][d][m]) for d in depth_names], dtype=float)
        ax.bar(
            x + offsets[m],
            y,
            width=w,
            color=colors[m],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            label=m.capitalize(),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(depth_names)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Score")
    ax.set_xlabel("Hierarchy level")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="lower right", ncol=2, handletextpad=0.35, columnspacing=0.8, labelspacing=0.2)

    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.15)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation metrics plots across hierarchy levels.")
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/eval_benchmark"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compute_all(pred_dir)
    if not results:
        raise SystemExit(f"No predictions_*.jsonl found in {pred_dir}")

    (out_dir / "validation_metrics_by_depth.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    plot_all_models(results, out_dir / "fig_validation_metrics_all_models")
    plot_ft_validation(results, out_dir / "fig_validation_metrics_finetune")

    print(
        f"Wrote {out_dir / 'fig_validation_metrics_all_models.png'} and "
        f"{out_dir / 'fig_validation_metrics_finetune.png'}"
    )


if __name__ == "__main__":
    main()

