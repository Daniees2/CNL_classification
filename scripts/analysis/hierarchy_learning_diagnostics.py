#!/usr/bin/env python3
"""
Hierarchy-aware diagnostics for hierarchical classification performance.

Outputs:
  - fig_assumption_error_locality_models.(png|pdf)
  - fig_diagnostic_error_vs_support_models.(png|pdf)
  - companion *_stats.json files
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
            "axes.grid": False,
        }
    )


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def parse_key(path: Path) -> List[dict]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[dict] = []
    for ln in lines[1:]:
        cid, cat, n = ln.split("\t")
        out.append({"C": cid, "category": cat, "n": int(n)})
    return out


def norm_label(s: str) -> str:
    return " ".join(s.strip().lower().split())


def split_display_path(s: str) -> Tuple[str, ...]:
    return tuple(norm_label(p) for p in s.split(">") if p.strip())


def shared_prefix_depth(pred: str, gold: str) -> int:
    pp = split_display_path(pred)
    gg = split_display_path(gold)
    d = 0
    for a, b in zip(pp, gg):
        if a != b:
            break
        d += 1
    return d


def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return p, p
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = (z / denom) * float(np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return lo, hi


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def locality_breakdown(rows: List[dict]) -> Dict[str, float]:
    total = 0
    total_wrong = 0
    counts = {
        "wrong_same_component": 0,  # shared prefix depth >=2
        "wrong_same_domain": 0,  # shared prefix depth ==1
        "wrong_other_domain": 0,  # shared prefix depth ==0
    }

    for r in rows:
        g = str(r.get("gold", "")).strip()
        p = str(r.get("pred_matched", "")).strip()
        if not g:
            continue
        total += 1
        if p == g:
            continue
        total_wrong += 1
        d = shared_prefix_depth(p, g) if p else 0
        if d >= 2:
            counts["wrong_same_component"] += 1
        elif d == 1:
            counts["wrong_same_domain"] += 1
        else:
            counts["wrong_other_domain"] += 1

    frac = {k: (counts[k] / total_wrong if total_wrong else 0.0) for k in counts}
    return {
        "n_total": total,
        "n_wrong_total": total_wrong,
        **counts,
        **{f"frac_{k}": v for k, v in frac.items()},
    }


def make_error_locality_plot(
    rows_by_model: Dict[str, List[dict]],
    model_display: Dict[str, str],
    out_base: Path,
) -> Dict[str, object]:
    keys = [k for k in ("finetune", "finetune_equal", "finetune_balance") if k in rows_by_model]
    if not keys:
        raise SystemExit("No model rows available for locality plot.")

    labels = ["Wrong: same component", "Wrong: same domain", "Wrong: other domain"]
    parts = ["wrong_same_component", "wrong_same_domain", "wrong_other_domain"]
    colors = ["#2B6CB0", "#B7791F", "#DC2626"]

    stats: Dict[str, Dict[str, float]] = {k: locality_breakdown(rows_by_model[k]) for k in keys}

    _rc()
    fig, ax = plt.subplots(figsize=(170 / 25.4, 62 / 25.4))
    y = np.arange(len(keys), dtype=float)
    left = np.zeros(len(keys), dtype=float)

    for part, lab, col in zip(parts, labels, colors):
        vals = np.array([float(stats[k].get(f"frac_{part}", 0.0)) for k in keys], dtype=float)
        ax.barh(
            y,
            vals,
            left=left,
            height=0.62,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            label=lab,
            zorder=2,
        )
        for i, v in enumerate(vals):
            if v >= 0.07:
                ax.text(left[i] + v / 2.0, y[i], f"{100*v:.0f}%", va="center", ha="center", color="white", fontsize=7)
        left += vals

    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("Fraction of incorrect predictions")
    ax.set_yticks(y)
    ax.set_yticklabels([model_display.get(k, k) for k in keys])
    ax.set_title("When wrong, where does the misclassification land?", fontsize=8.8)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, columnspacing=0.85, handletextpad=0.35)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {"models": {k: stats[k] for k in keys}}


def per_class_error(rows: List[dict]) -> Tuple[Dict[str, int], Dict[str, float]]:
    n_by: Dict[str, int] = defaultdict(int)
    e_by: Dict[str, int] = defaultdict(int)
    for r in rows:
        g = str(r.get("gold", "")).strip()
        if not g:
            continue
        n_by[g] += 1
        e_by[g] += 0 if bool(r.get("exact_match")) else 1

    err = {c: (float(e_by[c] / n_by[c]) if n_by[c] else 0.0) for c in n_by}
    return dict(n_by), err


def make_error_support_plot(
    rows_by_model: Dict[str, List[dict]],
    key_rows: List[dict],
    model_display: Dict[str, str],
    out_base: Path,
) -> Dict[str, object]:
    model_keys = [k for k in ("finetune", "finetune_equal", "finetune_balance") if k in rows_by_model]
    if "finetune" not in model_keys:
        raise SystemExit("Need predictions_finetune.jsonl for support diagnostic.")

    cid_by_cat = {r["category"]: r["C"] for r in key_rows}
    support_by_cat = {r["category"]: int(r["n"]) for r in key_rows}
    cats = [r["category"] for r in key_rows if support_by_cat.get(r["category"], 0) > 0]

    n_ref, err_ref = per_class_error(rows_by_model["finetune"])
    cats = [c for c in cats if c in n_ref]
    support = np.array([float(n_ref[c]) for c in cats], dtype=float)
    x = np.log10(np.maximum(support, 1.0))
    xf = np.linspace(float(np.min(x)), float(np.max(x)), 120)

    model_colors = {
        "finetune": "#2F855A",
        "finetune_equal": "#B7791F",
        "finetune_balance": "#6B46C1",
    }

    # Per-model error arrays
    err_by_model: Dict[str, np.ndarray] = {}
    trend_by_model: Dict[str, Dict[str, float]] = {}
    for k in model_keys:
        _, err_k = per_class_error(rows_by_model[k])
        y = np.array([float(err_k.get(c, 0.0)) for c in cats], dtype=float)
        err_by_model[k] = y
        slope, intercept = np.polyfit(x, y, deg=1)
        trend_by_model[k] = {
            "spearman_log_support_vs_error": float(spearman_rho(x, y)),
            "trend_slope_error_vs_log_support": float(slope),
            "trend_intercept": float(intercept),
            "mean_error_rate": float(np.mean(y)),
        }

    # Improvement against FT
    delta_equal = err_by_model["finetune"] - err_by_model.get("finetune_equal", err_by_model["finetune"])
    delta_balance = err_by_model["finetune"] - err_by_model.get("finetune_balance", err_by_model["finetune"])

    _rc()
    fig, axes = plt.subplots(1, 2, figsize=(240 / 25.4, 82 / 25.4), sharex=True)

    # Panel A: raw error vs support
    ax = axes[0]
    jitter = {"finetune": -0.006, "finetune_equal": 0.0, "finetune_balance": 0.006}
    for k in model_keys:
        y = err_by_model[k]
        ax.scatter(
            x + jitter.get(k, 0.0),
            y,
            s=26,
            alpha=0.88,
            color=model_colors.get(k, "#334155"),
            edgecolor="white",
            linewidth=0.45,
            label=model_display.get(k, k),
            zorder=3,
        )
        sl = trend_by_model[k]["trend_slope_error_vs_log_support"]
        ic = trend_by_model[k]["trend_intercept"]
        ax.plot(xf, sl * xf + ic, color=model_colors.get(k, "#334155"), linewidth=1.2, alpha=0.92, zorder=2)

    ax.set_xlabel("log10(class support n)")
    ax.set_ylabel("Class error rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-class error vs support", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.35, labelspacing=0.22)

    # Panel B: improvement vs support (positive = improvement)
    ax2 = axes[1]
    if "finetune_equal" in model_keys:
        sl_e, ic_e = np.polyfit(x, delta_equal, deg=1)
        ax2.scatter(
            x,
            delta_equal,
            s=26,
            alpha=0.9,
            color=model_colors["finetune_equal"],
            edgecolor="white",
            linewidth=0.45,
            label="FT+Synthetic (uniform) - FT",
            zorder=3,
        )
        ax2.plot(xf, sl_e * xf + ic_e, color=model_colors["finetune_equal"], linewidth=1.2, zorder=2)
    else:
        sl_e, ic_e = 0.0, 0.0

    if "finetune_balance" in model_keys:
        sl_b, ic_b = np.polyfit(x, delta_balance, deg=1)
        ax2.scatter(
            x + 0.006,
            delta_balance,
            s=26,
            alpha=0.9,
            color=model_colors["finetune_balance"],
            edgecolor="white",
            linewidth=0.45,
            label="FT+Synthetic (balanced) - FT",
            zorder=3,
        )
        ax2.plot(xf, sl_b * xf + ic_b, color=model_colors["finetune_balance"], linewidth=1.2, zorder=2)
    else:
        sl_b, ic_b = 0.0, 0.0

    ax2.axhline(0.0, color="#9CA3AF", linestyle="--", linewidth=1.0, zorder=1)
    ax2.set_xlabel("log10(class support n)")
    ax2.set_ylabel("Error-rate improvement vs FT")
    ax2.set_title("Do added data help low-support classes?", fontsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(False)
    ax2.legend(frameon=False, loc="lower right", handletextpad=0.35, labelspacing=0.22)

    # annotate a few most-improved labels (balanced)
    if "finetune_balance" in model_keys:
        idx = np.argsort(delta_balance)[-3:]
        for i in idx:
            cid = cid_by_cat.get(cats[i], "")
            ax2.text(x[i] + 0.008, delta_balance[i], cid, fontsize=6.7, color="#475569", va="center")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.13, wspace=0.22)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "n_categories": int(len(cats)),
        "model_trends": trend_by_model,
        "delta_trends": {
            "uniform_minus_ft_slope": float(sl_e),
            "uniform_minus_ft_intercept": float(ic_e),
            "balanced_minus_ft_slope": float(sl_b),
            "balanced_minus_ft_intercept": float(ic_b),
        },
        "rows": [
            {
                "category": c,
                "category_id": cid_by_cat.get(c, ""),
                "n": int(n_ref.get(c, 0)),
                "error_ft": float(err_by_model["finetune"][i]),
                "error_uniform": float(err_by_model["finetune_equal"][i]) if "finetune_equal" in model_keys else None,
                "error_balanced": float(err_by_model["finetune_balance"][i]) if "finetune_balance" in model_keys else None,
                "delta_uniform_minus_ft": float(delta_equal[i]) if "finetune_equal" in model_keys else None,
                "delta_balanced_minus_ft": float(delta_balance[i]) if "finetune_balance" in model_keys else None,
            }
            for i, c in enumerate(cats)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Hierarchy + learning diagnostics across fine-tuned models.")
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/eval_benchmark"))
    args = ap.parse_args()

    pred_dir = args.pred_dir.resolve()
    key_path = pred_dir / "category_id_key.tsv"
    if not key_path.is_file():
        raise SystemExit(f"Missing {key_path}")

    model_display = {
        "finetune": "Fine-tuned (original)",
        "finetune_equal": "Fine-tuned + synthetic (uniform)",
        "finetune_balance": "Fine-tuned + synthetic (balanced)",
    }
    rows_by_model: Dict[str, List[dict]] = {}
    for k in ("finetune", "finetune_equal", "finetune_balance"):
        p = pred_dir / f"predictions_{k}.jsonl"
        if p.is_file():
            rows_by_model[k] = load_jsonl(p)
    if "finetune" not in rows_by_model:
        raise SystemExit(f"Missing {pred_dir / 'predictions_finetune.jsonl'}")

    key_rows = parse_key(key_path)

    locality_stats = make_error_locality_plot(rows_by_model, model_display, pred_dir / "fig_assumption_error_locality_models")
    support_stats = make_error_support_plot(
        rows_by_model,
        key_rows,
        model_display,
        pred_dir / "fig_diagnostic_error_vs_support_models",
    )

    (pred_dir / "fig_assumption_error_locality_models_stats.json").write_text(
        json.dumps(locality_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (pred_dir / "fig_diagnostic_error_vs_support_models_stats.json").write_text(
        json.dumps(support_stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {pred_dir / 'fig_assumption_error_locality_models.png'} and "
        f"{pred_dir / 'fig_diagnostic_error_vs_support_models.png'}"
    )


if __name__ == "__main__":
    main()

