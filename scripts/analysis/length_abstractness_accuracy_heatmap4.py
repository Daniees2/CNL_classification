#!/usr/bin/env python3
"""
4-panel heatmap: excerpt length vs abstractness vs accuracy (all 4 models).

Panel design (one panel per model):
  x-axis: excerpt length bin (token count)
  y-axis: abstractness score bin
  color : accuracy
  text  : sample count in each cell
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from wordfreq import zipf_frequency


VARIANTS = [
    ("baseline", "Baseline"),
    ("finetune", "Fine-tuned (original)"),
    ("finetune_equal", "Fine-tuned + synthetic (uniform)"),
    ("finetune_balance", "Fine-tuned + synthetic (class-balanced)"),
]

TOKEN_RE = re.compile(r"[A-Za-z']+")


def _rc() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
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


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def abstractness_score(text: str) -> float:
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0
    nominal_ct = sum(t.endswith(("tion", "sion", "ment", "ness", "ity", "ship", "ance", "ence")) for t in toks)
    nominal_rate = nominal_ct / max(1.0, n)
    nominal_scaled = min(1.0, nominal_rate / 0.18)
    z = [max(0.0, float(zipf_frequency(t, "en"))) for t in toks]
    z_mean = float(np.mean(z)) if z else 0.0
    z_scaled = max(0.0, min(1.0, (z_mean - 2.0) / 5.0))
    return float(max(0.0, min(1.0, 0.6 * nominal_scaled + 0.4 * z_scaled)))


def main() -> None:
    ap = argparse.ArgumentParser(description="4-panel heatmap: length/abstractness vs accuracy")
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/eval_benchmark"))
    ap.add_argument("--abs-bins", type=int, default=6)
    ap.add_argument("--len-bins", type=int, default=6)
    ap.add_argument(
        "--out-base",
        type=Path,
        default=Path("outputs/eval_benchmark/fig_assumption_length_abstractness_heatmap4"),
    )
    args = ap.parse_args()

    pred_dir = args.pred_dir.resolve()
    abs_bins = max(4, int(args.abs_bins))
    len_bins = max(4, int(args.len_bins))

    # Global length quantile bins so all panels share x mapping.
    all_lengths: List[float] = []
    for key, _ in VARIANTS:
        p = pred_dir / f"predictions_{key}.jsonl"
        if not p.is_file():
            continue
        rows = load_jsonl(p)
        all_lengths.extend(float(len(tokenize(str(r.get("passage", ""))))) for r in rows)
    if not all_lengths:
        raise SystemExit(f"No prediction files found under {pred_dir}")

    q = np.linspace(0, 1, len_bins + 1)
    len_edges = np.quantile(np.array(all_lengths, dtype=float), q)
    len_edges = np.unique(len_edges)  # guard duplicate quantiles
    if len(len_edges) < 3:
        # fallback to linear edges
        lo, hi = float(min(all_lengths)), float(max(all_lengths))
        len_edges = np.linspace(lo, hi + 1e-6, len_bins + 1)
    abs_edges = np.linspace(0.0, 1.0, abs_bins + 1)

    _rc()
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 130 / 25.4), sharex=True, sharey=True)
    axes = axes.ravel()
    im = None

    stats: Dict[str, object] = {"len_edges": [float(x) for x in len_edges], "abs_edges": [float(x) for x in abs_edges], "models": {}}

    for ax, (key, title) in zip(axes, VARIANTS):
        p = pred_dir / f"predictions_{key}.jsonl"
        if not p.is_file():
            ax.set_axis_off()
            stats["models"][key] = {"missing": True}
            continue
        rows = load_jsonl(p)

        lengths = np.array([float(len(tokenize(str(r.get("passage", ""))))) for r in rows], dtype=float)
        abs_s = np.array([abstractness_score(str(r.get("passage", ""))) for r in rows], dtype=float)
        y = np.array([1.0 if bool(r.get("exact_match")) else 0.0 for r in rows], dtype=float)

        xi = np.clip(np.digitize(lengths, len_edges) - 1, 0, len(len_edges) - 2)
        yi = np.clip(np.digitize(abs_s, abs_edges) - 1, 0, abs_bins - 1)

        cnt = np.zeros((len(len_edges) - 1, abs_bins), dtype=int)
        sm = np.zeros((len(len_edges) - 1, abs_bins), dtype=float)
        for i in range(len(rows)):
            bx, by = xi[i], yi[i]
            cnt[bx, by] += 1
            sm[bx, by] += y[i]
        acc = np.full_like(sm, np.nan, dtype=float)
        m = cnt > 0
        acc[m] = sm[m] / cnt[m]

        im = ax.imshow(
            acc.T,
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            extent=(0.0, float(len(len_edges) - 1), 0.0, 1.0),
            aspect="auto",
        )

        # Cell counts
        xstep = 1.0
        ystep = 1.0 / abs_bins
        for i in range(acc.shape[0]):
            for j in range(acc.shape[1]):
                n = int(cnt[i, j])
                if n <= 0:
                    continue
                xc = i + 0.5 * xstep
                yc = (j + 0.5) * ystep
                ax.text(xc, yc, str(n), ha="center", va="center", fontsize=6.2, color="white")

        ax.set_title(title, fontsize=8)
        ax.set_xlim(0, float(len(len_edges) - 1))
        ax.set_ylim(0, 1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        stats["models"][key] = {
            "n_predictions": int(len(rows)),
            "mean_accuracy": float(np.mean(y)),
            "mean_length_tokens": float(np.mean(lengths)),
            "mean_abstractness": float(np.mean(abs_s)),
        }

    # x ticks as token-range labels
    centers = np.arange(len(len_edges) - 1) + 0.5
    labels = [f"{int(round(len_edges[i]))}-{int(round(len_edges[i+1]))}" for i in range(len(len_edges) - 1)]
    for ax in axes:
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=0)

    axes[2].set_xlabel("Excerpt length bin (tokens)")
    axes[3].set_xlabel("Excerpt length bin (tokens)")
    axes[0].set_ylabel("Abstractness score")
    axes[2].set_ylabel("Abstractness score")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("Accuracy")

    out_base = args.out_base.resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    stats_path = out_base.parent / f"{out_base.name}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_base.with_suffix('.png')} and {stats_path}")


if __name__ == "__main__":
    main()

