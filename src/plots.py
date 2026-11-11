"""出图。图中文字用英文，避免容器缺中文字体导致乱码。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from config import FIG_DIR

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = ["#2f6f9f", "#c1553b", "#4b8f6b", "#8a6fb0"]
P = {"exp": "#2f6f9f", "sli": "#4b8f6b", "rand": "#c1553b", "hold": "#8a6fb0"}


def save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def data_overview(ts_all, y_all, fold_slices):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    ax.scatter(ts_all[y_all == 0], np.zeros((y_all == 0).sum()), s=3, c="#b9c6d1", marker="|")
    ax.scatter(ts_all[y_all == 1], np.zeros((y_all == 1).sum()), s=16, c="#c1553b", label="fail batch")
    for i, (lo, hi) in enumerate(fold_slices):
        ax.axvline(ts_all[lo], color="#555", lw=0.8, ls="--" if i else ":")
    ax.set_title("SECOM batches in time order (6 walk-forward test blocks)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    counts = [int((y_all == 0).sum()), int((y_all == 1).sum())]
    bars = ax.bar(["pass (-1)", "fail (+1)"], counts, color=["#b9c6d1", "#c1553b"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n{c / sum(counts):.2%}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(counts) * 1.3)
    ax.set_ylabel("batches")
    ax.set_title("Class imbalance")
    fig.tight_layout()
    return save(fig, "fig1_data_overview.png")


def fold_performance(res_exp, res_sli, k: int):
    folds = [f["fold"] for f in res_exp["folds"]]
    e = [f[f"roc_auc_k{k}"] for f in res_exp["folds"]]
    s = [f[f"roc_auc_k{k}"] for f in res_sli["folds"]]
    x = np.arange(len(folds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - w / 2, e, w, color=P["exp"], label="expanding")
    ax.bar(x + w / 2, s, w, color=P["sli"], label="sliding (W=350)")
    ax.axhline(res_exp["per_k"][k]["pooled_roc_auc"], color=P["exp"], ls="--", lw=1)
    ax.axhline(res_sli["per_k"][k]["pooled_roc_auc"], color=P["sli"], ls="--", lw=1)
    ax.axhline(0.5, color="#888", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in folds])
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Walk-forward ROC-AUC per fold (k={k} sensors); dashed = pooled")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return save(fig, "fig2_walkforward_folds.png")


def roc_pr_curves(y, p, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=PALETTE[0], lw=1.6)
    axes[0].plot([0, 1], [0, 1], ls=":", c="#888", lw=1)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title=f"ROC - {tag}")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, color=PALETTE[1], lw=1.6)
    axes[1].axhline(y.mean(), ls=":", c="#888", lw=1)
    axes[1].text(0.04, y.mean() + 0.02, f"random ({y.mean():.1%})", fontsize=7, color="#666")
    axes[1].set(xlabel="recall", ylabel="precision", title=f"PR - {tag}",
                xlim=(0, 1.02), ylim=(0, 1.02))
    fig.tight_layout()
    return save(fig, "fig3_roc_pr_late.png")


def threshold_tradeoff(y, p, chosen: float):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    beta = 2.0
    den = beta ** 2 * prec + rec
    f2 = np.divide((1 + beta ** 2) * prec * rec, den, out=np.zeros_like(prec), where=den > 0)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(thr, prec, color=PALETTE[0], label="precision")
    ax.plot(thr, rec, color=PALETTE[1], label="recall")
    ax.plot(thr, f2, color=PALETTE[2], label="F2")
    ax.axvline(chosen, color="#333", ls="--", lw=1.1)
    ax.text(chosen, 0.03, f" threshold={chosen:.2f}", fontsize=8)
    ax.set(xlabel="decision threshold", ylabel="score",
           title="Threshold trade-off (late walk-forward blocks)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, "fig4_threshold_tradeoff.png")


def protocol_comparison(stats: dict):
    labels = ["random split\n(leaky)", "far holdout\n(last 30%)",
              "walk-forward\n(expanding)", "walk-forward\n(sliding W=350)"]
    keys = ["rand", "hold", "exp", "sli"]
    def get(d, field):
        return d.get(field, d.get(f"pooled_{field}"))
    auc = [get(stats[k], "roc_auc") for k in keys]
    ap = [get(stats[k], "pr_auc") for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar(x - w / 2, auc, w, color="#5d7fa3", label="ROC-AUC")
    b2 = ax.bar(x + w / 2, ap, w, color="#c1553b", label="PR-AUC")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color="#888", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Evaluation protocol decides the headline number")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    return save(fig, "fig5_protocol_comparison.png")


def sensor_stability(freq: dict[str, int], top: int = 15, n_folds: int = 6):
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:top][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    colors = ["#2f6f9f" if v == n_folds else "#8fb0c9" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, n_folds)
    ax.set_xlabel(f"selected in how many of {n_folds} folds")
    ax.set_title("Sensor stability across walk-forward folds")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return save(fig, "fig6_sensor_stability.png")


def cost_curve(ratios, costs, baseline):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(ratios, costs, marker="o", ms=4, color=PALETTE[1], label="tuned threshold")
    ax.plot(ratios, baseline, ls="--", lw=1, color="#555", label="threshold = 0.5")
    ax.set_xscale("log")
    ax.set(xlabel="cost ratio  C_miss : C_inspect  (log)",
           ylabel="expected cost per batch",
           title="Expected cost vs miss/inspection cost ratio")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save(fig, "fig7_cost_ratio.png")
