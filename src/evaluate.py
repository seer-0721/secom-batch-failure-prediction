"""评估指标与阈值选择。缺陷检测关心的是：漏检（把缺陷判成良品）代价远高于误报。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """ROC-AUC / PR-AUC 与固定 0.5 阈值下的指标。"""
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def pick_threshold(y: np.ndarray, p: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    """在训练集（或折外预测）上选使 F_beta 最大的阈值。

    beta=2 表示召回权重是精确率的 2 倍：产线上漏掉一个缺陷批次的代价，
    远高于把一个良品批次误判为风险批次（后者只是多一道复检）。
    """
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    denom = (beta ** 2) * prec + rec
    fbeta = np.divide((1 + beta ** 2) * prec * rec, denom, out=np.zeros_like(prec), where=denom > 0)
    k = int(np.nanargmax(fbeta))
    return float(thr[k]), float(fbeta[k])


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    """给定阈值下的分类指标。"""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged": int(tp + fp),
        "flag_rate": float((tp + fp) / len(y)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: str = "pr_auc",
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """测试集正例很少（<30），单点估计不可靠，给出自助法置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, len(idx)):
            continue
        if metric == "pr_auc":
            vals.append(average_precision_score(y[idx], p[idx]))
        else:
            vals.append(roc_auc_score(y[idx], p[idx]))
    vals = np.asarray(vals)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def cost_optimal(y: np.ndarray, p: np.ndarray, ratio: float) -> tuple[float, float]:
    """在给定代价比下选最优阈值并返回该阈值在样本上的单位代价。

    cost = (ratio * FN + FP) / n_batches，ratio = 漏检代价 / 单批复检代价。
    """
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    n = len(y)
    pos = int(y.sum())
    # FN = pos*(1-rec), FP = tp*(1/prec - 1)，其中 tp = pos*rec
    fn = pos * (1 - rec)
    tp = pos * rec
    fp = np.divide(tp * (1 - prec), prec, out=np.zeros_like(prec), where=prec > 0)
    cost = (ratio * fn + fp) / n
    k = int(np.argmin(cost))
    return float(thr[k]), float(cost[k])
