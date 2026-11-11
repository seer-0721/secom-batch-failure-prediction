"""高维传感器特征筛选：冗余剔除 + 单变量排序 + 嵌入式重要性。"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from config import CORR_PRUNE, SEED


def prune_correlated(z: np.ndarray, names: list[str], thr: float = CORR_PRUNE) -> tuple[list[int], int]:
    """剔除高度线性相关的冗余传感器，保留组内第一个（列序靠前者）。

    产线上同一工位的多个传感器常常高度共线，保留全部会稀释特征重要性、
    并让系数不稳定。这里做贪心剔除：按列序扫描，若与已保留列 |r| > thr 则丢弃。
    """
    if z.shape[1] <= 1:
        return list(range(z.shape[1])), 0
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    keep: list[int] = []
    for j in range(z.shape[1]):
        if all(corr[j, k] <= thr for k in keep):
            keep.append(j)
    return keep, z.shape[1] - len(keep)


def rank_by_mutual_info(z: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """互信息排序（离散化后计算，适合非线性单调关系）。"""
    mi = mutual_info_classif(z, y, discrete_features=False, random_state=seed)
    return np.argsort(-mi)


def rank_by_forest(z: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """随机森林不纯度重要性排序，能捕捉交互与非线性。"""
    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(z, y)
    return np.argsort(-rf.feature_importances_)


def rank_by_abs_t(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """两样本 t 统计量绝对值排序，作为线性视角的对照。"""
    a, b = z[y == 0], z[y == 1]
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    se = np.sqrt(va / len(a) + vb / len(b))
    se[se < 1e-12] = 1e-12
    t = np.abs((b.mean(axis=0) - a.mean(axis=0)) / se)
    return np.argsort(-t)


def consensus_rank(rankings: list[np.ndarray]) -> np.ndarray:
    """多种准则的平均名次（Borda 计数），降低单一准则的偏置。"""
    n = len(rankings[0])
    acc = np.zeros(n, dtype=np.float64)
    for r in rankings:
        pos = np.empty(n, dtype=np.float64)
        pos[r] = np.arange(n)
        acc += pos
    return np.argsort(acc)
