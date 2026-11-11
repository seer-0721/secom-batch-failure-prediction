"""滚动时间验证（walk-forward / rolling origin）。

工业产线的数据随时间到达，模型只能在"过去"上训练、在"紧接着的未来"上检验。
单块远期留出会把最远一段产线漂移全部算进误差，既不能反映模型实际使用时
的表现，也无法区分"模型不行"与"产线变了"。

主协议采用 expanding-window walk-forward：第 k 折用前 k 段训练、预测第 k+1 段，
逐段滚动，合并折外预测算指标。对照协议用 sliding-window（只用最近 W 个批次
重训练），检验滑窗能否跟上漂移。

每折内部完整重跑：中位数填补 → 标准化 → 相关性剔除 → 三准则共识排序 → 预测，
所有统计量只来自该折训练段，杜绝跨折泄漏。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from config import SEED
from prepare import Cleaner
from screening import (
    consensus_rank,
    prune_correlated,
    rank_by_abs_t,
    rank_by_forest,
    rank_by_mutual_info,
)

DEFAULT_KS = (20, 50)


def _models(seed: int) -> dict:
    """软投票集成：RF + ExtraTrees + 直方图梯度提升。

    样本只有 1567 批、缺陷仅 104 个，单模型方差很大；三个偏差方向不同的模型
    取平均概率能明显压低方差，这在少数类检测上是常见的低成本改进。
    """
    return {
        "rf": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
        "et": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15, random_state=seed),
    }


def _fit_one_fold(xs: pd.DataFrame, y: np.ndarray, i_tr: np.ndarray, i_te: np.ndarray,
                  ks: tuple[int, ...], seed: int) -> dict:
    """单折：清洗 + 剔除 + 共识排序只做一次，再对每个 k 训三个模型并软投票。"""
    cleaner = Cleaner().fit(xs.iloc[i_tr])
    z_tr, z_te = cleaner.transform(xs.iloc[i_tr]), cleaner.transform(xs.iloc[i_te])
    names = cleaner.feature_names

    keep, _ = prune_correlated(z_tr, names)
    z_tr, z_te = z_tr[:, keep], z_te[:, keep]
    kept_names = [names[j] for j in keep]

    order = consensus_rank([
        rank_by_mutual_info(z_tr, y[i_tr], seed=seed),
        rank_by_forest(z_tr, y[i_tr], seed=seed),
        rank_by_abs_t(z_tr, y[i_tr]),
    ])

    out = {"n_sensors_kept": int(z_tr.shape[1])}
    for k in ks:
        idx = order[: min(k, z_tr.shape[1])]
        probs = []
        for name, model in _models(seed).items():
            model.fit(z_tr[:, idx], y[i_tr])
            probs.append(model.predict_proba(z_te[:, idx])[:, 1])
        out[k] = {
            "p": np.mean(probs, axis=0),
            "sensors": [kept_names[j] for j in idx],
        }
    return out


def run_regime(xs: pd.DataFrame, y: np.ndarray, n_splits: int = 6,
               window: int | None = None, ks: tuple[int, ...] = DEFAULT_KS,
               seed: int = SEED) -> dict:
    """window=None -> expanding；window=W -> 只用最近 W 个批次训练。"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {k: np.full(len(y), np.nan) for k in ks}
    folds: list[dict] = []
    slices: list[list[int]] = []
    freq: dict[int, dict[str, int]] = {k: {} for k in ks}

    for fold, (i_tr, i_te) in enumerate(tscv.split(np.arange(len(y)))):
        if window is not None:
            i_tr = i_tr[-window:]
        res = _fit_one_fold(xs, y, i_tr, i_te, ks, seed=seed + fold)
        slices.append([int(i_te[0]), int(i_te[-1]) + 1])
        row = {
            "fold": fold + 1,
            "train_batches": int(len(i_tr)),
            "test_batches": int(len(i_te)),
            "test_fail": int(y[i_te].sum()),
            "n_sensors_kept": res["n_sensors_kept"],
        }
        for k in ks:
            p = res[k]["p"]
            oof[k][i_te] = p
            row[f"roc_auc_k{k}"] = float(roc_auc_score(y[i_te], p)) if 0 < y[i_te].sum() < len(i_te) else float("nan")
            row[f"pr_auc_k{k}"] = float(average_precision_score(y[i_te], p)) if y[i_te].sum() else float("nan")
            for nm in res[k]["sensors"]:
                freq[k][nm] = freq[k].get(nm, 0) + 1
        folds.append(row)

    per_k = {}
    for k in ks:
        mask = ~np.isnan(oof[k])
        per_k[k] = {
            "pooled_roc_auc": float(roc_auc_score(y[mask], oof[k][mask])),
            "pooled_pr_auc": float(average_precision_score(y[mask], oof[k][mask])),
            "mean_fold_roc_auc": float(np.nanmean([f[f"roc_auc_k{k}"] for f in folds])),
            "mean_fold_pr_auc": float(np.nanmean([f[f"pr_auc_k{k}"] for f in folds])),
            "n_scored": int(mask.sum()),
        }
    return {
        "regime": "expanding" if window is None else f"sliding(W={window})",
        "n_splits": n_splits,
        "folds": folds,
        "fold_slices": slices,
        "per_k": per_k,
        "oof": oof,
        "sensor_frequency": {str(k): freq[k] for k in ks},
    }


def slice_metrics(y: np.ndarray, p: np.ndarray, lo: int, hi: int) -> dict:
    yy, pp = y[lo:hi], p[lo:hi]
    ok = ~np.isnan(pp)
    yy, pp = yy[ok], pp[ok]
    return {
        "n": int(len(yy)),
        "n_fail": int(yy.sum()),
        "roc_auc": float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else float("nan"),
        "pr_auc": float(average_precision_score(yy, pp)) if yy.sum() else float("nan"),
    }
