"""模型库与交叉验证工具。

所有模型都显式处理类别不平衡（class_weight），不在训练集上做重采样：
SMOTE 这类过采样若在切分前做会泄漏测试集信息，是 SECOM 相关文献里常见的陷阱。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from config import SEED


def make_model(name: str, seed: int = SEED):
    if name == "lr":
        return LogisticRegression(
            C=0.1, solver="liblinear",
            class_weight="balanced", max_iter=5000, random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced",
            early_stopping=True, validation_fraction=0.15,
            random_state=seed,
        )
    raise ValueError(f"未知模型: {name}")


MODEL_NAMES = ["lr", "rf", "et", "hgb"]


def oof_predict(model_name: str, z: np.ndarray, y: np.ndarray,
                n_splits: int = 5, seed: int = SEED) -> np.ndarray:
    """按时间序列交叉验证生成折外概率，用于无泄漏地选模型与阈值。

    训练集本身是时间有序的，因此用 TimeSeriesSplit（只用过去预测未来），
    而不是随机 KFold —— 后者会让模型见到未来批次，抬高交叉验证分数。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (i_tr, i_va) in enumerate(tscv.split(z)):
        if y[i_tr].sum() == 0 or y[i_va].sum() == 0:
            continue
        model = make_model(model_name, seed=seed + fold)
        model.fit(z[i_tr], y[i_tr])
        oof[i_va] = model.predict_proba(z[i_va])[:, 1]
    return oof
