"""SECOM 数据读取与清洗。

SECOM 是半导体晶圆制造产线的批次级过程数据：1567 个批次、590 个传感器通道，
标签 -1 表示良品、1 表示缺陷品。原始数据存在大量缺失与常量列，且批次按时间
顺序排列，因此本研究按时间切分训练/测试集，避免用未来数据训练。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MISSING_DROP, PROC_DIR, RAW_DIR, SEED, TEST_FRAC


def load_raw() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """读取原始 secom.data / secom_labels.data。"""
    xs = pd.read_csv(
        RAW_DIR / "secom.data",
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        dtype=np.float64,
    )
    xs.columns = [f"s{i:03d}" for i in range(xs.shape[1])]

    lab = pd.read_csv(
        RAW_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "ts"],
        quotechar='"',
    )
    y = (lab["label"].astype(int) == 1).to_numpy(dtype=np.int64)
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    return xs, y, ts


def time_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """按时间顺序切分，返回训练/测试下标。"""
    cut = int(round(n * (1.0 - TEST_FRAC)))
    return np.arange(cut), np.arange(cut, n)


class Cleaner:
    """列筛选 + 中位数填补 + 标准化，所有统计量只在训练集上估计。"""

    def __init__(self, missing_drop: float = MISSING_DROP, add_missing_count: bool = True):
        self.missing_drop = missing_drop
        self.add_missing_count = add_missing_count
        self.keep_cols_: list[str] = []
        self.medians_: pd.Series | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.dropped_: dict[str, list[str]] = {}

    def fit(self, xs: pd.DataFrame) -> "Cleaner":
        n = len(xs)
        miss = xs.isna().mean()
        high_missing = miss[miss > self.missing_drop].index.tolist()
        rest = xs.drop(columns=high_missing)

        nunique = rest.nunique(dropna=True)
        constant = nunique[nunique <= 1].index.tolist()
        rest = rest.drop(columns=constant)

        self.keep_cols_ = rest.columns.tolist()
        self.medians_ = rest.median(numeric_only=True)
        filled = rest.fillna(self.medians_)
        self.mean_ = filled.mean().to_numpy(dtype=np.float64).copy()
        self.std_ = filled.std(ddof=0).to_numpy(dtype=np.float64).copy()
        self.std_[self.std_ < 1e-12] = 1.0
        self.dropped_ = {
            "high_missing": high_missing,
            "constant": constant,
        }
        assert len(self.keep_cols_) + len(high_missing) + len(constant) == xs.shape[1]
        return self

    def transform(self, xs: pd.DataFrame) -> np.ndarray:
        sub = xs[self.keep_cols_]
        miss_count = sub.isna().sum(axis=1).to_numpy(dtype=np.float64)[:, None]
        filled = sub.fillna(self.medians_).to_numpy(dtype=np.float64)
        z = (filled - self.mean_) / self.std_
        if self.add_missing_count:
            z = np.hstack([z, (miss_count - miss_count.mean()) / max(miss_count.std(ddof=0), 1e-12)])
        return z

    def fit_transform(self, xs: pd.DataFrame) -> np.ndarray:
        return self.fit(xs).transform(xs)

    @property
    def feature_names(self) -> list[str]:
        names = list(self.keep_cols_)
        if self.add_missing_count:
            names.append("n_missing")
        return names


def main() -> None:
    xs, y, ts = load_raw()
    assert ts.is_monotonic_increasing, "SECOM 批次未按时间排序，时间切分协议不成立"

    tr, te = time_split(len(xs))
    cleaner = Cleaner().fit(xs.iloc[tr])
    z_tr = cleaner.transform(xs.iloc[tr])
    z_te = cleaner.transform(xs.iloc[te])

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROC_DIR / "secom_clean.npz",
        z_tr=z_tr,
        z_te=z_te,
        y_tr=y[tr],
        y_te=y[te],
        tr_idx=tr,
        te_idx=te,
        names=np.array(cleaner.feature_names),
    )

    print(f"原始: {xs.shape[0]} 批次 x {xs.shape[1]} 传感器 | 缺陷 {y.sum()} ({y.mean():.2%})")
    print(f"丢弃高缺失列 (> {MISSING_DROP:.0%}): {len(cleaner.dropped_['high_missing'])} 个")
    print(f"丢弃常量列: {len(cleaner.dropped_['constant'])} 个")
    print(f"保留特征: {z_tr.shape[1]}  (含缺失计数特征 1 个)")
    print(f"时间切分: 训练 {len(tr)} 批次 / 缺陷 {y[tr].sum()} | 测试 {len(te)} 批次 / 缺陷 {y[te].sum()}")
    print(f"时间范围: 训练 {ts.iloc[tr[0]]:%Y-%m-%d} ~ {ts.iloc[tr[-1]]:%Y-%m-%d}"
          f" | 测试 {ts.iloc[te[0]]:%Y-%m-%d} ~ {ts.iloc[te[-1]]:%Y-%m-%d}")
    print(f"整体缺失率: {xs.isna().to_numpy().mean():.2%}")
    print(f"缺陷率: 训练 {y[tr].mean():.2%} / 测试 {y[te].mean():.2%}")
    print(f"[saved] {PROC_DIR / 'secom_clean.npz'}")


if __name__ == "__main__":
    main()
