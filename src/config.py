"""全局配置：路径、随机种子、切分比例。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"

SEED = 20260415
TEST_FRAC = 0.30          # 按时间切分：后 30% 作为测试集
MISSING_DROP = 0.50       # 缺失率高于此值的传感器整列丢弃
CORR_PRUNE = 0.95         # 相关性高于此值的冗余传感器剔除
