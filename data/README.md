# 数据目录

原始数据不随仓库分发（`.gitignore` 排除了 `data/raw/` 与 `data/processed/`），
用以下命令获取：

```bash
python data/download.py    # 从 UCI 下载并解压到 data/raw/
python src/prepare.py      # 清洗 + 时间切分，产出 data/processed/secom_clean.npz
```

- `download.py`：下载 `https://archive.ics.uci.edu/static/public/179/secom.zip`，
  解压出 `secom.data`（1567 × 590，空格分隔，缺失为 `NaN`）与
  `secom_labels.data`（`-1` 良品 / `+1` 缺陷品，附时间戳）。
- `prepare.py`：丢弃缺失率 > 50% 的 24 列与常量列 122 列，中位数填补 + 标准化
  （统计量只用训练段估计），产出 445 维特征矩阵与按时间切分的索引。

数据规模小（原始约 5.4 MB），之所以不入库是为了保持仓库轻量、且遵循
"数据与代码分离"的惯例。
