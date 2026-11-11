"""下载并解压 UCI SECOM 数据集到 data/raw/。

用法:
    python data/download.py
数据来源: https://archive.ics.uci.edu/dataset/179/secom
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
RAW = Path(__file__).resolve().parent / "raw"
EXPECT = ["secom.data", "secom_labels.data"]


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in EXPECT):
        print(f"[skip] 原始数据已存在: {RAW}")
        return 0
    print(f"[get ] {UCI_URL}")
    with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            target = RAW / Path(name).name
            if target.suffix in {".data", ".names"}:
                target.write_bytes(zf.read(name))
                print(f"[unzip] {target.name}  {target.stat().st_size / 1e6:.2f} MB")
    missing = [f for f in EXPECT if not (RAW / f).exists()]
    if missing:
        print(f"[error] 缺少文件: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
