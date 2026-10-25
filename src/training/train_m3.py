"""
train_m3.py - M3: con bao nhieu ngay nua mua lu BAT DAU / KET THUC.

Dat theo ngay chu khong theo nam: moi ngay trong mua la mot mau,
nho vay co hang nghin mau thay vi chi 10 mau nhu cach dat theo nam.

Baseline: trung binh lich su theo thang tu tap TRAIN.
Ket qua: 05_Ket_qua/M3/  |  Model: 04_Model/M3_muaLu/batdau|ketthuc/
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_splits, tao_thu_muc  # noqa: E402
from pipeline import train_hoi_quy  # noqa: E402
from train_m2 import tao_baseline_climatology  # noqa: E402


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    data = load_splits(cfg)

    model_root = os.path.join(cfg["duong_dan"]["model_dir"], "M3_muaLu")
    kq_dir = os.path.join(cfg["duong_dan"]["ket_qua_dir"], "M3")
    tao_thu_muc(model_root, kq_dir)

    cau_hinh = [
        ("ngay_den_batdau", "batdau", [5, 6, 7, 8], "M3_ngay_den_batdau"),
        ("ngay_den_ketthuc", "ketthuc", [9, 10, 11, 12], "M3_ngay_den_ketthuc"),
    ]

    tat_ca = []
    for target, thu_muc, thang, ten in cau_hinh:
        bang = train_hoi_quy(
            cfg=cfg, data=data, target=target, thang_loc=thang,
            model_dir=os.path.join(model_root, thu_muc),
            ket_qua_dir=kq_dir, ten_bai_toan=ten,
            baseline_fn=tao_baseline_climatology(target), ten_baseline="Climatology",
        )
        tat_ca.append(bang)

    metrics = pd.concat(tat_ca, ignore_index=True)
    metrics.to_csv(os.path.join(kq_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    metrics.to_csv(os.path.join(kq_dir, "xep_hang_model.csv"), index=False, encoding="utf-8-sig")
    print("\n===== M3: XEP HANG =====")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
