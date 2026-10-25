"""
train_m4.py - M4: warning of abnormally low water in the dry season (Jan-May).

Classification: will the water level fall below 0.5 m after 7 / 15 days?
Evaluated with POD / FAR / CSI / Brier score, not accuracy, because the classes are imbalanced.

Baseline: monthly historical frequency from the TRAIN set.
Output: results/M4/  |  Models: models/M4_han/han07|han15/
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_splits, tao_thu_muc  # noqa: E402
from pipeline import train_phan_loai  # noqa: E402


def tao_baseline_tan_suat(target: str):
    """Baseline: probability = share of low-water days in that month in the train set."""

    def _fn(df_test, df_train, cfg):
        date_col = cfg["cot"]["date"]
        d = df_train.dropna(subset=[target])
        ty_le = d.groupby(d[date_col].dt.month)[target].mean()
        chung = d[target].mean()
        return df_test[date_col].dt.month.map(ty_le).fillna(chung).to_numpy()

    return _fn


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    data = load_splits(cfg)

    model_root = os.path.join(cfg["duong_dan"]["model_dir"], "M4_han")
    kq_dir = os.path.join(cfg["duong_dan"]["ket_qua_dir"], "M4")
    tao_thu_muc(model_root, kq_dir)

    tat_ca = []
    for h in cfg["m4"]["horizons"]:
        target = f"han_{h}"
        bang = train_phan_loai(
            cfg=cfg, data=data, target=target,
            thang_loc=cfg["mua_vu"]["thang_mua_kho"],
            model_dir=os.path.join(model_root, f"han{h:02d}"),
            ket_qua_dir=kq_dir, ten_bai_toan=f"M4_han_{h}ngay",
            baseline_fn=tao_baseline_tan_suat(target), ten_baseline="TanSuatLichSu",
        )
        bang["horizon"] = f"{h} ngay"
        tat_ca.append(bang)

    metrics = pd.concat(tat_ca, ignore_index=True)
    cot_dau = ["horizon", "model", "POD", "FAR", "CSI", "Brier"]
    metrics = metrics[[c for c in cot_dau if c in metrics.columns]
                      + [c for c in metrics.columns if c not in cot_dau]]
    metrics.to_csv(os.path.join(kq_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    metrics.to_csv(os.path.join(kq_dir, "xep_hang_model.csv"), index=False, encoding="utf-8-sig")
    print("\n===== M4: RANKING =====")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
