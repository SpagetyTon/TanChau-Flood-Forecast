"""
train_m1.py - M1: du bao muc nuoc Tan Chau tai 6 tam du bao (1, 2, 3, 5, 7, 14 ngay).

Baseline: Persistence (muc nuoc ngay mai = muc nuoc hom nay).
Ket qua: 05_Ket_qua/M1/metrics_theo_horizon.csv, xep_hang_model.csv, du_bao_test_*.csv
Model:   04_Model/M1_mucnuoc/ngayXX/*.pkl
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_splits, tao_thu_muc, xep_hang  # noqa: E402
from pipeline import train_hoi_quy  # noqa: E402


def baseline_persistence(df_test, df_train, cfg):
    """Du bao = muc nuoc hien tai."""
    return df_test[cfg["cot"]["muc_nuoc_hien_tai"]].to_numpy()


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    data = load_splits(cfg)

    model_root = os.path.join(cfg["duong_dan"]["model_dir"], "M1_mucnuoc")
    kq_dir = os.path.join(cfg["duong_dan"]["ket_qua_dir"], "M1")
    tao_thu_muc(model_root, kq_dir)

    tat_ca = []
    for h in cfg["m1"]["horizons"]:
        target = f"target_ngay{h}"
        bang = train_hoi_quy(
            cfg=cfg, data=data, target=target, thang_loc=None,
            model_dir=os.path.join(model_root, f"ngay{h:02d}"),
            ket_qua_dir=kq_dir, ten_bai_toan=f"M1_ngay{h}",
            baseline_fn=baseline_persistence, ten_baseline="Persistence",
        )
        bang["horizon"] = f"Ngay {h}"
        tat_ca.append(bang)

    metrics = pd.concat(tat_ca, ignore_index=True)
    cot_dau = ["horizon", "model", "RMSE", "MAE", "R2", "MAPE_%", "WAPE_%"]
    metrics = metrics[cot_dau + [c for c in metrics.columns if c not in cot_dau]]
    metrics.to_csv(os.path.join(kq_dir, "metrics_theo_horizon.csv"),
                   index=False, encoding="utf-8-sig")

    # bang xep hang tong: trung binh cac chi so qua 6 tam du bao
    tb = (metrics.groupby("model")[["RMSE", "MAE", "R2", "MAPE_%", "WAPE_%"]]
          .mean().reset_index())
    tb = xep_hang(tb, "RMSE", tang_dan=True)
    if "Persistence" in set(tb["model"]):
        base = float(tb.loc[tb["model"] == "Persistence", "RMSE"].iloc[0])
        tb["cai_thien_vs_baseline_%"] = ((base - tb["RMSE"]) / base * 100).round(2)
    tb.to_csv(os.path.join(kq_dir, "xep_hang_model.csv"), index=False, encoding="utf-8-sig")

    print("\n===== M1: XEP HANG TRUNG BINH QUA 6 TAM DU BAO =====")
    print(tb.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
