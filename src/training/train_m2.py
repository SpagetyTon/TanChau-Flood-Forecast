"""
train_m2.py - M2: du bao so ngay muc nuoc vuot nguong trong 10 ngay toi (mua lu, thang 8-11).

Baseline: trung binh lich su theo thang (climatology) tinh tu tap TRAIN.
Ket qua: 05_Ket_qua/M2/  |  Model: 04_Model/M2_soNgayVuotNguong/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_splits, tao_thu_muc  # noqa: E402
from pipeline import train_hoi_quy  # noqa: E402


def tao_baseline_climatology(target: str):
    """Tra ve ham baseline: du bao = trung binh cua thang do tren tap train."""

    def _fn(df_test, df_train, cfg):
        date_col = cfg["cot"]["date"]
        tb_thang = df_train.dropna(subset=[target]).groupby(
            df_train.dropna(subset=[target])[date_col].dt.month)[target].mean()
        tb_chung = df_train[target].mean()
        return df_test[date_col].dt.month.map(tb_thang).fillna(tb_chung).to_numpy()

    return _fn


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    data = load_splits(cfg)

    k = cfg["m2"]["cua_so_ngay"]
    target = f"so_ngay_tren_nguong_{k}"
    model_dir = os.path.join(cfg["duong_dan"]["model_dir"], "M2_soNgayVuotNguong")
    kq_dir = os.path.join(cfg["duong_dan"]["ket_qua_dir"], "M2")
    tao_thu_muc(model_dir, kq_dir)

    bang = train_hoi_quy(
        cfg=cfg, data=data, target=target,
        thang_loc=cfg["mua_vu"]["thang_mua_lu"],
        model_dir=model_dir, ket_qua_dir=kq_dir,
        ten_bai_toan=f"M2_so_ngay_tren_{cfg['nguong']['m2_vuot_nguong_m']}m",
        baseline_fn=tao_baseline_climatology(target), ten_baseline="Climatology",
    )

    bang.to_csv(os.path.join(kq_dir, "xep_hang_model.csv"), index=False, encoding="utf-8-sig")
    bang.to_csv(os.path.join(kq_dir, "metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n===== M2: XEP HANG =====")
    print(bang.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
