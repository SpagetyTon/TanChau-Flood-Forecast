"""
summarize.py - Gop ket qua 4 model thanh mot bang duy nhat + ve hinh cho bao cao.

Ket qua: 05_Ket_qua/tong_hop_4_model.csv  va  06_Hinh_anh/*.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, tao_thu_muc  # noqa: E402


def _doc(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def tong_hop(cfg: dict) -> pd.DataFrame:
    kq = cfg["duong_dan"]["ket_qua_dir"]
    rows = []

    m1 = _doc(os.path.join(kq, "M1", "xep_hang_model.csv"))
    if m1 is not None:
        tot = m1.iloc[0]
        rows.append({"bai_toan": "M1 - Muc nuoc 1/2/3/5/7/14 ngay", "model_tot_nhat": tot["model"],
                     "chi_so_chinh": "RMSE (m)", "gia_tri": round(tot["RMSE"], 4),
                     "cai_thien_vs_baseline_%": tot.get("cai_thien_vs_baseline_%", None)})

    m2 = _doc(os.path.join(kq, "M2", "xep_hang_model.csv"))
    if m2 is not None:
        tot = m2.iloc[0]
        rows.append({"bai_toan": "M2 - So ngay vuot nguong trong 10 ngay", "model_tot_nhat": tot["model"],
                     "chi_so_chinh": "RMSE (ngay)", "gia_tri": round(tot["RMSE"], 4),
                     "cai_thien_vs_baseline_%": tot.get("cai_thien_vs_baseline_%", None)})

    m3 = _doc(os.path.join(kq, "M3", "xep_hang_model.csv"))
    if m3 is not None:
        for bt, nhom in m3.groupby("bai_toan"):
            tot = nhom.sort_values("RMSE").iloc[0]
            rows.append({"bai_toan": f"M3 - {bt}", "model_tot_nhat": tot["model"],
                         "chi_so_chinh": "RMSE (ngay)", "gia_tri": round(tot["RMSE"], 4),
                         "cai_thien_vs_baseline_%": tot.get("cai_thien_vs_baseline_%", None)})

    m4 = _doc(os.path.join(kq, "M4", "xep_hang_model.csv"))
    if m4 is not None:
        for hz, nhom in m4.groupby("horizon"):
            tot = nhom.sort_values("CSI", ascending=False).iloc[0]
            rows.append({"bai_toan": f"M4 - Canh bao han {hz}", "model_tot_nhat": tot["model"],
                         "chi_so_chinh": "CSI", "gia_tri": round(tot["CSI"], 4),
                         "cai_thien_vs_baseline_%": None})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(kq, "tong_hop_4_model.csv"), index=False, encoding="utf-8-sig")
    return df


def ve_hinh(cfg: dict):
    kq, hinh = cfg["duong_dan"]["ket_qua_dir"], cfg["duong_dan"]["hinh_anh_dir"]
    tao_thu_muc(hinh)

    m1 = _doc(os.path.join(kq, "M1", "metrics_theo_horizon.csv"))
    if m1 is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model, nhom in m1.groupby("model"):
            nhom = nhom.copy()
            nhom["h"] = nhom["horizon"].str.extract(r"(\d+)").astype(int)
            nhom = nhom.sort_values("h")
            ax.plot(nhom["h"], nhom["RMSE"], marker="o", label=model)
        ax.set_xlabel("Tam du bao (ngay)")
        ax.set_ylabel("RMSE (m)")
        ax.set_title("M1 - Sai so theo tam du bao")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(hinh, "M1_sai_so_theo_horizon.png"), dpi=150)
        plt.close(fig)

    for h in cfg["m1"]["horizons"]:
        f = os.path.join(kq, "M1", f"du_bao_test_target_ngay{h}.csv")
        d = _doc(f)
        if d is None or "du_bao_Blend" not in d.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(pd.to_datetime(d["date"]), d["thuc_te"], label="Thuc te", lw=1.2)
        ax.plot(pd.to_datetime(d["date"]), d["du_bao_Blend"], label="Blend", lw=1.0, alpha=0.85)
        ax.set_title(f"M1 - Du bao {h} ngay tren tap test (2023-2024)")
        ax.set_ylabel("Muc nuoc (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(hinh, f"M1_duong_du_bao_ngay{h:02d}.png"), dpi=150)
        plt.close(fig)

    for h in cfg["m4"]["horizons"]:
        d = _doc(os.path.join(kq, "M4", f"du_bao_test_han_{h}.csv"))
        if d is None or "xac_suat_Blend" not in d.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(pd.to_datetime(d["date"]), d["xac_suat_Blend"], label="Xac suat canh bao")
        ax.fill_between(pd.to_datetime(d["date"]), 0, d["thuc_te"], alpha=0.25, label="Thuc te co han")
        ax.set_title(f"M4 - Xac suat canh bao han {h} ngay (mua kho tap test)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(hinh, f"M4_xac_suat_han_{h:02d}ngay.png"), dpi=150)
        plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    df = tong_hop(cfg)
    ve_hinh(cfg)
    print("\n===== TONG HOP 4 MODEL =====")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
