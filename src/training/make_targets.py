"""
make_targets.py - Tao toan bo bien muc tieu cho M1, M2, M3, M4 tu chuoi muc nuoc goc.

Chay dau tien, truoc moi file train_*.py.
Ket qua: data/targets/all_targets.csv  (mot dong moi ngay, khop theo cot date)

    M1: target_ngay1/2/3/5/7/14        - muc nuoc sau h ngay
    M2: so_ngay_tren_nguong_10         - so ngay vuot 2.5 m trong 10 ngay toi
    M3: ngay_den_batdau / ngay_den_ketthuc - con bao nhieu ngay nua mua lu bat dau / ket thuc
    M4: han_7 / han_15                 - muc nuoc sau 7 / 15 ngay co duoi 0.5 m khong
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, tao_thu_muc  # noqa: E402


def doc_chuoi_muc_nuoc(cfg: dict) -> pd.DataFrame:
    """Ghep chuoi muc nuoc 2015-2024 tu 3 file da tien xu ly (train + valid + test).

    Chi lay cot date va H_TanChau de tinh bien muc tieu.
    Toan bo 104 cot dac trung van duoc doc nguyen ven tu 3 file do trong luc train.
    """
    date_col, h_col = cfg["cot"]["date"], cfg["cot"]["muc_nuoc"]
    d = cfg["duong_dan"]["processed_dir"]

    ung_vien = [h_col, cfg["cot"]["muc_nuoc_hien_tai"], "H_TanChau", "H_hientai", "H", "muc_nuoc"]

    parts, cot_dung = [], None
    for fn in ["train_final.csv", "valid_final.csv", "test_final.csv"]:
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Khong tim thay {path} - kiem tra lai processed_dir trong config.yaml")
        part = pd.read_csv(path, parse_dates=[date_col])

        if cot_dung is None:
            for c in ung_vien:
                if c and c in part.columns:
                    cot_dung = c
                    break
            if cot_dung is None and "target_ngay1" in part.columns:
                cot_dung = "__tu_target__"
            if cot_dung is None:
                raise KeyError(
                    f"{fn} khong co cot muc nuoc nao trong {ung_vien}.\n"
                    f"Cac cot dang co: {list(part.columns)[:40]} ...\n"
                    f"Sua 'cot: muc_nuoc' trong config.yaml cho dung ten cot muc nuoc cua ban."
                )
            print(f"  dung cot muc nuoc: {cot_dung}")

        if cot_dung == "__tu_target__":
            # H(t) = target_ngay1(t-1): suy nguoc muc nuoc tu bien muc tieu 1 ngay
            tmp = part[[date_col, "target_ngay1"]].copy()
            tmp[h_col] = tmp["target_ngay1"].shift(1)
            parts.append(tmp[[date_col, h_col]])
        else:
            parts.append(part[[date_col, cot_dung]].rename(columns={cot_dung: h_col}))
        print(f"  doc {fn}: {len(part)} dong")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(date_col).sort_values(date_col)

    lo, hi = float(df[h_col].min()), float(df[h_col].max())
    print(f"  muc nuoc: min={lo:.3f}, max={hi:.3f}")
    if hi > 20 or lo < -5:
        print("  [canh bao] gia tri khong giong met - kiem tra lai don vi truoc khi dung nguong trong config")
    # dam bao chuoi ngay lien tuc, khong bi nhay ngay
    full = pd.DataFrame({date_col: pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")})
    return full.merge(df, on=date_col, how="left").reset_index(drop=True)


def target_M1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h_col = cfg["cot"]["muc_nuoc"]
    for h in cfg["m1"]["horizons"]:
        df[f"target_ngay{h}"] = df[h_col].shift(-h)
    return df


def target_M2(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Dem so ngay vuot nguong trong cua so k ngay TOI (khong tinh ngay hien tai)."""
    h_col = cfg["cot"]["muc_nuoc"]
    k = cfg["m2"]["cua_so_ngay"]
    nguong = cfg["nguong"]["m2_vuot_nguong_m"]

    vuot = (df[h_col] > nguong).astype(float)
    # dao chuoi -> rolling -> dao lai: dem k ngay phia truoc
    dem = vuot.shift(-1).iloc[::-1].rolling(k, min_periods=k).sum().iloc[::-1]
    df[f"so_ngay_tren_nguong_{k}"] = dem
    return df


def target_M3(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Moi nam: ngay dau tien va ngay cuoi cung muc nuoc >= nguong mua lu."""
    date_col, h_col = cfg["cot"]["date"], cfg["cot"]["muc_nuoc"]
    nguong = cfg["nguong"]["mua_lu_m"]

    df["ngay_den_batdau"] = np.nan
    df["ngay_den_ketthuc"] = np.nan

    for nam, nhom in df.groupby(df[date_col].dt.year):
        tren = nhom[nhom[h_col] >= nguong]
        if tren.empty:
            print(f"  [canh bao] nam {nam} khong co ngay nao vuot {nguong} m - bo qua")
            continue
        ngay_bd = tren[date_col].iloc[0]
        ngay_kt = tren[date_col].iloc[-1]

        idx = nhom.index
        so_ngay_bd = (ngay_bd - nhom[date_col]).dt.days
        so_ngay_kt = (ngay_kt - nhom[date_col]).dt.days
        # chi giu cac ngay TRUOC moc, sau moc thi khong con y nghia du bao
        df.loc[idx, "ngay_den_batdau"] = so_ngay_bd.where(so_ngay_bd >= 0)
        df.loc[idx, "ngay_den_ketthuc"] = so_ngay_kt.where(so_ngay_kt >= 0)

    return df


def target_M4(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h_col = cfg["cot"]["muc_nuoc"]
    nguong = cfg["nguong"]["han_m"]
    for h in cfg["m4"]["horizons"]:
        tuong_lai = df[h_col].shift(-h)
        df[f"han_{h}"] = np.where(tuong_lai.isna(), np.nan, (tuong_lai < nguong).astype(float))
    return df


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    date_col = cfg["cot"]["date"]

    print("Doc chuoi muc nuoc...")
    df = doc_chuoi_muc_nuoc(cfg)
    print(f"  {len(df)} ngay, tu {df[date_col].min().date()} den {df[date_col].max().date()}")

    print("Tao target M1 (muc nuoc theo tam du bao)...")
    df = target_M1(df, cfg)
    print("Tao target M2 (so ngay vuot nguong)...")
    df = target_M2(df, cfg)
    print("Tao target M3 (bat dau / ket thuc mua lu)...")
    df = target_M3(df, cfg)
    print("Tao target M4 (canh bao han)...")
    df = target_M4(df, cfg)

    cot_target = [c for c in df.columns if c not in (date_col, cfg["cot"]["muc_nuoc"])]
    out_dir = cfg["duong_dan"]["targets_dir"]
    tao_thu_muc(out_dir)

    df[[date_col] + cot_target].to_csv(
        os.path.join(out_dir, "all_targets.csv"), index=False, encoding="utf-8-sig"
    )
    # luu them ban rieng cho tung model de de kiem tra
    df[[date_col] + [c for c in cot_target if c.startswith("target_ngay")]].to_csv(
        os.path.join(out_dir, "M1_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col] + [c for c in cot_target if c.startswith("so_ngay_tren_nguong")]].to_csv(
        os.path.join(out_dir, "M2_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col, "ngay_den_batdau", "ngay_den_ketthuc"]].to_csv(
        os.path.join(out_dir, "M3_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col] + [c for c in cot_target if c.startswith("han_")]].to_csv(
        os.path.join(out_dir, "M4_targets.csv"), index=False, encoding="utf-8-sig")

    print(f"\nDa luu {len(cot_target)} cot target vao {out_dir}/all_targets.csv")
    print(df[cot_target].notna().sum().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
