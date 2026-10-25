"""
make_targets.py - Build all targets for M1, M2, M3, M4 from the water-level series.

Run first, before any train_*.py script.
Output: data/targets/all_targets.csv  (one row per day, joined on the date column)

    M1: target_ngay1/2/3/5/7/14            - water level h days ahead
    M2: so_ngay_tren_nguong_10             - number of days above 2.5 m in the next 10 days
    M3: ngay_den_batdau / ngay_den_ketthuc - days until the flood season starts / ends
    M4: han_7 / han_15                     - whether the water level is below 0.5 m after 7 / 15 days
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import load_config, tao_thu_muc  # noqa: E402


def doc_chuoi_muc_nuoc(cfg: dict) -> pd.DataFrame:
    """Concatenate the 2015-2024 water-level series from the 3 preprocessed files (train + valid + test).

    Only the date and H_TanChau columns are used to build the targets.
    All feature columns are still read unchanged from those files during training.
    """
    date_col, h_col = cfg["cot"]["date"], cfg["cot"]["muc_nuoc"]
    d = cfg["duong_dan"]["processed_dir"]

    ung_vien = [h_col, cfg["cot"]["muc_nuoc_hien_tai"], "H_TanChau", "H_hientai", "H", "muc_nuoc"]

    parts, cot_dung = [], None
    for fn in ["train_final.csv", "valid_final.csv", "test_final.csv"]:
        path = os.path.join(d, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - check processed_dir in config.yaml")
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
                    f"{fn} has none of the water-level columns {ung_vien}.\n"
                    f"Available columns: {list(part.columns)[:40]} ...\n"
                    f"Set 'cot: muc_nuoc' in config.yaml to the name of your water-level column."
                )
            print(f"  using water-level column: {cot_dung}")

        if cot_dung == "__tu_target__":
            # H(t) = target_ngay1(t-1): recover the water level from the 1-day target
            tmp = part[[date_col, "target_ngay1"]].copy()
            tmp[h_col] = tmp["target_ngay1"].shift(1)
            parts.append(tmp[[date_col, h_col]])
        else:
            parts.append(part[[date_col, cot_dung]].rename(columns={cot_dung: h_col}))
        print(f"  read {fn}: {len(part)} rows")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(date_col).sort_values(date_col)

    lo, hi = float(df[h_col].min()), float(df[h_col].max())
    print(f"  water level: min={lo:.3f}, max={hi:.3f}")
    if hi > 20 or lo < -5:
        print("  [warning] values do not look like metres - check the units before using the thresholds in the config")
    # make sure the daily series is continuous, with no gaps
    full = pd.DataFrame({date_col: pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")})
    return full.merge(df, on=date_col, how="left").reset_index(drop=True)


def target_M1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    h_col = cfg["cot"]["muc_nuoc"]
    for h in cfg["m1"]["horizons"]:
        df[f"target_ngay{h}"] = df[h_col].shift(-h)
    return df


def target_M2(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Count days above the threshold in the NEXT k days (excluding the current day)."""
    h_col = cfg["cot"]["muc_nuoc"]
    k = cfg["m2"]["cua_so_ngay"]
    nguong = cfg["nguong"]["m2_vuot_nguong_m"]

    vuot = (df[h_col] > nguong).astype(float)
    # reverse -> rolling -> reverse back: counts the k days ahead
    dem = vuot.shift(-1).iloc[::-1].rolling(k, min_periods=k).sum().iloc[::-1]
    df[f"so_ngay_tren_nguong_{k}"] = dem
    return df


def target_M3(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """For each year: first and last day with water level >= the flood-season threshold."""
    date_col, h_col = cfg["cot"]["date"], cfg["cot"]["muc_nuoc"]
    nguong = cfg["nguong"]["mua_lu_m"]

    df["ngay_den_batdau"] = np.nan
    df["ngay_den_ketthuc"] = np.nan

    for nam, nhom in df.groupby(df[date_col].dt.year):
        tren = nhom[nhom[h_col] >= nguong]
        if tren.empty:
            print(f"  [warning] year {nam} has no day above {nguong} m - skipped")
            continue
        ngay_bd = tren[date_col].iloc[0]
        ngay_kt = tren[date_col].iloc[-1]

        idx = nhom.index
        so_ngay_bd = (ngay_bd - nhom[date_col]).dt.days
        so_ngay_kt = (ngay_kt - nhom[date_col]).dt.days
        # keep only days BEFORE the event; after it there is nothing to forecast
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

    print("Reading the water-level series...")
    df = doc_chuoi_muc_nuoc(cfg)
    print(f"  {len(df)} days, from {df[date_col].min().date()} to {df[date_col].max().date()}")

    print("Building M1 targets (water level by horizon)...")
    df = target_M1(df, cfg)
    print("Building M2 target (days above threshold)...")
    df = target_M2(df, cfg)
    print("Building M3 targets (flood-season onset / end)...")
    df = target_M3(df, cfg)
    print("Building M4 targets (low-water warning)...")
    df = target_M4(df, cfg)

    cot_target = [c for c in df.columns if c not in (date_col, cfg["cot"]["muc_nuoc"])]
    out_dir = cfg["duong_dan"]["targets_dir"]
    tao_thu_muc(out_dir)

    df[[date_col] + cot_target].to_csv(
        os.path.join(out_dir, "all_targets.csv"), index=False, encoding="utf-8-sig"
    )
    # also save one file per target group for easier checking
    df[[date_col] + [c for c in cot_target if c.startswith("target_ngay")]].to_csv(
        os.path.join(out_dir, "M1_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col] + [c for c in cot_target if c.startswith("so_ngay_tren_nguong")]].to_csv(
        os.path.join(out_dir, "M2_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col, "ngay_den_batdau", "ngay_den_ketthuc"]].to_csv(
        os.path.join(out_dir, "M3_targets.csv"), index=False, encoding="utf-8-sig")
    df[[date_col] + [c for c in cot_target if c.startswith("han_")]].to_csv(
        os.path.join(out_dir, "M4_targets.csv"), index=False, encoding="utf-8-sig")

    print(f"\nSaved {len(cot_target)} target columns to {out_dir}/all_targets.csv")
    print(df[cot_target].notna().sum().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
