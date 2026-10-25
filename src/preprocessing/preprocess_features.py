"""
Preprocessing pipeline for Mekong - Tan Chau (merged from 3 earlier scripts)

Flow: gap filling -> feature engineering -> 3-way split

Step 1: fill missing values of H_TanChau                 (from Code_preprocessing.ipynb)
Step 2: feature engineering (+ drop redundant columns)   (from 01_feature_engineering.py + 02_loc_cot.py)
Step 3: split into train / valid / test by year          (final step)

Output: train_final.csv, valid_final.csv, test_final.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ===============================================================
# SETTINGS (edit only here)
# ===============================================================
ROOT = Path(__file__).resolve().parents[2]   # repository root
RAW_FILE = ROOT / "data/final/Mekong_TanChau_2015_2024_extended_35cols.csv"   # 35-column table (formerly: Mekong_TanChau_2015_2024_extended-1.csv)

OUT_IMPUTED = "Mekong_TanChau_imputed.csv"      # after step 1
OUT_FEATURES = "Mekong_TanChau_features.csv"    # after step 2
SAVE_INTERMEDIATE = True    # True: also save intermediate files (imputed, features)
OUT_DIR = ROOT / "data/processed"   # output folder (config.yaml: processed_dir)

# True : step 2 drops redundant columns (helper columns + COT_XOA) -> 81 columns
# False: keep all 105 columns as in the old train/valid/test files
LOC_COT = True

TARGET_COL = "H_TanChau"
RANDOM_STATE = 42

# Split years (step 3)
TRAIN_END_YEAR = 2021       # train: <= 2021
VALID_YEAR = 2022           # valid: 2022
                            # test : >= 2023

# True: drop rows that still contain NaN (from rolling/shift at the start and targets at the end of the series)
# False: keep them, as in the original code
DROP_NAN = False

# Helper columns dropped at the end of step 2 (when LOC_COT = True)
COT_PHU = ["nam", "thang", "ngay", "la_mua_lu_7_11"]

# Redundant columns dropped at the end of step 2 (when LOC_COT = True)
COT_XOA = [
    # Exact duplicates of another column (r = 1)
    "H_TanChau",                                                 # = H_hientai
    "mua_tanchau_tong1", "mua_kratie_tong1", "mua_pakse_tong1",  # = raw rainfall

    # GLDAS variables very highly correlated with another column
    "ECanop_tavg", "Tair_f_inst",
    "SoilMoi0_10cm_inst", "SoilMoi40_100cm_inst",

    # Water-level lags nearly identical to H_lag1
    "H_lag2", "H_lag3",

    # Redundant rainfall / runoff sums
    "mua_tanchau_tong10", "mua_tanchau_tong30", "so_ngay_mua_30",
    "Qsb_acc_tong3", "Qsb_acc_tong15",

    # Highly correlated discharge (r = 0.98 with Q_LuangPrabang)
    "Q_ChiangSaen",

    # Q and dQ of the same stations -> keep Stung Treng only
    "Q_Pakse", "Q_Kratie",
    "dQ_Kratie_3ngay", "dQ_Kratie_7ngay", "dQ_Kratie_15ngay",
    "dQ_Pakse_3ngay", "dQ_Pakse_7ngay", "dQ_Pakse_15ngay",
]


# ===============================================================
# STEP 1: FILL MISSING VALUES OF H_TanChau
# ===============================================================
def buoc1_dien_thieu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_thieu = df[TARGET_COL].isna().sum()
    print(f"[B1] Rows: {len(df)} | missing {TARGET_COL}: {n_thieu}")
    if n_thieu == 0:
        return df

    cot_du_doan = [
        "Q_Pakse", "Q_StungTreng", "Q_Kratie", "Q_Vientiane",
        "Q_LuangPrabang", "Q_ChiangSaen",
        "SoilMoi100_200cm_inst", "RootMoist_inst",
    ]
    known = df[df[TARGET_COL].notna()]
    missing = df[df[TARGET_COL].isna()]

    # Compare two models with 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    lin_model = make_pipeline(StandardScaler(), LinearRegression())
    rf_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=RANDOM_STATE, n_jobs=-1)
    lin_r2 = cross_val_score(lin_model, known[cot_du_doan], known[TARGET_COL],
                             cv=kf, scoring="r2").mean()
    rf_r2 = cross_val_score(rf_model, known[cot_du_doan], known[TARGET_COL],
                            cv=kf, scoring="r2").mean()
    print(f"[B1] R2 Linear: {lin_r2:.4f} | R2 Random Forest: {rf_r2:.4f}")

    best_model, best_name = ((rf_model, "Random Forest") if rf_r2 > lin_r2
                             else (lin_model, "Linear Regression"))
    print(f"[B1] -> Selected model: {best_name}")

    # Fit and fill
    best_model.fit(known[cot_du_doan], known[TARGET_COL])
    pred = best_model.predict(missing[cot_du_doan])
    df.loc[missing.index, TARGET_COL] = pred
    for d, v in zip(pd.to_datetime(missing["date"]).dt.strftime("%Y-%m-%d"), pred):
        print(f"[B1]   {d}: {v:.3f}")

    print(f"[B1] Missing values left: {df[TARGET_COL].isna().sum()}")
    return df


# ===============================================================
# STEP 2: FEATURE ENGINEERING (on the full series, before splitting)
# ===============================================================
def buoc2_feature(df: pd.DataFrame) -> pd.DataFrame:
    H = df[TARGET_COL]
    f = {}

    # 2.1 Accumulated rainfall: rolling sum over w days
    for tram in ["tanchau", "kratie", "pakse"]:
        for w in [1, 3, 7, 10, 15, 30]:
            f[f"mua_{tram}_tong{w}"] = df[f"mua_{tram}"].rolling(w).sum()

    # 2.2 Change in upstream discharge: Q(t) - Q(t-k)
    for tram in ["StungTreng", "Kratie", "Pakse"]:
        for k in [3, 7, 15]:
            f[f"dQ_{tram}_{k}ngay"] = df[f"Q_{tram}"].diff(k)

    # 2.3 Water level: current value, lags and differences
    f["H_hientai"] = H
    for k in [1, 2, 3, 15, 30]:
        f[f"H_lag{k}"] = H.shift(k)
        f[f"dH_{k}ngay"] = H.diff(k)

    # 2.4 GLDAS surface / subsurface runoff: rolling sums
    for col in ["Qs_acc", "Qsb_acc"]:
        for w in [3, 7, 15]:
            f[f"{col}_tong{w}"] = df[col].rolling(w).sum()

    # 2.5 Change in soil moisture by layer
    for col in ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst",
                "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst"]:
        for k in [3, 7, 15]:
            f[f"d{col}_{k}ngay"] = df[col].diff(k)

    # 2.6 Calendar & season
    date = pd.to_datetime(df["date"])
    doy = date.dt.dayofyear
    thang = date.dt.month

    f["nam"] = date.dt.year
    f["thang"] = thang
    f["ngay"] = date.dt.day
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    f["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)
    f["la_mua_lu_7_11"] = thang.between(7, 11).astype(int)
    f["la_mua_lu_8_11"] = thang.between(8, 11).astype(int)
    # Flood phase: 1 = rising (Jul-Sep), -1 = receding (Oct-Dec), 0 = dry season (Jan-Jun)
    f["pha_lu"] = np.select([thang.between(7, 9), thang >= 10], [1, -1], default=0)

    # 2.7 Water-level acceleration: change of dH
    f["giatoc_H_3ngay"] = H.diff(3).diff(3)
    f["giatoc_H_7ngay"] = H.diff(7).diff(7)

    # 2.8 Number of rainy days (Tan Chau rainfall > 1 mm) in w days
    co_mua = (df["mua_tanchau"] > 1).astype(int)
    for w in [7, 15, 30]:
        f[f"so_ngay_mua_{w}"] = co_mua.rolling(w).sum()

    # 2.9 Targets: water level 1/2/3 days ahead
    for k in [1, 2, 3]:
        f[f"target_ngay{k}"] = H.shift(-k)

    df_feat = pd.concat([df, pd.DataFrame(f)], axis=1)
    print(f"[B2] After feature engineering: {df_feat.shape}")      # expected (3653, 109)
    if SAVE_INTERMEDIATE:
        df_feat.to_csv(f"{OUT_DIR}/{OUT_FEATURES}", index=False)

    # Drop redundant columns (still part of feature engineering, before splitting)
    if LOC_COT:
        df_feat = df_feat.drop(columns=COT_PHU + COT_XOA)    # KeyError if a column is missing
        print(f"[B2] After dropping columns: {df_feat.shape}")      # expected (3653, 81)
    return df_feat


# ===============================================================
# STEP 3: SPLIT INTO 3 SETS BY TIME (final step)
# ===============================================================
def buoc3_tach_tap(d: pd.DataFrame):
    if DROP_NAN:
        d = d.dropna().reset_index(drop=True)
    nam = pd.to_datetime(d["date"]).dt.year

    train = d[nam <= TRAIN_END_YEAR]
    valid = d[nam == VALID_YEAR]
    test = d[nam > VALID_YEAR]
    assert len(train) + len(valid) + len(test) == len(d)   # nothing lost, nothing duplicated

    n = len(d)
    for ten, x in [("Train", train), ("Valid", valid), ("Test", test)]:
        print(f"[B3] {ten}: {x.shape} ({len(x)/n*100:.1f}%) | "
              f"{str(x['date'].iloc[0])[:10]} -> {str(x['date'].iloc[-1])[:10]} | "
              f"rows with NaN: {x.isna().any(axis=1).sum()}")
    return train, valid, test


# ===============================================================
# RUN THE WHOLE PIPELINE
# ===============================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Step 1: gap filling
    df = buoc1_dien_thieu(df)
    if SAVE_INTERMEDIATE:
        df.to_csv(f"{OUT_DIR}/{OUT_IMPUTED}", index=False)

    # Step 2: feature engineering
    df_feat = buoc2_feature(df)

    # Step 3: split into 3 sets -> final output
    train, valid, test = buoc3_tach_tap(df_feat)
    train.to_csv(f"{OUT_DIR}/train_final.csv", index=False)
    valid.to_csv(f"{OUT_DIR}/valid_final.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test_final.csv", index=False)
    print("Written: train_final.csv, valid_final.csv, test_final.csv")
