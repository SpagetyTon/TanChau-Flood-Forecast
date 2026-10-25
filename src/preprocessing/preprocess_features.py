"""
Pipeline tiền xử lý Mekong - Tân Châu (gộp từ 3 file)

Flow: Điền thiếu -> Feature engineering -> Split 3 tập

Bước 1: Điền giá trị thiếu của H_TanChau        (từ Code_preprocessing.ipynb)
Bước 2: Feature engineering (+ lọc bớt cột dư)  (từ 01_feature_engineering.py + 02_loc_cot.py)
Bước 3: Tách train / valid / test theo năm      (bước cuối)

Kết quả: train_final.csv, valid_final.csv, test_final.csv
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
# CẤU HÌNH (chỉ sửa ở đây)
# ===============================================================
ROOT = Path(__file__).resolve().parents[2]   # thư mục gốc repo
RAW_FILE = ROOT / "data/final/Mekong_TanChau_2015_2024_extended_35cols.csv"   # bảng 35 cột (trước đây: Mekong_TanChau_2015_2024_extended-1.csv)

OUT_IMPUTED = "Mekong_TanChau_imputed.csv"      # sau bước 1
OUT_FEATURES = "Mekong_TanChau_features.csv"    # sau bước 2
SAVE_INTERMEDIATE = True    # True: lưu thêm file trung gian (imputed, features)
OUT_DIR = ROOT / "data/processed"   # thư mục ghi kết quả (config.yaml: processed_dir)

# True : bước 2 bỏ bớt cột dư (cột phụ + COT_XOA) -> 81 cột
# False: giữ đủ 105 cột như file train/valid/test cũ
LOC_COT = True

TARGET_COL = "H_TanChau"
RANDOM_STATE = 42

# Mốc tách theo năm (bước 3)
TRAIN_END_YEAR = 2021       # train: <= 2021
VALID_YEAR = 2022           # valid: 2022
                            # test : >= 2023

# True: bỏ các dòng còn NaN (do rolling/shift ở đầu chuỗi và target ở cuối chuỗi)
# False: giữ nguyên như code cũ
DROP_NAN = False

# Cột phụ bỏ ở cuối bước 2 (khi LOC_COT = True)
COT_PHU = ["nam", "thang", "ngay", "la_mua_lu_7_11"]

# Cột dư bỏ ở cuối bước 2 (khi LOC_COT = True)
COT_XOA = [
    # Trùng hoàn toàn với cột khác (r = 1)
    "H_TanChau",                                                 # = H_hientai
    "mua_tanchau_tong1", "mua_kratie_tong1", "mua_pakse_tong1",  # = mưa gốc

    # Biến GLDAS tương quan rất cao với cột khác
    "ECanop_tavg", "Tair_f_inst",
    "SoilMoi0_10cm_inst", "SoilMoi40_100cm_inst",

    # Lag mực nước gần trùng H_lag1
    "H_lag2", "H_lag3",

    # Mưa / dòng chảy dư thừa
    "mua_tanchau_tong10", "mua_tanchau_tong30", "so_ngay_mua_30",
    "Qsb_acc_tong3", "Qsb_acc_tong15",

    # Lưu lượng tương quan cao (r = 0.98 với Q_LuangPrabang)
    "Q_ChiangSaen",

    # Quy tắc Q và dQ cùng trạm -> chỉ giữ Stung Treng
    "Q_Pakse", "Q_Kratie",
    "dQ_Kratie_3ngay", "dQ_Kratie_7ngay", "dQ_Kratie_15ngay",
    "dQ_Pakse_3ngay", "dQ_Pakse_7ngay", "dQ_Pakse_15ngay",
]


# ===============================================================
# BƯỚC 1: ĐIỀN GIÁ TRỊ THIẾU CỦA H_TanChau
# ===============================================================
def buoc1_dien_thieu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_thieu = df[TARGET_COL].isna().sum()
    print(f"[B1] Tổng số dòng: {len(df)} | thiếu {TARGET_COL}: {n_thieu}")
    if n_thieu == 0:
        return df

    cot_du_doan = [
        "Q_Pakse", "Q_StungTreng", "Q_Kratie", "Q_Vientiane",
        "Q_LuangPrabang", "Q_ChiangSaen",
        "SoilMoi100_200cm_inst", "RootMoist_inst",
    ]
    known = df[df[TARGET_COL].notna()]
    missing = df[df[TARGET_COL].isna()]

    # So sánh 2 mô hình bằng 5-fold CV
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
    print(f"[B1] -> Chọn mô hình: {best_name}")

    # Huấn luyện và điền
    best_model.fit(known[cot_du_doan], known[TARGET_COL])
    pred = best_model.predict(missing[cot_du_doan])
    df.loc[missing.index, TARGET_COL] = pred
    for d, v in zip(pd.to_datetime(missing["date"]).dt.strftime("%Y-%m-%d"), pred):
        print(f"[B1]   {d}: {v:.3f}")

    print(f"[B1] Số giá trị thiếu còn lại: {df[TARGET_COL].isna().sum()}")
    return df


# ===============================================================
# BƯỚC 2: FEATURE ENGINEERING (tính trên toàn bộ chuỗi, chưa tách tập)
# ===============================================================
def buoc2_feature(df: pd.DataFrame) -> pd.DataFrame:
    H = df[TARGET_COL]
    f = {}

    # 2.1 Mưa tích lũy: tổng trượt w ngày
    for tram in ["tanchau", "kratie", "pakse"]:
        for w in [1, 3, 7, 10, 15, 30]:
            f[f"mua_{tram}_tong{w}"] = df[f"mua_{tram}"].rolling(w).sum()

    # 2.2 Biến thiên lưu lượng thượng nguồn: Q(t) - Q(t-k)
    for tram in ["StungTreng", "Kratie", "Pakse"]:
        for k in [3, 7, 15]:
            f[f"dQ_{tram}_{k}ngay"] = df[f"Q_{tram}"].diff(k)

    # 2.3 Mực nước: giá trị hiện tại, lag và chênh lệch
    f["H_hientai"] = H
    for k in [1, 2, 3, 15, 30]:
        f[f"H_lag{k}"] = H.shift(k)
        f[f"dH_{k}ngay"] = H.diff(k)

    # 2.4 Dòng chảy mặt / ngầm GLDAS: tổng trượt
    for col in ["Qs_acc", "Qsb_acc"]:
        for w in [3, 7, 15]:
            f[f"{col}_tong{w}"] = df[col].rolling(w).sum()

    # 2.5 Biến thiên độ ẩm đất theo tầng
    for col in ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst",
                "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst"]:
        for k in [3, 7, 15]:
            f[f"d{col}_{k}ngay"] = df[col].diff(k)

    # 2.6 Thời gian & mùa vụ
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
    # Pha lũ: 1 = lũ lên (T7-9), -1 = lũ rút (T10-12), 0 = mùa khô (T1-6)
    f["pha_lu"] = np.select([thang.between(7, 9), thang >= 10], [1, -1], default=0)

    # 2.7 Gia tốc mực nước: thay đổi của dH
    f["giatoc_H_3ngay"] = H.diff(3).diff(3)
    f["giatoc_H_7ngay"] = H.diff(7).diff(7)

    # 2.8 Số ngày có mưa (mưa Tân Châu > 1) trong w ngày
    co_mua = (df["mua_tanchau"] > 1).astype(int)
    for w in [7, 15, 30]:
        f[f"so_ngay_mua_{w}"] = co_mua.rolling(w).sum()

    # 2.9 Target: mực nước 1/2/3 ngày tới
    for k in [1, 2, 3]:
        f[f"target_ngay{k}"] = H.shift(-k)

    df_feat = pd.concat([df, pd.DataFrame(f)], axis=1)
    print(f"[B2] Sau khi tạo feature: {df_feat.shape}")      # kỳ vọng (3653, 109)
    if SAVE_INTERMEDIATE:
        df_feat.to_csv(f"{OUT_DIR}/{OUT_FEATURES}", index=False)

    # Lọc bớt cột dư (vẫn thuộc bước feature engineering, chưa tách tập)
    if LOC_COT:
        df_feat = df_feat.drop(columns=COT_PHU + COT_XOA)    # KeyError nếu thiếu cột
        print(f"[B2] Sau khi lọc cột: {df_feat.shape}")      # kỳ vọng (3653, 81)
    return df_feat


# ===============================================================
# BƯỚC 3: TÁCH THÀNH 3 TẬP THEO THỜI GIAN (bước cuối)
# ===============================================================
def buoc3_tach_tap(d: pd.DataFrame):
    if DROP_NAN:
        d = d.dropna().reset_index(drop=True)
    nam = pd.to_datetime(d["date"]).dt.year

    train = d[nam <= TRAIN_END_YEAR]
    valid = d[nam == VALID_YEAR]
    test = d[nam > VALID_YEAR]
    assert len(train) + len(valid) + len(test) == len(d)   # không sót, không trùng

    n = len(d)
    for ten, x in [("Train", train), ("Valid", valid), ("Test", test)]:
        print(f"[B3] {ten}: {x.shape} ({len(x)/n*100:.1f}%) | "
              f"{str(x['date'].iloc[0])[:10]} -> {str(x['date'].iloc[-1])[:10]} | "
              f"số dòng còn NaN: {x.isna().any(axis=1).sum()}")
    return train, valid, test


# ===============================================================
# CHẠY TOÀN BỘ PIPELINE
# ===============================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Bước 1: điền thiếu
    df = buoc1_dien_thieu(df)
    if SAVE_INTERMEDIATE:
        df.to_csv(f"{OUT_DIR}/{OUT_IMPUTED}", index=False)

    # Bước 2: feature engineering
    df_feat = buoc2_feature(df)

    # Bước 3: tách 3 tập -> kết quả cuối
    train, valid, test = buoc3_tach_tap(df_feat)
    train.to_csv(f"{OUT_DIR}/train_final.csv", index=False)
    valid.to_csv(f"{OUT_DIR}/valid_final.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test_final.csv", index=False)
    print("Đã ghi: train_final.csv, valid_final.csv, test_final.csv")
