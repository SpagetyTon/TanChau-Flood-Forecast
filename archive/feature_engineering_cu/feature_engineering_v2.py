r"""
feature_engineering.py - Tao 105 cot dac trung tu file goc 35 cot.

Dau vao : Mekong_TanChau_2015_2024_extended 1.csv   (3653 dong x 35 cot)
Dau ra  : Mekong_TanChau_features.csv               (109 cot, ban day du)
          train.csv / valid.csv / test.csv           (105 cot, da bo 4 cot khong dung)
          train 2015-2021 | valid 2022 | test 2023-2024

Nguyen tac:
  - Bien TICH LUY (mua, dong chay mat/ngam) -> rolling TONG
  - Bien TRANG THAI (luu luong, muc nuoc, do am dat) -> HIEU so voi n ngay truoc
  - Rolling va lag chi nhin ve qua khu nen khong ro ri du lieu
  - Target = shift AM cua muc nuoc (nguoc dau voi lag)
"""

import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# ------------------------------------------------------------------ cau hinh
FILE_GOC = "Mekong_TanChau_2015_2024_extended 1.csv"
THU_MUC_RA = "."
# File da co san gia tri muc nuoc dien khuyet (vd ban features cu cua nhom).
# De None thi script tu chay MICE. LUU Y: khong tro vao chinh FILE_FEATURES
# ben duoi, vi script se ghi de len file do.
FILE_DA_FILL = None
MICE_MAX_ITER = 10      # dien 4 ngay thieu muc nuoc bang MICE
MICE_SEED = 0

CUA_SO_MUA = [1, 3, 7, 10, 15, 30]          # tong mua
CUA_SO_DQ = [3, 7, 15]                      # hieu luu luong
LAG_H = [1, 2, 3, 15, 30]                   # lag + hieu muc nuoc
CUA_SO_DONGCHAY = [3, 7, 15]                # tong Qs_acc / Qsb_acc
CUA_SO_DAT = [3, 7, 15]                     # hieu do am dat
CUA_SO_GIATOC = [3, 7]                      # gia toc muc nuoc
CUA_SO_NGAYMUA = [7, 15, 30]                # dem ngay co mua
NGUONG_NGAY_MUA = 1.0                       # mm, tren muc nay tinh la "co mua"
HORIZONS = [1, 2, 3]                        # target

# 4 cot tao ra de tham khao nhung KHONG dua vao model:
#   nam/thang/ngay   - da duoc doy_sin/cos mo ta tot hon, de model hoc thuoc nam
#   la_mua_lu_7_11   - trung y nghia voi la_mua_lu_8_11, giu 1 trong 2
COT_BO_TRUOC_KHI_CHIA = ["nam", "thang", "ngay", "la_mua_lu_7_11"]
FILE_FEATURES = "Mekong_TanChau_features.csv"

TRAM_MUA = ["mua_tanchau", "mua_kratie", "mua_pakse"]
TRAM_Q = ["Q_StungTreng", "Q_Kratie", "Q_Pakse"]
TANG_DAT = ["SoilMoi0_10cm_inst", "SoilMoi10_40cm_inst",
            "SoilMoi40_100cm_inst", "SoilMoi100_200cm_inst"]

CAT_TRAIN = "2021-12-31"
CAT_VALID = "2022-12-31"


# ------------------------------------------------------------------ doc & fill
def doc_du_lieu(path: str) -> pd.DataFrame:
    if FILE_DA_FILL and os.path.abspath(FILE_DA_FILL) == os.path.abspath(
            os.path.join(THU_MUC_RA, FILE_FEATURES)):
        raise ValueError(
            f"FILE_DA_FILL khong duoc tro vao '{FILE_FEATURES}' - script se ghi de len file nay.\n"
            f"Hay doi ten ban goc (vd goc_features.csv) roi tro FILE_DA_FILL vao do.")

    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    # bao dam chuoi ngay lien tuc, khong nhay ngay
    full = pd.DataFrame({"date": pd.date_range(df.date.min(), df.date.max(), freq="D")})
    df = full.merge(df, on="date", how="left")

    thieu = int(df["H_TanChau"].isna().sum())
    if thieu:
        if FILE_DA_FILL:
            fill = pd.read_csv(FILE_DA_FILL, parse_dates=["date"])[["date", "H_TanChau"]]
            df["H_TanChau"] = df["date"].map(fill.set_index("date")["H_TanChau"]).fillna(df["H_TanChau"])
            print(f"  da lay {thieu} gia tri muc nuoc tu {FILE_DA_FILL}")
        else:
            # MICE: dien muc nuoc dua tren cac bien khi tuong - thuy van CUNG NGAY
            # (luu luong thuong nguon, mua, do am dat...). Day la quan he ngang theo
            # bien, khong phai noi suy theo thoi gian.
            print(f"  dien {thieu} ngay thieu muc nuoc bang MICE:")
            for d_ in df.loc[df["H_TanChau"].isna(), "date"]:
                print(f"      {d_.date()}")
            num = df.select_dtypes(include=[np.number])
            imp = IterativeImputer(max_iter=MICE_MAX_ITER, random_state=MICE_SEED)
            df[num.columns] = imp.fit_transform(num)
            print("      [luu y] ban goc cua nhom dung MICE voi tham so khac,")
            print("      gia tri co the lech vai cm. Muon trung khop tuyet doi")
            print("      thi dat FILE_DA_FILL o dau file.")
    return df


# ------------------------------------------------------------------ dac trung
def tao_dac_trung(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # A. Mua tich luy - TONG. 100 mm trong 3 ngay la 100 mm nuoc that do xuong.
    for tram in TRAM_MUA:
        for k in CUA_SO_MUA:
            d[f"{tram}_tong{k}"] = d[tram].rolling(k, min_periods=k).sum()

    # B. Bien dong luu luong - HIEU. Cong don luu luong khong co y nghia vat ly,
    #    cai dang quan tam la song lu dang len hay dang rut.
    for tram in TRAM_Q:
        for k in CUA_SO_DQ:
            d[f"d{tram}_{k}ngay"] = d[tram] - d[tram].shift(k)

    # C. Muc nuoc Tan Chau - muc hien tai + lag + hieu
    H = d["H_TanChau"]
    d["H_hientai"] = H
    for k in LAG_H:
        d[f"H_lag{k}"] = H.shift(k)
        d[f"dH_{k}ngay"] = H - H.shift(k)

    # D. Dong chay cuc bo - TONG
    for col in ["Qs_acc", "Qsb_acc"]:
        for k in CUA_SO_DONGCHAY:
            d[f"{col}_tong{k}"] = d[col].rolling(k, min_periods=k).sum()

    # E. Do am dat - HIEU. Luu vuc no nuoc thi mua chay tran thay vi tham.
    for tang in TANG_DAT:
        for k in CUA_SO_DAT:
            d[f"d{tang}_{k}ngay"] = d[tang] - d[tang].shift(k)

    # F. Mua vu - tach ngay/thang/nam + chu ky nam va nua nam
    d["nam"] = d["date"].dt.year
    d["thang"] = d["date"].dt.month
    d["ngay"] = d["date"].dt.day
    doy = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["doy_sin2"] = np.sin(4 * np.pi * doy / 365.25)
    d["doy_cos2"] = np.cos(4 * np.pi * doy / 365.25)

    # G. Co mua vu
    thang = d["date"].dt.month
    d["la_mua_lu_7_11"] = thang.isin([7, 8, 9, 10, 11]).astype(int)
    d["la_mua_lu_8_11"] = thang.isin([8, 9, 10, 11]).astype(int)
    d["pha_lu"] = np.select([thang.isin([7, 8, 9]), thang.isin([10, 11, 12])],
                            [1, -1], default=0).astype(int)   # 1 len, -1 rut, 0 kho

    # H. Gia toc muc nuoc = thay doi cua muc thay doi
    for k in CUA_SO_GIATOC:
        dh = H - H.shift(k)
        d[f"giatoc_H_{k}ngay"] = dh - dh.shift(k)

    # I. So ngay co mua tai Tan Chau trong cua so
    co_mua = (d["mua_tanchau"] > NGUONG_NGAY_MUA).astype(float)
    for k in CUA_SO_NGAYMUA:
        d[f"so_ngay_mua_{k}"] = co_mua.rolling(k, min_periods=k).sum()

    # J. Target - shift AM (nguoc dau voi lag)
    for h in HORIZONS:
        d[f"target_ngay{h}"] = H.shift(-h)

    return d


def chia_theo_thoi_gian(d: pd.DataFrame) -> dict:
    """Chia theo moc thoi gian, KHONG xao tron - tranh ro ri tuong lai vao qua khu."""
    return {
        "train": d[d.date <= CAT_TRAIN],
        "valid": d[(d.date > CAT_TRAIN) & (d.date <= CAT_VALID)],
        "test": d[d.date > CAT_VALID],
    }


def main():
    print("Doc du lieu goc...")
    df = doc_du_lieu(FILE_GOC)
    print(f"  {df.shape[0]} dong x {df.shape[1]} cot | {df.date.min().date()} -> {df.date.max().date()}")

    print("Tao dac trung...")
    d = tao_dac_trung(df)
    moi = d.shape[1] - df.shape[1]
    print(f"  {df.shape[1]} -> {d.shape[1]} cot (them {moi})")

    os.makedirs(THU_MUC_RA, exist_ok=True)
    d.to_csv(os.path.join(THU_MUC_RA, FILE_FEATURES), index=False)
    print(f"  da luu ban day du {d.shape[1]} cot -> {FILE_FEATURES}")

    d = d.drop(columns=COT_BO_TRUOC_KHI_CHIA)
    print(f"  bo {len(COT_BO_TRUOC_KHI_CHIA)} cot khong dua vao model -> {d.shape[1]} cot")

    for ten, phan in chia_theo_thoi_gian(d).items():
        path = os.path.join(THU_MUC_RA, f"{ten}.csv")
        phan.to_csv(path, index=False)
        print(f"  {ten:5s}: {len(phan):5d} dong  {phan.date.min().date()} -> {phan.date.max().date()}")

    print("\nLuu y: 30 dong dau co gia tri thieu (cua so rolling 30 ngay),")
    print("3 dong cuoi thieu target. Bo chung o buoc train, khong dien bang trung vi.")


if __name__ == "__main__":
    main()
