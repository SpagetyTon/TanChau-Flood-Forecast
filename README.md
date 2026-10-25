# Tan Chau forecast – flood map – population exposure chain

> Bản tiếng Việt: [README.vi.md](README.vi.md)

Code accompanying the paper *"From station water level to commune-level population exposure: evaluating a forecast – flood map – exposure chain around Tan Chau station, Vietnamese Mekong Delta"* (manuscript in preparation).

The repository covers **data collection**, **preprocessing and feature engineering**, and **training of four forecast targets M1–M4** (five algorithms plus a Blend).

> The version used for the paper is tagged `v1.0-paper`. Identifiers in the code (variable names, config keys, column and folder names) are in Vietnamese without diacritics; see the [glossary](#glossary-of-vietnamese-identifiers) below.

## Forecast targets

| Target | Definition | Season used |
|---|---|---|
| M1 | Water level at Tan Chau 1, 2, 3, 5, 7 and 14 days ahead | all year |
| M2 | Number of days with water level > 2.5 m in the next 10 days | Aug–Nov |
| M3b | Days remaining until the end of the flood season (last day ≥ 2.0 m) | Sep–Dec |
| M4 | Probability that water level falls below 0.5 m in 7 and 15 days | Jan–May |

M3a (days until flood-season onset) was trained with the same pipeline but discarded: its cross-validated RMSE (≈ 28 days) is too large to be useful. Its test predictions remain in `results/M3/`, but no model files are kept.

## Repository layout

```
tanchau-flood-chain/
├── config.yaml                 Shared settings (relative paths, thresholds, horizons, tuning)
├── requirements.txt            Library versions verified for reproduction
├── src/
│   ├── data_collection/
│   │   ├── gldas/              Download GLDAS-2.1 3-hourly → daily means over the upstream basin
│   │   ├── earth_engine/       IMERG V07 + ERA5-Land via Google Earth Engine (reconstructed code, see its README)
│   │   └── merge_and_validation/  Build the 28-variable and 35-column tables; validate the data package
│   ├── preprocessing/
│   │   └── preprocess_features.py  Gap-fill water level → feature engineering → keep 77 features → train/valid/test split
│   └── training/
│       ├── common.py, pipeline.py  Shared functions: data loading, tuning, Blend, model saving
│       ├── make_targets.py         Build targets for M1–M4
│       ├── train_m1.py … train_m4.py
│       ├── summarize.py            Summarise results and draw figures
│       └── kiem_tra_model_da_luu.py  Reload saved models and compare predictions with results/
├── data/                       See data/README.md (part of the data cannot be redistributed)
├── models/                     Per target and horizon: feature_list, blend_weights, threshold, training_log (.json)
├── results/                    Test-set (2023–2024) metrics and feature importance
└── docs/data_provenance/       Provenance, checksums and reproducibility checks of the data package
```

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run every command from the repository root.

```bash
# 0. (Optional) Re-collect the inputs – see the README in each src/data_collection/* folder
python src/data_collection/merge_and_validation/validate_package.py

# 1. Preprocessing: data/final/*_35cols.csv → data/processed/{train,valid,test}_final.csv
python src/preprocessing/preprocess_features.py

# 2. Targets for M1–M4 → data/targets/
python src/training/make_targets.py

# 3a. Quick check of the saved models (no retraining)
python src/training/kiem_tra_model_da_luu.py

# 3b. Retrain (overwrites models/ and results/)
python src/training/train_m1.py
python src/training/train_m2.py
python src/training/train_m3.py
python src/training/train_m4.py
python src/training/summarize.py
```

A different config file can be passed as an argument: `python src/training/train_m1.py my_config.yaml`.

## Main settings

| Item | Value |
|---|---|
| Data split | train 2015–2021 (2,557 days), validation 2022 (365), test 2023–2024 (731) |
| Features | 77, derived from 34 raw variables; all rolling windows and lags look backward only |
| Algorithms | DT, RF, LightGBM, CatBoost, SVR (M1–M3) / SVC (M4) |
| Tuning | RandomizedSearchCV, 20 candidates, TimeSeriesSplit with 5 folds, gap 14, seed 42 |
| Blend | Non-negative weights summing to 1, fitted with SLSQP on the validation set |
| M4 | Probability threshold chosen on the validation set to maximise CSI |

Note: in `models/M4_han/`, the file `svr_clf.pkl` is the SVC model (the name produced by the code is kept).

## Reproducibility check (October 2026)

With the versions in `requirements.txt`, steps 1–2 produce exactly the 77 features listed in `feature_list.json`, and the gap-filled table matches the file used by the team to floating-point precision (differences ≤ 2e-16, from parallel random-forest summation). `kiem_tra_model_da_luu.py` reproduces the predictions in `results/` for DT, RF, CatBoost and SVR (differences < 1e-11) and for every model of M2, M3 and M4. M1 LightGBM differs on 2–8 of about 730 days (at most 0.009 m; RMSE identical to four decimals), most likely because some values fall exactly on a split threshold on a different platform.

## Glossary of Vietnamese identifiers

**Config keys (`config.yaml`)**

| Identifier | Meaning |
|---|---|
| `duong_dan` | paths (`processed_dir`, `targets_dir`, `model_dir`, `ket_qua_dir` = results, `hinh_anh_dir` = figures) |
| `cot` | column names: `muc_nuoc` = water level, `muc_nuoc_hien_tai` = current water level (persistence baseline) |
| `nguong` | thresholds: `bao_dong_m` = Alarm Level I (m, unused), `m2_vuot_nguong_m` = M2 threshold, `mua_lu_m` = flood-season threshold (M3), `han_m` = low-water threshold (M4) |
| `mua_vu` | seasons: `thang_mua_lu` = flood-season months (M2), `thang_mua_lu_rong` = extended flood season (M3), `thang_mua_kho` = dry-season months (M4) |
| `cua_so_ngay` | window length in days (M2) |
| `chia_du_lieu` | data split: `train_den` / `valid_den` = end date of training / validation |
| `thuat_toan` | algorithms to train |
| `scoring_hoi_quy` / `scoring_phan_loai` | tuning metric for regression / classification |

**Columns in data and results**

| Identifier | Meaning |
|---|---|
| `H_TanChau`, `H_hientai` | water level at Tan Chau; current-day water level |
| `H_lagN`, `dH_Nngay` | water level N days earlier; change over N days |
| `giatoc_H_Nngay` | acceleration of water level over N days |
| `*_tongN`, `so_ngay_mua_N` | N-day rolling sum; number of rainy days in N days |
| `dQ_*`, `dSoilMoi*` | change in discharge / soil moisture |
| `doy_sin`, `doy_cos` | seasonal terms of day of year |
| `la_mua_lu_8_11`, `pha_lu` | flood-season flag (Aug–Nov); flood phase |
| `target_ngayN` | M1 target: water level N days ahead |
| `so_ngay_tren_nguong_10` | M2 target: days above threshold in the next 10 days |
| `ngay_den_batdau`, `ngay_den_ketthuc` | M3a / M3b targets: days to onset / end of flood season |
| `han_7`, `han_15` | M4 targets: low water (< 0.5 m) after 7 / 15 days |
| `thuc_te` | observed value |
| `du_bao_<model>`, `xac_suat_<model>` | forecast / predicted probability of a model |
| `TanSuatLichSu` | historical-frequency baseline (M4) |
| `hang`, `bai_toan` | rank; task name |
| `cai_thien_vs_baseline_%` | improvement over the baseline (%) |
| `nguong` (in M4 metrics) | probability threshold |
| `hit`, `miss`, `false_alarm`, `correct_neg` | contingency-table counts |
| `model_tot_nhat`, `chi_so_chinh`, `gia_tri` | best model, main metric, value |

**Files and folders**

| Identifier | Meaning |
|---|---|
| `M1_mucnuoc/ngayNN` | M1 (water level), horizon NN days |
| `M2_soNgayVuotNguong` | M2 (days above threshold) |
| `M3_muaLu/ketthuc` | M3b (end of flood season) |
| `M4_han/hanNN` | M4 (low water), horizon NN days |
| `metrics_theo_horizon.csv`, `xep_hang_model.csv` | metrics by horizon; model ranking |
| `tong_hop_4_model.csv` | summary of the four targets |
| `du_bao_test_*.csv` | test-set predictions (not in the public repository) |
| `kiem_tra_model_da_luu.py` | "check saved models" |
| `hoi_quy` / `phan_loai` | regression / classification |

## Data and licences

- **Code**: MIT (see `LICENSE`).
- **Tan Chau water level (MRC)**: distributed under the Mekong River Commission's non-commercial licence and **may not be redistributed** without written permission. This repository therefore does not contain the water-level series, any table with an `H_TanChau` column, the trained models (`*.pkl`), or predictions that include observed values. Users can obtain the series from the [MRC Data Portal](https://portal.mrcmekong.org) and place it under `data/` as described in `data/README.md`.
- **Other open data** (GloFAS, GLDAS-2.1, IMERG V07, ERA5-Land, HydroBASINS): see `data/README.md` for sources and licences.

## Citation

See `CITATION.cff`.
