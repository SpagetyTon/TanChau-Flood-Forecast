# Chuỗi dự báo mực nước – bản đồ ngập – dân số phơi nhiễm quanh trạm Tân Châu

> English version: [README.md](README.md)

Mã nguồn đi kèm bài báo *"Từ mực nước tại trạm đến dân số phơi nhiễm theo xã/phường: đánh giá chuỗi dự báo – bản đồ ngập – phơi nhiễm quanh trạm Tân Châu, đồng bằng sông Cửu Long"*.

Kho này gồm phần **thu thập dữ liệu**, **tiền xử lý và tạo đặc trưng**, và **huấn luyện bốn chỉ tiêu dự báo M1–M4** (5 thuật toán + Blend).

> Phiên bản dùng cho bài báo được gắn tag `v1.0-paper`.

## Cấu trúc thư mục

```
tanchau-flood-chain/
├── config.yaml                 Cấu hình chung (đường dẫn tương đối, ngưỡng, tầm dự báo, tinh chỉnh)
├── requirements.txt            Phiên bản thư viện đã kiểm tra
├── src/
│   ├── data_collection/
│   │   ├── gldas/              Tải GLDAS-2.1 3 giờ → trung bình ngày theo lưu vực thượng nguồn
│   │   ├── earth_engine/       IMERG V07 + ERA5-Land qua Google Earth Engine (mã phục dựng, xem README trong thư mục)
│   │   └── merge_and_validation/  Ghép bảng 28 biến, 35 cột và kiểm tra gói dữ liệu
│   ├── preprocessing/
│   │   └── preprocess_features.py  Điền khuyết mực nước → tạo đặc trưng → lọc còn 77 đặc trưng → tách train/valid/test
│   └── training/
│       ├── common.py, pipeline.py  Hàm dùng chung: tải dữ liệu, tinh chỉnh, Blend, lưu mô hình
│       ├── make_targets.py         Tạo nhãn M1–M4
│       ├── train_m1.py … train_m4.py
│       ├── summarize.py            Tổng hợp kết quả, vẽ hình
│       └── kiem_tra_model_da_luu.py  Tải mô hình đã lưu và so dự báo với results/
├── data/                       Xem data/README.md (một phần không được đưa lên kho công khai)
├── models/                     Mô hình theo chỉ tiêu và tầm: feature_list, blend_weights, threshold, training_log (.json)
├── results/                    Chỉ số đánh giá trên tập kiểm tra 2023–2024, độ quan trọng đặc trưng
└── docs/data_provenance/       Nguồn gốc dữ liệu, checksum, kiểm tra tái lập của gói dữ liệu
```

## Cài đặt
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

## Chạy lại

Mọi lệnh chạy từ thư mục gốc repo.

```bash
# 0. (Tùy chọn) Thu thập lại dữ liệu đầu vào – xem README trong từng thư mục src/data_collection/*
python src/data_collection/merge_and_validation/validate_package.py

# 1. Tiền xử lý: data/final/*_35cols.csv → data/processed/{train,valid,test}_final.csv
python src/preprocessing/preprocess_features.py

# 2. Tạo nhãn M1–M4 → data/targets/
python src/training/make_targets.py

# 3a. Kiểm tra nhanh mô hình đã lưu (không huấn luyện lại)
python src/training/kiem_tra_model_da_luu.py

# 3b. Huấn luyện lại (ghi đè models/ và results/)
python src/training/train_m1.py
python src/training/train_m2.py
python src/training/train_m3.py
python src/training/train_m4.py
python src/training/summarize.py
```

Có thể truyền đường dẫn config khác làm tham số: `python src/training/train_m1.py config_khac.yaml`.

## Thiết lập chính

| Mục | Giá trị |
|---|---|
| Chia dữ liệu | train 2015–2021 (2.557 ngày), validation 2022 (365), test 2023–2024 (731) |
| Đặc trưng | 77, tạo từ 34 biến gốc; mọi phép trượt và trễ chỉ nhìn về quá khứ |
| Thuật toán | DT, RF, LightGBM, CatBoost, SVR (M1–M3) / SVC (M4) |
| Tinh chỉnh | RandomizedSearchCV, 20 tổ hợp, TimeSeriesSplit 5 phần, gap 14, seed 42 |
| Blend | Trọng số không âm, tổng bằng 1, tìm bằng SLSQP trên validation |
| M4 | Ngưỡng xác suất chọn trên validation để cực đại CSI |

Ghi chú: trong `models/M4_han/`, tệp `svr_clf.pkl` là mô hình SVC (giữ nguyên tên do mã sinh ra).

## Kiểm tra tái lập (10/2026)

Với các phiên bản trong `requirements.txt`, chạy bước 1–2 cho ra đúng 77 đặc trưng như `feature_list.json`, và tệp điền khuyết trùng từng byte với tệp nhóm đã dùng. `kiem_tra_model_da_luu.py` cho dự báo trùng với `results/` ở DT, RF, CatBoost, SVR (sai khác < 1e-11) và ở mọi mô hình của M2, M3, M4. LightGBM của M1 lệch ở 2–8 trên khoảng 730 ngày (tối đa 0,009 m; RMSE trùng tới 4 chữ số), nhiều khả năng do giá trị nằm đúng ngưỡng tách cây trên nền tảng khác.

## Giải thích các định danh tiếng Việt

**Khóa cấu hình (`config.yaml`)**

| Định danh | Ý nghĩa |
|---|---|
| `duong_dan` | Các đường dẫn (`processed_dir`, `targets_dir`, `model_dir`, `ket_qua_dir` = kết quả, `hinh_anh_dir` = hình ảnh) |
| `cot` | Tên cột: `muc_nuoc` = mực nước, `muc_nuoc_hien_tai` = mực nước hiện tại (đường cơ sở duy trì) |
| `nguong` | Các ngưỡng: `bao_dong_m` = báo động cấp I (m, chưa sử dụng), `m2_vuot_nguong_m` = ngưỡng M2, `mua_lu_m` = ngưỡng mùa lũ (M3), `han_m` = ngưỡng hạn (M4) |
| `mua_vu` | Các mùa: `thang_mua_lu` = tháng mùa lũ (M2), `thang_mua_lu_rong` = mùa lũ mở rộng (M3), `thang_mua_kho` = tháng mùa khô (M4) |
| `cua_so_ngay` | Độ dài cửa sổ tính theo ngày (M2) |
| `chia_du_lieu` | Chia dữ liệu: `train_den` / `valid_den` = ngày kết thúc tập huấn luyện / xác thực |
| `thuat_toan` | Các thuật toán cần huấn luyện |
| `scoring_hoi_quy` / `scoring_phan_loai` | Chỉ số tinh chỉnh cho bài toán hồi quy / phân loại |

**Các cột trong dữ liệu và kết quả**

| Định danh | Ý nghĩa |
|---|---|
| `H_TanChau`, `H_hientai` | Mực nước tại Tân Châu; mực nước ngày hiện tại |
| `H_lagN`, `dH_Nngay` | Mực nước N ngày trước; mức thay đổi trong N ngày |
| `giatoc_H_Nngay` | Gia tốc mực nước trong N ngày |
| `*_tongN`, `so_ngay_mua_N` | Tổng trượt N ngày; số ngày mưa trong N ngày |
| `dQ_*`, `dSoilMoi*` | Mức thay đổi lưu lượng / độ ẩm đất |
| `doy_sin`, `doy_cos` | Thành phần mùa vụ theo ngày trong năm |
| `la_mua_lu_8_11`, `pha_lu` | Cờ mùa lũ (tháng 8–11); pha lũ |
| `target_ngayN` | Nhãn M1: mực nước sau N ngày |
| `so_ngay_tren_nguong_10` | Nhãn M2: số ngày vượt ngưỡng trong 10 ngày tiếp theo |
| `ngay_den_batdau`, `ngay_den_ketthuc` | Nhãn M3a / M3b: số ngày đến khi bắt đầu / kết thúc mùa lũ |
| `han_7`, `han_15` | Nhãn M4: mực nước thấp (< 0,5 m) sau 7 / 15 ngày |
| `thuc_te` | Giá trị thực đo |
| `du_bao_<model>`, `xac_suat_<model>` | Dự báo / xác suất dự đoán của mô hình |
| `TanSuatLichSu` | Đường cơ sở tần suất lịch sử (M4) |
| `hang`, `bai_toan` | Thứ hạng; tên bài toán |
| `cai_thien_vs_baseline_%` | Mức cải thiện so với đường cơ sở (%) |
| `nguong` (trong chỉ số M4) | Ngưỡng xác suất |
| `hit`, `miss`, `false_alarm`, `correct_neg` | Số lượng trong bảng liên hợp |
| `model_tot_nhat`, `chi_so_chinh`, `gia_tri` | Mô hình tốt nhất, chỉ số chính, giá trị |

**Tệp và thư mục**

| Định danh | Ý nghĩa |
|---|---|
| `M1_mucnuoc/ngayNN` | M1 (mực nước), dự báo trước NN ngày |
| `M2_soNgayVuotNguong` | M2 (số ngày vượt ngưỡng) |
| `M3_muaLu/ketthuc` | M3b (kết thúc mùa lũ) |
| `M4_han/hanNN` | M4 (mực nước thấp), dự báo trước NN ngày |
| `metrics_theo_horizon.csv`, `xep_hang_model.csv` | Chỉ số theo thời hạn dự báo; xếp hạng mô hình |
| `tong_hop_4_model.csv` | Tổng hợp bốn nhãn |
| `du_bao_test_*.csv` | Dự báo trên tập kiểm tra (không có trong kho công khai) |
| `kiem_tra_model_da_luu.py` | Kiểm tra mô hình đã lưu |
| `hoi_quy` / `phan_loai` | Hồi quy / phân loại |

## Dữ liệu và giấy phép

- **Mã nguồn**: MIT (xem `LICENSE`).
- **Mực nước Tân Châu (MRC)**: giấy phép phi thương mại của Ủy hội sông Mê Kông, **không được phân phối lại** khi chưa có văn bản cho phép. Kho này không chứa chuỗi mực nước, các bảng có cột `H_TanChau`, mô hình đã huấn luyện (`*.pkl`) hay dự báo có cột thực đo. Người dùng tự tải từ [MRC Data Portal](https://portal.mrcmekong.org) rồi đặt vào `data/` theo hướng dẫn ở `data/README.md`.
- **Dữ liệu mở khác** (GloFAS, GLDAS-2.1, IMERG V07, ERA5-Land, HydroBASINS): theo giấy phép của từng nguồn, xem `data/README.md`.

## Trích dẫn

Xem `CITATION.cff`.
