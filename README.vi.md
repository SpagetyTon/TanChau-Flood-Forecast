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

Với các phiên bản trong `requirements.txt`, chạy bước 1–2 cho ra đúng 77 đặc trưng như `feature_list.json`, và tệp điền khuyết trùng với tệp nhóm đã dùng tới sai số làm tròn (≤ 2e-16, do rừng ngẫu nhiên tính song song). `kiem_tra_model_da_luu.py` cho dự báo trùng với `results/` ở DT, RF, CatBoost, SVR (sai khác < 1e-11) và ở mọi mô hình của M2, M3, M4. LightGBM của M1 lệch ở 2–8 trên khoảng 730 ngày (tối đa 0,009 m; RMSE trùng tới 4 chữ số), nhiều khả năng do giá trị nằm đúng ngưỡng tách cây trên nền tảng khác.

## Dữ liệu và giấy phép

- **Mã nguồn**: MIT (xem `LICENSE`).
- **Mực nước Tân Châu (MRC)**: giấy phép phi thương mại của Ủy hội sông Mê Kông, **không được phân phối lại** khi chưa có văn bản cho phép. Kho này không chứa chuỗi mực nước, các bảng có cột `H_TanChau`, mô hình đã huấn luyện (`*.pkl`) hay dự báo có cột thực đo. Người dùng tự tải từ [MRC Data Portal](https://portal.mrcmekong.org) rồi đặt vào `data/` theo hướng dẫn ở `data/README.md`.
- **Dữ liệu mở khác** (GloFAS, GLDAS-2.1, IMERG V07, ERA5-Land, HydroBASINS): theo giấy phép của từng nguồn, xem `data/README.md`.

## Trích dẫn

Xem `CITATION.cff`.
