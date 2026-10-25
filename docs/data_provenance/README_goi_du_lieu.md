# MEKONG – TÂN CHÂU 2015–2024 — DATA + CODE PACKAGE

Package được gom từ các file người dùng đã cung cấp trong Project **SL | PTHT**, cộng với phần code Earth Engine được **phục dựng có đánh dấu rõ** vì notebook Colab gốc không còn được xác định/tải lên.

## 1. Dataset chính

| File | Kích thước | Vai trò |
|---|---:|---|
| `data/raw/earth_engine/dataset_raw_2015_2024.csv.gz` | 3653 × 19 | IMERG V07 + ERA5-Land raw daily table |
| `data/intermediate/gldas/GLDAS_daily.csv` | 3653 × 22 | 21 GLDAS predictors + date |
| `data/intermediate/glofas/GloFAS_6stations_daily.csv` | 3653 × 7 | 6 upstream GloFAS discharge predictors + date |
| `data/intermediate/mrc/TanChau_daily.csv` | 3653 × 2 | MRC target `H_TanChau` + date |
| `data/merged/Mekong_TanChau_2015_2024_28vars.csv` | 3653 × 29 | GLDAS + GloFAS + MRC: 27 predictors + target + date |
| `data/final/Mekong_TanChau_2015_2024_extended_35cols.csv` | 3653 × 35 | Final extended: 33 predictors + `H_TanChau` + date |

## 2. Dòng dữ liệu

`IMERG + ERA5-Land -> dataset_raw_2015_2024.csv.gz`

`GLDAS 3-hourly manifest -> daily area-weighted aggregation -> GLDAS_daily.csv`

`GLDAS_daily + GloFAS_6stations_daily + TanChau_daily -> 28-variable model table`

`28-variable table + 6 selected raw Earth Engine variables (3 IMERG + dewpoint + U/V wind) -> 35-column extended final table`

## 3. Nhiều dataset trung gian

`data/intermediate/earth_engine/yearly_reconstructed/` chứa 20 file theo năm (`imerg_YYYY.csv`, `era5_YYYY.csv`). Chúng được tách từ file raw 2015–2024 đã upload, vì các export theo năm gốc không còn được cung cấp. README trong thư mục này ghi rõ provenance.

Các intermediate thực tế từ pipeline V3 được giữ nguyên: GloFAS 6 trạm, station mapping, Tân Châu daily, Tân Châu QA, GLDAS daily và GLDAS metadata.

## 4. Code

- `code/source_v3_sanitized/`: pipeline V3 Bearer Token do người dùng upload, **đã loại bỏ credential hard-code**. Logic GLDAS/merge còn nguyên về chức năng.
- `code/earth_engine_reconstructed/`: code phục dựng cho IMERG V07 + ERA5-Land. Không khẳng định byte-for-byte giống notebook Colab cũ.
- `code/merge_and_validation/`: script merge raw, merge 28-variable, merge extended 35-column và validate package.

## 5. Bảo mật

ZIP V3 người dùng upload có credential Earthdata hard-code trong source. Credential đó **không được đưa vào package này**. Bản sanitized hỏi password khi chạy hoặc đọc từ `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

## 6. Kiểm tra tái lập

Xem `docs/reproducibility_check.csv`. Các script rebuild đã được chạy trong lúc tạo package để kiểm tra shape/cột và sai khác số học.

## 7. Thời gian và target

- Common date window: `2015-01-01` → `2024-12-31` = 3,653 ngày.
- GLDAS hydrological day trong config: `07:00 ICT -> 07:00 ICT`, tương đương `00:00–24:00 UTC`.
- `H_TanChau` còn 4 ngày thiếu trong bảng 28-variable: 2019-03-26, 2019-03-27, 2019-03-28, 2024-11-12; giữ `NaN` theo pipeline.
