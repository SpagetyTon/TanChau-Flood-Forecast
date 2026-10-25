# Dữ liệu

> English version: [README.md](README.md)

Giai đoạn chung: 01/01/2015 – 31/12/2024 (3.653 ngày). Ngày thủy văn 07:00–07:00 giờ Việt Nam.

| Thư mục | Nội dung | Có trên kho công khai |
|---|---|---|
| `raw/earth_engine/` | IMERG V07 + ERA5-Land theo ngày (19 cột) | Có |
| `raw/mrc_portal/` | Mực nước Tân Châu tải thủ công từ MRC Data Portal (trạm VN_019803) | **Không** – giấy phép MRC |
| `intermediate/earth_engine/yearly_reconstructed/` | Tách theo năm từ bảng raw (xem README trong thư mục) | Có |
| `intermediate/gldas/` | 21 biến GLDAS-2.1, trung bình ngày theo lưu vực thượng nguồn | Có |
| `intermediate/glofas/` | Lưu lượng tái phân tích GloFAS tại 6 trạm thượng nguồn | Có |
| `intermediate/mrc/` | `TanChau_daily.csv`: mực nước trung bình ngày | **Không** |
| `merged/` | Bảng 28 biến (GLDAS + GloFAS + mực nước) | **Không** (có cột `H_TanChau`) |
| `final/` | Bảng 35 cột – đầu vào của `src/preprocessing/preprocess_features.py` | **Không** |
| `processed/` | `train_final.csv`, `valid_final.csv`, `test_final.csv` (sinh ra ở bước 1) | **Không** |
| `targets/` | Nhãn M1–M4 (sinh ra ở bước 2) | **Không** |
| `support/manifest/` | Danh sách URL GLDAS dùng để tải | Có |
| `support/spatial/` | Lưu vực thượng nguồn Tân Châu, HydroBASINS cấp 8 (GeoJSON) | Có |
| `descriptions/` | Mô tả biến (19 cột raw, 28 biến, 34 biến mô hình), từ điển đặc trưng | Có |

## Dựng lại phần dữ liệu MRC

1. Tải chuỗi mực nước trạm Tân Châu (VN_019803) từ https://portal.mrcmekong.org, đồng ý điều khoản sử dụng của MRC.
2. Tính trung bình ngày thủy văn 07:00–07:00 và lưu `intermediate/mrc/TanChau_daily.csv` (cột `date`, `H_TanChau`).
3. Chạy `src/data_collection/merge_and_validation/merge_final_28vars.py` rồi `merge_extended_35cols.py`, đổi tên tệp `*_rebuilt.csv` thành tên chuẩn (hoặc dùng `src/data_collection/gldas/merge_final.py` cho bảng 28 biến).

## Còn thiếu

- Mã tải GloFAS 6 trạm và mã xử lý mực nước MRC thành trung bình ngày chưa có trong kho; hiện chỉ có kết quả trung gian.
- Mã Earth Engine là bản phục dựng, không khẳng định trùng từng byte với notebook gốc.

## Nguồn

- GloFAS-ERA5 v4 (Harrigan et al. 2020), Copernicus EMS – CC BY 4.0
- GLDAS Noah 2.1 3 giờ (Beaudoing & Rodell 2020), NASA GES DISC – cần tài khoản Earthdata
- GPM IMERG Final V07 (Huffman et al. 2023), NASA GES DISC
- ERA5-Land (Muñoz-Sabater et al. 2021), Copernicus C3S
- HydroBASINS (Lehner & Grill 2013)
- MRC Data Portal – giấy phép phi thương mại, không phân phối lại
