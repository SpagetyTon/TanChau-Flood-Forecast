> **Ghi chú sau khi sắp xếp lại kho (10/2026):** `config.json` nay đọc manifest và lưu vực từ `data/support/`, ghi `GLDAS_daily.csv` vào `data/intermediate/gldas/`; `merge_final.py` đọc GloFAS và mực nước từ `data/intermediate/` và ghi bảng 28 biến vào `data/merged/`. Các thư mục `output/`, `processed/`, `manifest/`, `spatial/` nhắc tới bên dưới không còn nằm trong thư mục này.

# MEKONG DAILY PIPELINE — 2015–2024

Bộ này được chuẩn bị cho dataset dự báo mực nước Tân Châu.

## Những gì đã có sẵn trong ZIP

- `processed/GloFAS_6stations_daily.csv`
  - 3,653 ngày: 2015-01-01 → 2024-12-31
  - 6 predictor discharge
  - không thiếu ngày
- `processed/TanChau_daily.csv`
  - 3,653 ngày
  - target `H_TanChau`
  - daily mean theo **hydrological day 07:00 → 07:00 ICT**
  - còn 4 ngày target bị thiếu và được giữ `NaN`, không tự nội suy
- `spatial/tanchau_upstream_hydrobasins_lev08.geojson`
  - upstream basin của Tân Châu đã trace từ HydroBASINS Level 8
- `manifest/subset_GLDAS_NOAH025_3H_2.1_20260904_154257_.txt`
  - manifest GLDAS đã kiểm tra: 29,224 URL, 2015–2024, 8 timestep/ngày, 21 biến

## GLDAS còn phải làm gì?

GLDAS cần Earthdata Login nên không thể nhúng sẵn các giá trị `.nc4` vào ZIP này.
Script `run_gldas_to_daily.py` sẽ:

1. đọc 29,224 URL;
2. tải từng ngày 8 file 3-hourly;
3. dùng upstream basin Tân Châu để area-weighted spatial aggregation;
4. chuyển 3-hourly → daily theo đúng loại biến;
5. xóa 8 file raw của ngày đó sau khi xử lý để tiết kiệm ổ cứng;
6. ghi kết quả từng ngày vào `output/GLDAS_daily_partial.csv`;
7. có thể resume nếu mạng/máy bị dừng;
8. hoàn tất thành `output/GLDAS_daily.csv`.

### Quy tắc 3-hourly → daily

- `Qs_acc`, `Qsb_acc`: SUM 8 timestep.
- `Rainf_f_tavg`, `Evap_tavg`: `value × 10800 giây`, rồi SUM.
- 17 biến còn lại: daily mean.

## Cách chạy trên Windows

### Bước 1 — test 1 ngày trước
Double-click:

`RUN_WINDOWS_TEST_1_DAY.bat`

Script sẽ cài thư viện, hỏi Earthdata username/password và chỉ xử lý 1 ngày.
Password **không được lưu vào bộ pipeline**.

Nếu test thành công, bạn sẽ thấy:

`output/GLDAS_daily_partial.csv`

### Bước 2 — chạy toàn bộ
Double-click:

`RUN_WINDOWS_FULL.bat`

Nó sẽ tự resume từ ngày đã test và chạy đến hết 2024.

Khi xong sẽ có:

- `output/GLDAS_daily.csv`
- `output/GLDAS_daily_metadata.csv`
- `output/Mekong_TanChau_2015_2024.csv`
- `output/final_quality_report.txt`

## Cấu trúc file cuối

- 3,653 rows
- 1 cột `date`
- 21 GLDAS predictors
- 6 GloFAS predictors
- 1 MRC Tân Châu target

=> **27 predictors + 1 target = 28 model variables**
=> tổng số cột CSV = **29**, tính cả `date`.

## Ngày thiếu target Tân Châu

Do dùng hydrological day 07:00–07:00 ICT, các ngày target thiếu là:

- 2019-03-26
- 2019-03-27
- 2019-03-28
- 2024-11-12

Giữ nguyên `NaN`. Không tự nội suy trước khi quyết định phương pháp modeling.

## Lưu ý Earthdata

Nếu script báo tải về HTML thay vì NetCDF:
- kiểm tra Earthdata username/password;
- đăng nhập Earthdata trên trình duyệt;
- đảm bảo đã pre-authorize ứng dụng GES DISC cho tài khoản.

Pipeline không lưu username/password.

## LƯU Ý BẢO MẬT CHO BẢN ĐÓNG GÓI

Bản trong package này đã được làm sạch credential. `run_gldas_to_daily.py` đọc username/password từ biến môi trường `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, hoặc hỏi trực tiếp khi chạy. Không có mật khẩu/tokens được lưu trong source code.

## V3 — EARTHDATA BEARER TOKEN

Ảnh Authorized Apps cho thấy `GESDISC` và `NASA GESDISC DATA ARCHIVE` đã được authorize.
Vì vậy V3 không còn mặc định coi 401 là lỗi chưa authorize.

V3 làm 2 bước tách biệt:

1. Xác minh username/password trực tiếp với API chính thức của Earthdata.
2. Nếu PASS, tự lấy EDL Bearer Token và dùng token đó để tải GLDAS.

### Test riêng tài khoản

Chạy:
`TEST_EARTHDATA_CREDENTIALS.bat`

- Nếu **PASS**: username/password đúng; tiếp tục chạy `RUN_WINDOWS_TEST_1_DAY.bat`.
- Nếu **FAIL HTTP 401**: kiểm tra lại chính xác username trong Earthdata Profile và thử logout/login lại trên browser.

Bản package đã loại bỏ password nhúng; token chỉ được tạo trong lúc chạy.
