# Data provenance — Mekong – Tan Chau 2015–2024 data and code package

> Vietnamese version: [README_goi_du_lieu.md](README_goi_du_lieu.md)

This package was assembled from the files supplied by the team, plus Earth Engine collection code that was **explicitly reconstructed**, because the original Colab notebook could no longer be located. Paths below refer to the reorganised repository.

## 1. Main datasets

| File | Size | Role |
|---|---:|---|
| `data/raw/earth_engine/dataset_raw_2015_2024.csv.gz` | 3653 × 19 | Raw daily IMERG V07 + ERA5-Land table |
| `data/intermediate/gldas/GLDAS_daily.csv` | 3653 × 22 | 21 GLDAS predictors + date |
| `data/intermediate/glofas/GloFAS_6stations_daily.csv` | 3653 × 7 | 6 upstream GloFAS discharge predictors + date |
| `data/intermediate/mrc/TanChau_daily.csv` | 3653 × 2 | MRC target `H_TanChau` + date (not public) |
| `data/merged/Mekong_TanChau_2015_2024_28vars.csv` | 3653 × 29 | GLDAS + GloFAS + MRC: 27 predictors + target + date (not public) |
| `data/final/Mekong_TanChau_2015_2024_extended_35cols.csv` | 3653 × 35 | Final extended table: 33 predictors + `H_TanChau` + date (not public) |

## 2. Data flow

`IMERG + ERA5-Land -> dataset_raw_2015_2024.csv.gz`

`GLDAS 3-hourly manifest -> daily area-weighted aggregation -> GLDAS_daily.csv`

`GLDAS_daily + GloFAS_6stations_daily + TanChau_daily -> 28-variable model table`

`28-variable table + 6 selected raw Earth Engine variables (3 IMERG rainfall series + dew point + U/V wind) -> 35-column extended table`

## 3. Intermediate datasets

`data/intermediate/earth_engine/yearly_reconstructed/` contains 20 yearly files (`imerg_YYYY.csv`, `era5_YYYY.csv`). They were split from the uploaded 2015–2024 raw file because the original yearly exports were not available; the README in that folder documents this.

The real intermediate outputs of the V3 pipeline are kept unchanged: GloFAS 6 stations, station mapping, Tan Chau daily, Tan Chau QA, GLDAS daily and GLDAS metadata.

## 4. Code

- `src/data_collection/gldas/`: the V3 Bearer-token pipeline supplied by the team, **with hard-coded credentials removed**. The GLDAS and merge logic is functionally unchanged.
- `src/data_collection/earth_engine/`: reconstructed code for IMERG V07 + ERA5-Land. It is not claimed to be byte-identical to the original Colab notebook.
- `src/data_collection/merge_and_validation/`: scripts to merge the raw table, the 28-variable table and the 35-column table, and to validate the package.

## 5. Security

The V3 ZIP originally supplied contained hard-coded Earthdata credentials. They are **not included** in this package. The cleaned version prompts for the password at run time or reads `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

## 6. Reproducibility check

See `reproducibility_check.csv`. The rebuild scripts were run while assembling the package to check shapes, columns and numerical differences.

## 7. Period and target

- Common window: `2015-01-01` → `2024-12-31` = 3,653 days.
- GLDAS hydrological day in the config: `07:00 ICT -> 07:00 ICT`, i.e. `00:00–24:00 UTC`.
- `H_TanChau` has 4 missing days in the 28-variable table: 2019-03-26, 2019-03-27, 2019-03-28, 2024-11-12; they are kept as `NaN` there.
