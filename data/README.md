# Data

> Bản tiếng Việt: [README.vi.md](README.vi.md)

Common period: 1 January 2015 – 31 December 2024 (3,653 days). Hydrological day 07:00–07:00 Vietnam time (00:00–24:00 UTC).

| Folder | Content | In the public repository |
|---|---|---|
| `raw/earth_engine/` | Daily IMERG V07 + ERA5-Land (19 columns) | Yes |
| `raw/mrc_portal/` | Tan Chau water level downloaded manually from the MRC Data Portal (station VN_019803) | **No** – MRC licence |
| `intermediate/earth_engine/yearly_reconstructed/` | Yearly files split from the raw table (see the README in that folder) | Yes |
| `intermediate/gldas/` | 21 GLDAS-2.1 variables, daily means over the upstream basin | Yes |
| `intermediate/glofas/` | GloFAS reanalysis discharge at six upstream stations | Yes |
| `intermediate/mrc/` | `TanChau_daily.csv`: daily mean water level | **No** |
| `merged/` | 28-variable table (GLDAS + GloFAS + water level) | **No** (contains `H_TanChau`) |
| `final/` | 35-column table – input to `src/preprocessing/preprocess_features.py` | **No** |
| `processed/` | `train_final.csv`, `valid_final.csv`, `test_final.csv` (created in step 1) | **No** |
| `targets/` | M1–M4 targets (created in step 2) | **No** |
| `support/manifest/` | List of GLDAS URLs used for download | Yes |
| `support/spatial/` | Tan Chau upstream basin, HydroBASINS level 8 (GeoJSON) | Yes |
| `descriptions/` | Variable descriptions (19 raw columns, 28 variables, 34 model variables) and feature dictionary (Vietnamese) | Yes |

## Rebuilding the MRC-dependent files

1. Download the Tan Chau water-level series (station VN_019803) from https://portal.mrcmekong.org and accept the MRC terms of use.
2. Compute daily means over the 07:00–07:00 hydrological day and save them as `intermediate/mrc/TanChau_daily.csv` (columns `date`, `H_TanChau`).
3. Run `src/data_collection/merge_and_validation/merge_final_28vars.py`, then `merge_extended_35cols.py`, and rename the `*_rebuilt.csv` outputs to the standard names (or use `src/data_collection/gldas/merge_final.py` for the 28-variable table).

## Not yet included

- The scripts that download the six GloFAS stations and turn the MRC series into daily means; only their intermediate outputs are provided.
- The Earth Engine code is a reconstruction and is not claimed to be byte-identical to the original notebook.

## Sources

- GloFAS-ERA5 v4 (Harrigan et al. 2020), Copernicus Emergency Management Service – CC BY 4.0
- GLDAS Noah 2.1 3-hourly (Beaudoing & Rodell 2020), NASA GES DISC – requires an Earthdata account
- GPM IMERG Final V07 (Huffman et al. 2023), NASA GES DISC
- ERA5-Land (Muñoz-Sabater et al. 2021), Copernicus Climate Change Service
- HydroBASINS (Lehner & Grill 2013)
- MRC Data Portal – non-commercial licence, no redistribution
