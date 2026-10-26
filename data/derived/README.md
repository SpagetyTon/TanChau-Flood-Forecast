# data/derived — gridded inputs of the flood and exposure model

`grids_public.npz` holds the 1024 x 1024 grid (104.95–105.55 E, 10.50–11.08 N, cells ≈ 64 m) used by
`src/analysis/flood_model.py`. All layers come from openly licensed sources and may be redistributed with attribution:

| Array | Content | Source / licence |
|---|---|---|
| `elev` | terrain height (m), obstacle-filtered | Copernicus GLO-30 DEM (ESA, Copernicus licence) |
| `occ`, `lost` | % of Landsat observations with water 1984–2021; "lost seasonal" flag | JRC Global Surface Water v1.4 (CC BY 4.0) |
| `pop` | people per cell, 2020, constrained 100 m redistributed to the grid | WorldPop (CC BY 4.0) |
| `river` | river / canal mask (2 = main river, 1 = canal) | © OpenStreetMap contributors (ODbL) |
| `ghs_pop_2020_crop.npz` | GHS-POP R2023A 2020 (3 arc-second) cropped to the study area, used in E5 | GHSL, European Commission JRC (CC BY 4.0) |
| `sentinel1_water_masks.npz` | Water masks of 9 Sentinel-1 VV scenes (E3), rebuilt from the Planetary Computer archive | Copernicus Sentinel data 2022–2025 (free and open) |
| `adm`, `adm_names` | commune id per cell (0 = outside), 54 communes after the 2025 reorganisation | vietnamese-provinces-database (MIT) |

The array was extracted from the project's demo (`convert_demo_grids` in `flood_model.py`).

`../mrc_derived/calib.json` holds the calibration derived from MRC gauge data: the stage
exceedance curve 1984–2021, water-surface gradients between Tan Chau, Chau Doc, My Thuan and Can Tho, annual maxima and
Kratie discharge classes. See `../mrc_derived/README.md` for its licence status.
