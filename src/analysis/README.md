# src/analysis — recomputation of the paper's results (E1–E8)

Run `python -m analysis.reproduce_all` from `src/` (add `--full` for E7 and the Monte Carlo checks, a few minutes).
It recomputes the tables below and compares **every value** with the paper; see "Status" for what matches.

| Module | Experiments |
|---|---|
| `flood_model.py` | Flood map from satellite-calibrated trigger levels, exposure per cell and commune (Table `explev`, E5) |
| `forecast_skill.py` | E1 (RMSE, MAE, bias, NSE, block bootstrap, Diebold–Mariano, alert detection), E2 (M2, M3b, M4), E7 (forecast error → exposure) |
| `sensitivity.py` | E4 (threshold, satellite-frequency, stage, population and DEM uncertainty) and the layer ablation of E8 |
| `population_sources.py` | E5 comparison WorldPop vs GHS-POP R2023A |
| `sar_validation.py` | E3: Sentinel-1 scenes from Microsoft Planetary Computer → water masks (Otsu) → comparison with the map |
| `reproduce_table.py`, `reproduce_all.py` | Checks against the paper |

Notebooks: `notebooks/01_flood_map_and_exposure`, `02_forecast_skill`, `03_sensitivity_and_ablation`, `04_sentinel1_validation`, `05_feature_group_ablation`.

## Status (371 values compared in the default run)

* 281 identical to the paper; 88 close; 2 differ. (On the machine used for the first run the counts were 270 / 84 / 2 over 356 values; the 15 new values are the feature-group ablation, 14 identical and 1 within rounding. Exposure totals can differ by one person because of float summation order across numpy versions.)
* Close: 24 bootstrap confidence limits (within 1 percentage point, random-stream dependent), RF CSI of M4 at 15 days
  (0.593 vs 0.596), six GHS-POP totals (within 0.02 %), and the E3 values (below).
* **E3** was rebuilt from the open Sentinel-1 archive, not from the authors' original masks. IoU agrees within 0.01
  (0.501–0.591 vs 0.498–0.589), the comparison domain is 2,147 km² (paper 2,126), the SAR extent differs by up to
  27 km², commune rank correlations by up to 0.07. The claim that moving the Otsu threshold by ±2 dB changes IoU by
  at most 0.04 holds for five scenes; for 20 Oct 2023 the change is +0.05 / −0.08.
* 2 differ: E8, "DEM only, flat surface" at 4.70 m gives 1,027,917 people (Spearman 0.827); the paper has 1,058,948
  (0.834). The other seven E8 rows are identical.
* **E8 feature-group ablation** (Table S9): `feature_ablation.py`, notebook `05_feature_group_ablation`. It refits M1 with the
  hyper-parameters stored in `models/M1_mucnuoc/*/training_log.json` and needs `data/processed/` (derived from MRC data) and
  scikit-learn, lightgbm, catboost (about 6 min). 14 of the 15 Blend RMSEs equal the paper at four decimals; the fifteenth,
  full model at 7 d, is 0.15694 (paper 0.157). The "soil moisture / GLDAS runoff" group is 22 columns: `SoilMoi*`, `dSoilMoi*`,
  `RootMoist_inst`, `CanopInt_inst`, `Qs_acc*`, `Qsb_acc*`. Results are stored in `results/feature_ablation.csv`.
* Not covered: **E6** (compares the 3-D demo with a recomputation, a property of the demo, not of the model).

Inputs: `data/derived/` (open data), `data/mrc_derived/` (derived from MRC data; read its README).
