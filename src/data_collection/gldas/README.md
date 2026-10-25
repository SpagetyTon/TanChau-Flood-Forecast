# GLDAS daily pipeline — 2015–2024

> Vietnamese version: [README_VI.md](README_VI.md)

This folder builds the daily GLDAS-2.1 predictors for the Tan Chau water-level dataset and merges them with the GloFAS discharge and the Tan Chau water level into the 28-variable table.

> **Paths after the repository reorganisation (October 2026):** `config.json` reads the manifest and the upstream basin from `data/support/` and writes `GLDAS_daily.csv` to `data/intermediate/gldas/`; `merge_final.py` reads GloFAS and the water level from `data/intermediate/` and writes the 28-variable table to `data/merged/`. The `output/`, `processed/`, `manifest/` and `spatial/` folders of the original package no longer live in this folder.

## Inputs already provided

- `data/intermediate/glofas/GloFAS_6stations_daily.csv` — 3,653 days (2015-01-01 → 2024-12-31), 6 discharge predictors, no missing days.
- `data/intermediate/mrc/TanChau_daily.csv` — 3,653 days, target `H_TanChau`, daily mean over the **hydrological day 07:00 → 07:00 ICT**; 4 days are missing and kept as `NaN` (no interpolation). *Not in the public repository (MRC licence).*
- `data/support/spatial/tanchau_upstream_hydrobasins_lev08.geojson` — Tan Chau upstream basin traced from HydroBASINS level 8.
- `data/support/manifest/subset_GLDAS_NOAH025_3H_2.1_20260904_154257_.txt` — checked GLDAS manifest: 29,224 URLs, 2015–2024, 8 time steps per day, 21 variables.

## What `run_gldas_to_daily.py` does

GLDAS requires an Earthdata Login, so the `.nc4` files cannot be bundled. The script:

1. reads the 29,224 URLs;
2. downloads the eight 3-hourly files of each day;
3. computes area-weighted means over the Tan Chau upstream basin;
4. converts 3-hourly values to daily values according to the variable type;
5. deletes the eight raw files of the day after processing to save disk space;
6. appends each day to `GLDAS_daily_partial.csv`;
7. can resume after an interruption;
8. finishes with `GLDAS_daily.csv` and `GLDAS_daily_metadata.csv`.

### 3-hourly → daily rules

- `Qs_acc`, `Qsb_acc`: sum of the 8 time steps.
- `Rainf_f_tavg`, `Evap_tavg`: `value × 10800 s`, then summed.
- The other 17 variables: daily mean.

## Running

Credentials are read from the environment variables `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, or prompted at run time. **Nothing is stored in the code.** The script validates the username/password with the official Earthdata API and then obtains an EDL Bearer token for the GES DISC downloads.

```bash
pip install -r requirements.txt
python TEST_EARTHDATA_CREDENTIALS.py      # optional: check the account
python run_gldas_to_daily.py --max-days 1 # test with one day
python run_gldas_to_daily.py              # full run (resumes automatically)
python merge_final.py                     # build the 28-variable table
```

On Windows, the same steps are available as `TEST_EARTHDATA_CREDENTIALS.bat`, `RUN_WINDOWS_TEST_1_DAY.bat` and `RUN_WINDOWS_FULL.bat`.

## Final table

3,653 rows × 29 columns: `date`, 21 GLDAS predictors, 6 GloFAS predictors and the MRC Tan Chau target, i.e. **27 predictors + 1 target = 28 model variables**.

Missing target days (hydrological day 07:00–07:00 ICT): 2019-03-26, 2019-03-27, 2019-03-28, 2024-11-12. They are kept as `NaN` here and filled later in `src/preprocessing/preprocess_features.py`.

## Earthdata troubleshooting

If the download returns HTML instead of NetCDF:

- check the Earthdata username (the profile username, not the e-mail address) and password;
- log in to Earthdata in a browser;
- make sure the GES DISC application is authorised for the account (Profile → Authorized Apps should list `GESDISC` and `NASA GESDISC DATA ARCHIVE`);
- open one GLDAS link from the manifest in the same browser session, accept any GES DISC prompt, and rerun.

## Security note

An earlier version of this pipeline contained hard-coded Earthdata credentials. They were removed before packaging; the current scripts only create a token at run time.
