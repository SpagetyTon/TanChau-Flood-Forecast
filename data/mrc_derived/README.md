# data/mrc_derived — files derived from Mekong River Commission (MRC) data

| File | Content |
|---|---|
| `forecast_test.json` | Observed daily-mean Tan Chau stage for 2023-01-01 .. 2024-12-31 and the project's model forecasts (M1–M4) and day-of-year climatology band for the same days |
| `calib.json` | Calibration derived from MRC gauge series: exceedance curve of the Tan Chau stage 1984–2021, water-surface gradients (Tan Chau – Chau Doc – My Thuan – Can Tho, 2000–2026), annual maximum stages 1979–2025, discharge classes at Kratie |

**Source and acknowledgement.** Data provided by the Mekong River Commission (water level, manual, stations
VN_019803 Tan Chau, VN_039801 Chau Doc, VN_039803 Can Tho, VN_019804 My Thuan), obtained under the MRC Non-commercial
Data Use Licence (order mrc20260822084617, 22 August 2026). The raw gauge series are **not** included here; they can
be requested from MRC (mrcdata@mrcmekong.org, https://portal.mrcmekong.org). The files above are products derived
from those data and are to be used for non-commercial purposes only.

**Licence status.** Publication of these derived files in this repository was decided by the repository owner, who
is responsible for it. The licence (clauses 3, 5 and 6) requires written permission from MRC before derived products
are passed to third parties; at the time of writing that permission was requested from MRC but not yet received.
If MRC objects, delete this folder; the code in `src/analysis` then still runs on the public inputs in
`data/derived/` plus a calibration file supplied by the user.

**Also derived from MRC data:** `data/processed/*_final.csv` (features built from the Tan Chau gauge series),
`models/` and `results/feature_ablation.csv` (feature-group ablation, Table S9). The same licence status applies.
