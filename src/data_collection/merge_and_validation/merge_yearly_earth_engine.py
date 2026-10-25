#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]  # repository root
YEARLY = PKG / "data/intermediate/earth_engine/yearly_reconstructed"
OUT = PKG / "data/raw/earth_engine/dataset_raw_2015_2024_rebuilt.csv.gz"

imerg = pd.concat([pd.read_csv(YEARLY/f"imerg_{y}.csv") for y in range(2015, 2025)], ignore_index=True)
era5 = pd.concat([pd.read_csv(YEARLY/f"era5_{y}.csv") for y in range(2015, 2025)], ignore_index=True)
raw = imerg.merge(era5, on="ngay", how="inner", validate="one_to_one").sort_values("ngay")
raw.to_csv(OUT, index=False, compression="gzip")
print(OUT, raw.shape)
