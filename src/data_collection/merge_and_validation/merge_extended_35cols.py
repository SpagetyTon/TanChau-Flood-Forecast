#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]  # thư mục gốc repo
BASE = PKG/'data/merged/Mekong_TanChau_2015_2024_28vars.csv'
RAW = PKG/'data/raw/earth_engine/dataset_raw_2015_2024.csv.gz'
OUT = PKG/'data/final/Mekong_TanChau_2015_2024_extended_35cols_rebuilt.csv'

base=pd.read_csv(BASE)
raw=pd.read_csv(RAW)
raw=raw.rename(columns={'ngay':'date'})
extra=['date','mua_tanchau','mua_kratie','mua_pakse','diem_suong','gio_u','gio_v']
out=base.merge(raw[extra],on='date',how='left',validate='one_to_one')
out.to_csv(OUT,index=False)
print(OUT, out.shape)
