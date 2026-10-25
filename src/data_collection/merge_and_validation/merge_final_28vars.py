#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[3]  # thư mục gốc repo
GLDAS = PKG/'data/intermediate/gldas/GLDAS_daily.csv'
GLOFAS = PKG/'data/intermediate/glofas/GloFAS_6stations_daily.csv'
TANCHAU = PKG/'data/intermediate/mrc/TanChau_daily.csv'
OUT = PKG/'data/merged/Mekong_TanChau_2015_2024_28vars_rebuilt.csv'

def rd(p):
    d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); return d
final=(rd(GLDAS).merge(rd(GLOFAS),on='date',how='inner',validate='one_to_one')
       .merge(rd(TANCHAU),on='date',how='left',validate='one_to_one')
       .sort_values('date').reset_index(drop=True))
final['date']=final['date'].dt.strftime('%Y-%m-%d')
final.to_csv(OUT,index=False)
print(OUT, final.shape)
