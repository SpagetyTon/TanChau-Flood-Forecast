#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
PKG=HERE.parents[3]  # repository root
checks={
 'raw_earth_engine': PKG/'data/raw/earth_engine/dataset_raw_2015_2024.csv.gz',
 'gldas': PKG/'data/intermediate/gldas/GLDAS_daily.csv',
 'glofas': PKG/'data/intermediate/glofas/GloFAS_6stations_daily.csv',
 'tanchau': PKG/'data/intermediate/mrc/TanChau_daily.csv',
 'merged_28vars': PKG/'data/merged/Mekong_TanChau_2015_2024_28vars.csv',
 'extended_35cols': PKG/'data/final/Mekong_TanChau_2015_2024_extended_35cols.csv',
}
for name,p in checks.items():
    d=pd.read_csv(p)
    print(f'{name:18s} rows={len(d):4d} cols={len(d.columns):2d} missing={int(d.isna().sum().sum())}')
assert pd.read_csv(checks['raw_earth_engine']).shape==(3653,19)
assert pd.read_csv(checks['gldas']).shape==(3653,22)
assert pd.read_csv(checks['merged_28vars']).shape==(3653,29)
assert pd.read_csv(checks['extended_35cols']).shape==(3653,35)
print('PASS')
