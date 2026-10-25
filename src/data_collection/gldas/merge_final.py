#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]  # repository root
gldas_path = REPO / "data/intermediate/gldas/GLDAS_daily.csv"
glofas_path = REPO / "data/intermediate/glofas/GloFAS_6stations_daily.csv"
tanchau_path = REPO / "data/intermediate/mrc/TanChau_daily.csv"
out_path = REPO / "data/merged/Mekong_TanChau_2015_2024_28vars.csv"
qa_path = REPO / "data/merged/final_quality_report_28vars.txt"

if not gldas_path.exists():
    raise SystemExit(
        "GLDAS_daily.csv does not exist yet. Run run_gldas_to_daily.py first."
    )

def read_daily(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise RuntimeError(f"Duplicate dates found in {path.name}")
    return df

gldas = read_daily(gldas_path)
glofas = read_daily(glofas_path)
tanchau = read_daily(tanchau_path)

final = (
    gldas
    .merge(glofas, on="date", how="inner", validate="one_to_one")
    .merge(tanchau, on="date", how="left", validate="one_to_one")
    .sort_values("date")
    .reset_index(drop=True)
)

expected = pd.date_range("2015-01-01", "2024-12-31", freq="D")
if len(final) != len(expected):
    raise RuntimeError(f"Expected {len(expected)} rows, got {len(final)}.")
if not final["date"].equals(pd.Series(expected, name="date")):
    missing = expected.difference(final["date"])
    raise RuntimeError(f"Date alignment failed; missing dates include: {missing[:20].tolist()}")

gldas_cols = [c for c in gldas.columns if c != "date"]
glofas_cols = [c for c in glofas.columns if c != "date"]

report = [
    "MEKONG–TAN CHAU FINAL DATA QUALITY REPORT",
    "==========================================",
    f"Rows: {len(final)}",
    f"Date range: {final.date.min().date()} -> {final.date.max().date()}",
    f"GLDAS predictors: {len(gldas_cols)}",
    f"GloFAS predictors: {len(glofas_cols)}",
    f"Total predictors: {len(gldas_cols) + len(glofas_cols)}",
    "Target columns: 1 (H_TanChau)",
    f"Total model variables: {len(gldas_cols) + len(glofas_cols) + 1}",
    f"Duplicate dates: {int(final.date.duplicated().sum())}",
    f"Missing GLDAS predictor cells: {int(final[gldas_cols].isna().sum().sum())}",
    f"Missing GloFAS predictor cells: {int(final[glofas_cols].isna().sum().sum())}",
    f"Missing H_TanChau days: {int(final['H_TanChau'].isna().sum())}",
]

missing_target_dates = final.loc[final["H_TanChau"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
report.append("Missing H_TanChau dates: " + ", ".join(missing_target_dates))

final["date"] = final["date"].dt.strftime("%Y-%m-%d")
final.to_csv(out_path, index=False)
qa_path.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nDONE: {out_path}")
