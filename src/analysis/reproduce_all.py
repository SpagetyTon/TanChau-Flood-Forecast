"""Recompute the paper's tables and compare each value with the published one.

    cd src && python -m analysis.reproduce_all            # fast checks (E1, E2, E4 deterministic, E5, E8)
    cd src && python -m analysis.reproduce_all --full     # also E7 (about one minute) and E4 Monte Carlo

Status per check: OK (equal within rounding), CLOSE (within the stated tolerance: bootstrap and Monte Carlo
quantities depend on the random stream; GHS-POP totals on the regridding details), DIFF (differs from the paper).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .flood_model import FloodModel
from .forecast_skill import Forecasts
from .reproduce_table import PAPER
from .population_sources import e5_ghs_table, ghs_on_grid
from .sar_validation import Sentinel1Validation
from .sensitivity import Sensitivity

ROOT = Path(__file__).resolve().parents[2]
CHECKS = []


def check(name, got, paper, tol=0.0):
    status = "OK" if abs(got - paper) <= max(tol if tol == 0 else 0, 1e-9) or abs(got - paper) <= 0 else \
        ("CLOSE" if abs(got - paper) <= tol else "DIFF")
    CHECKS.append((name, got, paper, status))


def rounded(name, got, paper, nd, tol=None):
    ok = round(got, nd) == paper
    status = "OK" if ok else ("CLOSE" if tol is not None and abs(got - paper) <= tol else "DIFF")
    CHECKS.append((name, round(got, nd + 1), paper, status))


def main(full=False) -> int:
    fm = FloodModel(ROOT / "data/derived/grids_public.npz", ROOT / "data/mrc_derived/calib.json")
    fc = Forecasts(ROOT / "data/mrc_derived/forecast_test.json")

    # E5 / E4 table: exposure by stage
    for exp in PAPER:
        got = fm.table_row(exp[0])
        for k, v in zip(["people", "pct_grid_pop", "land_km2", "communes_with_land", "communes_affected", "people_gt_1m"], exp[2:]):
            check(f"E5 stage {exp[0]:.2f} m {k}", got[k], v, tol=1 if k == "people" else 0)  # +-1 person: float summation order

    # E1: RMSE table
    paper_rmse = {"DT": [.090, .149, .184, .215, .230, .269], "RF": [.082, .115, .139, .178, .192, .216],
                  "LightGBM": [.083, .113, .134, .158, .179, .210], "CatBoost": [.079, .107, .132, .153, .176, .211],
                  "SVR": [.077, .104, .126, .148, .157, .187], "Blend": [.075, .102, .123, .147, .157, .187],
                  "Persistence": [.088, .136, .178, .248, .295, .345], "Climatology": [.238, .239, .239, .239, .239, .240]}
    t = fc.e1_rmse_table()
    for mdl, vals in paper_rmse.items():
        for h, v in zip([1, 2, 3, 5, 7, 14], vals):
            rounded(f"E1 RMSE {mdl} {h} d", t.loc[mdl, f"{h} d"], v, 3)
    b = fc.e1_blend_table()
    paper_blend = [(0.056, .005, .991, 15.2, 68.6), (.080, .010, .984, 25.2, 57.3), (.097, .010, .976, 30.6, 48.3),
                   (.117, .012, .966, 40.7, 38.5), (.126, .015, .962, 46.8, 34.4), (.140, .014, .946, 45.9, 22.2)]
    paper_ci = [((11.8, 19.6), (60.2, 75.5)), ((20.0, 30.9), (46.4, 66.2)), ((24.8, 36.3), (36.3, 58.6)),
                ((35.0, 45.4), (26.5, 48.2)), ((40.9, 51.4), (22.7, 44.6)), ((36.6, 52.6), (13.6, 29.9))]
    for i, h in enumerate([1, 2, 3, 5, 7, 14]):
        r = b.iloc[i]; pb = paper_blend[i]
        rounded(f"E1 Blend MAE {h} d", r["MAE"], pb[0], 3); rounded(f"E1 Blend bias {h} d", r["bias"], pb[1], 3)
        rounded(f"E1 Blend NSE {h} d", r["NSE"], pb[2], 3)
        rounded(f"E1 reduction vs persistence {h} d", r["persist_red_%"], pb[3], 1)
        rounded(f"E1 reduction vs climatology {h} d", r["clim_red_%"], pb[4], 1)
        (plo, phi), (clo, chi) = paper_ci[i]
        check(f"E1 CI lo vs persistence {h} d", r["persist_ci_lo"], plo, 1.0); check(f"E1 CI hi vs persistence {h} d", r["persist_ci_hi"], phi, 1.0)
        check(f"E1 CI lo vs climatology {h} d", r["clim_ci_lo"], clo, 1.5); check(f"E1 CI hi vs climatology {h} d", r["clim_ci_hi"], chi, 1.5)
    a = fc.e1_alert_table()
    for i, (hit, miss, fa, tn, pod, far, csi) in enumerate([(6, 10, 3, 705, .38, .33, .32), (9, 7, 7, 701, .56, .44, .39),
                                                          (18, 5, 0, 701, .78, 0.0, .78), (16, 7, 0, 701, .70, 0.0, .70)]):
        r = a.iloc[i]
        for k, v in zip(["hit", "miss", "false_alarm", "correct_neg"], [hit, miss, fa, tn]):
            check(f"E1 alert row {i + 1} {k}", r[k], v)
        for k, v in zip(["POD", "FAR", "CSI"], [pod, far, csi]):
            rounded(f"E1 alert row {i + 1} {k}", r[k], v, 2)

    # E2
    e2 = fc.e2_table()
    paper_e2 = {"DT": (.98, 7.44, .217, .75, .489, .243, .73, .462), "RF": (.94, 5.21, .178, .83, .518, .177, .84, .596),
                "LightGBM": (.89, 4.51, .152, .87, .557, .178, .84, .598), "CatBoost": (.85, 4.79, .169, .86, .546, .170, .85, .609),
                "SVR": (1.86, 4.42, .166, .83, .589, .166, .83, .613), "Blend": (1.01, 4.46, .151, .87, .545, .166, .83, .613),
                "Baseline": (2.98, 7.68, .210, .76, .558, .213, .79, .606)}
    cols = ["M2 MAE", "M3b MAE", "M4-7d Brier", "M4-7d AUC", "M4-7d CSI", "M4-15d Brier", "M4-15d AUC", "M4-15d CSI"]
    nds = [2, 2, 3, 2, 3, 3, 2, 3]
    for mdl, vals in paper_e2.items():
        for c, v, nd in zip(cols, vals, nds):
            rounded(f"E2 {mdl} {c}", e2.loc[mdl, c], v, nd, tol=0.004)

    # E4 deterministic + Monte Carlo, E8
    s = Sensitivity(fm)
    e4a = s.e4(3.5, n_mc=100 if full else 5)
    for src, pv in [("satellite frequency x0.8", 122652), ("satellite frequency x1.2", 200802),
                    ("stage -0.157 m", 136920), ("stage +0.157 m", 200802)]:
        check(f"E4 3.5 m {src}", round(e4a[e4a.source == src].people_lo.iloc[0]), pv)
    e4b = s.e4(4.7, n_mc=1)
    for src, pv in [("stage -0.157 m", 495697), ("stage +0.157 m", 607942), ("satellite frequency x0.8", 607942)]:
        check(f"E4 4.7 m {src}", round(e4b[e4b.source == src].people_lo.iloc[0]), pv)
    if full:
        r = e4a[e4a.source.str.contains("0.25")].iloc[0]
        check("E4 3.5 m sigma 0.25 people P5", r.people_lo, 152680, 5000); check("E4 3.5 m sigma 0.25 people P95", r.people_hi, 181861, 10000)
        r = e4a[e4a.source.str.contains("population")].iloc[0]
        check("E4 3.5 m population noise P5", r.people_lo, 164237, 300); check("E4 3.5 m population noise P95", r.people_hi, 166383, 300)
        dd = s.e4_depth(4.7, 50)
        check("E4 4.7 m DEM error people>1 m median", dd["median"], 242000, 5000)

    # E5: GHS-POP comparison (paper Table 'e5ghs'); people agree within 0.02 % (regridding detail)
    paper_ghs = {2.5: (28809, 14878, 0.72, 0.71, 6), 3.17: (110324, 65569, 0.81, 0.86, 7), 3.5: (165286, 99750, 0.79, 0.90, 6),
                 4.0: (297524, 199964, 0.84, 0.92, 8), 4.5: (495697, 370071, 0.88, 0.94, 7), 4.7: (607942, 475128, 0.89, 0.94, 7)}
    ghs = ghs_on_grid(ROOT / "data/derived/ghs_pop_2020_crop.npz", fm)
    t5 = e5_ghs_table(fm, ghs, tuple(paper_ghs))
    for _, r in t5.iterrows():
        pw, pg, rp, rs, top = paper_ghs[round(r.level_m, 2)]
        check(f"E5 {r.level_m} m WorldPop people", round(r["WorldPop"]), pw)
        check(f"E5 {r.level_m} m GHS-POP people", round(r["GHS-POP"]), pg, 0.0002 * pg + 1)
        rounded(f"E5 {r.level_m} m rho people", r.rho_people, rp, 2); rounded(f"E5 {r.level_m} m rho share", r.rho_share, rs, 2)
        check(f"E5 {r.level_m} m top-10 overlap", r.top10_overlap, top)

    # E3: Sentinel-1 validation (paper Table 'e3'); the SAR masks were rebuilt from Planetary Computer scenes
    v = Sentinel1Validation(fm, ROOT / "data/derived/sentinel1_water_masks.npz")
    check("E3 comparison domain km2", round(v.domain_km2), 2126, 25)
    paper_e3 = {"2023-10-20": (.539, .700, .831, .605, .246, 552, 759, .54, .28), "2024-10-02": (.589, .741, .748, .735, .415, 713, 726, .74, .49),
                "2025-10-03": (.550, .709, .652, .779, .462, 832, 696, .67, .47), "2022-10-13": (.569, .725, .690, .765, .500, 869, 785, .71, .53),
                "2025-10-15": (.498, .665, .553, .834, .437, 977, 648, .62, .64), "2025-10-09": (.520, .684, .570, .856, .447, 1012, 674, .61, .55)}
    for _, r in v.table().iterrows():
        pe = paper_e3[r.date]
        for k, pv, tol in [("IoU", pe[0], .01), ("F1", pe[1], .01), ("precision", pe[2], .015), ("recall", pe[3], .015), ("IoU_DEM", pe[4], .01),
                           ("map_km2", pe[5], 25), ("SAR_km2", pe[6], 30), ("rho_map", pe[7], .07), ("rho_DEM", pe[8], .07)]:
            check(f"E3 {r.date} {k}", round(float(r[k]), 3), pv, tol)
    e8 = s.e8()
    paper_e8 = {(3.5, 1): (258581, 56.4, 2350, .924), (3.5, 2): (215387, 30.3, 2231, .762), (3.5, 3): (445441, 169.5, 2404, .668),
                (4.7, 1): (924712, 52.1, 3471, .937), (4.7, 2): (797784, 31.2, 3310, .885), (4.7, 3): (1058948, 74.2, 3208, .834)}
    for (L, i), (ppl, ch, km2, rho) in paper_e8.items():
        r = e8[e8.level == L].iloc[i]
        check(f"E8 {L} m {r.configuration[:28]} people", round(r.people), ppl, 0)
        check(f"E8 {L} m {r.configuration[:28]} spearman", round(r.spearman, 3), rho, 0)

    # E8 part 2: feature-group ablation (Table S9). Needs scikit-learn, lightgbm, catboost and data/processed;
    # uses the stored results/feature_ablation.csv unless it is missing (python src/analysis/feature_ablation.py to refit, ~6 min).
    import pandas as pd
    from .feature_ablation import PAPER_BLEND
    abl_csv = ROOT / "results/feature_ablation.csv"
    if abl_csv.exists():
        pv = pd.read_csv(abl_csv).pivot(index="variant", columns="horizon_d", values="blend")
        for v, vals in PAPER_BLEND.items():
            for h, x in zip((1, 7, 14), vals):
                rounded(f"E8 ablation {v} {h} d Blend RMSE", float(pv.loc[v, h]), x, 4, tol=1.1e-4)

    if full:
        e7 = fc.e7_table(fm)
        paper_e7 = [(730, 691, 73, 3381, 5.5, 1730, 1651, .971), (729, 1188, 73, 5748, 9.3, 2611, 3137, .953),
                    (728, 1352, 73, 6910, 11.2, 4558, 2352, .947), (726, 2146, 73, 12553, 20.3, 8519, 4034, .899),
                    (724, 2250, 73, 12270, 19.8, 8522, 3749, .901), (717, 2924, 73, 13618, 22.0, 11046, 2572, .891)]
        for i, h in enumerate([1, 2, 3, 5, 7, 14]):
            r = e7.iloc[i]
            for k, v in zip(["n_all", "dP_mean_all", "n_high", "dP_mean_high", "dP_pct_high", "missed_high", "spurious_high", "IoU_high"], paper_e7[i]):
                if k in ("n_all", "n_high"):
                    check(f"E7 {h} d {k}", r[k], v)
                else:
                    nd = 3 if k == "IoU_high" else (1 if k == "dP_pct_high" else 0)
                    rounded(f"E7 {h} d {k}", r[k], v, nd)

    n = {k: sum(1 for c in CHECKS if c[3] == k) for k in ("OK", "CLOSE", "DIFF")}
    for c in CHECKS:
        if c[3] != "OK":
            print(f"{c[3]:5} {c[0]}: got {c[1]}, paper {c[2]}")
    print(f"\n{len(CHECKS)} values checked: {n['OK']} identical, {n['CLOSE']} close (within tolerance), {n['DIFF']} different")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
    raise SystemExit(main(ap.parse_args().full))
