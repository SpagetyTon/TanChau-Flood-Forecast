"""E8 (part 2) - feature-group ablation of the M1 water-level model (Table S9).

Refits the five base algorithms of M1 at horizons 1, 7 and 14 days with the
hyper-parameters FIXED to those stored in models/M1_mucnuoc/ngayXX/training_log.json
(no new search), once with all 77 features and once with each feature group removed.
Medians for imputation come from the training years, SVR uses standardised inputs,
and Blend weights are found by SLSQP on the validation year (2022) - exactly as in
src/training. Metrics are computed on the test set (2023-2024).

Run from the repository root:  python src/analysis/feature_ablation.py
Output: results/feature_ablation.csv
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HORIZONS = [1, 7, 14]
SEED = 42
ALGOS = ["dt", "rf", "lgbm", "catboost", "svr"]

PAPER_BLEND = {   # Table S9 (Blend RMSE, m)
    "Full (77)": (0.0749, 0.1570, 0.1870),
    "No upstream discharge": (0.0762, 0.1726, 0.1975),
    "No rain": (0.0756, 0.1576, 0.1965),
    "No soil moisture / GLDAS runoff": (0.0754, 0.1677, 0.1857),
    "No seasonal terms": (0.0750, 0.1582, 0.1884),
}


def groups(features):
    g = {}
    g["No upstream discharge"] = [c for c in features
        if c in ("Q_LuangPrabang", "Q_Vientiane", "Q_StungTreng") or c.startswith("dQ_StungTreng")]
    g["No rain"] = [c for c in features
        if c.startswith(("mua_", "so_ngay_mua")) or c == "Rainf_f_tavg"]
    g["No soil moisture / GLDAS runoff"] = [c for c in features
        if c.startswith(("SoilMoi", "dSoilMoi", "Qs_acc", "Qsb_acc")) or c in ("RootMoist_inst", "CanopInt_inst")]
    g["No seasonal terms"] = [c for c in features
        if c.startswith("doy_") or c in ("la_mua_lu_8_11", "pha_lu")]
    return g


def load():
    d = os.path.join(ROOT, "data", "processed")
    tg = pd.read_csv(os.path.join(ROOT, "data", "targets", "all_targets.csv"), parse_dates=["date"])
    out = {}
    for k in ("train", "valid", "test"):
        df = pd.read_csv(os.path.join(d, f"{k}_final.csv"), parse_dates=["date"])
        cols = [c for c in tg.columns if c == "date" or c not in df.columns]
        out[k] = df.merge(tg[cols], on="date", how="left").sort_values("date").reset_index(drop=True)
    drop = lambda c: c == "date" or c.startswith(("target_", "han_", "ngay_den_", "so_ngay_tren_nguong"))
    feats = [c for c in out["train"].columns if not drop(c)]
    return out, feats


def make(name, p):
    if name == "dt":
        return DecisionTreeRegressor(random_state=SEED, **p)
    if name == "rf":
        return RandomForestRegressor(random_state=SEED, n_jobs=-1, **p)
    if name == "lgbm":
        return LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1, **p)
    if name == "catboost":
        return CatBoostRegressor(random_state=SEED, verbose=0, allow_writing_files=False, **p)
    return SVR(**p)


def blend_weights(P, y):
    n = P.shape[1]
    r = minimize(lambda w: np.sqrt(np.mean((y - P @ w) ** 2)), np.ones(n) / n, method="SLSQP",
                 bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                 options={"maxiter": 500, "ftol": 1e-9})
    w = np.clip(r.x if r.success else np.ones(n) / n, 0, None)
    return np.round(w / w.sum(), 6)


def run_one(data, feats, target, params):
    tr, va, te = [data[k].dropna(subset=[target]) for k in ("train", "valid", "test")]
    med = tr[feats].median(numeric_only=True)
    Xtr, Xva, Xte = [d[feats].fillna(med) for d in (tr, va, te)]
    sc = StandardScaler().fit(Xtr)
    pv, pt = {}, {}
    for a in ALGOS:
        m = make(a, params[a])
        A, B, C = (sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)) if a == "svr" else (Xtr, Xva, Xte)
        m.fit(A, tr[target])
        pv[a], pt[a] = m.predict(B), m.predict(C)
    w = blend_weights(np.column_stack([pv[a] for a in ALGOS]), va[target].to_numpy())
    pt["blend"] = np.column_stack([pt[a] for a in ALGOS]) @ w
    y = te[target].to_numpy()
    return {k: float(np.sqrt(np.mean((y - v) ** 2))) for k, v in pt.items()}


def main(budget_s: float = 1e9):
    """Resumable: rows already in results/feature_ablation.csv are skipped; stops after budget_s seconds."""
    t0 = time.time()
    out_csv = os.path.join(ROOT, "results", "feature_ablation.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
    data, feats = load()
    assert len(feats) == 77, len(feats)
    gr = groups(feats)
    print({k: len(feats) - len(v) for k, v in gr.items()})
    rows = done.to_dict("records")
    have = {(r["horizon_d"], r["variant"]) for r in rows}
    for h in HORIZONS:
        lg = json.load(open(os.path.join(ROOT, "models", "M1_mucnuoc", f"ngay{h:02d}", "training_log.json")))
        params = lg["best_params"]
        for name, drop in [("Full (77)", [])] + list(gr.items()):
            if (h, name) in have:
                continue
            if time.time() - t0 > budget_s:
                print("time budget reached - run again to continue"); return
            t = time.time()
            f = [c for c in feats if c not in drop]
            r = run_one(data, f, f"target_ngay{h}", params)
            rows.append({"horizon_d": h, "variant": name, "n_features": len(f), **r})
            print(h, name, len(f), {k: round(v, 4) for k, v in r.items()}, f"{time.time()-t:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "feature_ablation.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1e9)
