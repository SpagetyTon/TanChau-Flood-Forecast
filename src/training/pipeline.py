"""
pipeline.py - Training procedure SHARED by all four targets.

M1, M2, M3 call train_hoi_quy() (regression); M4 calls train_phan_loai() (classification).
Tuning, data splitting, blending and evaluation are therefore identical
across targets; only the target variable differs.

Procedure:
  1. Filter by season (if any) and drop rows without a target
  2. Fill missing values with TRAIN medians
  3. Tune each algorithm on TRAIN (RandomizedSearchCV + TimeSeriesSplit with a gap)
  4. Find Blend weights on VALID
  5. Evaluate everything on TEST against the baseline
  6. Save models, metrics, predictions and feature importance
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import (
    BoDienKhuyet,
    can_scale,
    chon_nguong_theo_CSI,
    chuan_bi_XY,
    cot_dac_trung,
    du_bao_blend,
    khoi_tao_models,
    luu_bo_model,
    luu_feature_importance,
    metrics_hoi_quy,
    metrics_phan_loai,
    tao_thu_muc,
    them_cai_thien,
    toi_uu_trong_so_blend,
    tune_model,
    xep_hang,
)


def _chuan_bi_3_tap(data, features, target, thang_loc, cfg):
    tap = {}
    for ten in ["train", "valid", "test"]:
        X, y, ngay = chuan_bi_XY(data[ten], features, target, thang_loc, cfg)
        tap[ten] = {"X": X, "y": y, "ngay": ngay}
    return tap


def _fit_va_du_bao(tap, cfg, task):
    """Tune each algorithm on train; return the models and their valid/test predictions."""
    imputer = BoDienKhuyet().fit(tap["train"]["X"])
    Xtr = imputer.transform(tap["train"]["X"])
    Xva = imputer.transform(tap["valid"]["X"])
    Xte = imputer.transform(tap["test"]["X"])

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=Xtr.columns, index=Xtr.index)
    Xva_s = pd.DataFrame(scaler.transform(Xva), columns=Xva.columns, index=Xva.index)
    Xte_s = pd.DataFrame(scaler.transform(Xte), columns=Xte.columns, index=Xte.index)

    zoo = khoi_tao_models(task, cfg["tuning"]["random_state"])
    models, best_params, cv_scores = {}, {}, {}
    pred_valid, pred_test = {}, {}

    for ten in cfg["thuat_toan"]:
        if ten not in zoo:
            print(f"    - skipping {ten} (library not installed)")
            continue
        mdl, grid = zoo[ten]
        A, B, C = (Xtr_s, Xva_s, Xte_s) if can_scale(ten) else (Xtr, Xva, Xte)

        print(f"    - tune {ten} ...", end=" ", flush=True)
        # CatBoost is multithreaded; extra joblib parallelism causes errors on Windows
        n_jobs_rieng = 1 if ten == "catboost" else None
        try:
            best, params, score = tune_model(
                mdl, grid, A, tap["train"]["y"], cfg, task, n_jobs=n_jobs_rieng
            )
        except Exception as e:
            print(f"ERROR - skipping {ten}: {type(e).__name__}: {str(e)[:200]}")
            continue
        print(f"cv_score={score:.4f}")

        models[ten] = best
        best_params[ten] = params
        cv_scores[ten] = score

        if task == "hoi_quy":
            pred_valid[ten] = best.predict(B)
            pred_test[ten] = best.predict(C)
        else:
            pred_valid[ten] = best.predict_proba(B)[:, 1]
            pred_test[ten] = best.predict_proba(C)[:, 1]

    return models, best_params, cv_scores, pred_valid, pred_test, scaler, imputer


# ------------------------------------------------------------------ REGRESSION
def train_hoi_quy(
    cfg: dict, data: dict, target: str, thang_loc: list[int] | None,
    model_dir: str, ket_qua_dir: str, ten_bai_toan: str,
    baseline_fn=None, ten_baseline: str = "Baseline",
) -> pd.DataFrame:
    print(f"\n=== {ten_bai_toan} | target = {target} ===")
    features = cot_dac_trung(data["train"], cfg)
    tap = _chuan_bi_3_tap(data, features, target, thang_loc, cfg)
    print(f"  rows: train={len(tap['train']['y'])}, valid={len(tap['valid']['y'])}, test={len(tap['test']['y'])}")

    models, best_params, cv_scores, pv, pt, scaler, imputer = _fit_va_du_bao(tap, cfg, "hoi_quy")

    weights = toi_uu_trong_so_blend(pv, tap["valid"]["y"], "hoi_quy")
    print(f"  Blend weights: {weights}")
    pv["Blend"] = du_bao_blend({k: v for k, v in pv.items() if k in weights}, weights)
    pt["Blend"] = du_bao_blend({k: v for k, v in pt.items() if k in weights}, weights)

    if baseline_fn is not None:
        df_test = data["test"]
        if thang_loc:
            df_test = df_test[df_test[cfg["cot"]["date"]].dt.month.isin(thang_loc)]
        df_test = df_test.loc[tap["test"]["X"].index]
        pt[ten_baseline] = np.asarray(baseline_fn(df_test, data["train"], cfg), dtype=float)

    rows = []
    y_test = tap["test"]["y"].to_numpy()
    for ten, p in pt.items():
        rows.append({"model": ten, **metrics_hoi_quy(y_test, p)})
    bang = xep_hang(pd.DataFrame(rows), "RMSE", tang_dan=True)
    bang = them_cai_thien(bang, "RMSE", ten_baseline)
    bang.insert(1, "bai_toan", ten_bai_toan)

    tao_thu_muc(ket_qua_dir)
    luu_bo_model(
        model_dir, models, weights, best_params, cv_scores, list(features), scaler, imputer,
        ghi_chu={"bai_toan": ten_bai_toan, "target": target, "thang_loc": thang_loc,
                 "n_train": len(tap["train"]["y"]), "task": "hoi_quy"},
    )

    du_bao = pd.DataFrame({"date": tap["test"]["ngay"].values, "thuc_te": y_test})
    for ten, p in pt.items():
        du_bao[f"du_bao_{ten}"] = p
    du_bao.to_csv(os.path.join(ket_qua_dir, f"du_bao_test_{target}.csv"),
                  index=False, encoding="utf-8-sig")
    luu_feature_importance(ket_qua_dir, models, list(features), f"feature_importance_{target}.csv")

    print(bang.to_string(index=False))
    return bang


# ---------------------------------------------------------------- CLASSIFICATION
def train_phan_loai(
    cfg: dict, data: dict, target: str, thang_loc: list[int] | None,
    model_dir: str, ket_qua_dir: str, ten_bai_toan: str,
    baseline_fn=None, ten_baseline: str = "Baseline",
) -> pd.DataFrame:
    print(f"\n=== {ten_bai_toan} | target = {target} ===")
    features = cot_dac_trung(data["train"], cfg)
    tap = _chuan_bi_3_tap(data, features, target, thang_loc, cfg)
    ty_le = float(np.mean(tap["train"]["y"]))
    print(f"  rows: train={len(tap['train']['y'])}, valid={len(tap['valid']['y'])}, "
          f"test={len(tap['test']['y'])} | positive rate in train={ty_le:.1%}")

    models, best_params, cv_scores, pv, pt, scaler, imputer = _fit_va_du_bao(tap, cfg, "phan_loai")

    weights = toi_uu_trong_so_blend(pv, tap["valid"]["y"], "phan_loai")
    print(f"  Blend weights: {weights}")
    pv["Blend"] = du_bao_blend({k: v for k, v in pv.items() if k in weights}, weights)
    pt["Blend"] = du_bao_blend({k: v for k, v in pt.items() if k in weights}, weights)

    # the baseline is also computed on valid so that its threshold is chosen the same way as for the models
    if baseline_fn is not None:
        for ten_tap, kho in [("valid", pv), ("test", pt)]:
            df_ = data[ten_tap]
            if thang_loc:
                df_ = df_[df_[cfg["cot"]["date"]].dt.month.isin(thang_loc)]
            df_ = df_.loc[tap[ten_tap]["X"].index]
            kho[ten_baseline] = np.asarray(baseline_fn(df_, data["train"], cfg), dtype=float)

    # probability threshold chosen on VALID, not the default 0.5
    nguong = {ten: chon_nguong_theo_CSI(tap["valid"]["y"], p) for ten, p in pv.items()}
    print(f"  probability thresholds chosen on valid: {nguong}")

    rows = []
    y_test = tap["test"]["y"].to_numpy()
    for ten, p in pt.items():
        rows.append({"model": ten, **metrics_phan_loai(y_test, p, nguong.get(ten, 0.5))})
    bang = xep_hang(pd.DataFrame(rows), "CSI", tang_dan=False)
    bang.insert(1, "bai_toan", ten_bai_toan)

    tao_thu_muc(ket_qua_dir)
    luu_bo_model(
        model_dir, models, weights, best_params, cv_scores, list(features), scaler, imputer,
        ghi_chu={"bai_toan": ten_bai_toan, "target": target, "thang_loc": thang_loc,
                 "ty_le_duong_train": ty_le, "task": "phan_loai"},
        hau_to="_clf",
    )
    import json
    with open(os.path.join(model_dir, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump(nguong, f, indent=2, ensure_ascii=False)

    du_bao = pd.DataFrame({"date": tap["test"]["ngay"].values, "thuc_te": y_test})
    for ten, p in pt.items():
        du_bao[f"xac_suat_{ten}"] = p
    du_bao.to_csv(os.path.join(ket_qua_dir, f"du_bao_test_{target}.csv"),
                  index=False, encoding="utf-8-sig")

    cm = bang[["model", "hit", "miss", "false_alarm", "correct_neg"]]
    cm.to_csv(os.path.join(ket_qua_dir, f"confusion_matrix_{target}.csv"),
              index=False, encoding="utf-8-sig")
    luu_feature_importance(ket_qua_dir, models, list(features), f"feature_importance_{target}.csv")

    print(bang.to_string(index=False))
    return bang
