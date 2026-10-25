"""
common.py - Ham dung chung cho ca 4 model M1, M2, M3, M4.

Nguyen tac:
  - Moi thuat toan deu tune GIONG NHAU: RandomizedSearchCV + TimeSeriesSplit co gap.
  - Fit tren TRAIN, chon trong so Blend tren VALID, danh gia tren TEST.
  - Moi thu luu ra file de tai lap lai duoc.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    CO_LGBM = True
except ImportError:
    CO_LGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor

    CO_CATBOOST = True
except ImportError:
    CO_CATBOOST = False


# ---------------------------------------------------------------- config
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # thư mục gốc repo


def load_config(path: str = "config.yaml") -> dict:
    """Đọc config. Đường dẫn tương đối trong config được tính từ thư mục chứa config."""
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(ROOT, path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    goc = os.path.dirname(os.path.abspath(path))
    for k, v in cfg.get("duong_dan", {}).items():
        if not os.path.isabs(v):
            cfg["duong_dan"][k] = os.path.normpath(os.path.join(goc, v))
    return cfg


def tao_thu_muc(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------- du lieu
def load_splits(cfg: dict) -> dict:
    """Doc 3 file train/valid/test da chia san va gan them cac cot target moi."""
    d = cfg["duong_dan"]["processed_dir"]
    tgt_dir = cfg["duong_dan"]["targets_dir"]
    date_col = cfg["cot"]["date"]

    out = {}
    for ten, fn in [
        ("train", "train_final.csv"),
        ("valid", "valid_final.csv"),
        ("test", "test_final.csv"),
    ]:
        df = pd.read_csv(os.path.join(d, fn), parse_dates=[date_col])
        tgt_path = os.path.join(tgt_dir, "all_targets.csv")
        if os.path.exists(tgt_path):
            tgt = pd.read_csv(tgt_path, parse_dates=[date_col])
            cot_moi = [c for c in tgt.columns if c == date_col or c not in df.columns]
            df = df.merge(tgt[cot_moi], on=date_col, how="left")
        out[ten] = df.sort_values(date_col).reset_index(drop=True)
    return out


def cot_dac_trung(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Lay danh sach cot dau vao: bo cot date va moi cot target."""
    date_col = cfg["cot"]["date"]
    bo = {date_col}
    for c in df.columns:
        if c.startswith("target_") or c.startswith("han_") or c.startswith("ngay_den_"):
            bo.add(c)
        if c.startswith("so_ngay_tren_nguong"):
            bo.add(c)
    return [c for c in df.columns if c not in bo]


def chuan_bi_XY(
    df: pd.DataFrame, features: list[str], target: str, thang_loc: list[int] | None, cfg: dict
):
    """Loc theo mua (neu co), bo dong thieu target, tra ve X, y, ngay."""
    date_col = cfg["cot"]["date"]
    d = df.copy()
    if thang_loc:
        d = d[d[date_col].dt.month.isin(thang_loc)]
    d = d.dropna(subset=[target])
    return d[features], d[target], d[date_col]


class BoDienKhuyet:
    """Dien gia tri thieu bang trung vi cua TAP TRAIN (khong dung du lieu tuong lai)."""

    def __init__(self):
        self.medians_: pd.Series | None = None

    def fit(self, X: pd.DataFrame):
        self.medians_ = X.median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------- model zoo
def khoi_tao_models(task: str, random_state: int = 42) -> dict:
    """task = 'hoi_quy' hoac 'phan_loai'. Tra ve dict {ten: (model, param_grid)}."""
    rs = random_state
    zoo: dict = {}

    if task == "hoi_quy":
        zoo["dt"] = (
            DecisionTreeRegressor(random_state=rs),
            {
                "max_depth": [4, 6, 8, 10, 12, None],
                "min_samples_leaf": [1, 5, 10, 20, 50],
                "min_samples_split": [2, 5, 10, 20],
            },
        )
        zoo["rf"] = (
            RandomForestRegressor(random_state=rs, n_jobs=-1),
            {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [6, 10, 14, 20, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.3, 0.5, 0.8],
            },
        )
        if CO_LGBM:
            zoo["lgbm"] = (
                LGBMRegressor(random_state=rs, n_jobs=-1, verbose=-1),
                {
                    "n_estimators": [300, 500, 800, 1200],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "num_leaves": [15, 31, 63, 127],
                    "min_child_samples": [5, 10, 20, 40],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.5, 0.7, 0.9],
                    "reg_lambda": [0, 1, 5, 10],
                },
            )
        if CO_CATBOOST:
            zoo["catboost"] = (
                CatBoostRegressor(random_state=rs, verbose=0, allow_writing_files=False),
                {
                    "iterations": [300, 500, 800],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 10],
                },
            )
        zoo["svr"] = (
            SVR(),
            {
                "C": [0.1, 1, 10, 50, 100],
                "gamma": ["scale", 0.001, 0.01, 0.1],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
                "kernel": ["rbf"],
            },
        )

    else:  # phan_loai
        zoo["dt"] = (
            DecisionTreeClassifier(random_state=rs, class_weight="balanced"),
            {
                "max_depth": [3, 4, 6, 8, 10, None],
                "min_samples_leaf": [1, 5, 10, 20, 50],
                "min_samples_split": [2, 5, 10, 20],
            },
        )
        zoo["rf"] = (
            RandomForestClassifier(random_state=rs, n_jobs=-1, class_weight="balanced"),
            {
                "n_estimators": [200, 400, 600, 800],
                "max_depth": [4, 8, 12, 20, None],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", 0.3, 0.5],
            },
        )
        if CO_LGBM:
            zoo["lgbm"] = (
                LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1, class_weight="balanced"),
                {
                    "n_estimators": [300, 500, 800],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "num_leaves": [15, 31, 63],
                    "min_child_samples": [5, 10, 20, 40],
                    "subsample": [0.7, 0.9, 1.0],
                    "colsample_bytree": [0.5, 0.7, 0.9],
                    "reg_lambda": [0, 1, 5, 10],
                },
            )
        if CO_CATBOOST:
            zoo["catboost"] = (
                CatBoostClassifier(
                    random_state=rs, verbose=0, allow_writing_files=False, auto_class_weights="Balanced"
                ),
                {
                    "iterations": [300, 500, 800],
                    "learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5, 10],
                },
            )
        zoo["svr"] = (  # giu ten 'svr' cho dong bo, ban chat la SVC
            SVC(probability=True, class_weight="balanced", random_state=rs),
            {
                "C": [0.1, 1, 10, 50, 100],
                "gamma": ["scale", 0.001, 0.01, 0.1],
                "kernel": ["rbf"],
            },
        )

    return zoo


def can_scale(ten_model: str) -> bool:
    """Chi SVR/SVC can chuan hoa. Cac model cay dung du lieu goc."""
    return ten_model == "svr"


# ---------------------------------------------------------------- tuning
def tune_model(model, param_grid: dict, X, y, cfg: dict, task: str, n_jobs: int | None = None):
    """Cach tune GIONG NHAU cho moi thuat toan va moi model M1-M4.

    n_jobs: de None thi lay tu config. CatBoost tu chay da luong ben trong,
    nen truyen n_jobs=1 de tranh xung dot tien trinh tren Windows.
    """
    t = cfg["tuning"]
    cv = TimeSeriesSplit(n_splits=t["n_splits"], gap=t["gap"])
    scoring = t["scoring_hoi_quy"] if task == "hoi_quy" else t["scoring_phan_loai"]

    n_to_hop = 1
    for v in param_grid.values():
        n_to_hop *= len(v)
    n_iter = min(t["n_iter"], n_to_hop)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=t["random_state"],
        n_jobs=t["n_jobs"] if n_jobs is None else n_jobs,
        refit=True,
        error_score=np.nan,   # mot to hop tham so loi thi bo qua, khong dung ca qua trinh
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


# ---------------------------------------------------------------- metrics
def wape(y_true, y_pred) -> float:
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    m = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100)


def metrics_hoi_quy(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
    }


def metrics_phan_loai(y_true, prob, nguong: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= nguong).astype(int)

    hit = int(((pred == 1) & (y_true == 1)).sum())
    miss = int(((pred == 0) & (y_true == 1)).sum())
    false_alarm = int(((pred == 1) & (y_true == 0)).sum())
    correct_neg = int(((pred == 0) & (y_true == 0)).sum())

    pod = hit / (hit + miss) if (hit + miss) else np.nan          # bat duoc bao nhieu su kien that
    far = false_alarm / (hit + false_alarm) if (hit + false_alarm) else np.nan  # ty le bao dong gia
    csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) else np.nan

    out = {
        "POD": float(pod), "FAR": float(far), "CSI": float(csi),
        "Brier": float(brier_score_loss(y_true, prob)),
        "hit": hit, "miss": miss, "false_alarm": false_alarm, "correct_neg": correct_neg,
        "nguong": float(nguong),
    }
    if len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = float(roc_auc_score(y_true, prob))
        out["PR_AUC"] = float(average_precision_score(y_true, prob))
    return out


def chon_nguong_theo_CSI(y_true, prob) -> float:
    """Chon nguong xac suat toi uu CSI tren tap VALID (khong dung mac dinh 0.5)."""
    best_thr, best_csi = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        m = metrics_phan_loai(y_true, prob, thr)
        if not np.isnan(m["CSI"]) and m["CSI"] > best_csi:
            best_csi, best_thr = m["CSI"], float(thr)
    return round(best_thr, 2)


# ---------------------------------------------------------------- blend
def toi_uu_trong_so_blend(preds: dict, y_true, task: str = "hoi_quy") -> dict:
    """Tim trong so khong am, tong bang 1, toi thieu sai so tren tap VALID."""
    ten = list(preds.keys())
    P = np.column_stack([preds[k] for k in ten])
    y = np.asarray(y_true, dtype=float)

    def loss(w):
        p = P @ w
        if task == "hoi_quy":
            return np.sqrt(np.mean((y - p) ** 2))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    w0 = np.ones(len(ten)) / len(ten)
    res = minimize(
        loss, w0, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(ten),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"maxiter": 500, "ftol": 1e-9},
    )
    w = res.x if res.success else w0
    w = np.clip(w, 0, None)
    w = w / w.sum()
    return {k: float(round(v, 6)) for k, v in zip(ten, w)}


def du_bao_blend(preds: dict, weights: dict) -> np.ndarray:
    return sum(np.asarray(preds[k], dtype=float) * weights.get(k, 0.0) for k in preds)


# ---------------------------------------------------------------- luu tru
def luu_bo_model(
    thu_muc: str, models: dict, blend_weights: dict, best_params: dict,
    cv_scores: dict, features: list[str], scaler, imputer, ghi_chu: dict, hau_to: str = "",
):
    """Luu model + scaler + feature_list + blend_weights + training_log vao dung thu muc."""
    tao_thu_muc(thu_muc)
    for ten, mdl in models.items():
        joblib.dump(mdl, os.path.join(thu_muc, f"{ten}{hau_to}.pkl"), compress=3)

    with open(os.path.join(thu_muc, "blend_weights.json"), "w", encoding="utf-8") as f:
        json.dump(blend_weights, f, indent=2, ensure_ascii=False)

    with open(os.path.join(thu_muc, "feature_list.json"), "w", encoding="utf-8") as f:
        json.dump({"n_features": len(features), "features": features}, f, indent=2)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(thu_muc, "scaler.pkl"), compress=3)
    if imputer is not None:
        joblib.dump(imputer, os.path.join(thu_muc, "imputer.pkl"), compress=3)

    log = {
        "ngay_train": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_params": best_params,
        "cv_score_tren_train": cv_scores,
        **ghi_chu,
    }
    with open(os.path.join(thu_muc, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=str)


def luu_feature_importance(thu_muc_kq: str, models: dict, features: list[str], ten_file: str):
    rows = []
    for ten, mdl in models.items():
        if hasattr(mdl, "feature_importances_"):
            imp = np.asarray(mdl.feature_importances_, dtype=float)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            for f, v in zip(features, imp):
                rows.append({"model": ten, "feature": f, "importance": v})
    if rows:
        df = pd.DataFrame(rows).sort_values(["model", "importance"], ascending=[True, False])
        tao_thu_muc(thu_muc_kq)
        df.to_csv(os.path.join(thu_muc_kq, ten_file), index=False, encoding="utf-8-sig")


def xep_hang(df: pd.DataFrame, cot_sap_xep: str, tang_dan: bool = True) -> pd.DataFrame:
    df = df.sort_values(cot_sap_xep, ascending=tang_dan).reset_index(drop=True)
    df.insert(0, "hang", range(1, len(df) + 1))
    return df


def them_cai_thien(df: pd.DataFrame, cot: str, ten_baseline: str = "Baseline") -> pd.DataFrame:
    """Them cot % cai thien so voi baseline (cang nho cang tot doi voi RMSE)."""
    if ten_baseline not in set(df["model"]):
        return df
    base = float(df.loc[df["model"] == ten_baseline, cot].iloc[0])
    df["cai_thien_vs_baseline_%"] = ((base - df[cot]) / base * 100).round(2)
    return df
