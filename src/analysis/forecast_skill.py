"""Forecast-skill experiments E1, E2 and E7 (paper Sect. 4.2, 4.3 and 4.8).

Input: ``data/mrc_derived/forecast_test.json`` with, for the test period 2023-01-01 .. 2024-12-31 (731 days):
observed Tan Chau stage ``H`` (daily mean), the forecasts of M1 (six horizons, five algorithms, Blend, persistence),
M2, M3b, M4 and the day-of-year climatology band. Observations come from the Mekong River Commission (see
``data/mrc_derived/README.md``); the forecasts are outputs of the project's training code.
Pair convention: forecast issued on day t for day t + h, compared with the observation of day t + h.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HORIZONS = [1, 2, 3, 5, 7, 14]
M1_MODELS = ["dt", "rf", "lgbm", "catboost", "svr", "Blend", "Persistence"]
LABELS = {"dt": "DT", "rf": "RF", "lgbm": "LightGBM", "catboost": "CatBoost", "svr": "SVR", "Blend": "Blend",
          "Persistence": "Persistence", "Climatology": "Climatology"}
# probability thresholds chosen on the validation year 2022 (paper Table 'e2')
M4_THRESHOLDS = {
    "m4_7": {"dt": .05, "rf": .32, "lgbm": .07, "catboost": .29, "svr": .18, "Blend": .15, "TanSuatLichSu": .13},
    "m4_15": {"dt": .30, "rf": .43, "lgbm": .28, "catboost": .46, "svr": .24, "Blend": .24, "TanSuatLichSu": .18},
}


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def convert_demo_forecast(demo_forecast_js: str | Path, out_json: str | Path) -> None:
    """Keep only the fields used here from the demo's ``forecast.js``."""
    txt = Path(demo_forecast_js).read_text(encoding="utf-8")
    d = json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))
    keep = {k: d[k] for k in ["dates", "H", "m1", "m2", "m3b", "m4_7", "m4_15", "band"]}
    Path(out_json).write_text(json.dumps(keep), encoding="utf-8")


class Forecasts:
    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.d = d
        self.H = np.array(d["H"], float)
        self.dates = [dt.date.fromisoformat(x) for x in d["dates"]]
        doy = np.array([min(x.timetuple().tm_yday, 365) - 1 for x in self.dates])
        self.clim = np.array(d["band"], float)[doy, 2]          # climatological median of the day of year
        self.n = len(self.H)

    def pairs(self, h: int, model: str):
        """(observed, forecast) for horizon h. Climatology is the median of the target's day of year."""
        y = self.H[h:]
        if model == "Climatology":
            return y, self.clim[h:]
        return y, np.array(self.d["m1"][str(h)][model], float)[: self.n - h]

    # ---------------------------------------------------------------------------------- E1
    def e1_rmse_table(self) -> pd.DataFrame:
        rows = {}
        for m in M1_MODELS + ["Climatology"]:
            rows[LABELS[m]] = [float(np.sqrt(np.mean(np.subtract(*self.pairs(h, m)[::-1]) ** 2))) for h in HORIZONS]
        rows["n pairs"] = [self.n - h for h in HORIZONS]
        return pd.DataFrame(rows, index=[f"{h} d" for h in HORIZONS]).T

    @staticmethod
    def _block_bootstrap_reduction(e1, e2, B=2000, seed=20261017, block=30):
        rng = np.random.default_rng(seed)
        n = len(e1)
        nb = int(np.ceil(n / block))
        out = np.empty(B)
        for i in range(B):
            st = rng.integers(0, n - block + 1, nb)
            idx = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
            out[i] = 1 - np.sqrt(np.mean(e1[idx] ** 2)) / np.sqrt(np.mean(e2[idx] ** 2))
        return np.percentile(out, [2.5, 97.5]) * 100

    @staticmethod
    def diebold_mariano(e1, e2, h):
        """Two-sided DM test on squared errors with the Harvey-Leybourne-Newbold small-sample correction."""
        dd = e1 ** 2 - e2 ** 2
        n, m = len(dd), dd.mean()
        g = [np.sum((dd[k:] - m) * (dd[: n - k] - m)) / n for k in range(h)]
        var = (g[0] + 2 * sum(g[1:])) / n
        stat = m / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        return stat, 2 * stats.t.sf(abs(stat), n - 1)

    def e1_blend_table(self, B=2000, seed=20261017) -> pd.DataFrame:
        rows = []
        for h in HORIZONS:
            y, b = self.pairs(h, "Blend")
            eb = b - y
            row = {"horizon": h, "MAE": np.mean(np.abs(eb)), "bias": np.mean(eb),
                   "NSE": 1 - np.sum(eb ** 2) / np.sum((y - y.mean()) ** 2)}
            for tag, base in [("persist", "Persistence"), ("clim", "Climatology")]:
                eo = self.pairs(h, base)[1] - y
                red = 100 * (1 - np.sqrt(np.mean(eb ** 2)) / np.sqrt(np.mean(eo ** 2)))
                lo, hi = self._block_bootstrap_reduction(eb, eo, B, seed)
                row.update({f"{tag}_red_%": red, f"{tag}_ci_lo": lo, f"{tag}_ci_hi": hi,
                            f"{tag}_p_DM": self.diebold_mariano(eb, eo, h)[1]})
            rows.append(row)
        return pd.DataFrame(rows)

    def e1_alert_table(self, threshold=3.0, h=7) -> pd.DataFrame:
        def conf(ev, pr):
            a, b, c = int((ev & pr).sum()), int((ev & ~pr).sum()), int((~ev & pr).sum())
            e = int((~ev & ~pr).sum())
            return {"hit": a, "miss": b, "false_alarm": c, "correct_neg": e, "POD": a / (a + b),
                    "FAR": c / (a + c) if a + c else 0.0, "CSI": a / (a + b + c)}
        out, n = {}, self.n - h
        for m in ["Blend", "Persistence"]:
            y, p = self.pairs(h, m)
            out[f"{m}, value at day {h}"] = conf(y >= threshold, p >= threshold)
        for m in ["Blend", "Persistence"]:
            peak = np.max([self.pairs(k, m)[1][:n] for k in [1, 2, 3, 5, 7]] + [self.H[:n]], axis=0)
            ev = np.array([self.H[i:i + h + 1].max() >= threshold for i in range(n)])
            out[f"{m}, operational rule (peak in {h} days)"] = conf(ev, peak >= threshold)
        return pd.DataFrame(out).T

    # ---------------------------------------------------------------------------------- E2
    def e2_table(self) -> pd.DataFrame:
        d = self.d
        rows = {}
        y2, y3 = _arr(d["m2"]["y"]), _arr(d["m3b"]["y"])
        for m in ["dt", "rf", "lgbm", "catboost", "svr", "Blend", "Climatology"]:
            row = {"M2 MAE": np.nanmean(np.abs(np.clip(_arr(d["m2"][m]), 0, 10) - y2)),      # M2 is a count in [0, 10]
                   "M3b MAE": np.nanmean(np.abs(_arr(d["m3b"][m]) - y3))}
            for key, tag in [("m4_7", "M4-7d"), ("m4_15", "M4-15d")]:
                y = _arr(d[key]["y"]); ok = ~np.isnan(y); yy = y[ok]
                name = "TanSuatLichSu" if m == "Climatology" else m
                p = _arr(d[key][name])[ok]
                pr = p >= M4_THRESHOLDS[key][name]
                a, b, c = (pr & (yy == 1)).sum(), (~pr & (yy == 1)).sum(), (pr & (yy == 0)).sum()
                row.update({f"{tag} Brier": np.mean((p - yy) ** 2), f"{tag} AUC": roc_auc_score(yy, p),
                            f"{tag} CSI": a / (a + b + c)})
            rows["Baseline" if m == "Climatology" else LABELS[m]] = row
        return pd.DataFrame(rows).T

    # ---------------------------------------------------------------------------------- E7
    def e7_table(self, flood_model, min_level=2.5) -> pd.DataFrame:
        """Run the flood map on observed and on Blend-forecast stages for every test day and horizon."""
        N = flood_model.elev.size
        pop = flood_model.pop.ravel()
        land = ((flood_model.river == 0) & ~(flood_model.occ >= 90)).ravel()
        cache = {}

        def mask(L):
            k = round(float(L), 3)
            if k not in cache:
                _, depth = flood_model.flood(k)
                cache[k] = np.packbits((depth > 0).ravel())
            return np.unpackbits(cache[k], count=N).astype(bool)

        rows = []
        for h in HORIZONS:
            y, p = self.pairs(h, "Blend")
            res = []
            for lo, lp in zip(y, p):
                a, b = mask(lo), mask(lp)
                la, lb = a & land, b & land
                res.append((lo, pop[a].sum(), pop[a & ~b].sum(), pop[b & ~a].sum(),
                            (la & lb).sum() / max((la | lb).sum(), 1)))
            r = np.array(res)
            hi = r[:, 0] >= min_level
            rows.append({"horizon": h, "n_all": len(r), "dP_mean_all": (r[:, 2] + r[:, 3]).mean(),
                         "n_high": int(hi.sum()), "dP_mean_high": (r[hi, 2] + r[hi, 3]).mean(),
                         "dP_pct_high": 100 * (r[hi, 2] + r[hi, 3]).mean() / r[hi, 1].mean(),
                         "missed_high": r[hi, 2].mean(), "spurious_high": r[hi, 3].mean(), "IoU_high": r[hi, 4].mean()})
        return pd.DataFrame(rows)
