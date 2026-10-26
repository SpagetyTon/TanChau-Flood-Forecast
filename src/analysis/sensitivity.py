"""Sensitivity and uncertainty (E4) and layer ablation (E8) of the flood and exposure maps (paper Sect. 4.5, 4.9).

Totals are people in the 54 communes unless stated otherwise. Spearman correlations compare the ranking of the 54
communes (by exposed people) with that of the baseline configuration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from .flood_model import FloodModel

CELL_M = 64.0
RMSE_7D = 0.157          # m, Blend RMSE at the 7-day horizon (E1)


class Sensitivity:
    def __init__(self, model: FloodModel, seed: int = 42):
        self.m = model
        self.seed = seed
        self.in_c = model.adm > 0
        self.n = len(model.adm_names) + 1

    # -- helpers -------------------------------------------------------------------------------------------------------
    def _exposure(self, depth, pop=None):
        pop = self.m.pop if pop is None else pop
        wet = depth > 0
        total = pop[wet & self.in_c].sum()
        per = np.bincount(self.m.adm[wet].ravel(), weights=pop[wet].ravel(), minlength=self.n)[1:]
        return total, per

    def _field(self, rng, sigma, corr_m):
        z = gaussian_filter(rng.standard_normal(self.m.elev.shape), corr_m / CELL_M, mode="wrap")
        return (sigma * z / z.std()).astype(np.float32)

    # -- E4 ------------------------------------------------------------------------------------------------------------
    def e4(self, L: float, n_mc: int = 100) -> pd.DataFrame:
        m = self.m
        _, d0 = m.flood(L)
        base, base_rank = self._exposure(d0)
        rows = [{"level": L, "source": "baseline", "people_lo": base, "people_hi": base, "shift_%": 0.0,
                 "rho_median": 1.0, "rho_p5": 1.0}]

        def add(source, totals, rhos):
            totals, rhos = np.array(totals), np.array(rhos)
            rows.append({"level": L, "source": source, "people_lo": np.percentile(totals, 5),
                         "people_hi": np.percentile(totals, 95), "shift_%": 100 * (np.median(totals) / base - 1),
                         "rho_median": np.median(rhos), "rho_p5": np.percentile(rhos, 5)})

        def single(source, **kw):
            Lx = kw.pop("level", L)
            t, per = self._exposure(m.flood(Lx, **kw)[1])
            add(source, [t], [spearmanr(per, base_rank)[0]])

        rng = np.random.default_rng(self.seed)
        for sigma in (0.10, 0.25):                                   # threshold error, correlation length 2 km
            T, R = [], []
            for _ in range(n_mc):
                t, per = self._exposure(m.flood(L, th_noise=self._field(rng, sigma, 2000))[1])
                T.append(t); R.append(spearmanr(per, base_rank)[0])
            add(f"threshold error sigma={sigma} m, 2 km correlation", T, R)
        single("satellite frequency x0.8", occ_scale=0.8)
        single("satellite frequency x1.2", occ_scale=1.2)
        single(f"stage -{RMSE_7D} m", level=L - RMSE_7D)
        single(f"stage +{RMSE_7D} m", level=L + RMSE_7D)
        T, R = [], []
        _, dd = m.flood(L)
        wet = dd > 0
        for _ in range(n_mc):                                        # mean-preserving log-normal noise per cell
            p = m.pop * np.exp(rng.normal(0, 0.3, m.pop.shape) - 0.045)
            t, per = self._exposure(dd, pop=p)
            T.append(t); R.append(spearmanr(per, base_rank)[0])
        add("population noise sigma_log=0.3", T, R)
        return pd.DataFrame(rows)

    def e4_depth(self, L: float = 4.7, n_mc: int = 50) -> dict:
        """DEM error (sigma 2 m, correlation 1 km): people in cells deeper than 1 m."""
        rng = np.random.default_rng(self.seed)
        base = self.m.pop[self.m.flood(L)[1] > 1].sum()
        T = [self.m.pop[self.m.flood(L, elev=self.m.elev + self._field(rng, 2.0, 1000))[1] > 1].sum()
             for _ in range(n_mc)]
        return {"level": L, "baseline_people_gt_1m": base, "p5": np.percentile(T, 5),
                "median": np.median(T), "p95": np.percentile(T, 95)}

    # -- E8 (spatial layers) -------------------------------------------------------------------------------------------
    def e8(self, levels=(3.5, 4.7)) -> pd.DataFrame:
        m = self.m
        land = lambda d: (d > 0) & (m.river == 0) & ~(m.occ >= 90)
        rows = []
        for L in levels:
            _, d0 = m.flood(L)
            b_tot, b_rank = self._exposure(d0)
            b_land = land(d0)
            for name, kw in [("Full: satellite thresholds, with dikes", {}),
                             ("No dike layer", dict(dike=False)),
                             ("DEM only, tilted surface, connected", dict(mode="dem")),
                             ("DEM only, flat surface, connected", dict(mode="dem", tilt=False))]:
                _, d = m.flood(L, **kw)
                t, per = self._exposure(d)
                ld = land(d)
                rows.append({"level": L, "configuration": name, "people": t,
                             "change_%": None if not kw else 100 * (t / b_tot - 1),
                             "land_km2": ld.sum() * m.grid.cell_km2,
                             "IoU_vs_full": (ld & b_land).sum() / (ld | b_land).sum(),
                             "spearman": spearmanr(per, b_rank)[0]})
        return pd.DataFrame(rows)
