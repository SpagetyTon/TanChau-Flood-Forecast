"""Validation of the flood map against Sentinel-1 water masks (E3, paper Sect. 4.4 and Supplement S5).

Workflow (paper S5)
  1. Six flood-season scenes (Sentinel-1 GRD, IW, VV, relative orbit 26, ~18:11 local time) outside the JRC period and
     three dry-season scenes of April 2024 (same orbit), obtained from the Microsoft Planetary Computer.
  2. Each scene is projected to the model grid (64 m, average resampling), converted to dB, median-filtered (3 x 3),
     thresholded with Otsu's method, then opened and closed (3 x 3).
  3. Cells wet in at least two of the three April scenes are permanent water and, with river cells, excluded.
  4. The map is run at the daily-mean Tan Chau stage of the scene date and compared on land cells of the 54 communes.

``fetch_scenes`` downloads and projects the scenes (needs ``rasterio``; about 25 s per scene), ``water_masks`` turns
them into masks, and ``masks.npz`` in ``data/derived`` stores the result so the comparison runs without downloading.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd"
# date -> (platform, Tan Chau daily-mean stage in m)  [paper Table 'e3']
SCENES = {"2023-10-20": ("S1A", 2.74), "2024-10-02": ("S1A", 3.11), "2025-10-03": ("S1C", 3.40),
          "2022-10-13": ("S1A", 3.49), "2025-10-15": ("S1C", 3.67), "2025-10-09": ("S1A", 3.75)}
DRY_SEASON = ["2024-04-05", "2024-04-17", "2024-04-29"]
ORBIT = 26


def _search(date: str, bbox):
    body = {"collections": ["sentinel-1-grd"], "bbox": list(bbox), "limit": 50,
            "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z"}
    req = urllib.request.Request(STAC, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    feats = json.loads(urllib.request.urlopen(req, timeout=60).read())["features"]
    return [f for f in feats if f["properties"].get("sat:relative_orbit") == ORBIT][0]


def fetch_scenes(model, out_dir: str | Path) -> None:
    """Download the nine scenes and project each to the model grid (``<date>.npy``, mean DN amplitude, uint-like float)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    g = model.grid
    dlon, dlat = (g.east - g.west) / (g.ncols - 1), (g.north - g.south) / (g.nrows - 1)
    dst_tr = from_origin(g.west - dlon / 2, g.north + dlat / 2, dlon, dlat)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for date in list(SCENES) + DRY_SEASON:
        item = _search(date, (g.west, g.south, g.east, g.north))
        token = json.loads(urllib.request.urlopen(SAS, timeout=60).read())["token"]
        with rasterio.open(item["assets"]["vv"]["href"] + "?" + token) as src:
            gcps, crs = src.gcps
            dst = np.zeros((g.nrows, g.ncols), np.float32)
            reproject(source=rasterio.band(src, 1), destination=dst, gcps=gcps, src_crs=crs, dst_transform=dst_tr,
                      dst_crs="EPSG:4326", resampling=Resampling.average, src_nodata=0, dst_nodata=0, num_threads=4)
        np.save(out / f"{date}.npy", dst)


def otsu(x: np.ndarray, bins: int = 256) -> float:
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w1 = np.cumsum(h); w2 = w1[-1] - w1
    m1 = np.cumsum(h * c) / np.maximum(w1, 1)
    m2 = (np.cumsum((h * c)[::-1]) / np.maximum(np.cumsum(h[::-1]), 1))[::-1]
    return float(c[:-1][np.argmax(w1[:-1] * w2[:-1] * (m1[:-1] - m2[1:]) ** 2)])


def water_mask(dn: np.ndarray, shift_db: float = 0.0) -> np.ndarray:
    """Water mask from a projected VV amplitude image (steps 2 of the workflow)."""
    db = ndimage.median_filter(20 * np.log10(np.maximum(dn, 1)), size=3)
    w = db < otsu(db) + shift_db
    st = np.ones((3, 3), bool)
    return ndimage.binary_closing(ndimage.binary_opening(w, st), st)


def build_masks(scene_dir: str | Path, out_npz: str | Path) -> None:
    d = Path(scene_dir)
    masks = {k: water_mask(np.load(d / f"{k}.npy")) for k in list(SCENES) + DRY_SEASON}
    np.savez_compressed(out_npz, **{k: v for k, v in masks.items()})


class Sentinel1Validation:
    def __init__(self, model, masks_npz: str | Path):
        self.m = model
        z = np.load(masks_npz)
        self.masks = {k: z[k] for k in z.files}
        self.permanent = np.sum([self.masks[k] for k in DRY_SEASON], axis=0) >= 2
        self.domain = (model.adm > 0) & (model.river == 0) & ~self.permanent

    @staticmethod
    def _metrics(pred, obs):
        tp, fp, fn = (pred & obs).sum(), (pred & ~obs).sum(), (~pred & obs).sum()
        return tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn)

    def table(self):
        import pandas as pd
        m, rows = self.m, []
        n = len(m.adm_names) + 1
        for date, (sat, L) in SCENES.items():
            sar = self.masks[date] & self.domain
            _, dm = m.flood(L)
            _, dd = m.flood(L, mode="dem")
            pm, pd_ = (dm > 0) & self.domain, (dd > 0) & self.domain
            iou, f1, prec, rec = self._metrics(pm, sar)
            ppl = lambda p: np.bincount(m.adm[p].ravel(), weights=m.pop[p].ravel(), minlength=n)[1:]
            rows.append({"date": date, "satellite": sat, "stage_m": L, "IoU": iou, "F1": f1, "precision": prec, "recall": rec,
                         "IoU_DEM": self._metrics(pd_, sar)[0], "map_km2": pm.sum() * m.grid.cell_km2,
                         "SAR_km2": sar.sum() * m.grid.cell_km2,
                         "rho_map": spearmanr(ppl(pm), ppl(sar))[0], "rho_DEM": spearmanr(ppl(pd_), ppl(sar))[0]})
        return pd.DataFrame(rows).sort_values("stage_m").reset_index(drop=True)

    def threshold_sensitivity(self, scene_dir: str | Path, shifts=(-2.0, 2.0)):
        """Largest change of IoU when the Otsu threshold is moved by +/- 2 dB (paper: not more than 0.04)."""
        out = []
        for date, (_, L) in SCENES.items():
            dn = np.load(Path(scene_dir) / f"{date}.npy")
            _, dm = self.m.flood(L)
            pm = (dm > 0) & self.domain
            base = self._metrics(pm, water_mask(dn) & self.domain)[0]
            for s in shifts:
                out.append({"date": date, "shift_dB": s, "dIoU": self._metrics(pm, water_mask(dn, s) & self.domain)[0] - base})
        import pandas as pd
        return pd.DataFrame(out)

    @property
    def domain_km2(self):
        return self.domain.sum() * self.m.grid.cell_km2
