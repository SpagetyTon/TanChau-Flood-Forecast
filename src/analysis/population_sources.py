"""Second population source (GHS-POP R2023A, 2020) for the exposure comparison E5 (paper Sect. 4.6).

GHS-POP replaces WorldPop on the same 64 m grid by the same two rules: (1) pixel values are added to grid cells in
proportion to the overlapping area (population-conserving regridding), (2) people on river and canal cells are moved
to the nearest land cell.  Requires ``tifffile`` and ``imagecodecs`` for the one-off download.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

GHS_TILE_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
                "GHS_POP_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_{tile}.zip")


def fetch_ghs_crop(out_npz: str | Path, extent=(104.95, 105.55, 10.5, 11.08), tile="R8_C29", pad=3) -> None:
    """Download the GHS-POP 2020 tile that covers the study area (about 34 MB) and keep a small crop."""
    import tifffile
    data = urllib.request.urlopen(GHS_TILE_URL.format(tile=tile), timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(".tif"))
        page = tifffile.TiffFile(io.BytesIO(z.read(name))).pages[0]
        ox, oy = page.tags["ModelTiepointTag"].value[3:5]
        ps = page.tags["ModelPixelScaleTag"].value[0]
        w, e, s, n = extent
        c0, c1 = int((w - ox) / ps) - pad, int((e - ox) / ps) + pad + 1
        r0, r1 = int((oy - n) / ps) - pad, int((oy - s) / ps) + pad + 1
        a = page.asarray()[r0:r1, c0:c1].astype("float32")
    np.savez_compressed(out_npz, a=a, ox=ox + c0 * ps, oy=oy - r0 * ps, ps=ps)


def _overlap(t_edges, s_edges):
    """Matrix (targets x sources) of overlap length divided by source length (edges increasing)."""
    nt, ns = len(t_edges) - 1, len(s_edges) - 1
    M = np.zeros((nt, ns))
    for i in range(nt):
        lo, hi = t_edges[i], t_edges[i + 1]
        j0 = max(0, np.searchsorted(s_edges, lo, "right") - 1)
        j1 = min(ns, np.searchsorted(s_edges, hi, "left"))
        for j in range(j0, j1):
            ov = min(hi, s_edges[j + 1]) - max(lo, s_edges[j])
            if ov > 0:
                M[i, j] = ov / (s_edges[j + 1] - s_edges[j])
    return M


def ghs_on_grid(crop_npz: str | Path, model) -> np.ndarray:
    """GHS-POP people per grid cell (same grid and river rule as the WorldPop layer)."""
    z = np.load(crop_npz)
    a, ox, oy, ps = z["a"], float(z["ox"]), float(z["oy"]), float(z["ps"])
    g = model.grid
    t_lon = g.west + np.arange(g.ncols + 1) * (g.east - g.west) / g.ncols          # ncols equal cells over the extent
    t_lat = g.north - np.arange(g.nrows + 1) * (g.north - g.south) / g.nrows
    s_lon = ox + np.arange(a.shape[1] + 1) * ps
    s_lat = oy - np.arange(a.shape[0] + 1) * ps
    M_lon = _overlap(t_lon, s_lon)
    M_lat = _overlap(t_lat[::-1], s_lat[::-1])[::-1, ::-1]
    raw = M_lat @ a @ M_lon.T
    riv = model.river > 0
    near = ndimage.distance_transform_edt(riv, return_distances=False, return_indices=True)
    moved = np.zeros_like(raw)
    np.add.at(moved, (near[0][riv], near[1][riv]), raw[riv])
    return (np.where(riv, 0, raw) + moved).astype(np.float32)


def e5_ghs_table(model, ghs_pop, levels=(2.5, 3.17, 3.5, 4.0, 4.5, 4.7)):
    import pandas as pd
    in_c, n = model.adm > 0, len(model.adm_names) + 1
    rows = []
    for L in levels:
        _, d = model.flood(L)
        wet = d > 0
        w, g = model.pop[wet & in_c].sum(), ghs_pop[wet & in_c].sum()
        pw = np.bincount(model.adm[wet].ravel(), weights=model.pop[wet].ravel(), minlength=n)[1:]
        pg = np.bincount(model.adm[wet].ravel(), weights=ghs_pop[wet].ravel(), minlength=n)[1:]
        tw = np.bincount(model.adm.ravel(), weights=model.pop.ravel(), minlength=n)[1:]
        tg = np.bincount(model.adm.ravel(), weights=ghs_pop.ravel(), minlength=n)[1:]
        top = lambda x: set(np.argsort(-x)[:10])
        rows.append({"level_m": L, "WorldPop": w, "GHS-POP": g, "difference_%": 100 * (g / w - 1),
                     "rho_people": spearmanr(pw, pg)[0], "rho_share": spearmanr(pw / tw, pg / tg)[0],
                     "top10_overlap": len(top(pw) & top(pg))})
    return pd.DataFrame(rows)
