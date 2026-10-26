"""Flood map and exposure model for the Tan Chau study area (Python port of the demo's JavaScript).

Pipeline (paper Sect. 3.4-3.6):
  1. DEM (Copernicus GLO-30, resampled to a 1024 x 1024 grid, ~64 m cells), river mask (OpenStreetMap),
     JRC Global Surface Water occurrence (1984-2021), WorldPop 2020 population, commune ids.
  2. Each cell gets a trigger level h*: the Tan Chau stage whose exceedance frequency over 1984-2021 equals the
     satellite-observed water frequency O of that cell (``trigger_levels``).
  3. A cell is flooded at stage L if L > h*; "lost seasonal" cells (probable dike-protected) need L > 5.2 m.
  4. Above the highest satellite-observed stage (5.06 m) flooding also spreads through the DEM from rivers.
  5. Exposure = WorldPop people in flooded cells; aggregated per commune.

The gridded inputs are stored in ``data/derived/grids_public.npz`` (see ``convert_demo_grids``).
"""
from __future__ import annotations

import gzip
import json
import base64
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

# ---- constants (identical to the demo) -----------------------------------------------------------------------------
SAT_MAX = 5.06        # m, highest stage in the satellite period (2000 flood)
DIKE_LEVEL = 5.2      # m, assumed protection level of closed dike systems
BASEFLOW = 0.8        # m at Tan Chau, minimum dry-season stage (keeps rivers wet)
CHANNEL_DEPTH = 10.0  # m below dry-season surface, main rivers (assumed)
CANAL_DEPTH = 3.0     # m, canals and small branches (assumed)
OCC_PERMANENT = 90    # % occurrence above which a cell is treated as permanent water


@dataclass
class Grid:
    ncols: int
    nrows: int
    west: float
    east: float
    south: float
    north: float
    gauge_lat: float
    gauge_lon: float

    @property
    def km(self):
        lat_mid = (self.south + self.north) / 2
        kmlon = 111.320 * np.cos(np.radians(lat_mid))
        kmlat = 110.574
        return kmlon, kmlat

    @property
    def cell_km2(self):
        kmlon, kmlat = self.km
        wkm = (self.east - self.west) * kmlon
        hkm = (self.north - self.south) * kmlat
        return (wkm / (self.ncols - 1)) * (hkm / (self.nrows - 1))


# ---- decoding of the demo's embedded rasters -----------------------------------------------------------------------
def _js_payload(path: Path) -> dict:
    """Read ``window.X={...};`` into a dict (the payloads are plain JSON objects with unquoted keys for some)."""
    txt = path.read_text(encoding="utf-8")
    body = txt[txt.index("=") + 1:].strip().rstrip(";")
    body = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', body)
    return json.loads(body)


def _gunzip(b64: str) -> bytes:
    return gzip.decompress(base64.b64decode(b64))


def decode_dem(p: dict) -> np.ndarray:
    n = p["ncols"] * p["nrows"]
    raw = np.frombuffer(_gunzip(p["b64"]), dtype=np.uint8)
    hi, lo = raw[:n].astype(np.int64), raw[n:2 * n].astype(np.int64)
    zz = (hi << 8) | lo
    d = (zz >> 1) ^ (-(zz & 1))                       # zig-zag delta decoding
    q = np.cumsum(d.reshape(p["nrows"], p["ncols"]), axis=1)
    return np.where(q <= 1200, q / 10.0 - 20.0, 100.0 + (q - 1200)).astype(np.float32)


def decode_sat(p: dict, ncols: int):
    raw = np.frombuffer(_gunzip(p["b64"]), dtype=np.uint8).reshape(p["n"], p["n"])
    if p["n"] != ncols:
        raise ValueError("SAT grid must match the DEM grid")
    q = (raw & 31).astype(np.int32)
    occ = np.where(q <= 20, q, np.where(q >= 31, 98, 20 + (q - 20) * 8 - 4)).astype(np.uint8)
    lost = ((raw >> 5) & 1).astype(bool)
    return occ, lost


def decode_pop(p: dict, shape) -> np.ndarray:
    u = np.frombuffer(_gunzip(p["b64"]), dtype="<u2")
    return (u.astype(np.float32) * p["scale"]).reshape(shape)


def decode_river(p: dict, shape) -> np.ndarray:
    raw = np.frombuffer(_gunzip(p["b64"]), dtype=np.uint8)
    n = shape[0] * shape[1]
    i = np.arange(n)
    return ((raw[i >> 2] >> ((i & 3) * 2)) & 3).reshape(shape).astype(np.uint8)


def decode_adm(p: dict, shape):
    ids = np.frombuffer(_gunzip(p["ids"]), dtype=np.uint8).reshape(shape)
    return ids, p["names"]


def convert_demo_grids(demo_dir: str | Path, out_npz: str | Path, out_calib_json: str | Path | None = None) -> None:
    """Convert the demo's embedded JavaScript payloads to a compact .npz (public data) and a calibration .json.

    Public inputs (Copernicus GLO-30 DEM, JRC Global Surface Water, WorldPop 2020, OpenStreetMap rivers, commune
    boundaries) go into the .npz. The calibration (exceedance curve, water-surface gradients, annual maxima) is
    derived from MRC gauge data and is written separately because it is subject to the MRC licence.
    """
    d = Path(demo_dir)
    dem = _js_payload(d / "dem.js")
    shape = (dem["nrows"], dem["ncols"])
    occ, lost = decode_sat(_js_payload(d / "sat.js"), dem["ncols"])
    adm, names = decode_adm(_js_payload(d / "adm.js"), shape)
    np.savez_compressed(out_npz, elev=decode_dem(dem), occ=occ, lost=lost,
                        pop=decode_pop(_js_payload(d / "pop.js"), shape),
                        river=decode_river(_js_payload(d / "river.js"), shape),
                        adm=adm, adm_names=np.array(names),
                        extent=np.array([dem["west"], dem["east"], dem["south"], dem["north"]]))
    if out_calib_json is not None:
        Path(out_calib_json).write_text(json.dumps(_js_payload(d / "calib.js")), encoding="utf-8")


class FloodModel:
    """``FloodModel(grids_npz, calib_json)``; see the module docstring for the method."""

    def __init__(self, grids_npz: str | Path, calib_json: str | Path):
        g = np.load(grids_npz)
        cal = json.loads(Path(calib_json).read_text(encoding="utf-8"))
        shape = g["elev"].shape
        west, east, south, north = g["extent"]
        self.grid = Grid(shape[1], shape[0], float(west), float(east), float(south), float(north),
                         cal["gauge"][0], cal["gauge"][1])
        self.elev0 = g["elev"]
        self.occ, self.lost = g["occ"], g["lost"].astype(bool)
        self.pop, self.river = g["pop"], g["river"]
        self.adm, self.adm_names = g["adm"], [str(x) for x in g["adm_names"]]
        self.cal = cal
        self.H = np.array(cal["H"]); self.exc = np.array(cal["exc"])
        self.grad = np.array(cal["grad"])             # rows: stage, east gradient, north gradient (m/km)
        lat = self.grid.north - (self.grid.north - self.grid.south) * (np.arange(shape[0]) / (shape[0] - 1))
        lon = self.grid.west + (self.grid.east - self.grid.west) * (np.arange(shape[1]) / (shape[1] - 1))
        kmlon, kmlat = self.grid.km
        self._dx = ((lon - self.grid.gauge_lon) * kmlon)[None, :]
        self._dy = ((lat - self.grid.gauge_lat) * kmlat)[:, None]
        self.elev = self.elev0.copy()                 # carved DEM (rivers lowered)
        self._carve_river()
        self.h_star = self._trigger_levels()

    # -- water surface: tilted plane, gradient depends on the Tan Chau stage ------------------------------------------
    def _gradient(self, L: float):
        g = self.grad
        gx = np.interp(L, g[:, 0], g[:, 1])
        gy = np.interp(L, g[:, 0], g[:, 2])
        return gx, gy

    def water_surface(self, L: float) -> np.ndarray:
        gx, gy = self._gradient(L)
        return (L + gx * self._dx + gy * self._dy).astype(np.float32)

    def _carve_river(self):
        bed = self.water_surface(BASEFLOW) - np.where(self.river == 2, CHANNEL_DEPTH, CANAL_DEPTH)
        m = self.river > 0
        self.elev[m] = np.minimum(self.elev[m], bed[m])

    # -- trigger level from the satellite frequency -------------------------------------------------------------------
    def _level_for_exceed(self, p: float) -> float:
        H, X = self.H, self.exc
        if p >= X[0]:
            return float(H[0])
        for t in range(1, len(H)):
            if X[t] <= p:
                f = (X[t - 1] - p) / ((X[t - 1] - X[t]) or 1.0)
                return float(H[t - 1] + (H[t] - H[t - 1]) * f)
        return SAT_MAX

    def _trigger_levels(self) -> np.ndarray:
        lut = np.zeros(101, dtype=np.float32)
        for o in range(1, 101):
            lut[o] = self._level_for_exceed(o / 100.0)
        h = np.where(self.occ > 0, np.minimum(lut[self.occ], SAT_MAX), 99.0).astype(np.float32)
        h[self.river > 0] = -99.0
        return h

    # -- DEM-connected flooding (only used above SAT_MAX) -------------------------------------------------------------
    def _connected(self, wl: np.ndarray, elev: np.ndarray | None = None) -> np.ndarray:
        wet = wl > (self.elev if elev is None else elev)
        lab, _ = ndimage.label(wet, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        seeds = np.zeros_like(wet)
        seeds[self.river > 0] = True
        seeds[0, :] = seeds[-1, :] = True
        seeds[:, 0] = seeds[:, -1] = True
        keep = np.unique(lab[seeds & wet])
        keep = keep[keep > 0]
        return np.isin(lab, keep)

    # -- flood mask and depth -----------------------------------------------------------------------------------------
    def trigger_levels_scaled(self, occ_scale: float = 1.0) -> np.ndarray:
        """Trigger levels after scaling the satellite frequency (sensitivity test, E4)."""
        if occ_scale == 1.0:
            return self.h_star.copy()
        lut = np.zeros(101, dtype=np.float32)
        for o in range(1, 101):
            lut[o] = self._level_for_exceed(o / 100.0)
        occ = np.clip(np.rint(self.occ.astype(np.float32) * occ_scale), 0, 100).astype(int)
        h = np.where(occ > 0, np.minimum(lut[occ], SAT_MAX), 99.0).astype(np.float32)
        h[self.river > 0] = -99.0
        return h

    def flood(self, L: float, dike: bool = True, mode: str = "sat", tilt: bool = True,
              th_noise: np.ndarray | None = None, occ_scale: float = 1.0, elev: np.ndarray | None = None):
        """Return (flooded mask, depth in m).

        mode  'sat': satellite-calibrated trigger levels (the paper's method);
              'dem': flood where the water surface exceeds the DEM and is connected to a river or the border (E8).
        tilt  False replaces the tilted water surface by a flat one (E8).
        th_noise  additive noise on the trigger levels (E4); elev  replacement DEM (E4 DEM-error test).
        """
        elev = self.elev if elev is None else elev
        riv = self.river > 0
        wl = self.water_surface(L) if tilt else np.full(self.elev.shape, L, np.float32)
        wbase = self.water_surface(BASEFLOW) if tilt else np.full(self.elev.shape, BASEFLOW, np.float32)
        wl = np.where(riv & (wbase > wl), wbase, wl)
        if mode == "dem":
            f = self._connected(wl, elev)
            depth = np.where(f, wl - elev, 0.0)
            return f, depth.astype(np.float32)
        th = self.trigger_levels_scaled(occ_scale)
        if th_noise is not None:
            m = (th < 99) & ~riv
            th = np.where(m, th + th_noise, th)
        if dike:
            m = self.lost & ~riv & (th < DIKE_LEVEL)
            th[m] = DIKE_LEVEL
        f = riv | (L > th)
        if L > SAT_MAX:
            dem_wet = self._connected(wl, elev)
            f |= (th == 99) & dem_wet
            if dike:
                f |= self.lost & (L > DIKE_LEVEL) & dem_wet
        depth = np.where(f, wl - elev, 0.0)
        depth = np.where(f & (depth < 0.15), 0.15, depth)    # minimal film on satellite-flooded cells
        return f, depth.astype(np.float32)

    # -- statistics ---------------------------------------------------------------------------------------------------
    def stats(self, L: float, dike: bool = True) -> dict:
        f, depth = self.flood(L, dike)
        wet = depth > 0
        land = wet & (self.river == 0) & ~(self.occ >= OCC_PERMANENT)
        cell = self.grid.cell_km2
        return {
            "level_m": L,
            "area_km2": float(land.sum() * cell),
            "pct_of_domain": float(100 * land.sum() / land.size),
            "mean_depth_m": float(depth[land].mean()) if land.any() else 0.0,
            "max_depth_m": float(depth[land].max()) if land.any() else 0.0,
            "people": float(self.pop[wet].sum()),
            "people_gt_1m": float(self.pop[wet & (depth > 1)].sum()),
            "total_people": float(self.pop.sum()),
        }

    def by_commune(self, L: float, dike: bool = True):
        f, depth = self.flood(L, dike)
        wet = depth > 0
        land = wet & (self.river == 0) & ~(self.occ >= OCC_PERMANENT)
        n = len(self.adm_names) + 1
        ppl = np.bincount(self.adm[wet].ravel(), weights=self.pop[wet].ravel(), minlength=n)
        tot = np.bincount(self.adm.ravel(), weights=self.pop.ravel(), minlength=n)
        area = np.bincount(self.adm[land].ravel(), minlength=n) * self.grid.cell_km2
        return ppl, tot, area

    def table_row(self, L: float, dike: bool = True) -> dict:
        """One row of the exposure-by-stage table (paper Table 'explev')."""
        st = self.stats(L, dike)
        ppl, tot, area = self.by_commune(L, dike)
        ppl, tot, area = ppl[1:], tot[1:], area[1:]          # drop id 0 = outside the 54 communes
        has_land = (ppl >= 5) | (area >= 0.1)
        affected = (ppl / np.maximum(tot, 1) >= 0.05) | (area >= 2.0)
        return {"level_m": L, "people": round(st["people"]), "pct_grid_pop": round(100 * st["people"] / st["total_people"], 1),
                "land_km2": round(st["area_km2"]), "communes_with_land": int(has_land.sum()),
                "communes_affected": int(affected.sum()), "people_gt_1m": round(st["people_gt_1m"])}
