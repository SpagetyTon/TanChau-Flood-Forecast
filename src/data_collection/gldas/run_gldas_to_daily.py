#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import getpass
import json
import math
import netrc
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset
from pyproj import Transformer
from shapely.geometry import Polygon, shape
from shapely.ops import transform as shp_transform


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

MANIFEST = ROOT / CONFIG["manifest"]
UPSTREAM_GEOJSON = ROOT / CONFIG["upstream_geojson"]
OUT_DIR = ROOT / CONFIG["output_dir"]
CACHE_DIR = ROOT / CONFIG["cache_dir"]

OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

FINAL_CSV = OUT_DIR / "GLDAS_daily.csv"
PARTIAL_CSV = OUT_DIR / "GLDAS_daily_partial.csv"
META_CSV = OUT_DIR / "GLDAS_daily_metadata.csv"
WEIGHTS_CACHE = CACHE_DIR / "gldas_upstream_weights.npz"

VARS = CONFIG["selected_variables"]
SUM_VARS = set(CONFIG["sum_variables"])
INTEGRATE_VARS = set(CONFIG["integrate_10800s_variables"])
MEAN_VARS = set(CONFIG["mean_variables"])

START = pd.Timestamp(CONFIG["start_date"])
END = pd.Timestamp(CONFIG["end_date"])


class EarthdataBearerSession(requests.Session):
    """Session authenticated with an Earthdata user Bearer token."""
    def __init__(self, token: str):
        super().__init__()
        self.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mekong-TanChau-GLDAS-Pipeline/3.0",
        })

    def rebuild_auth(self, prepared_request, response):
        # Keep the Bearer token on redirects only for NASA Earthdata/GES DISC hosts.
        host = (urlparse(prepared_request.url).hostname or "").lower()
        allowed = (
            host == "urs.earthdata.nasa.gov"
            or host.endswith(".gesdisc.eosdis.nasa.gov")
            or host.endswith(".earthdata.nasa.gov")
        )
        if not allowed:
            prepared_request.headers.pop("Authorization", None)


def get_or_create_edl_token(username: str, password: str):
    """Validate credentials and obtain an EDL user token using the official API."""
    url = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
    print("Checking Earthdata username/password with the official EDL API...")
    r = requests.post(
        url,
        auth=(username, password),
        timeout=60,
        headers={"User-Agent": "Mekong-TanChau-GLDAS-Pipeline/3.0"},
    )

    if r.status_code == 401:
        raise RuntimeError(
            "EARTHDATA CREDENTIAL CHECK FAILED (HTTP 401). "
            "The username or password is not accepted by Earthdata Login. "
            "Please verify the exact USERNAME shown in your Earthdata profile "
            "(not your email/display name) and verify that you can log out and "
            "log back in on the Earthdata website with the same credentials."
        )
    r.raise_for_status()

    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(
            "Earthdata credential check succeeded, but no access_token was returned."
        )

    print("Earthdata credential check: PASS")
    print(f"EDL token expiry: {data.get('expiration_date', 'unknown')}")
    return token



def get_credentials():
    """Read Earthdata credentials without storing secrets in source code."""
    import getpass
    username = os.environ.get("EARTHDATA_USERNAME") or input("Earthdata username: " ).strip()
    if not username:
        raise RuntimeError("Earthdata username cannot be empty.")
    password = os.environ.get("EARTHDATA_PASSWORD") or getpass.getpass("Earthdata password: " )
    if not password:
        raise RuntimeError("Earthdata password cannot be empty.")
    return username, password


def parse_manifest():
    urls = []
    pat = re.compile(r"A(\d{8})\.(\d{4})")
    for line in MANIFEST.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("http"):
            continue
        if "GLDAS_NOAH025_3H" not in line:
            continue
        m = pat.search(line)
        if not m:
            continue
        ts = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M")
        if START <= ts.normalize() <= END:
            urls.append((ts, line))

    urls.sort(key=lambda x: x[0])

    expected = int(CONFIG["expected_urls"])
    if len(urls) != expected:
        raise RuntimeError(f"Manifest has {len(urls):,} GLDAS URLs; expected {expected:,}.")

    groups = defaultdict(list)
    for ts, url in urls:
        groups[ts.normalize()].append((ts, url))

    bad = [(d, len(v)) for d, v in groups.items() if len(v) != 8]
    if bad:
        raise RuntimeError(f"Some dates do not have exactly 8 timesteps. First issues: {bad[:10]}")

    expected_days = pd.date_range(START, END, freq="D")
    missing_days = expected_days.difference(pd.DatetimeIndex(sorted(groups)))
    if len(missing_days):
        raise RuntimeError(f"Manifest is missing {len(missing_days)} days.")

    return groups


def output_name_from_url(url: str):
    qs = parse_qs(urlparse(url).query)
    if "LABEL" in qs:
        return qs["LABEL"][0]
    m = re.search(r"GLDAS_NOAH025_3H\.A\d{8}\.\d{4}\.021\.nc4", url)
    return (m.group(0) + ".SUB.nc4") if m else "gldas_download.nc4"


def is_netcdf(path: Path):
    with path.open("rb") as f:
        sig = f.read(8)
    return sig.startswith(b"CDF") or sig == b"\x89HDF\r\n\x1a\n"


def download_file(session, url: str, dest: Path, retries=5):
    if dest.exists() and dest.stat().st_size > 1000 and is_netcdf(dest):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(30, 300), allow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            if tmp.stat().st_size < 1000 or not is_netcdf(tmp):
                text = tmp.read_bytes()[:500].decode("utf-8", errors="ignore")
                if "<!DOCTYPE html" in text or "<html" in text.lower():
                    raise RuntimeError(
                        "EARTHDATA LOGIN HTML WAS RETURNED INSTEAD OF NETCDF. "
                        "The username/password check already passed, so this is a "
                        "GES DISC access/OAuth issue rather than a bad password. "
                        "Your Authorized Apps screenshot should include GESDISC and "
                        "NASA GESDISC DATA ARCHIVE. If they are already authorized, "
                        "open one GLDAS subset link from the manifest in the same "
                        "browser session once, accept any GES DISC prompt, and rerun. "
                        f"Response starts with: {text[:180]!r}"
                    )
                raise RuntimeError(
                    "Downloaded response is not NetCDF. "
                    f"Response starts with: {text[:300]!r}"
                )

            tmp.replace(dest)
            return dest

        except Exception as e:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  download retry {attempt}/{retries}: {e}; waiting {wait}s")
            time.sleep(wait)


def detect_lat_lon(ds: Dataset):
    lat_candidates = ["lat", "latitude", "Latitude"]
    lon_candidates = ["lon", "longitude", "Longitude"]

    lat_name = next((x for x in lat_candidates if x in ds.variables), None)
    lon_name = next((x for x in lon_candidates if x in ds.variables), None)
    if not lat_name or not lon_name:
        raise RuntimeError(f"Could not detect lat/lon variables. Found: {list(ds.variables)[:30]}")

    lat = np.asarray(ds.variables[lat_name][:], dtype=float).squeeze()
    lon = np.asarray(ds.variables[lon_name][:], dtype=float).squeeze()
    return lat_name, lon_name, lat, lon


def load_upstream_geometry():
    obj = json.loads(UPSTREAM_GEOJSON.read_text(encoding="utf-8"))
    return shape(obj["features"][0]["geometry"])


def build_weights(lat: np.ndarray, lon: np.ndarray):
    if WEIGHTS_CACHE.exists():
        cached = np.load(WEIGHTS_CACHE)
        if np.allclose(cached["lat"], lat) and np.allclose(cached["lon"], lon):
            print("Using cached HydroBASINS/GLDAS cell weights.")
            return cached["weights"]

    print("Building area-intersection weights from HydroBASINS upstream polygon...")
    upstream = load_upstream_geometry()

    to_equal_area = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    upstream_ea = shp_transform(to_equal_area.transform, upstream)

    dlat = float(np.median(np.abs(np.diff(lat))))
    dlon = float(np.median(np.abs(np.diff(lon))))

    weights = np.zeros((len(lat), len(lon)), dtype=np.float64)
    minx, miny, maxx, maxy = upstream.bounds

    for i, la in enumerate(lat):
        if la + dlat / 2 < miny or la - dlat / 2 > maxy:
            continue
        for j, lo in enumerate(lon):
            if lo + dlon / 2 < minx or lo - dlon / 2 > maxx:
                continue

            cell = Polygon([
                (lo - dlon/2, la - dlat/2),
                (lo + dlon/2, la - dlat/2),
                (lo + dlon/2, la + dlat/2),
                (lo - dlon/2, la + dlat/2),
            ])
            if not cell.intersects(upstream):
                continue

            cell_ea = shp_transform(to_equal_area.transform, cell)
            inter_area = cell_ea.intersection(upstream_ea).area
            if inter_area > 0:
                weights[i, j] = inter_area

    if weights.sum() <= 0:
        raise RuntimeError("No GLDAS cells intersect the upstream basin.")

    np.savez_compressed(WEIGHTS_CACHE, lat=lat, lon=lon, weights=weights)
    print(f"Weighted GLDAS cells: {(weights > 0).sum():,}")
    print(f"Intersected upstream area: {weights.sum()/1e6:,.1f} km²")
    return weights


def read_2d_variable(ds: Dataset, var_name: str, lat_name: str, lon_name: str, nlat: int, nlon: int):
    if var_name not in ds.variables:
        raise RuntimeError(f"Variable {var_name} is missing in {getattr(ds, 'filepath', lambda: '')()}")

    var = ds.variables[var_name]
    dims = list(var.dimensions)

    index = []
    for d in dims:
        if d == lat_name or d == lon_name:
            index.append(slice(None))
        else:
            # GLDAS subset files contain singleton time dimension.
            index.append(0)

    arr = var[tuple(index)]
    arr = np.ma.filled(arr, np.nan).astype(np.float64)
    arr = np.squeeze(arr)

    if arr.shape == (nlat, nlon):
        return arr
    if arr.shape == (nlon, nlat):
        return arr.T

    raise RuntimeError(
        f"Unexpected shape for {var_name}: {arr.shape}; expected {(nlat, nlon)}."
    )


def weighted_spatial_mean(arr: np.ndarray, weights: np.ndarray):
    valid = np.isfinite(arr) & (weights > 0)
    if not np.any(valid):
        return np.nan
    w = weights[valid]
    return float(np.sum(arr[valid] * w) / np.sum(w))


def aggregate_day(step_values):
    # step_values: dict[var] -> list of 8 basin-mean timestep values
    row = {}
    for v, values in step_values.items():
        x = np.asarray(values, dtype=float)
        if np.isnan(x).any():
            raise RuntimeError(f"{v} has missing timestep value within the day.")

        if v in SUM_VARS:
            row[v] = float(np.sum(x))
        elif v in INTEGRATE_VARS:
            # kg m-2 s-1 × 10800 s per 3-hour step.
            row[v] = float(np.sum(x * 10800.0))
        elif v in MEAN_VARS:
            row[v] = float(np.mean(x))
        else:
            raise RuntimeError(f"No daily aggregation rule declared for {v}")
    return row


def write_metadata():
    rows = []
    for v in VARS:
        if v in SUM_VARS:
            method = "SUM of 8 three-hour accumulations"
            daily_unit = "kg m-2 day-1"
        elif v in INTEGRATE_VARS:
            method = "SUM(value × 10800 s) across 8 timesteps"
            daily_unit = "kg m-2 day-1 (≈ mm/day for water)"
        else:
            method = "MEAN of 8 three-hour values"
            daily_unit = "same as source"
        rows.append({"feature": v, "daily_aggregation": method, "daily_unit": daily_unit})
    pd.DataFrame(rows).to_csv(META_CSV, index=False)


def finalize():
    df = pd.read_csv(PARTIAL_CSV, parse_dates=["date"])
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    expected = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})
    df = expected.merge(df, on="date", how="left", validate="one_to_one")

    missing = df[VARS].isna().any(axis=1)
    if missing.any():
        bad = df.loc[missing, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise RuntimeError(
            f"GLDAS daily output has {len(bad)} incomplete dates. First: {bad[:20]}"
        )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(FINAL_CSV, index=False)
    write_metadata()
    print(f"\nDONE: {FINAL_CSV}")
    print(f"Rows: {len(df):,}; variables: {len(VARS)}; missing predictor cells: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Process only N unprocessed days (useful for a one-day test)."
    )
    args = parser.parse_args()

    print("Reading and validating GLDAS manifest...")
    groups = parse_manifest()
    print(f"Manifest PASS: {sum(map(len, groups.values())):,} URLs; {len(groups):,} days.")

    already = set()
    if PARTIAL_CSV.exists():
        try:
            old = pd.read_csv(PARTIAL_CSV, usecols=["date"])
            already = set(pd.to_datetime(old["date"]).dt.normalize())
            print(f"Resume mode: {len(already):,} daily rows already completed.")
        except Exception:
            pass

    remaining = [d for d in sorted(groups) if d not in already]
    if args.max_days is not None:
        remaining = remaining[:args.max_days]

    if not remaining:
        if PARTIAL_CSV.exists():
            finalize()
        else:
            print("Nothing to process.")
        return

    username, password = get_credentials()
    token = get_or_create_edl_token(username, password)
    session = EarthdataBearerSession(token)
    print("Using Earthdata Bearer-token authentication for GES DISC download.")

    # Download one sample file to detect actual GLDAS grid and build weights.
    sample_date = remaining[0]
    sample_ts, sample_url = groups[sample_date][0]
    day_dir = CACHE_DIR / "current_day"
    day_dir.mkdir(exist_ok=True)
    sample_path = day_dir / output_name_from_url(sample_url)

    print(f"\nDownloading one sample file ({sample_ts}) to verify the actual NetCDF structure...")
    download_file(session, sample_url, sample_path)

    with Dataset(sample_path, "r") as ds:
        lat_name, lon_name, lat, lon = detect_lat_lon(ds)
        missing_vars = [v for v in VARS if v not in ds.variables]
        if missing_vars:
            raise RuntimeError(f"Sample GLDAS file is missing selected variables: {missing_vars}")
        print(f"GLDAS grid: {len(lat)} lat × {len(lon)} lon")
        print(f"lat {lat.min():.3f}..{lat.max():.3f}; lon {lon.min():.3f}..{lon.max():.3f}")

    weights = build_weights(lat, lon)

    # Process one hydrological day at a time.
    completed_this_run = 0
    for k, date in enumerate(remaining, start=1):
        print(f"\n[{k}/{len(remaining)}] {date.date()} — downloading/aggregating 8 timesteps")

        step_values = {v: [] for v in VARS}
        paths = []

        for ts, url in groups[date]:
            dest = day_dir / output_name_from_url(url)
            download_file(session, url, dest)
            paths.append(dest)

            with Dataset(dest, "r") as ds:
                lat_name2, lon_name2, lat2, lon2 = detect_lat_lon(ds)
                if not (np.allclose(lat2, lat) and np.allclose(lon2, lon)):
                    raise RuntimeError(f"Grid mismatch at {ts}")

                for v in VARS:
                    arr = read_2d_variable(ds, v, lat_name2, lon_name2, len(lat), len(lon))
                    step_values[v].append(weighted_spatial_mean(arr, weights))

        if any(len(x) != 8 for x in step_values.values()):
            raise RuntimeError(f"{date.date()} does not contain 8 successfully-read timesteps.")

        row = {"date": date.strftime("%Y-%m-%d")}
        row.update(aggregate_day(step_values))

        out_df = pd.DataFrame([row])
        out_df.to_csv(
            PARTIAL_CSV,
            mode="a",
            header=not PARTIAL_CSV.exists(),
            index=False
        )

        completed_this_run += 1

        if CONFIG.get("delete_downloaded_3hourly_after_each_day", True):
            for p in paths:
                p.unlink(missing_ok=True)

    # If this was only a test, keep partial output and exit.
    if args.max_days is not None:
        print(f"\nTEST COMPLETE: processed {completed_this_run} day(s).")
        print(f"Partial CSV: {PARTIAL_CSV}")
        print("Run again without --max-days to resume and complete all 2015–2024 dates.")
        return

    finalize()


if __name__ == "__main__":
    main()
