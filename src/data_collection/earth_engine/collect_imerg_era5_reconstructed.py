#!/usr/bin/env python3
"""
RECONSTRUCTED FROM PROJECT HISTORY — not the original lost Colab file.

Purpose:
- Recreate the IMERG V07 + ERA5-Land daily extraction that produced
  dataset_raw_2015_2024.csv.gz (3653 rows × 19 columns).
- Export yearly `imerg_YYYY.csv` and `era5_YYYY.csv`, then merge them.

Grounded project details recovered from earlier SL | PTHT conversations:
- Common window: 2015-01-01..2024-12-31, UTC daily.
- IMERG: NASA/GPM_L3/IMERG_V07, precipitation × 0.5 h, summed by day.
- ERA5-Land: ECMWF/ERA5_LAND/DAILY_AGGR.
- Tân Châu point: [105.2480164, 10.80062008]
- Kratie point: [106.03, 12.48]
- Pakse point: [105.783, 15.117]
- HydroBASINS level 7 + reverse NEXT_DOWN BFS (15 steps) from Tân Châu.
- Major rivers: WWF/HydroSHEDS/v1/FreeFlowingRivers, DIS_AV_CMS > 1000,
  buffered 50 km and intersected with the upstream basin.
- Kratie/Pakse regions: point buffer 50 km intersected with Mekong main basin.
- Scale: 11132 m.

Because the original notebook file itself was not recovered, keep this script labeled
"reconstructed" when citing provenance.
"""
from pathlib import Path
import os
import pandas as pd
import ee
import geemap

PROJECT_ID = os.getenv("EE_PROJECT", "lateral-pillar-507605-i7")
OUT_DIR = Path(os.getenv("EE_OUTPUT_DIR", "earth_engine_exports"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
START_YEAR, END_YEAR = 2015, 2024
SCALE = 11132
BUFFER_M = 50_000
UPSTREAM_STEPS = 15

TAN_CHAU = ee.Geometry.Point([105.2480164, 10.80062008])
KRATIE = ee.Geometry.Point([106.03, 12.48])
PAKSE = ee.Geometry.Point([105.783, 15.117])

ERA5_BANDS = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "runoff_sum",
    "surface_runoff_sum",
    "sub_surface_runoff_sum",
    "total_evaporation_sum",
    "potential_evaporation_sum",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "surface_solar_radiation_downwards_sum",
]
ERA5_RENAME = {
    "volumetric_soil_water_layer_1": "do_am_tang1",
    "volumetric_soil_water_layer_2": "do_am_tang2",
    "volumetric_soil_water_layer_3": "do_am_tang3",
    "volumetric_soil_water_layer_4": "do_am_tang4",
    "runoff_sum": "dong_chay_tong",
    "surface_runoff_sum": "dong_chay_mat",
    "sub_surface_runoff_sum": "dong_chay_ngam",
    "total_evaporation_sum": "boc_hoi",
    "potential_evaporation_sum": "boc_hoi_tiem_nang",
    "temperature_2m": "nhiet_do",
    "dewpoint_temperature_2m": "diem_suong",
    "surface_pressure": "ap_suat",
    "u_component_of_wind_10m": "gio_u",
    "v_component_of_wind_10m": "gio_v",
    "surface_solar_radiation_downwards_sum": "buc_xa",
}
RAW_ORDER = [
    "ngay", "mua_tanchau", "mua_kratie", "mua_pakse",
    "do_am_tang1", "do_am_tang2", "do_am_tang3", "do_am_tang4",
    "dong_chay_tong", "dong_chay_mat", "dong_chay_ngam",
    "boc_hoi", "boc_hoi_tiem_nang", "nhiet_do", "diem_suong",
    "ap_suat", "gio_u", "gio_v", "buc_xa",
]


def init_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


def build_regions():
    basins = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_7")
    seed = ee.Feature(basins.filterBounds(TAN_CHAU).sort("UP_AREA", False).first())
    main_bas = seed.get("MAIN_BAS")
    mekong = basins.filter(ee.Filter.eq("MAIN_BAS", main_bas))

    # Reverse NEXT_DOWN traversal for 15 steps. Small client-side ID transfers keep
    # the topology logic explicit and reproducible.
    frontier = [int(seed.get("HYBAS_ID").getInfo())]
    all_ids = set(frontier)
    for _ in range(UPSTREAM_STEPS):
        fc = mekong.filter(ee.Filter.inList("NEXT_DOWN", frontier))
        new_ids = [int(x) for x in fc.aggregate_array("HYBAS_ID").getInfo()]
        new_ids = [x for x in new_ids if x not in all_ids]
        if not new_ids:
            break
        all_ids.update(new_ids)
        frontier = new_ids

    upstream = mekong.filter(ee.Filter.inList("HYBAS_ID", sorted(all_ids))).geometry().dissolve()
    mekong_geom = mekong.geometry().dissolve()

    rivers = (ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers")
              .filter(ee.Filter.gt("DIS_AV_CMS", 1000))
              .filterBounds(upstream))
    tanchau_region = rivers.geometry().buffer(BUFFER_M).intersection(upstream, ee.ErrorMargin(100))
    kratie_region = KRATIE.buffer(BUFFER_M).intersection(mekong_geom, ee.ErrorMargin(100))
    pakse_region = PAKSE.buffer(BUFFER_M).intersection(mekong_geom, ee.ErrorMargin(100))
    return tanchau_region, kratie_region, pakse_region


def _daily_dates(year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = ee.Date.fromYMD(year + 1, 1, 1)
    n = end.difference(start, "day")
    return ee.List.sequence(0, n.subtract(1)).map(lambda i: start.advance(i, "day"))


def extract_imerg_year(year, regions):
    tanchau_region, kratie_region, pakse_region = regions
    col = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07")
           .filterDate(f"{year}-01-01", f"{year+1}-01-01")
           .filter(ee.Filter.eq("status", "permanent"))
           .select("precipitation"))

    def one_day(d):
        d = ee.Date(d)
        # precipitation is mm/h at 30-minute cadence -> multiply by 0.5 h, then sum.
        rain = col.filterDate(d, d.advance(1, "day")).map(
            lambda im: ee.Image(im).multiply(0.5)
        ).sum()
        def mean(region):
            return rain.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=region, scale=SCALE,
                maxPixels=1e13, tileScale=4
            ).get("precipitation")
        return ee.Feature(None, {
            "ngay": d.format("YYYY-MM-dd"),
            "mua_tanchau": mean(tanchau_region),
            "mua_kratie": mean(kratie_region),
            "mua_pakse": mean(pakse_region),
        })

    fc = ee.FeatureCollection(_daily_dates(year).map(one_day))
    df = geemap.ee_to_df(fc)[["ngay", "mua_tanchau", "mua_kratie", "mua_pakse"]]
    df.to_csv(OUT_DIR / f"imerg_{year}.csv", index=False)
    return df


def extract_era5_year(year, region):
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
           .filterDate(f"{year}-01-01", f"{year+1}-01-01")
           .select(ERA5_BANDS))

    def one_image(im):
        im = ee.Image(im)
        vals = im.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=SCALE,
            maxPixels=1e13, tileScale=4
        )
        return ee.Feature(None, vals).set("ngay", im.date().format("YYYY-MM-dd"))

    fc = ee.FeatureCollection(col.map(one_image))
    df = geemap.ee_to_df(fc)
    keep = ["ngay"] + ERA5_BANDS
    df = df[keep].rename(columns=ERA5_RENAME)
    df.to_csv(OUT_DIR / f"era5_{year}.csv", index=False)
    return df


def merge_yearly_exports():
    imerg = pd.concat([pd.read_csv(OUT_DIR/f"imerg_{y}.csv") for y in range(START_YEAR, END_YEAR+1)], ignore_index=True)
    era5 = pd.concat([pd.read_csv(OUT_DIR/f"era5_{y}.csv") for y in range(START_YEAR, END_YEAR+1)], ignore_index=True)
    out = imerg.merge(era5, on="ngay", how="inner", validate="one_to_one").sort_values("ngay")
    out = out[RAW_ORDER]
    expected = pd.date_range("2015-01-01", "2024-12-31", freq="D")
    got = pd.to_datetime(out["ngay"])
    if len(out) != len(expected) or not got.reset_index(drop=True).equals(pd.Series(expected)):
        raise RuntimeError("Date coverage/alignment check failed")
    out.to_csv(OUT_DIR/"dataset_raw_2015_2024.csv.gz", index=False, compression="gzip")
    print(out.shape, OUT_DIR/"dataset_raw_2015_2024.csv.gz")


def main():
    init_ee()
    regions = build_regions()
    for y in range(START_YEAR, END_YEAR+1):
        print("IMERG", y); extract_imerg_year(y, regions)
        print("ERA5", y); extract_era5_year(y, regions[0])
    merge_yearly_exports()


if __name__ == "__main__":
    main()
