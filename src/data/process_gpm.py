"""
process_gpm.py
--------------
Processes downloaded NASA GPM IMERG HDF5 files into the project's
environmental-grid format (3-hourly, 0.1-deg, Uttarakhand bbox).

Input:  data/raw/gpm/YYYY/DOY/3B-HHR.MS.MRG.3IMERG.*.HDF5
Output: data/processed/gpm_rainfall.parquet
        data/processed/environmental_grid.parquet  (GPM-based, replaces synthetic)

GPM IMERG V07 HDF5 structure:
  /Grid/precipitationCal   -- calibrated precip (mm/hr), shape (time, lon, lat)
  /Grid/lon                -- 0.1-deg centres, global
  /Grid/lat                -- 0.1-deg centres, global
  /Grid/time               -- seconds since 1970-01-01 00:00:00 UTC

Usage:
  python src/data/process_gpm.py

Run after download_data.py --source gpm
"""

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))

BBOX          = CFG["region"]["bbox"]
RES           = CFG["region"]["grid_resolution_deg"]
MONSOON_MONTHS = CFG["time"]["monsoon_months"]
RAW_GPM       = ROOT / CFG["paths"]["raw_gpm"]
OUT           = ROOT / CFG["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)



def parse_imerg_nc4(fpath: Path):
    """
    Parse an OPeNDAP-subset file (.HDF5.nc4) from GES DISC.

    Despite the .nc4 extension, GES DISC OPeNDAP returns HDF5 format.
    The file contains only the requested bbox subset (~20-50 KB).
    We detect the actual format from magic bytes and parse accordingly.

    Tries h5py first (HDF5), falls back to netCDF4 library if needed.
    """
    magic = fpath.read_bytes()[:4]
    if magic == b"\x89HDF":
        # GES DISC OPeNDAP returned HDF5 — parse same as full file
        # but the subset only contains bbox lat/lon range
        return parse_imerg_hdf5(fpath,
                                 BBOX["lat_min"], BBOX["lat_max"],
                                 BBOX["lon_min"], BBOX["lon_max"])
    else:
        # Try netCDF4 (classic CDF format)
        import netCDF4 as nc
        with nc.Dataset(str(fpath), "r") as ds:
            lats   = np.array(ds.variables["lat"][:])
            lons   = np.array(ds.variables["lon"][:])
            t_sec  = int(ds.variables["time"][0])
            precip = np.array(ds.variables["precipitation"][0, :, :], dtype=np.float32)
        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
        ts = GPS_EPOCH + timedelta(seconds=t_sec)
        precip[precip < -9000] = np.nan
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
        return pd.DataFrame({
            "timestamp":    ts,
            "latitude":     lat_grid.flatten().astype(np.float32),
            "longitude":    lon_grid.flatten().astype(np.float32),
            "precip_mm_hr": precip.flatten(),
        })


def parse_imerg_hdf5(fpath: Path, lat_min, lat_max, lon_min, lon_max):
    """
    Read one GPM IMERG V07B HDF5 file, subset to bbox, return DataFrame.

    V07B structure (differs from V06):
      Dataset : /Grid/precipitation        (V06: /Grid/precipitationCal)
      Axes    : (time, lat, lon)  shape (1, 1800, 3600)
                (V06 was (time, lon, lat))
      Units   : mm/hr (instantaneous rate at 30-min window centre)
      Fill    : -9999.9

    Returns columns: timestamp, latitude, longitude, precip_mm_hr
    """
    import h5py
    with h5py.File(fpath, "r") as hf:
        grp = hf["Grid"]

        # Coordinates — global 0.1-deg grid
        lats = grp["lat"][:]   # shape (1800,)  south->north
        lons = grp["lon"][:]   # shape (3600,)  west->east

        # Spatial mask
        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

        if lat_idx.size == 0 or lon_idx.size == 0:
            log.warning("No grid points in bbox for %s", fpath.name)
            return None

        # V07B dataset name and axis order: (time, lat, lon)
        if "precipitation" in grp:
            ds_name = "precipitation"
        elif "precipitationCal" in grp:
            ds_name = "precipitationCal"   # V06 fallback
        else:
            log.error("No precipitation dataset in %s. Keys: %s", fpath.name, list(grp.keys()))
            return None

        ds = grp[ds_name]
        # Detect axis order from shape: V07B=(1,1800,3600), V06=(1,3600,1800)
        if ds.shape[1] == 1800:
            # V07B: (time, lat, lon) -> slice [0, lat_idx, lon_idx]
            precip_raw = ds[0,
                            lat_idx[0]:lat_idx[-1] + 1,
                            lon_idx[0]:lon_idx[-1] + 1]  # (n_lat, n_lon)
            axis_order = "time,lat,lon"
        else:
            # V06: (time, lon, lat) -> slice [0, lon_idx, lat_idx]
            precip_raw = ds[0,
                            lon_idx[0]:lon_idx[-1] + 1,
                            lat_idx[0]:lat_idx[-1] + 1]  # (n_lon, n_lat)
            axis_order = "time,lon,lat"

        # Timestamp: GPM IMERG stores seconds since GPS epoch 1980-01-06 00:00:00 UTC
        # (NOT Unix epoch 1970-01-01)
        # GPM IMERG time = seconds since GPS epoch 1980-01-06 00:00:00 UTC
        GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
        t_sec = int(grp["time"][0])
        ts = GPS_EPOCH + timedelta(seconds=t_sec)

        sub_lats = lats[lat_idx]
        sub_lons = lons[lon_idx]

    # Build long-format DataFrame
    if axis_order == "time,lat,lon":
        # precip_raw shape: (n_lat, n_lon) — meshgrid to align
        lat_grid, lon_grid = np.meshgrid(sub_lats, sub_lons, indexing="ij")
    else:
        # precip_raw shape: (n_lon, n_lat)
        lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats, indexing="ij")

    precip_vals = precip_raw.flatten().astype(np.float32)
    precip_vals[precip_vals < -9000] = np.nan

    df = pd.DataFrame({
        "timestamp":    ts,
        "latitude":     lat_grid.flatten().astype(np.float32),
        "longitude":    lon_grid.flatten().astype(np.float32),
        "precip_mm_hr": precip_vals,
    })
    return df


def aggregate_to_3h(df_30min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 30-min IMERG records to 3-hourly accumulations (mm/3h).

    IMERG precip_mm_hr is a rate; multiply by 0.5h to get mm per 30-min period,
    then sum 6 consecutive 30-min periods -> mm per 3h.
    """
    df = df_30min.copy()
    df["precip_mm_30min"] = df["precip_mm_hr"] * 0.5  # rate -> depth

    # Floor timestamp to 3h bin
    df["ts_3h"] = df["timestamp"].dt.floor("3h")

    agg = (
        df.groupby(["ts_3h", "latitude", "longitude"])["precip_mm_30min"]
        .sum()
        .reset_index()
        .rename(columns={"ts_3h": "timestamp", "precip_mm_30min": "precip_3h_mm"})
    )
    agg["precip_3h_mm"] = agg["precip_3h_mm"].clip(lower=0)
    return agg


def build_environmental_grid_from_gpm(gpm_3h: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the full environmental_grid.parquet compatible with build_features.py.
    Weather columns (temperature, humidity, pressure, soil_moisture) are set to NaN
    since only GPM rainfall is available in this run. ERA5 can be merged later.
    """
    df = gpm_3h.copy()
    df["temperature_2m_c"]   = np.nan
    df["humidity_pct"]        = np.nan
    df["pressure_hpa"]        = np.nan
    df["soil_moisture_m3m3"]  = np.nan
    df["data_source"]         = "GPM_IMERG_V07"
    return df


def main():
    log.info("=== process_gpm.py ===")

    # Collect GPM files — OPeNDAP subsets (.HDF5.nc4) preferred, fall back to full HDF5
    nc4_files  = sorted(RAW_GPM.rglob("*.HDF5.nc4"))
    hdf5_files = sorted(RAW_GPM.rglob("*.HDF5"))
    # Exclude any HDF5 that already has a nc4 subset (avoid duplicates)
    hdf5_only  = [f for f in hdf5_files if not (f.parent / (f.name + ".nc4")).exists()]

    all_files = nc4_files + hdf5_only
    if not all_files:
        log.error("No GPM files found in %s. Run download_data.py --source gpm first.", RAW_GPM)
        return 1

    log.info("Found %d GPM files (%d OPeNDAP subsets, %d full HDF5) in %s",
             len(all_files), len(nc4_files), len(hdf5_only), RAW_GPM)

    try:
        import h5py
    except ImportError:
        log.error("h5py not installed. Run: pip install h5py")
        return 1

    # Parse each file
    frames = []
    lat_min, lat_max = BBOX["lat_min"], BBOX["lat_max"]
    lon_min, lon_max = BBOX["lon_min"], BBOX["lon_max"]

    for i, fpath in enumerate(all_files):
        if i % 480 == 0:
            log.info("  Parsing file %d/%d: %s", i + 1, len(all_files), fpath.name)
        try:
            if fpath.suffix == ".nc4":
                df = parse_imerg_nc4(fpath)
            else:
                df = parse_imerg_hdf5(fpath, lat_min, lat_max, lon_min, lon_max)
            if df is not None:
                frames.append(df)
        except Exception as e:
            log.error("  Failed to parse %s: %s", fpath.name, e)
            if fpath.exists():
                fpath.unlink()
                log.warning("  Deleted corrupt file: %s (will be re-downloaded)", fpath.name)
            continue

    if not frames:
        log.error("No data could be parsed from HDF5 files.")
        return 1

    log.info("Parsed %d half-hourly frames. Concatenating...", len(frames))
    df_30min = pd.concat(frames, ignore_index=True)
    df_30min["timestamp"] = pd.to_datetime(df_30min["timestamp"])

    log.info("  30-min records: %d | time range: %s to %s",
             len(df_30min), df_30min["timestamp"].min(), df_30min["timestamp"].max())

    # Save raw 30-min parquet
    raw_out = OUT / "gpm_30min.parquet"
    df_30min.to_parquet(raw_out, index=False)
    log.info("  30-min parquet saved: %s (%.1f MB)", raw_out.name, raw_out.stat().st_size / 1e6)

    # Aggregate to 3h
    log.info("Aggregating to 3-hourly...")
    df_3h = aggregate_to_3h(df_30min)
    log.info("  3h records: %d | unique timestamps: %d | grid cells: %d",
             len(df_3h),
             df_3h["timestamp"].nunique(),
             df_3h.groupby(["latitude", "longitude"]).ngroups)

    # Save 3h rainfall parquet
    gpm_out = OUT / "gpm_rainfall.parquet"
    df_3h.to_parquet(gpm_out, index=False)
    log.info("  GPM 3h parquet saved: %s (%.1f MB)", gpm_out.name, gpm_out.stat().st_size / 1e6)

    # Build full environmental_grid (GPM rainfall + NaN weather columns)
    env_df = build_environmental_grid_from_gpm(df_3h)
    env_out = OUT / "environmental_grid.parquet"
    env_df.to_parquet(env_out, index=False)
    log.info("  Environmental grid saved: %s (%.1f MB, %d rows)",
             env_out.name, env_out.stat().st_size / 1e6, len(env_df))

    # Quick sanity checks
    log.info("=== GPM DATA VERIFICATION ===")
    log.info("  Time range     : %s  to  %s", df_3h["timestamp"].min(), df_3h["timestamp"].max())
    log.info("  Lat range      : %.2f  to  %.2f", float(df_3h["latitude"].min()), float(df_3h["latitude"].max()))
    log.info("  Lon range      : %.2f  to  %.2f", float(df_3h["longitude"].min()), float(df_3h["longitude"].max()))
    log.info("  Precip min/mean/max: %.3f / %.3f / %.3f mm/3h",
             float(df_3h["precip_3h_mm"].min()),
             float(df_3h["precip_3h_mm"].mean()),
             float(df_3h["precip_3h_mm"].max()))
    log.info("  NaN precip rows: %d (%.2f%%)",
             int(df_3h["precip_3h_mm"].isna().sum()),
             100 * df_3h["precip_3h_mm"].isna().mean())
    log.info("  Data source    : GPM_IMERG_V07 (REAL)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
