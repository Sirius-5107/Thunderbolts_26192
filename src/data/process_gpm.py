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
from datetime import datetime, timezone
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


def parse_imerg_hdf5(fpath: Path, lat_min, lat_max, lon_min, lon_max):
    """
    Read one GPM IMERG HDF5 file, subset to bbox, return DataFrame.

    Returns columns: timestamp, latitude, longitude, precip_mm_hr
    precipitationCal units: mm/hr  (half-hourly accumulation rate)
    """
    import h5py
    with h5py.File(fpath, "r") as hf:
        grp = hf["Grid"]

        # Coordinates — global 0.1-deg grid
        lats = grp["lat"][:]     # shape (1800,)
        lons = grp["lon"][:]     # shape (3600,)

        # Spatial mask
        lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

        if lat_idx.size == 0 or lon_idx.size == 0:
            log.warning("No grid points in bbox for %s", fpath.name)
            return None

        # Precipitation: shape (time, lon, lat) in V07
        # Slice: [time_idx, lon_idx_min:lon_idx_max+1, lat_idx_min:lat_idx_max+1]
        precip_raw = grp["precipitationCal"][
            0,
            lon_idx[0]:lon_idx[-1] + 1,
            lat_idx[0]:lat_idx[-1] + 1,
        ]  # shape (n_lon, n_lat)

        # Timestamp: seconds since 1970-01-01 UTC
        t_sec = int(grp["time"][0])
        ts = datetime.fromtimestamp(t_sec, tz=timezone.utc).replace(tzinfo=None)

        sub_lats = lats[lat_idx]
        sub_lons = lons[lon_idx]

    # Build long-format DataFrame
    lon_grid, lat_grid = np.meshgrid(sub_lons, sub_lats, indexing="ij")
    precip_vals = precip_raw.flatten().astype(np.float32)

    # IMERG fill value is -9999.9
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

    # Collect all HDF5 files
    hdf5_files = sorted(RAW_GPM.rglob("*.HDF5"))
    if not hdf5_files:
        log.error("No HDF5 files found in %s. Run download_data.py --source gpm first.", RAW_GPM)
        return 1

    log.info("Found %d HDF5 files in %s", len(hdf5_files), RAW_GPM)

    try:
        import h5py
    except ImportError:
        log.error("h5py not installed. Run: pip install h5py")
        return 1

    # Parse each file
    frames = []
    lat_min, lat_max = BBOX["lat_min"], BBOX["lat_max"]
    lon_min, lon_max = BBOX["lon_min"], BBOX["lon_max"]

    for i, fpath in enumerate(hdf5_files):
        if i % 48 == 0:
            log.info("  Parsing file %d/%d: %s", i + 1, len(hdf5_files), fpath.name)
        try:
            df = parse_imerg_hdf5(fpath, lat_min, lat_max, lon_min, lon_max)
            if df is not None:
                frames.append(df)
        except Exception as e:
            log.error("  Failed to parse %s: %s", fpath.name, e)
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
