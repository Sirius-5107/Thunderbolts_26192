"""
process_data.py
---------------
Processes raw data sources into spatially and temporally aligned grids.

Steps:
  1. Process IMD flood event catalog → Uttarakhand flood label time series
  2. Process GPM IMERG NetCDF (if available) → rainfall grid
  3. Process ERA5 NetCDF (if available) → weather grid
  4. Process SRTM GeoTIFF (if available) → terrain features
  5. If real environmental data not available: generate physics-based synthetic
     data clearly labeled as SYNTHETIC, using published climatological statistics
     for Uttarakhand (documented in-code with references)

Output:
  data/processed/flood_labels.parquet
  data/processed/environmental_grid.parquet
  data/processed/terrain_grid.parquet
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))

BBOX = CFG["region"]["bbox"]
RES  = CFG["region"]["grid_resolution_deg"]
LATS = np.arange(BBOX["lat_min"], BBOX["lat_max"] + RES / 2, RES).round(1)
LONS = np.arange(BBOX["lon_min"], BBOX["lon_max"] + RES / 2, RES).round(1)
MONSOON_MONTHS = CFG["time"]["monsoon_months"]

OUT = ROOT / CFG["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Process IMD Flood Event Catalog
# ---------------------------------------------------------------------------

def process_flood_events():
    """
    Parse IMD flood event catalog, filter to Uttarakhand, produce daily binary labels.

    Source: varadtrivedi/Analysing-Flood-Risk-in-India (floods.xlsx)
    Coverage: 1967-2023
    """
    src = ROOT / CFG["paths"]["raw_events"] / "floods_india.xlsx"
    if not src.exists():
        log.error("Flood catalog not found: %s", src)
        return None

    log.info("Processing flood event catalog...")
    df = pd.read_excel(src)

    uk = df[df["State"].str.contains("Uttarakhand", case=False, na=False)].copy()
    log.info("  Uttarakhand events found: %d", len(uk))

    def parse_date(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                pass
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

    uk["start"] = uk["Start Date"].apply(parse_date)
    uk["end"]   = uk["End Date"].apply(parse_date)
    uk = uk.dropna(subset=["start"])

    FLASH_KEYWORDS = ["flash flood", "flash floods", "cloudburst", "cloud burst", "cloudbursts"]

    def classify_severity(cause):
        if pd.isna(cause):
            return 1
        c = str(cause).lower()
        if any(k in c for k in FLASH_KEYWORDS):
            return 2
        return 1

    uk["severity"] = uk["Main Cause"].apply(classify_severity)

    records = []
    for _, row in uk.iterrows():
        start = row["start"].normalize()
        end = pd.Timestamp(row["end"]).normalize() if pd.notna(row["end"]) else start
        for day in pd.date_range(start, end, freq="D"):
            records.append({"date": day, "severity": row["severity"]})

    labels_daily = pd.DataFrame(records)
    labels_daily = labels_daily.groupby("date")["severity"].max().reset_index()

    # flood_label=1 for any event; flash_flood_label=1 for severity==2
    labels_daily["flood_label"]       = 1  # all rows here are events
    labels_daily["flash_flood_label"] = (labels_daily["severity"] >= 2).astype(int)
    labels_daily = labels_daily.drop(columns=["severity"])

    log.info(
        "  Labeled days: %d | flood=1: %d | flash_flood=1: %d",
        len(labels_daily),
        labels_daily["flood_label"].sum(),
        labels_daily["flash_flood_label"].sum(),
    )

    out = OUT / "flood_labels.parquet"
    labels_daily.to_parquet(out, index=False)
    log.info("  Saved -> %s", out)
    return labels_daily


# ---------------------------------------------------------------------------
# Step 2: Process GPM IMERG (real NetCDF if available)
# ---------------------------------------------------------------------------

def process_gpm_rainfall():
    """
    If process_gpm.py has already produced gpm_rainfall.parquet, load it.
    Otherwise attempt to run process_gpm.py if raw HDF5 files exist.
    Returns the 3h rainfall DataFrame or None (falls back to synthetic).
    """
    gpm_parquet = OUT / "gpm_rainfall.parquet"
    gpm_dir     = ROOT / CFG["paths"]["raw_gpm"]
    hdf5_files  = list(gpm_dir.rglob("*.HDF5"))

    # Case 1: raw HDF5 files present — check if parquet is up-to-date
    if hdf5_files:
        n_hdf5 = len(hdf5_files)
        # Invalidate parquet cache if:
        #   a) parquet doesn't exist, OR
        #   b) parquet is older than the newest HDF5 file
        parquet_stale = True
        if gpm_parquet.exists():
            parquet_mtime = gpm_parquet.stat().st_mtime
            newest_hdf5   = max(f.stat().st_mtime for f in hdf5_files)
            parquet_stale = newest_hdf5 > parquet_mtime
            if not parquet_stale:
                log.info("GPM IMERG: parquet up-to-date, loading cached file")
                df = pd.read_parquet(gpm_parquet)
                log.info("GPM IMERG: %d rows, %s to %s",
                         len(df), df["timestamp"].min(), df["timestamp"].max())
                return df
            else:
                log.info("GPM IMERG: %d new/updated HDF5 files detected — reprocessing", n_hdf5)
        else:
            log.info("GPM IMERG: %d HDF5 files found. Running process_gpm.py...", n_hdf5)

        import subprocess, sys
        result = subprocess.run(
            [sys.executable, str(ROOT / "src" / "data" / "process_gpm.py")],
            capture_output=False,
        )
        if result.returncode == 0 and gpm_parquet.exists():
            df = pd.read_parquet(gpm_parquet)
            return df
        else:
            log.error("GPM IMERG: process_gpm.py failed.")
            return None

    log.warning("GPM IMERG: No HDF5 files in %s. Will use synthetic data.", gpm_dir)
    return None


# ---------------------------------------------------------------------------
# Step 3: Process ERA5 NetCDF (if available)
# ---------------------------------------------------------------------------

def process_era5():
    """
    Process per-year ERA5 NetCDF files into a single aligned DataFrame.
    Merges era5land_YYYY.nc (precip, soil) + era5_YYYY.nc (temp, dewpoint, pressure, wind).
    Resamples to 3-hourly to match GPM grid.
    """
    era5_dir = ROOT / CFG["paths"]["raw_era5"]
    land_files = sorted(era5_dir.glob("era5land_*.nc"))
    sfc_files  = sorted(era5_dir.glob("era5_*.nc"))

    if not land_files and not sfc_files:
        log.warning("ERA5: No files in %s. Will use synthetic data.", era5_dir)
        return None

    try:
        import xarray as xr

        def load_nc_files(files, label):
            if not files:
                return None
            ds_list = []
            for f in files:
                ds = xr.open_dataset(f)
                # Handle both old (latitude/longitude) and new (lat/lon) coord names
                rename = {}
                if "latitude" in ds.coords:  rename["latitude"]  = "lat"
                if "longitude" in ds.coords: rename["longitude"] = "lon"
                if rename:
                    ds = ds.rename(rename)
                # Subset to bbox
                ds = ds.sel(
                    lat=slice(BBOX["lat_min"], BBOX["lat_max"]),
                    lon=slice(BBOX["lon_min"], BBOX["lon_max"]),
                )
                ds_list.append(ds)
                log.info("ERA5 %s: loaded %s", label, f.name)
            return xr.concat(ds_list, dim="time").sortby("time")

        ds_land = load_nc_files(land_files, "land")
        ds_sfc  = load_nc_files(sfc_files,  "sfc")

        # Convert to DataFrames
        frames = []
        for ds, label in [(ds_land, "land"), (ds_sfc, "sfc")]:
            if ds is None:
                continue
            df = ds.to_dataframe().reset_index()
            # Standardise time column name
            if "valid_time" in df.columns:
                df = df.rename(columns={"valid_time": "time"})
            df["time"] = pd.to_datetime(df["time"])
            frames.append(df)
            log.info("ERA5 %s: %d rows, %s to %s",
                     label, len(df), df["time"].min(), df["time"].max())

        if not frames:
            return None

        # Merge land + sfc on time + lat + lon
        result = frames[0]
        for extra in frames[1:]:
            merge_keys = [k for k in ["time", "lat", "lon"] if k in result.columns and k in extra.columns]
            result = result.merge(extra, on=merge_keys, how="outer")

        # Rename to standard column names
        rename_map = {
            "tp":    "precip_era5_m",   # ERA5 precip in metres
            "swvl1": "soil_moisture_m3m3",
            "t2m":   "temperature_2m_k",
            "d2m":   "dewpoint_2m_k",
            "sp":    "pressure_pa",
            "u10":   "u10_ms",
            "v10":   "v10_ms",
            "lat":   "latitude",
            "lon":   "longitude",
            "time":  "timestamp",
        }
        result = result.rename(columns={k: v for k, v in rename_map.items() if k in result.columns})

        # Unit conversions
        if "temperature_2m_k" in result.columns:
            result["temperature_2m_c"] = result["temperature_2m_k"] - 273.15
        if "pressure_pa" in result.columns:
            result["pressure_hpa"] = result["pressure_pa"] / 100.0
        if "dewpoint_2m_k" in result.columns and "temperature_2m_k" in result.columns:
            # Magnus formula: relative humidity from dewpoint
            T  = result["temperature_2m_k"] - 273.15
            Td = result["dewpoint_2m_k"] - 273.15
            result["humidity_pct"] = 100 * (
                (17.625 * Td) / (243.04 + Td)
            ).pipe(lambda x: x - (17.625 * T) / (243.04 + T)).apply(lambda x: min(100, max(0, 100 * (2.718281828 ** (x)))))
        if "precip_era5_m" in result.columns:
            # Convert from metres/hour to mm (already accumulated per timestep for ERA5)
            result["precip_era5_mm"] = result["precip_era5_m"] * 1000

        log.info("ERA5: merged %d rows with columns: %s",
                 len(result), [c for c in result.columns if c not in ("latitude","longitude","timestamp")])

        out = ROOT / CFG["paths"]["processed"] / "era5_processed.parquet"
        result.to_parquet(out, index=False)
        log.info("ERA5: saved -> %s", out)
        return result

    except Exception as e:
        log.error("ERA5 processing error: %s", e)
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Step 4: Process SRTM DEM (if available)
# ---------------------------------------------------------------------------

def process_srtm():
    srtm_dir  = ROOT / CFG["paths"]["raw_srtm"]
    tif_files = list(srtm_dir.rglob("*.tif"))
    gz_files  = list(srtm_dir.rglob("*.hgt.gz"))
    zip_files = list(srtm_dir.rglob("*.hgt.zip"))

    # Unpack .gz and .zip files into .hgt so rasterio can read them
    if gz_files:
        import gzip, shutil
        for gz in gz_files:
            hgt = gz.with_suffix("")  # removes .gz -> .hgt
            if not hgt.exists():
                log.info("SRTM: Unpacking %s", gz.name)
                with gzip.open(gz, "rb") as fin, open(hgt, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        tif_files += list(srtm_dir.rglob("*.hgt"))

    if zip_files:
        import zipfile
        for zf in zip_files:
            log.info("SRTM: Unpacking %s", zf.name)
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(srtm_dir)
        tif_files += list(srtm_dir.rglob("*.hgt"))

    # Deduplicate
    tif_files = list(set(tif_files))

    if not tif_files:
        log.warning("SRTM: No terrain files in %s. Will use synthetic terrain.", srtm_dir)
        return None
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge

        datasets = [rasterio.open(f) for f in tif_files]
        mosaic, _ = rio_merge(datasets)
        elev = mosaic[0].astype(float)
        elev[elev < -9000] = np.nan

        dz_dy, dz_dx = np.gradient(elev, RES * 111000, RES * 111000)
        slope_deg  = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        aspect_deg = np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360

        lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")
        records = []
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                ri = int((lat - BBOX["lat_min"]) / (BBOX["lat_max"] - BBOX["lat_min"]) * (elev.shape[0] - 1))
                ci = int((lon - BBOX["lon_min"]) / (BBOX["lon_max"] - BBOX["lon_min"]) * (elev.shape[1] - 1))
                sl = slice(max(0, ri - 2), ri + 3)
                sc = slice(max(0, ci - 2), ci + 3)
                records.append({
                    "latitude": lat, "longitude": lon,
                    "elevation_m":       float(np.nanmean(elev[sl, sc])),
                    "slope_deg":         float(np.nanmean(slope_deg[sl, sc])),
                    "aspect_deg":        float(np.nanmean(aspect_deg[sl, sc])),
                })
        terrain_df = pd.DataFrame(records)
        terrain_df["terrain_ruggedness"] = terrain_df["slope_deg"] / 90.0
        terrain_df.to_parquet(OUT / "terrain_grid.parquet", index=False)
        log.info("SRTM: terrain grid %s", terrain_df.shape)
        return terrain_df
    except Exception as e:
        log.error("SRTM processing error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 5: Vectorized Synthetic Environmental Data Generator
# ---------------------------------------------------------------------------
# Physics-based synthetic data using published statistics:
#   Bhatt & Nakamura (2005): Uttarakhand JJAS mean ~1500-2000 mm/season
#   Pai et al. (2014): IMD 0.25deg gridded rainfall statistics
#   Dimri et al. (2017): SRTM terrain stats, median slope ~28 deg

def generate_synthetic_environmental_data(labels_df=None):
    """
    Fully vectorized generation of synthetic environmental data.

    DATA IS SYNTHETIC - labeled as data_source=SYNTHETIC.
    For pipeline testing only. Replace with real GPM/ERA5 when credentials available.
    """
    log.info("Generating synthetic environmental data (vectorized)...")
    log.warning("SYNTHETIC DATA - not for final model. See download_data.py for real data.")

    rng = np.random.default_rng(42)

    # --- Time index: monsoon months, 2019-2023, 3-hourly ---
    date_range   = pd.date_range(start="2019-01-01", end="2023-12-31", freq="3h")
    monsoon_mask = date_range.month.isin(MONSOON_MONTHS)
    timestamps   = date_range[monsoon_mask]
    n_times      = len(timestamps)
    log.info("  Monsoon 3h timesteps (2019-2023): %d", n_times)

    # --- Static terrain grid ---
    lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")
    n_cells = lat_grid.size

    elev_base  = 200 + (lat_grid - BBOX["lat_min"]) / (BBOX["lat_max"] - BBOX["lat_min"]) * 4000
    elev_base += (lon_grid - 79.0) * 200
    elevation  = np.clip(elev_base + rng.normal(0, 300, lat_grid.shape), 200, 7000)
    slope      = np.clip(
        5 + (np.clip(elevation, 500, 4000) - 500) / 3500 * 35 + rng.normal(0, 5, elevation.shape),
        0, 60,
    )
    aspect     = rng.uniform(90, 270, elevation.shape)

    terrain_df = pd.DataFrame({
        "latitude":           lat_grid.flatten(),
        "longitude":          lon_grid.flatten(),
        "elevation_m":        elevation.flatten(),
        "slope_deg":          slope.flatten(),
        "aspect_deg":         aspect.flatten(),
        "terrain_ruggedness": slope.flatten() / 90.0,
        "data_source":        "SYNTHETIC",
    })
    terrain_df.to_parquet(OUT / "terrain_grid.parquet", index=False)
    log.info("  Terrain: %d cells saved", n_cells)

    # --- Flood day lookup set ---
    flood_days = set()
    if labels_df is not None:
        flood_days = set(pd.to_datetime(labels_df["date"]).dt.normalize())

    # --- Vectorized environmental generation ---
    # Shape strategy:
    #   time-varying scalars: (n_times,) arrays
    #   cell-varying scalars: (n_cells,) arrays
    #   broadcast together to (n_times, n_cells), then flatten

    doy        = timestamps.day_of_year.values.astype(float)        # (T,)
    hour       = timestamps.hour.values.astype(float)                # (T,)
    is_flood   = np.array([
        pd.Timestamp(ts.date()) in flood_days for ts in timestamps
    ], dtype=bool)                                                    # (T,)

    seasonal_factor = 1.0 + 0.5 * np.sin((doy - 150) / 365 * 2 * np.pi)  # (T,)
    diurnal_factor  = 1.0 + 0.4 * np.sin((hour - 6)  / 24  * 2 * np.pi)  # (T,)
    time_factor     = seasonal_factor * diurnal_factor                      # (T,)

    elev_flat = elevation.flatten()                                          # (C,)
    orographic = 1.0 + 0.8 * np.exp(-((elev_flat - 1500) ** 2) / (1000 ** 2))  # (C,)
    base_rate  = 2.0 * orographic                                            # (C,)

    # Broadcast: (T,1) * (1,C) -> (T,C)
    tf  = time_factor[:, np.newaxis]     # (T,1)
    br  = base_rate[np.newaxis, :]       # (1,C)
    eff = tf * br                        # (T,C)

    log.info("  Generating precipitation array (%d x %d)...", n_times, n_cells)

    # Rain mask: ~35% wet periods on non-flood days
    rain_mask = rng.random((n_times, n_cells)) < 0.35

    # Non-flood rainfall: gamma with shape=1.2, scale=eff
    precip = np.where(rain_mask, rng.gamma(1.2, eff), 0.0)

    # Flood day rows: override with heavy gamma (shape=3, scale=5*eff)
    flood_idx = np.where(is_flood)[0]
    if flood_idx.size > 0:
        heavy = rng.gamma(3.0, 5.0 * eff[flood_idx, :])
        precip[flood_idx, :] = heavy

    precip = np.clip(precip, 0, 200).astype(np.float32)

    # Temperature: (T,1) broadcast with (1,C)
    t_base = 30 - 10 * (doy[:, np.newaxis] - 90) / 365  # seasonal
    t_base += 5 * np.sin((hour[:, np.newaxis] - 14) / 24 * 2 * np.pi)  # diurnal
    temp   = t_base - 6.5 * elev_flat[np.newaxis, :] / 1000
    temp  += rng.normal(0, 1.5, (n_times, n_cells)).astype(np.float32)
    temp   = np.clip(temp, -20, 40).astype(np.float32)

    # Humidity: (T,C)
    hum_base = 80.0  # monsoon months only
    hum      = (hum_base - 0.003 * elev_flat[np.newaxis, :]) * np.ones((n_times, 1), dtype=np.float32)
    hum     += rng.normal(0, 8, (n_times, n_cells)).astype(np.float32)
    hum      = np.clip(hum, 10, 100).astype(np.float32)

    # Pressure
    pres = (1013.25 * (1 - 2.26e-5 * elev_flat[np.newaxis, :]) ** 5.256
            + rng.normal(0, 1, (n_times, n_cells)).astype(np.float32))
    pres = np.clip(pres, 400, 1020).astype(np.float32)

    # Soil moisture
    sm = 0.2 + 0.15 * (precip / 50) + rng.normal(0, 0.03, (n_times, n_cells)).astype(np.float32)
    sm = np.clip(sm, 0.05, 0.5).astype(np.float32)

    log.info("  Building DataFrame (%d rows)...", n_times * n_cells)

    # Repeat timestamps for each cell; tile cell coords for each time
    ts_repeated  = np.repeat(timestamps, n_cells)
    lat_tiled    = np.tile(lat_grid.flatten(), n_times)
    lon_tiled    = np.tile(lon_grid.flatten(), n_times)

    env_df = pd.DataFrame({
        "timestamp":           ts_repeated,
        "latitude":            lat_tiled.astype(np.float32),
        "longitude":           lon_tiled.astype(np.float32),
        "precip_3h_mm":        precip.flatten(),
        "temperature_2m_c":    temp.flatten(),
        "humidity_pct":        hum.flatten(),
        "pressure_hpa":        pres.flatten(),
        "soil_moisture_m3m3":  sm.flatten(),
        "data_source":         "SYNTHETIC",
    })

    out = OUT / "environmental_grid.parquet"
    env_df.to_parquet(out, index=False)
    log.info("  Environmental grid: %d rows saved -> %s", len(env_df), out)
    return env_df, terrain_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== process_data.py ===")

    labels_df = process_flood_events()

    rainfall_ok = process_gpm_rainfall()
    era5_ok     = process_era5()
    terrain_ok  = process_srtm()

    # Route: GPM available → real env grid. Neither → synthetic fallback.
    gpm_env_out = OUT / "environmental_grid.parquet"

    if rainfall_ok is not None:
        log.info("Building environmental_grid from real GPM data...")
        env_df = rainfall_ok.copy()
        env_df["data_source"] = "GPM_IMERG_V07"

        # Merge ERA5 weather columns if available
        if era5_ok is not None:
            log.info("Merging ERA5 weather into environmental grid...")
            era5_cols = ["timestamp", "latitude", "longitude",
                         "temperature_2m_c", "humidity_pct",
                         "pressure_hpa", "soil_moisture_m3m3"]
            era5_sub = era5_ok[[c for c in era5_cols if c in era5_ok.columns]].copy()
            era5_sub["timestamp"] = pd.to_datetime(era5_sub["timestamp"]).dt.floor("3h")
            era5_sub["latitude"]  = era5_sub["latitude"].round(1)
            era5_sub["longitude"] = era5_sub["longitude"].round(1)
            era5_sub = era5_sub.drop_duplicates(["timestamp", "latitude", "longitude"])
            env_df["latitude"]  = env_df["latitude"].round(1)
            env_df["longitude"] = env_df["longitude"].round(1)
            env_df = env_df.merge(era5_sub, on=["timestamp", "latitude", "longitude"], how="left")
            fill_pct = 100 * env_df["temperature_2m_c"].notna().mean() if "temperature_2m_c" in env_df.columns else 0
            log.info("ERA5 merge: temperature fill=%.1f%%", fill_pct)
            env_df["data_source"] = "GPM_IMERG_V07+ERA5"
        else:
            env_df["temperature_2m_c"]  = float("nan")
            env_df["humidity_pct"]       = float("nan")
            env_df["pressure_hpa"]       = float("nan")
            env_df["soil_moisture_m3m3"] = float("nan")

        env_df.to_parquet(gpm_env_out, index=False)
        log.info("Environmental grid: %d rows -> %s (source: %s)",
                 len(env_df), gpm_env_out.name, env_df["data_source"].iloc[0])
    elif not gpm_env_out.exists():
        log.info("Real data unavailable. Generating synthetic data.")
        generate_synthetic_environmental_data(labels_df=labels_df)
        log.info(
            "\n*** SYNTHETIC DATA NOTICE ***\n"
            "Environmental data is SYNTHETIC (pipeline testing only).\n"
            "Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD and run:\n"
            "  python src/data/download_data.py --source gpm\n"
            "Then re-run this script. Flood labels are REAL (IMD catalog)."
        )
    else:
        log.info("Environmental grid already exists, skipping regeneration.")

    log.info("Processing complete. Output: %s", OUT)


if __name__ == "__main__":
    main()
