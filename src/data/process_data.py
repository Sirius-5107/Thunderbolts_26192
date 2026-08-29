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
  data/processed/rainfall_grid.parquet
  data/processed/weather_grid.parquet
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
RES = CFG["region"]["grid_resolution_deg"]
LATS = np.arange(BBOX["lat_min"], BBOX["lat_max"] + RES, RES).round(1)
LONS = np.arange(BBOX["lon_min"], BBOX["lon_max"] + RES, RES).round(1)
MONSOON_MONTHS = CFG["time"]["monsoon_months"]

OUT = ROOT / CFG["paths"]["processed"]
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Process IMD Flood Event Catalog
# ---------------------------------------------------------------------------

def process_flood_events():
    """
    Parse IMD flood event catalog, filter to Uttarakhand, produce daily binary labels.
    Flash-flood and heavy-rain events coded as flood_label=1.

    Source: varadtrivedi/Analysing-Flood-Risk-in-India (floods.xlsx)
    Coverage: 1967–2023
    """
    src = ROOT / CFG["paths"]["raw_events"] / "floods_india.xlsx"
    if not src.exists():
        log.error("Flood catalog not found: %s", src)
        return None

    log.info("Processing flood event catalog...")
    df = pd.read_excel(src)

    # Filter to Uttarakhand events
    uk = df[df["State"].str.contains("Uttarakhand", case=False, na=False)].copy()
    log.info("  Uttarakhand events found: %d", len(uk))

    # Parse dates — mixed formats in source
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
    uk["end"] = uk["End Date"].apply(parse_date)
    uk = uk.dropna(subset=["start"])

    # Map cause to flash-flood severity
    def classify_severity(cause):
        if pd.isna(cause):
            return 1
        c = str(cause).lower()
        if any(k in c for k in ["flash flood", "cloudburst", "cloud burst", "cloudbursts", "flash floods"]):
            return 2  # high severity
        return 1  # standard flood/heavy rain

    uk["severity"] = uk["Main Cause"].apply(classify_severity)

    # Expand each event to daily rows
    records = []
    for _, row in uk.iterrows():
        start = row["start"].normalize()
        end = row["end"] if pd.notna(row["end"]) else start + pd.Timedelta(days=1)
        end = pd.Timestamp(end).normalize()
        for day in pd.date_range(start, end, freq="D"):
            records.append({"date": day, "severity": row["severity"], "cause": row["Main Cause"]})

    labels_daily = pd.DataFrame(records)
    labels_daily = labels_daily.groupby("date")["severity"].max().reset_index()
    labels_daily.columns = ["date", "flood_label"]
    labels_daily["flood_label"] = (labels_daily["flood_label"] >= 1).astype(int)
    labels_daily["flash_flood_label"] = (labels_daily["flood_label"] >= 2).astype(int)

    log.info("  Labeled days: %d (flood=%d, flash_flood=%d)",
             len(labels_daily),
             labels_daily["flood_label"].sum(),
             labels_daily["flash_flood_label"].sum())

    out = OUT / "flood_labels.parquet"
    labels_daily.to_parquet(out, index=False)
    log.info("  Saved → %s", out)
    return labels_daily


# ---------------------------------------------------------------------------
# Step 2: Process GPM IMERG (real NetCDF if available)
# ---------------------------------------------------------------------------

def process_gpm_rainfall():
    """Process GPM IMERG NetCDF files if present."""
    gpm_dir = ROOT / CFG["paths"]["raw_gpm"]
    nc_files = list(gpm_dir.rglob("*.nc4")) + list(gpm_dir.rglob("*.HDF5"))
    if not nc_files:
        log.warning("GPM IMERG: No files found in %s. Using synthetic data.", gpm_dir)
        return None

    try:
        import xarray as xr
        ds_list = []
        for f in sorted(nc_files):
            ds = xr.open_dataset(f, group="Grid")
            # Spatial subset
            ds = ds.sel(
                lat=slice(BBOX["lat_min"], BBOX["lat_max"]),
                lon=slice(BBOX["lon_min"], BBOX["lon_max"]),
            )
            ds_list.append(ds["precipitationCal"])
        da = xr.concat(ds_list, dim="time")
        df = da.to_dataframe(name="precip_mm_30min").reset_index()
        df.to_parquet(OUT / "gpm_rainfall.parquet", index=False)
        log.info("GPM IMERG: Processed %d files → rainfall_grid.parquet", len(nc_files))
        return df
    except Exception as e:
        log.error("GPM IMERG processing error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 3: Process ERA5 NetCDF (if available)
# ---------------------------------------------------------------------------

def process_era5():
    """Process ERA5 NetCDF files if present."""
    era5_dir = ROOT / CFG["paths"]["raw_era5"]
    nc_files = list(era5_dir.rglob("*.nc"))
    if not nc_files:
        log.warning("ERA5: No files found in %s. Using synthetic data.", era5_dir)
        return None

    try:
        import xarray as xr
        frames = {}
        for f in nc_files:
            ds = xr.open_dataset(f)
            ds = ds.sel(
                latitude=slice(BBOX["lat_max"], BBOX["lat_min"]),
                longitude=slice(BBOX["lon_min"], BBOX["lon_max"]),
            )
            df = ds.to_dataframe().reset_index()
            frames[f.stem] = df
            log.info("ERA5: Processed %s, shape=%s", f.name, df.shape)
        return frames
    except Exception as e:
        log.error("ERA5 processing error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 4: Process SRTM DEM (if available)
# ---------------------------------------------------------------------------

def process_srtm():
    """Process SRTM GeoTIFF to derive elevation, slope, aspect per grid cell."""
    srtm_dir = ROOT / CFG["paths"]["raw_srtm"]
    tif_files = list(srtm_dir.rglob("*.tif"))
    if not tif_files:
        log.warning("SRTM: No GeoTIFF found in %s. Using synthetic terrain.", srtm_dir)
        return None

    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.warp import reproject, Resampling
        import numpy as np

        # Merge tiles if multiple
        datasets = [rasterio.open(f) for f in tif_files]
        mosaic, out_transform = merge(datasets)
        elev = mosaic[0].astype(float)
        elev[elev < -9000] = np.nan  # nodata

        # Derive slope and aspect using finite differences
        dz_dy, dz_dx = np.gradient(elev, RES * 111000, RES * 111000)  # approx meters
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        aspect_deg = np.degrees(np.arctan2(-dz_dy, dz_dx)) % 360

        # Build per-cell dataframe at target resolution
        lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")
        records = []
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                # Map lat/lon to pixel indices in the mosaic
                row_frac = (lat - BBOX["lat_min"]) / (BBOX["lat_max"] - BBOX["lat_min"])
                col_frac = (lon - BBOX["lon_min"]) / (BBOX["lon_max"] - BBOX["lon_min"])
                ri = int(row_frac * (elev.shape[0] - 1))
                ci = int(col_frac * (elev.shape[1] - 1))
                records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "elevation_m": float(np.nanmean(elev[max(0,ri-2):ri+3, max(0,ci-2):ci+3])),
                    "slope_deg": float(np.nanmean(slope_deg[max(0,ri-2):ri+3, max(0,ci-2):ci+3])),
                    "aspect_deg": float(np.nanmean(aspect_deg[max(0,ri-2):ri+3, max(0,ci-2):ci+3])),
                })

        terrain_df = pd.DataFrame(records)
        terrain_df["terrain_ruggedness"] = terrain_df["slope_deg"] / 90.0  # normalized
        terrain_df.to_parquet(OUT / "terrain_grid.parquet", index=False)
        log.info("SRTM: Terrain grid shape=%s → terrain_grid.parquet", terrain_df.shape)
        return terrain_df

    except Exception as e:
        log.error("SRTM processing error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Step 5: Generate Synthetic Environmental Data (when real data unavailable)
# ---------------------------------------------------------------------------
# Physics-based synthetic data for Uttarakhand using published statistics:
#
# Rainfall climatology:
#   Bhatt & Nakamura (2005): Uttarakhand monsoon (JJAS) mean ~1500–2000 mm/season
#   IMD gridded 0.25° data statistics from Pai et al. (2014)
#   Mean daily monsoon rainfall ~10–20 mm/day; extreme events 100–300 mm/day
#
# Temperature:
#   Valley stations (Dehradun ~700m): 25–35°C summer, 5–15°C winter
#   High altitude (>2000m): subtract ~6.5°C per 1000m lapse rate
#
# Terrain:
#   Uttarakhand elevation: Gangetic plain ~200m → High Himalaya ~7000m
#   SRTM statistics from Dimri et al. (2017): median slope ~28°, aspect mostly S/SW

def generate_synthetic_environmental_data(labels_df=None):
    """
    Generate physics-based synthetic environmental data for Uttarakhand.

    DATA IS SYNTHETIC — clearly labeled in output.
    Use this ONLY for pipeline testing until real GPM/ERA5/SRTM data is downloaded.

    Based on:
      - IMD observed statistics for Uttarakhand monsoon
      - Published ERA5 climatology for 28-32°N, 77-81°E
      - SRTM terrain statistics for Himalayan region
    """
    log.info("Generating synthetic environmental data (pipeline testing mode)...")
    log.warning("SYNTHETIC DATA — not for final model. Download real data using download_data.py")

    rng = np.random.default_rng(42)

    # 3-hourly time index, monsoon months 2001-2023
    date_range = pd.date_range(
        # For synthetic testing, use last 5 years to keep dataset manageable
        start="2019-01-01",
        end="2023-12-31",
        freq="3h",
    )
    monsoon_mask = date_range.month.isin(MONSOON_MONTHS)
    timestamps = date_range[monsoon_mask]
    n_times = len(timestamps)
    log.info("  Time steps (monsoon 3H, 2001-2023): %d", n_times)

    # --- Terrain grid (static) ---
    lat_grid, lon_grid = np.meshgrid(LATS, LONS, indexing="ij")
    n_lat, n_lon = lat_grid.shape
    n_cells = n_lat * n_lon

    # Elevation: increases from SW (plain) to NE (Himalaya)
    # Gangetic plain in south ~200-400m, high peaks in north ~5000-7000m
    elev_base = 200 + (lat_grid - BBOX["lat_min"]) / (BBOX["lat_max"] - BBOX["lat_min"]) * 4000
    elev_base += (lon_grid - 79.0) * 200  # some east-west variation
    elev_noise = rng.normal(0, 300, lat_grid.shape)
    elevation = np.clip(elev_base + elev_noise, 200, 7000).flatten()

    # Slope: higher at mid-altitudes (mountain flanks), lower on plains and peaks
    elev_flat = elevation
    slope = 5 + (np.clip(elev_flat, 500, 4000) - 500) / 3500 * 35
    slope += rng.normal(0, 5, n_cells)
    slope = np.clip(slope, 0, 60)

    # Aspect: roughly south-facing (180°) with noise
    aspect = rng.uniform(90, 270, n_cells)

    terrain_df = pd.DataFrame({
        "latitude": lat_grid.flatten(),
        "longitude": lon_grid.flatten(),
        "elevation_m": elevation,
        "slope_deg": slope,
        "aspect_deg": aspect,
        "terrain_ruggedness": slope / 90.0,
        "data_source": "SYNTHETIC",
    })
    terrain_df.to_parquet(OUT / "terrain_grid.parquet", index=False)
    log.info("  Terrain: %d grid cells saved", n_cells)

    # --- Rainfall + weather grid (time-varying) ---
    # For efficiency: generate for all grid cells × all time steps in chunks
    # Relationship with flood events:
    #   On flood event days: rainfall boosted (physically realistic)
    #   On non-event days: background monsoon climatology

    # Create set of flood days for label linkage
    flood_days = set()
    if labels_df is not None:
        flood_rows = labels_df[labels_df["flood_label"] == 1]
        flood_days = set(flood_rows["date"].dt.normalize())

    records = []
    lat_arr = lat_grid.flatten()
    lon_arr = lon_grid.flatten()

    for i_t, ts in enumerate(timestamps):
        if i_t % 2000 == 0:
            log.info("  Generating timestep %d/%d (%s)", i_t, n_times, ts.date())

        is_flood_day = pd.Timestamp(ts.date()) in flood_days
        hour_of_day = ts.hour
        day_of_year = ts.day_of_year

        # Seasonal monsoon strength: peaks in July-August
        seasonal_factor = 1.0 + 0.5 * np.sin((day_of_year - 150) / 365 * 2 * np.pi)
        # Diurnal: afternoon maximum (local convective peak)
        diurnal_factor = 1.0 + 0.4 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi)

        for i_c in range(n_cells):
            lat = lat_arr[i_c]
            lon = lon_arr[i_c]
            elev = elevation[i_c]
            slp = slope[i_c]

            # Rainfall (mm/3h)
            # Orographic enhancement at elevation 500-2500m
            orographic_factor = 1.0 + 0.8 * np.exp(-((elev - 1500) ** 2) / (1000 ** 2))
            base_precip_rate = 2.0 * seasonal_factor * diurnal_factor * orographic_factor

            if is_flood_day:
                # Flood day: heavy rainfall event (cloudburst/extreme)
                precip_3h = rng.gamma(shape=3.0, scale=base_precip_rate * 5)
            else:
                # Normal monsoon day: intermittent rainfall
                if rng.random() < 0.35:  # ~35% of 3h slots have rain during monsoon
                    precip_3h = rng.gamma(shape=1.2, scale=base_precip_rate)
                else:
                    precip_3h = 0.0
            precip_3h = float(np.clip(precip_3h, 0, 200))

            # Temperature (°C): decreases with elevation (6.5°C/1000m lapse)
            t_sea_level = 30 - 10 * (day_of_year - 90) / 365  # seasonal cycle
            t_sea_level += 5 * np.sin((hour_of_day - 14) / 24 * 2 * np.pi)  # diurnal
            temp_2m = t_sea_level - 6.5 * elev / 1000 + rng.normal(0, 1.5)
            temp_2m = float(np.clip(temp_2m, -20, 40))

            # Humidity (%): higher during monsoon, lower at altitude
            humidity_base = 80 if month_in_monsoon(ts.month) else 50
            humidity = humidity_base - 0.003 * elev + rng.normal(0, 8)
            humidity = float(np.clip(humidity, 10, 100))

            # Pressure (hPa): decreases with altitude
            pressure = 1013.25 * (1 - 0.0000226 * elev) ** 5.256 + rng.normal(0, 1)
            pressure = float(np.clip(pressure, 400, 1020))

            # Soil moisture (m³/m³): 0.1–0.5, higher after heavy rain
            soil_moisture = 0.2 + 0.15 * (precip_3h / 50) + rng.normal(0, 0.03)
            soil_moisture = float(np.clip(soil_moisture, 0.05, 0.5))

            records.append({
                "timestamp": ts,
                "latitude": round(lat, 1),
                "longitude": round(lon, 1),
                "precip_3h_mm": round(precip_3h, 3),
                "temperature_2m_c": round(temp_2m, 2),
                "humidity_pct": round(humidity, 1),
                "pressure_hpa": round(pressure, 1),
                "soil_moisture_m3m3": round(soil_moisture, 4),
                "data_source": "SYNTHETIC",
            })

    env_df = pd.DataFrame(records)
    env_df.to_parquet(OUT / "environmental_grid.parquet", index=False)
    log.info("  Environmental grid: %d rows saved", len(env_df))
    return env_df, terrain_df


def month_in_monsoon(m):
    return m in MONSOON_MONTHS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=== process_data.py ===")

    # Step 1: Flood event labels (real IMD data)
    labels_df = process_flood_events()

    # Step 2–4: Try real data, fall back to synthetic
    rainfall_ok = process_gpm_rainfall()
    era5_ok = process_era5()
    terrain_ok = process_srtm()

    real_env = all([rainfall_ok is not None, era5_ok is not None])

    if not real_env:
        log.info("Real environmental data not available. Generating synthetic data for pipeline testing.")
        generate_synthetic_environmental_data(labels_df=labels_df)
        log.info(
            "\n*** SYNTHETIC DATA NOTICE ***\n"
            "Environmental data (rainfall/weather) is SYNTHETIC.\n"
            "To use real data:\n"
            "  1. Set EARTHDATA_USERNAME/EARTHDATA_PASSWORD and run: python src/data/download_data.py --source gpm\n"
            "  2. Create ~/.cdsapirc and run: python src/data/download_data.py --source era5\n"
            "  3. Re-run: python src/data/process_data.py\n"
            "Flood event labels are REAL (IMD catalog)."
        )
    else:
        log.info("Real environmental data processed successfully.")

    log.info("Processing complete. Output: %s", OUT)


if __name__ == "__main__":
    main()
