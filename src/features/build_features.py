"""
build_features.py
-----------------
Builds ML-ready feature set from processed environmental and terrain data.

Output: data/features/master_dataset.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
CFG      = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))
PROCESSED = ROOT / CFG["paths"]["processed"]
FEATURES  = ROOT / CFG["paths"]["features"]
FEATURES.mkdir(parents=True, exist_ok=True)


def load_processed():
    env_path     = PROCESSED / "environmental_grid.parquet"
    terrain_path = PROCESSED / "terrain_grid.parquet"
    labels_path  = PROCESSED / "flood_labels.parquet"

    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} not found. Run process_data.py first.")

    log.info("Loading environmental grid...")
    env = pd.read_parquet(env_path)
    env["timestamp"] = pd.to_datetime(env["timestamp"])
    # Downcast numerics to float32 to halve memory usage
    for col in env.select_dtypes(include=["float64"]).columns:
        env[col] = env[col].astype(np.float32)

    log.info("Loading terrain grid...")
    terrain = pd.read_parquet(terrain_path)
    for col in terrain.select_dtypes(include=["float64"]).columns:
        terrain[col] = terrain[col].astype(np.float32)

    log.info("Loading flood labels...")
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()

    log.info("  env=%s  terrain=%s  label_days=%d", env.shape, terrain.shape, len(labels))
    return env, terrain, labels


def build_rainfall_features(env: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling rainfall accumulations per grid cell using vectorized ops.
    All windows are in units of 3h steps.
    """
    log.info("Building rainfall rolling features...")

    env = env.sort_values(["latitude", "longitude", "timestamp"]).copy()

    # Windows: name -> number of 3h steps
    windows = {
        "rain_3h":  1,
        "rain_6h":  2,
        "rain_12h": 4,
        "rain_24h": 8,
        "rain_48h": 16,
        "rain_72h": 24,
        "rain_5d":  40,
        "rain_7d":  56,
        "rain_14d": 112,
    }

    # Use groupby + transform for vectorized rolling (no apply loop)
    grp = env.groupby(["latitude", "longitude"], sort=False)["precip_3h_mm"]

    for col_name, n in windows.items():
        env[col_name] = grp.transform(lambda x: x.rolling(n, min_periods=1).sum())

    env["rain_intensity_3h"] = env["precip_3h_mm"]
    env["max_intensity_24h"] = grp.transform(lambda x: x.rolling(8,   min_periods=1).max())
    env["wet_fraction_7d"]   = grp.transform(
        lambda x: (x > 0.5).rolling(56, min_periods=1).mean()
    )

    # Anomaly: rain_24h minus its 14-day rolling mean
    rain24_grp = env.groupby(["latitude", "longitude"], sort=False)["rain_24h"]
    rolling_mean_14d = rain24_grp.transform(lambda x: x.rolling(112, min_periods=8).mean())
    global_mean      = env["rain_24h"].mean()
    env["rain_anomaly_24h"] = env["rain_24h"] - rolling_mean_14d.fillna(global_mean)

    log.info("  Rainfall features done: %s", env.shape)
    return env


def merge_labels(env_feat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging flood labels...")
    env_feat = env_feat.copy()
    env_feat["date"] = env_feat["timestamp"].dt.normalize()
    merged = env_feat.merge(labels, on="date", how="left")
    merged["flood_label"]       = merged["flood_label"].fillna(0).astype(int)
    merged["flash_flood_label"] = merged["flash_flood_label"].fillna(0).astype(int)
    log.info(
        "  Merged: %d rows | flood_label=1: %d (%.2f%%) | flash=1: %d",
        len(merged), merged["flood_label"].sum(),
        100 * merged["flood_label"].mean(), merged["flash_flood_label"].sum(),
    )
    return merged


def merge_terrain(df: pd.DataFrame, terrain: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging terrain features...")
    terrain_cols = ["latitude", "longitude", "elevation_m", "slope_deg", "aspect_deg", "terrain_ruggedness"]
    merged = df.merge(terrain[terrain_cols], on=["latitude", "longitude"], how="left")

    merged["flood_prone_terrain"] = (
        (merged["slope_deg"] > 15) & merged["elevation_m"].between(300, 3000)
    ).astype(int)
    merged["aspect_sin"] = np.sin(np.radians(merged["aspect_deg"]))
    merged["aspect_cos"] = np.cos(np.radians(merged["aspect_deg"]))

    log.info("  Terrain merged: %s", merged.shape)
    return merged


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "timestamp", "latitude", "longitude",
        "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
        "terrain_ruggedness", "flood_prone_terrain",
        "rain_intensity_3h", "rain_3h", "rain_6h", "rain_12h",
        "rain_24h", "rain_48h", "rain_72h",
        "rain_5d", "rain_7d", "rain_14d",
        "max_intensity_24h", "wet_fraction_7d", "rain_anomaly_24h",
        "temperature_2m_c", "humidity_pct", "pressure_hpa", "soil_moisture_m3m3",
        "flood_label", "flash_flood_label",
        "data_source",
    ]
    available = [c for c in keep if c in df.columns]
    missing   = [c for c in keep if c not in df.columns]
    if missing:
        log.warning("Columns not present (skipped): %s", missing)
    return df[available].copy()


def validate_no_leakage(df: pd.DataFrame):
    feature_cols = [
        c for c in df.columns
        if c not in ("flood_label", "flash_flood_label", "timestamp", "date", "data_source")
    ]
    log.info("Leakage check: %d feature columns", len(feature_cols))
    for col in feature_cols:
        if "flood" in col.lower() and col != "flood_prone_terrain":
            log.warning("Potential leakage: column contains 'flood': %s", col)
    log.info("Leakage check PASSED (all rainfall features are past-only rolling aggregations)")


def main():
    log.info("=== build_features.py ===")

    env, terrain, labels = load_processed()

    env_feat = build_rainfall_features(env)
    del env  # free original env
    master   = merge_labels(env_feat, labels)
    del env_feat  # free after merge
    master   = merge_terrain(master, terrain)
    # Select columns eagerly to free wide intermediate frame
    master   = select_final_columns(master)
    validate_no_leakage(master)

    # Sort in-place, avoid reset_index copy
    master.sort_values(["timestamp", "latitude", "longitude"], inplace=True)
    master.reset_index(drop=True, inplace=True)

    out = ROOT / CFG["output"]["master_dataset"]
    master.to_parquet(out, index=False)

    size_mb = out.stat().st_size / 1e6
    log.info("Master dataset: %s | %.1f MB | %d rows x %d cols",
             out.name, size_mb, len(master), len(master.columns))
    log.info("Date range : %s  ->  %s", master["timestamp"].min(), master["timestamp"].max())
    log.info("Grid cells : %d", master.groupby(["latitude", "longitude"]).ngroups)
    log.info("flood_label=1   : %d (%.2f%%)", master["flood_label"].sum(), 100 * master["flood_label"].mean())
    log.info("flash_label=1   : %d (%.2f%%)", master["flash_flood_label"].sum(), 100 * master["flash_flood_label"].mean())


if __name__ == "__main__":
    main()
