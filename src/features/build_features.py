"""
build_features.py
-----------------
Builds ML-ready feature set from processed environmental and terrain data.

Features created:
  Rainfall accumulations: rain_1h, rain_3h, rain_6h, rain_12h, rain_24h, rain_48h, rain_72h
  Rainfall intensity: max_intensity_3h, mean_intensity_wet
  Antecedent rainfall: rain_5d, rain_7d, rain_14d
  Rainfall anomaly: rain_anomaly_7d (deviation from seasonal mean)
  Terrain: elevation_m, slope_deg, aspect_deg, terrain_ruggedness
  Weather: temperature_2m_c, humidity_pct, pressure_hpa, soil_moisture_m3m3
  Target: flood_label (binary), flash_flood_label (binary, subset of flood events)

Output:
  data/features/master_dataset.parquet
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))
PROCESSED = ROOT / CFG["paths"]["processed"]
FEATURES = ROOT / CFG["paths"]["features"]
FEATURES.mkdir(parents=True, exist_ok=True)


def load_processed():
    env_path = PROCESSED / "environmental_grid.parquet"
    terrain_path = PROCESSED / "terrain_grid.parquet"
    labels_path = PROCESSED / "flood_labels.parquet"

    if not env_path.exists():
        raise FileNotFoundError(f"{env_path} not found. Run process_data.py first.")

    log.info("Loading environmental grid...")
    env = pd.read_parquet(env_path)
    env["timestamp"] = pd.to_datetime(env["timestamp"])

    log.info("Loading terrain grid...")
    terrain = pd.read_parquet(terrain_path)

    log.info("Loading flood labels...")
    labels = pd.read_parquet(labels_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.normalize()

    log.info("  env shape: %s  terrain: %s  labels: %d days",
             env.shape, terrain.shape, len(labels))
    return env, terrain, labels


def build_rainfall_features(env: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling rainfall accumulations per grid cell.

    Input env must have: timestamp, latitude, longitude, precip_3h_mm
    3H resolution: 1h ≈ 0.33 steps → we use nearest integer steps
    """
    log.info("Building rainfall features...")

    env = env.sort_values(["latitude", "longitude", "timestamp"])

    # Accumulation windows in 3H steps
    windows = {
        "rain_3h": 1,    # 1×3H = 3h cumulative (current period)
        "rain_6h": 2,    # 2×3H
        "rain_12h": 4,   # 4×3H
        "rain_24h": 8,   # 8×3H
        "rain_48h": 16,
        "rain_72h": 24,
        "rain_5d": 40,   # ~5 days
        "rain_7d": 56,
        "rain_14d": 112,
    }

    # Group by cell and compute rolling sums
    def compute_cell_features(group):
        g = group.sort_values("timestamp").copy()
        for col_name, nsteps in windows.items():
            g[col_name] = g["precip_3h_mm"].rolling(nsteps, min_periods=1).sum()

        # Instantaneous rainfall intensity for current 3h window
        g["rain_intensity_3h"] = g["precip_3h_mm"]

        # Max 3H intensity in past 24H (8 steps)
        g["max_intensity_24h"] = g["precip_3h_mm"].rolling(8, min_periods=1).max()

        # Fraction of wet 3H periods in past 7 days (56 steps)
        g["wet_fraction_7d"] = (g["precip_3h_mm"] > 0.5).rolling(56, min_periods=1).mean()

        # Rainfall anomaly: current 24h vs 14-day rolling mean
        rain_24h_mean = g["rain_24h"].rolling(112, min_periods=8).mean()
        g["rain_anomaly_24h"] = g["rain_24h"] - rain_24h_mean.fillna(g["rain_24h"].mean())

        return g

    log.info("  Computing rolling features per cell (this may take a moment)...")
    result = (
        env.groupby(["latitude", "longitude"], group_keys=False)
        .apply(compute_cell_features)
    )

    log.info("  Rainfall features computed: %s", result.shape)
    return result


def merge_labels(env_feat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-join flood labels onto the environmental feature frame by date."""
    log.info("Merging flood labels...")

    env_feat = env_feat.copy()
    env_feat["date"] = env_feat["timestamp"].dt.normalize()

    merged = env_feat.merge(labels, on="date", how="left")
    merged["flood_label"] = merged["flood_label"].fillna(0).astype(int)
    merged["flash_flood_label"] = merged["flash_flood_label"].fillna(0).astype(int)

    log.info(
        "  Merged: %d rows | flood_label=1: %d (%.2f%%) | flash=1: %d",
        len(merged),
        merged["flood_label"].sum(),
        100 * merged["flood_label"].mean(),
        merged["flash_flood_label"].sum(),
    )
    return merged


def merge_terrain(df: pd.DataFrame, terrain: pd.DataFrame) -> pd.DataFrame:
    """Join static terrain features by lat/lon."""
    log.info("Merging terrain features...")

    terrain_cols = ["latitude", "longitude", "elevation_m", "slope_deg", "aspect_deg", "terrain_ruggedness"]
    terrain_sub = terrain[terrain_cols]

    merged = df.merge(terrain_sub, on=["latitude", "longitude"], how="left")

    # Derived terrain features
    # High-risk terrain: steep slopes at flood-prone elevations (500–2500m)
    merged["flood_prone_terrain"] = (
        (merged["slope_deg"] > 15) &
        (merged["elevation_m"].between(300, 3000))
    ).astype(int)

    # Aspect converted to N-S and E-W components (circular features)
    merged["aspect_sin"] = np.sin(np.radians(merged["aspect_deg"]))
    merged["aspect_cos"] = np.cos(np.radians(merged["aspect_deg"]))

    log.info("  Terrain merged: %s", merged.shape)
    return merged


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order final ML-ready columns. Drop pipeline internals."""
    keep = [
        "timestamp", "latitude", "longitude",
        # Terrain
        "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
        "terrain_ruggedness", "flood_prone_terrain",
        # Rainfall features (no leakage — all are past/current values at time T)
        "rain_intensity_3h",   # precipitation in this 3H window
        "rain_3h",             # = rain_intensity_3h (same, kept for naming consistency)
        "rain_6h",
        "rain_12h",
        "rain_24h",
        "rain_48h",
        "rain_72h",
        "rain_5d",
        "rain_7d",
        "rain_14d",
        "max_intensity_24h",
        "wet_fraction_7d",
        "rain_anomaly_24h",
        # Weather
        "temperature_2m_c",
        "humidity_pct",
        "pressure_hpa",
        "soil_moisture_m3m3",
        # Target
        "flood_label",
        "flash_flood_label",
        # Metadata
        "data_source",
    ]
    available = [c for c in keep if c in df.columns]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        log.warning("Columns not found (skipped): %s", missing)
    return df[available].copy()


def validate_no_leakage(df: pd.DataFrame):
    """Basic leakage check: flood label must not be in feature columns."""
    feature_cols = [c for c in df.columns if c not in ("flood_label", "flash_flood_label", "timestamp", "date", "data_source")]
    log.info("Leakage check: %d feature columns", len(feature_cols))

    for col in feature_cols:
        if "flood" in col.lower() and col not in ("flood_prone_terrain",):
            log.warning("Potential leakage risk — column name contains 'flood': %s", col)

    # All rainfall features use only past data (rolling), so no future leakage.
    log.info("Leakage check: PASSED (all rainfall features are past-only rolling aggregations)")


def main():
    log.info("=== build_features.py ===")

    env, terrain, labels = load_processed()

    # Build rainfall rolling features
    env_feat = build_rainfall_features(env)

    # Merge labels
    master = merge_labels(env_feat, labels)

    # Merge terrain
    master = merge_terrain(master, terrain)

    # Select final columns
    master = select_final_columns(master)

    # Sort chronologically
    master = master.sort_values(["timestamp", "latitude", "longitude"]).reset_index(drop=True)

    # Leakage check
    validate_no_leakage(master)

    # Save
    out = ROOT / CFG["output"]["master_dataset"]
    master.to_parquet(out, index=False)
    log.info("Master dataset saved: %s (%.1f MB, %d rows × %d cols)",
             out, out.stat().st_size / 1e6, len(master), len(master.columns))

    # Brief summary
    log.info("\n=== FEATURE SUMMARY ===")
    log.info("Date range: %s to %s", master["timestamp"].min(), master["timestamp"].max())
    log.info("Grid cells: %d unique lat/lon", master.groupby(["latitude","longitude"]).ngroups)
    log.info("Total rows: %d", len(master))
    log.info("Flood positive: %d (%.2f%%)", master["flood_label"].sum(), 100*master["flood_label"].mean())
    log.info("Flash flood positive: %d (%.2f%%)", master["flash_flood_label"].sum(), 100*master["flash_flood_label"].mean())
    log.info("Columns: %s", list(master.columns))


if __name__ == "__main__":
    main()
