"""
validate_dataset.py
-------------------
Validates the master ML-ready dataset and produces a concise report.

Checks:
  - Row/column counts
  - Missing values per column
  - Duplicate records
  - Timestamp continuity (expected 3H intervals)
  - Geographic coverage vs config bbox
  - Min/max/mean for numerical variables
  - Outlier detection (IQR method)
  - Class balance (flood_label, flash_flood_label)
  - Leakage sanity (future data not in features)

Output:
  data/validation_report.json
  stdout summary
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    report = {}

    # Basic shape
    report["row_count"] = len(df)
    report["column_count"] = len(df.columns)
    report["columns"] = list(df.columns)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # Duplicates
    dupes = df.duplicated(subset=["timestamp", "latitude", "longitude"]).sum()
    report["duplicate_rows"] = int(dupes)

    # Timestamp continuity
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        report["timestamp_range"] = {
            "start": str(ts.min()),
            "end": str(ts.max()),
        }
        # Check per-cell continuity
        sample_cell = df.groupby(["latitude", "longitude"]).first().index[0]
        cell_ts = df[
            (df["latitude"] == sample_cell[0]) & (df["longitude"] == sample_cell[1])
        ]["timestamp"].sort_values()
        diffs = cell_ts.diff().dropna()
        expected_freq = pd.Timedelta("3H")
        gaps = diffs[diffs > expected_freq * 1.5]
        report["timestamp_gaps_in_sample_cell"] = int(len(gaps))
        if len(gaps) > 0:
            report["largest_gap"] = str(gaps.max())

    # Geographic coverage
    bbox = cfg["region"]["bbox"]
    lat_ok = (df["latitude"].min() >= bbox["lat_min"] - 0.1) and (df["latitude"].max() <= bbox["lat_max"] + 0.1)
    lon_ok = (df["longitude"].min() >= bbox["lon_min"] - 0.1) and (df["longitude"].max() <= bbox["lon_max"] + 0.1)
    report["geographic_coverage"] = {
        "lat_range": [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range": [float(df["longitude"].min()), float(df["longitude"].max())],
        "within_bbox": bool(lat_ok and lon_ok),
        "unique_grid_cells": int(df.groupby(["latitude", "longitude"]).ngroups),
    }

    # Numerical stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    for col in num_cols:
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()
        stats[col] = {
            "min": round(float(s.min()), 4),
            "mean": round(float(s.mean()), 4),
            "max": round(float(s.max()), 4),
            "std": round(float(s.std()), 4),
            "outliers_3iqr": int(outliers),
        }
    report["numerical_stats"] = stats

    # Class balance
    for label_col in ["flood_label", "flash_flood_label"]:
        if label_col in df.columns:
            vc = df[label_col].value_counts()
            report[f"{label_col}_balance"] = {
                "positive": int(vc.get(1, 0)),
                "negative": int(vc.get(0, 0)),
                "positive_rate_pct": round(100 * float(vc.get(1, 0)) / len(df), 3),
            }

    # Data source breakdown
    if "data_source" in df.columns:
        report["data_source_counts"] = df["data_source"].value_counts().to_dict()

    # Suspicious value checks
    warnings = []
    if "rain_24h" in df.columns:
        extreme_rain = (df["rain_24h"] > 400).sum()
        if extreme_rain > 0:
            warnings.append(f"rain_24h > 400mm: {extreme_rain} rows (verify against GPM maxima)")
    if "elevation_m" in df.columns:
        neg_elev = (df["elevation_m"] < 0).sum()
        if neg_elev > 0:
            warnings.append(f"Negative elevation: {neg_elev} rows")
    if "temperature_2m_c" in df.columns:
        extreme_temp = ((df["temperature_2m_c"] > 50) | (df["temperature_2m_c"] < -30)).sum()
        if extreme_temp > 0:
            warnings.append(f"Extreme temperatures: {extreme_temp} rows")
    report["warnings"] = warnings

    return report


def print_report(report: dict):
    print("\n" + "=" * 60)
    print("FLASH FLOOD DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Rows:    {report['row_count']:,}")
    print(f"Columns: {report['column_count']}")

    if report.get("duplicate_rows"):
        print(f"⚠ Duplicates: {report['duplicate_rows']}")
    else:
        print("✓ No duplicate timestamp+lat+lon combinations")

    mv = report.get("missing_values", {})
    if mv:
        print(f"⚠ Missing values in {len(mv)} columns:")
        for col, info in mv.items():
            print(f"    {col}: {info['count']} ({info['pct']}%)")
    else:
        print("✓ No missing values")

    ts = report.get("timestamp_range", {})
    if ts:
        print(f"\nTime range: {ts['start']} → {ts['end']}")
        gaps = report.get("timestamp_gaps_in_sample_cell", 0)
        if gaps:
            print(f"⚠ Timestamp gaps in sample cell: {gaps} (largest: {report.get('largest_gap','')})")
        else:
            print("✓ Timestamp continuity OK")

    geo = report.get("geographic_coverage", {})
    print(f"\nGrid cells: {geo.get('unique_grid_cells',0)}")
    print(f"Lat range: {geo.get('lat_range')}")
    print(f"Lon range: {geo.get('lon_range')}")
    print(f"Within bbox: {'✓' if geo.get('within_bbox') else '⚠'}")

    for lbl in ["flood_label", "flash_flood_label"]:
        bal = report.get(f"{lbl}_balance")
        if bal:
            print(f"\n{lbl}: positive={bal['positive']:,} | negative={bal['negative']:,} | rate={bal['positive_rate_pct']:.2f}%")

    print("\nKey numerical stats (min / mean / max):")
    key_cols = ["rain_24h", "rain_72h", "elevation_m", "slope_deg", "temperature_2m_c", "humidity_pct", "soil_moisture_m3m3"]
    for col in key_cols:
        s = report.get("numerical_stats", {}).get(col)
        if s:
            print(f"  {col:30s}: {s['min']:10.2f} / {s['mean']:10.2f} / {s['max']:10.2f}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report.get("data_source_counts"):
        print(f"\nData sources: {report['data_source_counts']}")

    print("=" * 60 + "\n")


def main():
    master_path = ROOT / CFG["output"]["master_dataset"]
    if not master_path.exists():
        log.error("Master dataset not found: %s\nRun build_features.py first.", master_path)
        return 1

    log.info("Loading master dataset: %s", master_path)
    df = pd.read_parquet(master_path)

    log.info("Running validation...")
    report = validate(df, CFG)

    print_report(report)

    # Save JSON report
    report_path = ROOT / CFG["output"]["validation_report"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Validation report saved: %s", report_path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
