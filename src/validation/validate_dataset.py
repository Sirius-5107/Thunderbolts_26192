"""
validate_dataset.py
-------------------
Validates the master ML-ready dataset and produces a concise report.

Output:
  data/validation_report.json
  stdout summary
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG  = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))


def validate(df: pd.DataFrame, cfg: dict) -> dict:
    report = {}

    report["row_count"]    = len(df)
    report["column_count"] = len(df.columns)
    report["columns"]      = list(df.columns)

    # Missing values
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing_values"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # Duplicates
    report["duplicate_rows"] = int(
        df.duplicated(subset=["timestamp", "latitude", "longitude"]).sum()
    )

    # Timestamp range and continuity
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        report["timestamp_range"] = {"start": str(ts.min()), "end": str(ts.max())}

        sample_cell = df.groupby(["latitude", "longitude"]).first().index[0]
        cell_ts = df[
            (df["latitude"] == sample_cell[0]) & (df["longitude"] == sample_cell[1])
        ]["timestamp"].sort_values()
        diffs    = cell_ts.diff().dropna()
        expected = pd.Timedelta("3h")
        gaps     = diffs[diffs > expected * 1.5]
        report["timestamp_gaps_sample_cell"] = int(len(gaps))
        if len(gaps):
            report["largest_gap"] = str(gaps.max())

    # Geographic coverage
    bbox   = cfg["region"]["bbox"]
    lat_ok = (df["latitude"].min() >= bbox["lat_min"] - 0.1) and (df["latitude"].max() <= bbox["lat_max"] + 0.1)
    lon_ok = (df["longitude"].min() >= bbox["lon_min"] - 0.1) and (df["longitude"].max() <= bbox["lon_max"] + 0.1)
    report["geographic_coverage"] = {
        "lat_range":        [float(df["latitude"].min()), float(df["latitude"].max())],
        "lon_range":        [float(df["longitude"].min()), float(df["longitude"].max())],
        "within_bbox":      bool(lat_ok and lon_ok),
        "unique_grid_cells": int(df.groupby(["latitude", "longitude"]).ngroups),
    }

    # Numerical stats
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    for col in num_cols:
        s   = df[col].dropna()
        q1  = float(s.quantile(0.25))
        q3  = float(s.quantile(0.75))
        iqr = q3 - q1
        stats[col] = {
            "min":           round(float(s.min()), 4),
            "mean":          round(float(s.mean()), 4),
            "max":           round(float(s.max()), 4),
            "std":           round(float(s.std()), 4),
            "outliers_3iqr": int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()),
        }
    report["numerical_stats"] = stats

    # Class balance
    for lbl in ["flood_label", "flash_flood_label"]:
        if lbl in df.columns:
            vc = df[lbl].value_counts()
            report[f"{lbl}_balance"] = {
                "positive":          int(vc.get(1, 0)),
                "negative":          int(vc.get(0, 0)),
                "positive_rate_pct": round(100 * float(vc.get(1, 0)) / len(df), 3),
            }

    # Data source
    if "data_source" in df.columns:
        report["data_source_counts"] = df["data_source"].value_counts().to_dict()

    # Suspicious values
    warnings_list = []
    if "rain_24h" in df.columns and (df["rain_24h"] > 400).any():
        warnings_list.append(f"rain_24h > 400mm: {int((df['rain_24h']>400).sum())} rows")
    if "elevation_m" in df.columns and (df["elevation_m"] < 0).any():
        warnings_list.append(f"Negative elevation: {int((df['elevation_m']<0).sum())} rows")
    if "temperature_2m_c" in df.columns:
        n = int(((df["temperature_2m_c"] > 50) | (df["temperature_2m_c"] < -30)).sum())
        if n:
            warnings_list.append(f"Extreme temperatures: {n} rows")
    report["warnings"] = warnings_list

    return report


def print_report(report: dict):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("FLASH FLOOD DATASET VALIDATION REPORT")
    print(SEP)
    print(f"Rows:    {report['row_count']:,}")
    print(f"Columns: {report['column_count']}")

    print("✓ No duplicates" if not report.get("duplicate_rows")
          else f"⚠ Duplicates: {report['duplicate_rows']}")

    mv = report.get("missing_values", {})
    if mv:
        print(f"⚠ Missing values in {len(mv)} columns:")
        for col, info in mv.items():
            print(f"    {col}: {info['count']} ({info['pct']}%)")
    else:
        print("✓ No missing values")

    ts = report.get("timestamp_range", {})
    if ts:
        print(f"\nTime range: {ts['start']} -> {ts['end']}")
        gaps = report.get("timestamp_gaps_sample_cell", 0)
        if gaps:
            print(f"⚠ Timestamp gaps (sample cell): {gaps}  largest={report.get('largest_gap','')}")
        else:
            print("✓ Timestamp continuity OK")

    geo = report.get("geographic_coverage", {})
    print(f"\nGrid cells : {geo.get('unique_grid_cells', 0)}")
    print(f"Lat range  : {geo.get('lat_range')}")
    print(f"Lon range  : {geo.get('lon_range')}")
    print(f"Within bbox: {'✓' if geo.get('within_bbox') else '⚠'}")

    for lbl in ["flood_label", "flash_flood_label"]:
        bal = report.get(f"{lbl}_balance")
        if bal:
            print(f"\n{lbl}: pos={bal['positive']:,} | neg={bal['negative']:,} | rate={bal['positive_rate_pct']:.3f}%")

    print("\nKey stats (min / mean / max):")
    for col in ["rain_24h", "rain_72h", "elevation_m", "slope_deg",
                "temperature_2m_c", "humidity_pct", "soil_moisture_m3m3"]:
        s = report.get("numerical_stats", {}).get(col)
        if s:
            print(f"  {col:30s}: {s['min']:9.2f} / {s['mean']:9.2f} / {s['max']:9.2f}")

    if report.get("warnings"):
        print("\nWarnings:")
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report.get("data_source_counts"):
        print(f"\nData sources: {report['data_source_counts']}")

    print(SEP + "\n")


def main():
    master_path = ROOT / CFG["output"]["master_dataset"]
    if not master_path.exists():
        log.error("Master dataset not found: %s\nRun build_features.py first.", master_path)
        return 1

    log.info("Loading %s ...", master_path)
    df = pd.read_parquet(master_path)

    report = validate(df, CFG)
    print_report(report)

    report_path = ROOT / CFG["output"]["validation_report"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Report saved: %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
