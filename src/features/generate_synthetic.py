"""
generate_synthetic.py
---------------------
Generates a training-ready dataset by perturbing real flash-flood observations
with district-level spatial filtering, then sampling real heavy-rain negatives.

PROBLEM WITH NAIVE LABELLING
-----------------------------
build_features.py joins flash_flood_label on DATE alone, so every one of the
1,116 grid cells gets label=1 on a flash-flood day — even cells 200 km from
the documented event. This inflates seed rows from ~2,300 to ~134,000 and
produces synthetic positives that are geographically nonsensical.

SPATIAL FILTERING USED HERE
----------------------------
Each flash-flood event in the catalog names at least one Uttarakhand district.
We map the 13 districts to approximate lat/lon bboxes and restrict positive
seeds to grid cells that fall inside the relevant district(s).

District bboxes are approximate (Census of India district boundaries,
cross-referenced with GADM). The catalog labels remain date-level; we are
only restricting WHICH CELLS are treated as positives, not asserting that
every cell in the district flooded.

PERTURBATION
------------
Each seed row is copied N times with independent Gaussian noise:
  Rainfall features  : noise_std = 2% of value (multiplicative — zeros stay zero)
  Atmospheric features: noise_std = 1% of value (with a floor to avoid 0*1%=0)
  Terrain features   : NEVER perturbed (deterministic geography)
Physical constraints are enforced after perturbation (rain ≥ 0, humidity 0-100, etc.)

NEGATIVES
---------
Sampled from real rows where flash_flood_label=0 AND rain_24h ≥ 90th percentile.
These are genuine "heavy rain, no flash flood" observations.

OUTPUT
------
  data/features/synthetic_train.parquet
  data/features/synthetic_train.csv
  data/features/synthetic_meta.json
  data/features/synthetic_seed_audit.csv

Usage:
  python src/features/generate_synthetic.py
  python src/features/generate_synthetic.py --copies 10 --neg-ratio 5
  python src/features/generate_synthetic.py --exclude-years 2022 --copies 10
  python src/features/generate_synthetic.py --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
CFG      = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))
FEATURES = ROOT / CFG["paths"]["features"]
FEATURES.mkdir(parents=True, exist_ok=True)

# ── Uttarakhand district → approximate bbox ───────────────────────────────────
# Source: Census of India district maps + GADM cross-check.
# Bboxes are intentionally generous (±0.1 deg) to avoid edge artefacts.
UK_DISTRICTS = {
    "Almora":            dict(lat=(29.3, 30.3), lon=(79.1, 80.0)),
    "Bageshwar":         dict(lat=(29.7, 30.3), lon=(79.6, 80.2)),
    "Chamoli":           dict(lat=(30.0, 31.1), lon=(79.0, 80.0)),
    "Champawat":         dict(lat=(29.0, 29.5), lon=(79.9, 80.6)),
    "Dehradun":          dict(lat=(29.9, 31.1), lon=(77.5, 78.5)),
    "Garhwal":           dict(lat=(29.4, 30.5), lon=(78.4, 79.4)),   # Pauri Garhwal
    "Haridwar":          dict(lat=(29.5, 30.1), lon=(77.7, 78.5)),
    "Nainital":          dict(lat=(29.1, 29.6), lon=(79.0, 79.7)),
    "Pithoragarh":       dict(lat=(29.5, 30.6), lon=(80.0, 81.0)),
    "Rudra Prayag":      dict(lat=(30.2, 31.0), lon=(78.8, 79.6)),
    "Tehri Garhwal":     dict(lat=(30.0, 30.8), lon=(78.2, 79.2)),
    "Udham Singh Nagar": dict(lat=(28.8, 29.3), lon=(79.0, 80.2)),
    "Uttar Kashi":       dict(lat=(30.5, 31.4), lon=(77.9, 79.0)),
}

# Flash-flood events 2018-2024 from the IMD/NDMA catalog.
# Each entry: (date_str, [district_names]).
# District names are matched fuzzily (substring) against UK_DISTRICTS keys.
FLASH_EVENTS = [
    ("2019-06-02",  ["Almora", "Chamoli"]),
    ("2019-06-21",  ["Uttar Kashi"]),
    ("2019-08-06",  ["Chamoli", "Garhwal", "Uttar Kashi"]),
    ("2019-08-08",  ["Chamoli", "Garhwal", "Rudra Prayag", "Tehri Garhwal"]),
    ("2019-08-18",  ["Dehradun", "Garhwal", "Tehri Garhwal", "Uttar Kashi"]),
    ("2019-09-02",  ["Almora"]),
    ("2019-09-06",  ["Chamoli", "Pithoragarh"]),
    ("2019-09-07",  ["Chamoli", "Pithoragarh"]),   # multi-day event
    ("2019-09-08",  ["Chamoli", "Pithoragarh"]),
    ("2019-09-27",  ["Dehradun"]),
    ("2020-07-18",  ["Pithoragarh"]),
    ("2020-07-19",  ["Pithoragarh"]),
    ("2020-08-25",  ["Chamoli"]),
    ("2022-08-19",  ["Dehradun", "Garhwal", "Tehri Garhwal"]),
]

# ── feature groups ────────────────────────────────────────────────────────────
TERRAIN_COLS = [
    "elevation_m", "slope_deg", "aspect_sin", "aspect_cos",
    "terrain_ruggedness", "flood_prone_terrain",
]

# noise_frac = Gaussian std as a fraction of the absolute value
RAIN_COLS = {
    "rain_intensity_3h": 0.02, "rain_3h":  0.02, "rain_6h":  0.02,
    "rain_12h": 0.02, "rain_24h": 0.02, "rain_48h": 0.02,
    "rain_72h": 0.02, "rain_5d":  0.02, "rain_7d":  0.02, "rain_14d": 0.02,
    "max_intensity_24h": 0.02,
    "wet_fraction_7d":   0.01,   # bounded 0-1, smaller perturbation
    "rain_anomaly_24h":  0.02,   # can be negative; noise applied to absolute
}
ATMO_COLS = {
    "temperature_2m_c":   0.01,
    "humidity_pct":       0.01,
    "pressure_hpa":       0.005,
    "soil_moisture_m3m3": 0.02,
}


# ── district → cells ──────────────────────────────────────────────────────────

def district_cells(terrain: pd.DataFrame, district_names: list) -> pd.DataFrame:
    """
    Return terrain rows whose (lat, lon) fall inside any of the named districts.
    District names are matched by case-insensitive substring.
    """
    mask = pd.Series(False, index=terrain.index)
    matched_districts = []
    for name in district_names:
        name_l = name.lower().strip()
        for dist_key, bb in UK_DISTRICTS.items():
            if name_l in dist_key.lower() or dist_key.lower() in name_l:
                in_bb = (
                    (terrain["latitude"]  >= bb["lat"][0]) &
                    (terrain["latitude"]  <= bb["lat"][1]) &
                    (terrain["longitude"] >= bb["lon"][0]) &
                    (terrain["longitude"] <= bb["lon"][1])
                )
                mask |= in_bb
                matched_districts.append(dist_key)
    if not matched_districts:
        log.warning("  No district match for: %s", district_names)
    return terrain[mask][["latitude", "longitude"]]


def build_event_cell_index(grid_coords: pd.DataFrame) -> dict:
    """
    For each flash event date, return the set of (lat, lon) tuples
    that fall in the documented districts.

    grid_coords must have columns latitude, longitude with the SAME
    coordinate values used in the master dataset (GPM cell centres:
    28.55, 28.65 ... not terrain corners 28.5, 28.6 ...).
    """
    idx = {}
    for date_str, districts in FLASH_EVENTS:
        cells = district_cells(grid_coords, districts)
        key_set = set(map(tuple, cells[["latitude", "longitude"]].round(3).values))
        idx[date_str] = key_set
        log.info("  %s  districts=%s  -> %d cells",
                 date_str, districts, len(key_set))
    return idx


# ── perturbation ──────────────────────────────────────────────────────────────

def perturb_copy(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = rows.copy()

    for col, frac in RAIN_COLS.items():
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=np.float64)
        v = v + rng.normal(0, frac * np.abs(v))
        v = np.clip(v, 0, None)
        if col == "wet_fraction_7d":
            v = np.clip(v, 0, 1)
        out[col] = v.astype(np.float32)

    for col, frac in ATMO_COLS.items():
        if col not in out.columns:
            continue
        v = out[col].to_numpy(dtype=np.float64)
        scale = np.maximum(np.abs(v), 1.0) * frac
        v = v + rng.normal(0, scale)
        if col == "humidity_pct":
            v = np.clip(v, 0, 100)
        if col == "soil_moisture_m3m3":
            v = np.clip(v, 0, 0.6)
        out[col] = v.astype(np.float32)

    return out


# ── seed extraction ───────────────────────────────────────────────────────────

def extract_seeds(
    df: pd.DataFrame,
    terrain: pd.DataFrame,
    exclude_years: list,
) -> pd.DataFrame:
    """
    Extract the positive seed rows using district-level spatial filtering.

    For each flash event:
      1. Filter master_dataset to that date.
      2. Restrict to cells in the documented districts.
      3. Optionally skip years in exclude_years.

    IMPORTANT: The district bbox lookup uses the master dataset's own unique
    (lat, lon) pairs — NOT terrain coordinates. Terrain uses grid corners
    (28.5, 28.6 ...) while GPM uses cell centres (28.55, 28.65 ...). Using
    terrain coords for the lookup produced zero matches.
    """
    # Extract unique grid coords from master dataset itself
    grid_coords = df[["latitude", "longitude"]].drop_duplicates().copy()
    log.info("Building district→cell index from %d master grid cells ...",
             len(grid_coords))
    event_cells = build_event_cell_index(grid_coords)

    seeds = []
    skipped_years = []
    for date_str, districts in FLASH_EVENTS:
        dt = pd.Timestamp(date_str)
        if dt.year in exclude_years:
            skipped_years.append(date_str)
            continue

        day_rows = df[df["timestamp"].dt.date == dt.date()].copy()
        if len(day_rows) == 0:
            log.warning("  %s: no rows in master_dataset (date out of range?)", date_str)
            continue

        valid_cells = event_cells[date_str]
        day_rows["_cell"] = list(zip(
            day_rows["latitude"].round(3),
            day_rows["longitude"].round(3)
        ))
        spatial = day_rows[day_rows["_cell"].isin(valid_cells)].drop(columns="_cell")
        log.info("  %s: %d district cells → %d master rows kept as seeds",
                 date_str, len(valid_cells), len(spatial))
        seeds.append(spatial)

    if skipped_years:
        log.info("Skipped (excluded years): %s", skipped_years)

    if not seeds:
        raise ValueError("No seed rows after spatial + year filtering.")

    return pd.concat(seeds, ignore_index=True)


# ── negative sampling ─────────────────────────────────────────────────────────

def sample_negatives(
    df: pd.DataFrame,
    n_needed: int,
    heavy_rain_pct: float,
    seed: int,
) -> pd.DataFrame:
    threshold = df["rain_24h"].quantile(heavy_rain_pct / 100)
    log.info("Negative pool: rain_24h ≥ %.2f mm (%.0fth pct)", threshold, heavy_rain_pct)

    pool = df[
        (df["flash_flood_label"] == 0) &
        (df["rain_24h"] >= threshold)
    ].copy()
    log.info("Heavy-rain non-flash pool: %d rows", len(pool))

    if len(pool) == 0:
        raise ValueError("No heavy-rain negatives found.")

    if len(pool) > n_needed:
        pool = pool.sample(n_needed, random_state=seed)
        log.info("Sampled %d negatives", n_needed)
    else:
        log.warning("Only %d negatives available (wanted %d) — using all",
                    len(pool), n_needed)

    pool["synthetic_copy"] = 0
    pool["data_source"] = pool.get("data_source", "REAL")
    return pool


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--neg-ratio", type=float, default=5)
    ap.add_argument("--heavy-rain-pct", type=float, default=90)
    ap.add_argument("--exclude-years", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--master", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    master_path = (Path(args.master) if args.master
                   else ROOT / CFG["output"]["master_dataset"])

    log.info("Loading master dataset ...")
    df = pd.read_parquet(master_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    log.info("  %d rows × %d cols", len(df), len(df.columns))

    log.info("Loading terrain ...")
    terrain = pd.read_parquet(ROOT / CFG["paths"]["processed"] / "terrain_grid.parquet")

    log.info("Extracting spatially-filtered seed rows ...")
    seeds = extract_seeds(df, terrain, args.exclude_years)
    n_seed_dates = seeds["timestamp"].dt.date.nunique()

    n_synth_pos  = len(seeds) * args.copies
    n_neg_target = int(n_synth_pos * args.neg_ratio)

    log.info("\n=== GENERATION PLAN ===")
    log.info("  Seed rows (district-filtered)  : %d", len(seeds))
    log.info("  Seed dates                     : %d", n_seed_dates)
    log.info("  Copies per seed row            : %d", args.copies)
    log.info("  Synthetic positives            : %d", n_synth_pos)
    log.info("  Neg:pos ratio target           : %.1f", args.neg_ratio)
    log.info("  Target negatives               : %d", n_neg_target)
    log.info("  Exclude years                  : %s", args.exclude_years or "none")

    if args.dry_run:
        log.info("DRY RUN — exiting without writing.")
        return 0

    # ── generate positives ───────────────────────────────────────────────────
    log.info("Generating synthetic positives ...")
    copies = []
    for i in range(args.copies):
        rng = np.random.default_rng(args.seed + i)
        c = perturb_copy(seeds, rng)
        c["synthetic_copy"] = i + 1
        c["data_source"] = "SYNTHETIC_PERTURBED"
        copies.append(c)
    synth_pos = pd.concat(copies, ignore_index=True)
    log.info("  Generated %d synthetic positive rows", len(synth_pos))

    # ── sample negatives ─────────────────────────────────────────────────────
    log.info("Sampling negatives ...")
    negatives = sample_negatives(df, n_neg_target, args.heavy_rain_pct,
                                 args.seed + 9999)

    # ── combine & validate ───────────────────────────────────────────────────
    train = pd.concat([synth_pos, negatives], ignore_index=True)
    train = train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    assert train["flash_flood_label"].isin([0, 1]).all(), "Label integrity failed"

    n_pos = int((train["flash_flood_label"] == 1).sum())
    n_neg = int((train["flash_flood_label"] == 0).sum())
    pos_rate = 100 * n_pos / len(train)
    missing = {c: int(train[c].isna().sum())
               for c in train.columns if train[c].isna().any()}

    # ── write ────────────────────────────────────────────────────────────────
    out_parquet = FEATURES / "synthetic_train.parquet"
    out_csv     = FEATURES / "synthetic_train.csv"
    out_meta    = FEATURES / "synthetic_meta.json"
    out_audit   = FEATURES / "synthetic_seed_audit.csv"

    train.to_parquet(out_parquet, index=False)
    train.to_csv(out_csv, index=False)

    audit = (seeds.groupby(seeds["timestamp"].dt.date)
             .agg(n_seed_cells=("latitude", "count"),
                  mean_rain_24h=("rain_24h", "mean"),
                  max_rain_24h=("rain_24h", "max"))
             .reset_index())
    audit.columns = ["seed_date","n_seed_cells","mean_rain_24h_mm","max_rain_24h_mm"]
    audit["n_copies"] = args.copies
    audit["n_synth_rows"] = audit["n_seed_cells"] * args.copies
    audit.to_csv(out_audit, index=False)

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "master_dataset": str(master_path),
        "parameters": vars(args),
        "spatial_filtering": {
            "method": "district_bbox",
            "source": "IMD/NDMA catalog Districts column + Census district bboxes",
            "note": "Bboxes are approximate. Labels remain date-level, not cell-level.",
        },
        "perturbation": {
            "rainfall_noise_frac": 0.02,
            "atmospheric_noise_frac": 0.01,
            "terrain_perturbed": False,
        },
        "dataset": {
            "total_rows": len(train),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "positive_rate_pct": round(pos_rate, 3),
            "missing_values": missing,
        },
        "seed_events": {
            "n_dates": n_seed_dates,
            "n_rows": len(seeds),
            "excluded_years": args.exclude_years,
            "events": [
                {"date": d, "districts": dist}
                for d, dist in FLASH_EVENTS
                if pd.Timestamp(d).year not in args.exclude_years
            ],
        },
        "warnings": [
            "Synthetic positives are perturbed copies of real events — not new events.",
            "Test ONLY on real held-out rows from master_dataset.parquet.",
            "District bboxes are approximate; ~10% of cells may be misassigned.",
            "flash_flood_label is date+district level, not individual cell level.",
        ],
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  SYNTHETIC TRAINING DATASET")
    print("=" * 60)
    print(f"  Total rows              : {len(train):,}")
    print(f"  Positives (synthetic)   : {n_pos:,}")
    print(f"  Negatives (real)        : {n_neg:,}")
    print(f"  Positive rate           : {pos_rate:.2f}%")
    print(f"  Seed dates              : {n_seed_dates}")
    print(f"  Seed rows (district)    : {len(seeds):,}")
    print(f"  Copies per seed row     : {args.copies}")
    print(f"  Neg threshold           : {args.heavy_rain_pct:.0f}th pct rain_24h")
    if args.exclude_years:
        print(f"  Excluded from seeds     : {args.exclude_years}")
    if missing:
        print(f"  Missing values          : {missing}")
    else:
        print(f"  Missing values          : none")
    print()
    print(f"  Files:")
    print(f"    {out_parquet.name}")
    print(f"    {out_csv.name}")
    print(f"    {out_meta.name}")
    print(f"    {out_audit.name}")
    print()
    print("  RULE: test set must contain ONLY real events.")
    print("  Use --exclude-years <year> to hold out a test year.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
