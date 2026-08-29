# Flash Flood Prediction — Uttarakhand
### SIH Problem 26192 | Data Engineering Pipeline

Reproducible data pipeline for flash-flood prediction in Indian hilly regions.
Produces a clean ML-ready dataset for the ML team.

---

## Study Region

**Uttarakhand, India** | Bbox: 28.5–31.5°N, 77.5–81.0°E | Grid: 0.1° (~11 km)

## Team Scope (This Repo)

- Data acquisition, processing, feature engineering
- ML-ready dataset output
- **Not included:** model training, frontend, backend

---

## Quick Start

```bash
pip install -r requirements.txt

# Step 1 — Download data (requires credentials for GPM/ERA5/SRTM — see below)
python src/data/download_data.py --source all

# Step 2 — Process raw data into aligned grids
python src/data/process_data.py

# Step 3 — Build ML-ready feature set
python src/features/build_features.py

# Step 4 — Validate the dataset
python src/validation/validate_dataset.py
```

Output: `data/features/master_dataset.parquet`

---

## Data Source Credentials

| Source | Register At | Env Var |
|--------|-------------|---------|
| NASA GPM IMERG | https://urs.earthdata.nasa.gov/ | `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD` |
| ERA5 / ERA5-Land | https://cds.climate.copernicus.eu/ | `~/.cdsapirc` file |
| SRTM 30m DEM | https://opentopography.org/ | `OPENTOPO_API_KEY` |

Set env vars before running `download_data.py`. All registrations are free.

**Without credentials:** `process_data.py` falls back to physics-based synthetic data
(labeled `data_source=SYNTHETIC`) so the full pipeline can be tested immediately.
Flood event labels (from IMD catalog) are **always real**.

---

## Repository Structure

```
config/
  data_config.yaml          # Bbox, dates, source URLs, paths

data/
  raw/
    flood_events/           # IMD flood event catalog (committed — small)
      floods_india.xlsx     # 6,876 events, 1967–2023, all India
      flood_labels_aditya.csv
    gpm/                    # GPM IMERG HDF5 files (gitignored — large)
    era5/                   # ERA5 NetCDF files (gitignored — large)
    srtm/                   # SRTM GeoTIFF tiles (gitignored — large)
  processed/
    flood_labels.parquet    # Uttarakhand daily flood labels (committed)
    environmental_grid.parquet   # gitignored — regenerate via pipeline
  features/
    master_dataset.parquet  # gitignored — regenerate via pipeline

src/
  data/
    download_data.py        # Download GPM / ERA5 / SRTM
    process_data.py         # Process raw → aligned grids; synthetic fallback
  features/
    build_features.py       # Rolling rainfall features, terrain merge, labels
  validation/
    validate_dataset.py     # Dataset QC and validation report

docs/
  data_plan.md              # Data sources, region, limitations
  ml_handoff.md             # Feature dictionary, split strategy, ML recommendations
```

---

## Features in Master Dataset

| Feature | Description | Unit |
|---------|-------------|------|
| `rain_3h … rain_72h` | Rolling rainfall accumulations | mm |
| `rain_5d, rain_7d, rain_14d` | Antecedent rainfall | mm |
| `max_intensity_24h` | Peak 3H intensity in past 24H | mm/3H |
| `wet_fraction_7d` | Fraction of rainy 3H periods, past 7 days | 0–1 |
| `rain_anomaly_24h` | rain_24h minus 14-day rolling mean | mm |
| `elevation_m` | Terrain elevation | m |
| `slope_deg` | Terrain slope | degrees |
| `aspect_sin / aspect_cos` | Aspect (circular encoding) | — |
| `terrain_ruggedness` | Normalized slope | 0–1 |
| `flood_prone_terrain` | Binary: steep slope at flood-prone elevation | 0/1 |
| `temperature_2m_c` | Air temperature | °C |
| `humidity_pct` | Relative humidity | % |
| `pressure_hpa` | Surface pressure | hPa |
| `soil_moisture_m3m3` | Volumetric soil water content | m³/m³ |
| **`flood_label`** | **Primary target** | 0/1 |
| **`flash_flood_label`** | **Secondary target (cloudburst)** | 0/1 |

---

## ML Team Notes

- Use **time-based train/val/test split** (not random): train ≤2020, val 2021–22, test 2023
- Labels are **state-level** (Uttarakhand), not grid-cell level
- Class imbalance ~3–5% positive during monsoon — use `scale_pos_weight` or weighted loss
- See `docs/ml_handoff.md` for full feature dictionary and recommended models
