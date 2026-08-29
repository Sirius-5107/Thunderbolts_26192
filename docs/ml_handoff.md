# ML Handoff — Flash Flood Prediction Dataset

## What Data Is Available

**Master dataset:** `data/features/master_dataset.parquet`

Two versions:
1. **Synthetic (pipeline testing)** — available now; `data_source=SYNTHETIC`; flood labels are REAL (IMD catalog)
2. **Real (production)** — available after downloading GPM IMERG + ERA5 + SRTM (see `README.md`)

**Flood event labels:** REAL (IMD catalog, 1967–2023, 121 Uttarakhand events)

---

## Feature Dictionary

| Column | Description | Unit | Source |
|--------|-------------|------|--------|
| `timestamp` | Observation time (3H intervals, monsoon months) | UTC | — |
| `latitude` | Grid cell center latitude | degrees N | — |
| `longitude` | Grid cell center longitude | degrees E | — |
| `elevation_m` | Terrain elevation (mean of 0.1° cell) | meters | SRTM |
| `slope_deg` | Terrain slope | degrees (0–90) | SRTM derived |
| `aspect_sin` | Sine of aspect angle | dimensionless | SRTM derived |
| `aspect_cos` | Cosine of aspect angle | dimensionless | SRTM derived |
| `terrain_ruggedness` | Normalized slope (slope/90) | 0–1 | SRTM derived |
| `flood_prone_terrain` | 1 if slope>15° AND 300m<elev<3000m | binary | SRTM derived |
| `rain_intensity_3h` | Precipitation in current 3H window | mm | GPM/ERA5 |
| `rain_3h` | Same as rain_intensity_3h | mm | GPM/ERA5 |
| `rain_6h` | Cumulative precipitation, past 6H | mm | GPM/ERA5 |
| `rain_12h` | Cumulative precipitation, past 12H | mm | GPM/ERA5 |
| `rain_24h` | Cumulative precipitation, past 24H | mm | GPM/ERA5 |
| `rain_48h` | Cumulative precipitation, past 48H | mm | GPM/ERA5 |
| `rain_72h` | Cumulative precipitation, past 72H | mm | GPM/ERA5 |
| `rain_5d` | Cumulative precipitation, past 5 days | mm | GPM/ERA5 |
| `rain_7d` | Cumulative precipitation, past 7 days | mm | GPM/ERA5 |
| `rain_14d` | Cumulative precipitation, past 14 days | mm | GPM/ERA5 |
| `max_intensity_24h` | Max 3H intensity in past 24H | mm/3H | GPM/ERA5 |
| `wet_fraction_7d` | Fraction of wet 3H periods in past 7 days | 0–1 | GPM/ERA5 |
| `rain_anomaly_24h` | rain_24h minus 14-day rolling mean | mm | GPM/ERA5 |
| `temperature_2m_c` | Air temperature at 2m | °C | ERA5 |
| `humidity_pct` | Relative humidity | % | ERA5 |
| `pressure_hpa` | Surface pressure | hPa | ERA5 |
| `soil_moisture_m3m3` | Volumetric soil water content (0–10cm) | m³/m³ | ERA5-Land |
| `flood_label` | **PRIMARY TARGET**: any flood/heavy-rain event | binary (0/1) | IMD |
| `flash_flood_label` | **SECONDARY TARGET**: cloudburst/flash flood | binary (0/1) | IMD |
| `data_source` | "SYNTHETIC" or data origin tag | string | pipeline |

---

## Spatial/Temporal Resolution

- **Spatial:** 0.1° × 0.1° grid (~11 km), 30×36 = 1,080 cells over Uttarakhand bbox
- **Temporal:** 3-hourly, monsoon season only (June–September)
- **Coverage:** 2001–2023 monsoon seasons = ~22 seasons × ~120 days × 8 timesteps/day = ~21,000 timesteps per cell

---

## Target Definition

```
features at time T (current + past only)
           ↓
flood_label = 1 if any flood event occurred in Uttarakhand on day(T)
```

**Important:** Labels are **state-level** (Uttarakhand), not cell-level. All grid cells get the same label for a given day. This is a known limitation — treat predictions as state-level flood probability, not cell-specific.

**Recommended approach:** Use `flood_label` for initial model; use `flash_flood_label` as harder/rarer target.

---

## Date Range

- Training: 2001–2020 (20 monsoon seasons, ~336,000 rows)
- Validation: 2021–2022 (2 seasons, ~33,600 rows)
- Test: 2023 (1 season, ~16,800 rows)

---

## Recommended Train/Validation/Test Split

**Use time-based splitting — NOT random splitting.**

Random splitting causes data leakage because:
- Rolling rainfall features (rain_24h, rain_72h...) encode temporal autocorrelation
- A test point at T can have training neighbors at T±3h

```python
df['year'] = pd.to_datetime(df['timestamp']).dt.year
train = df[df['year'] <= 2020]
val   = df[df['year'].isin([2021, 2022])]
test  = df[df['year'] == 2023]
```

---

## Missing Data Handling

- Rolling features will have `NaN` for the first N timesteps at the start of each monsoon season (warm-up period). **Impute with 0** (no prior rainfall assumed) or drop first 7 days of each season.
- Terrain features should have no missing values for cells within the bbox.
- If real GPM/ERA5 data has gaps, `process_data.py` logs them — do not impute rainfall with non-zero values.

---

## Potential Leakage Risks

1. **Same-day features:** `rain_24h` at 18:00 includes rainfall from 18:00–00:00 the same day. The flood label covers the full day. **This is acceptable** — the prediction task is whether conditions AT time T indicate a flood WILL occur, not whether rainfall after T causes it. However, if the event typically starts at 10:00 and `rain_24h` at 23:00 includes event rainfall — mild leakage. Mitigate by using T-12H lag for the prediction target.
2. **Spatial aggregation:** All cells share the same label. Cells far from the actual event still get `flood_label=1`. Consider spatial masking if district-level labels become available.
3. **Antecedent rainfall features** (`rain_14d`) — these are safe (past-only).

---

## Recommended Baseline Models

| Model | Why |
|-------|-----|
| Logistic Regression | Fast baseline; interpretable coefficients |
| Gradient Boosted Trees (XGBoost/LightGBM) | Handles nonlinear terrain × rainfall interactions; strong for tabular data |
| Random Forest | Good uncertainty estimation |
| LSTM / Temporal CNN | If time series structure per cell is used |

**Start with XGBoost or LightGBM.** They handle class imbalance well with `scale_pos_weight`.

**Recommended metric:** F1-score (positive class) and AUC-PR (precision-recall) — NOT accuracy, due to class imbalance (~95% negative during monsoon).

## Class Imbalance

Flood positive rate: ~3–5% of monsoon timesteps.

Options:
- `scale_pos_weight = n_negative / n_positive` in XGBoost
- `class_weight='balanced'` in sklearn
- Oversample flood timesteps (SMOTE on tabular features)
- Cost-sensitive loss (FN more costly than FP for early warning)

---

## What Is Still Needed

1. **Real GPM IMERG rainfall** — most important missing piece. Download with `src/data/download_data.py --source gpm` after setting NASA Earthdata credentials.
2. **Real ERA5 weather** — download with `--source era5` after creating `~/.cdsapirc`.
3. **Real SRTM terrain** — download with `--source srtm` after setting `OPENTOPO_API_KEY`.
4. **District-level or watershed-level flood labels** — contact NDMA / CWC for finer-grained labels.
5. **CWC river gauge data** — discharge thresholds would make `flood_label` more precise.
