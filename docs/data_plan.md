# Flash Flood Data Plan — SIH 26192

## Prediction Target

**Binary classification:** Will a flood/flash-flood event occur in Uttarakhand within the next 24 hours, given environmental conditions observed at time T?

- Primary target: `flood_label` (1 = any flood/heavy-rain event, 0 = no event)
- Secondary target: `flash_flood_label` (1 = cloudburst/flash flood specifically)
- Prediction horizon: event within [T, T+24h]
- Features at time T only — **no future data used as input**

---

## Study Region

**Uttarakhand, India** (including Garhwal and Kumaon Himalayas)

| Parameter | Value |
|-----------|-------|
| Bounding box | 28.5°N–31.5°N, 77.5°E–81.0°E |
| Grid resolution | 0.1° × 0.1° (~11 km) |
| Elevation range | 200 m (Gangetic plain) → 7,000 m (High Himalaya) |
| Rationale | Highest flash-flood frequency in India; Kedarnath 2013, Uttarkashi events well-documented; orographic rainfall well-characterized; IMD catalog extensive |

---

## Required Variables

| Variable | Source | Temporal Res | Spatial Res |
|----------|--------|-------------|-------------|
| Precipitation | GPM IMERG (primary) / ERA5-Land | 30 min / 1H | 0.1° |
| Temperature 2m | ERA5 | 1H | 0.25° |
| Dewpoint / Humidity | ERA5 | 1H | 0.25° |
| Surface pressure | ERA5 | 1H | 0.25° |
| Soil moisture (layer 1) | ERA5-Land | 1H | 0.1° |
| Elevation | SRTM GL1 (30m) / GL3 (90m) | static | 30–90 m |
| Slope, Aspect | Derived from SRTM | static | same as DEM |
| Flood event dates | IMD catalog | daily | point/state |

---

## Data Sources

### 1. Precipitation: NASA GPM IMERG Final Run V07

- **URL:** https://gpm.nasa.gov/data/imerg
- **Access:** Free, requires NASA Earthdata account (`EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`)
- **Coverage:** 2000–present (near-real-time available; Final Run has ~3.5 month latency)
- **Temporal res:** 30 min
- **Spatial res:** 0.1° × 0.1°
- **Format:** HDF5 via GES DISC OPeNDAP
- **Why:** Best satellite QPE over complex terrain; validated for Indian monsoon (Prakash et al. 2018)
- **Limitation:** Systematic underestimation of orographic extremes; Final Run calibrated with monthly rain gauge data

### 2. Precipitation fallback: ERA5-Land

- **URL:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- **Access:** Free, Copernicus CDS account + ~/.cdsapirc key
- **Coverage:** 1950–present
- **Temporal res:** 1H
- **Spatial res:** 0.1° × 0.1°
- **Format:** NetCDF4
- **Why:** Consistent long reanalysis; covers all required weather variables; same CDS key for ERA5

### 3. Weather: ERA5 Single-Level Reanalysis

- **URL:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- **Variables:** 2m temperature, 2m dewpoint, surface pressure, 10m wind components
- **Access:** Same CDS API key
- **Coverage:** 1940–present; 0.25° spatial
- **Why:** Only globally consistent reanalysis with full weather state; validated over South Asia

### 4. Terrain: NASA SRTM GL1 (30m preferred) / GL3 (90m open)

- **URL (30m):** https://portal.opentopography.org/API/globaldem (free API key)
- **URL (90m):** https://srtm.csi.cgiar.org/srtmdata/ (no auth required)
- **Format:** GeoTIFF
- **Why:** Standard terrain for hydrological studies; slope/aspect are critical flood predictors
- **Derived:** elevation, slope (degrees), aspect (degrees), terrain ruggedness index

### 5. Flood Event Labels: IMD Event Catalog

- **File:** `data/raw/flood_events/floods_india.xlsx`
- **Source:** NDMA/IMD via GitHub (varadtrivedi/Analysing-Flood-Risk-in-India)
- **Coverage:** 1967–2023, 121 Uttarakhand events
- **Variables:** Start date, end date, main cause, state, districts, fatalities
- **Limitation:** State-level only (not grid-level); under-reporting for older events; cause categorization inconsistent pre-2000
- **Flash flood events in Uttarakhand:** 12 (cloudburst/flash flood cause category)

---

## Historical Coverage

- **Flood labels:** 1967–2023 (121 events), but rainfall satellite data starts 2000
- **Recommended training period:** 2001–2020 (GPM IMERG available, sufficient events)
- **Validation:** 2021–2022
- **Test:** 2023

---

## Limitations

1. **Flood labels are state-level** — not gridded. All grid cells in Uttarakhand receive the same daily label, which overstates positive coverage. Ideal: district or watershed-level labels.
2. **GPM IMERG requires NASA Earthdata credentials** — pipeline ships download scripts but not data.
3. **ERA5 requires CDS API key** — also not bundled.
4. **No SRTM included in repo** — must download via OpenTopography or CGIAR-CSI.
5. **Synthetic data used in pipeline testing** — explicitly labeled `data_source=SYNTHETIC`. Replace with real data before training.
6. **No river gauge / streamflow data** — ideal addition would be CWC (Central Water Commission) gauge data, but not openly accessible via API.
7. **Class imbalance** — even within monsoon season, flood events cover <5% of timesteps. Use weighted loss, SMOTE, or stratified sampling.
