"""
process_gpm.py
--------------
Processes downloaded NASA GPM IMERG files into the project's environmental-grid
format (3-hourly, 0.1-deg, Uttarakhand bbox).

Input:  data/raw/gpm/**/3B-HHR.MS.MRG.3IMERG.*.HDF5       (full granules)
        data/raw/gpm/**/3B-HHR.MS.MRG.3IMERG.*.HDF5.nc4   (OPeNDAP bbox subsets)
Output: data/processed/gpm_rainfall.parquet
        data/processed/environmental_grid.parquet

TWO FILE LAYOUTS
----------------
A full granule (direct HTTPS download) keeps the HDF5 group hierarchy:
    /Grid/precipitation   (time, lon, lat)  shape (1, 3600, 1800)
    /Grid/lat  /Grid/lon  /Grid/time

An OPeNDAP subset (.HDF5.nc4) is still HDF5 by magic bytes, but Hyrax's DAP2
response FLATTENS the group hierarchy. There is no "Grid" group at all:
    precipitation   (time, lon, lat)  shape (1, n_lon, n_lat)   root level
    lat  lon  time                                              root level
The dataset keeps attrs fullnamepath="/Grid/precipitation" and
DimensionNames="time,lon,lat", which is how the layout is detected.

Reading a subset via hf["Grid"] raises
    "Unable to synchronously open object (object 'Grid' doesn't exist)"
That is a LAYOUT MISMATCH, not corruption. Unreadable files are moved to
data/raw/gpm/_quarantine/ and are never deleted.

Units: IMERG `precipitation` is a RATE in mm/hr. Each granule covers 30 min,
so depth_mm = rate * 0.5. Six granules sum to one 3-hourly accumulation.

Usage:
  python src/data/process_gpm.py
  python src/data/process_gpm.py --dry-run     # inspect layouts, write nothing
  python src/data/process_gpm.py --all-months  # skip the monsoon-month filter
"""

import argparse
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "config" / "data_config.yaml"))

BBOX = CFG["region"]["bbox"]
RES = CFG["region"]["grid_resolution_deg"]
MONSOON_MONTHS = set(CFG["time"]["monsoon_months"])

# Reproduce the GES DISC OPeNDAP subsetter's cell selection exactly.
#
# A request for lat 28.5-31.5 returns centres 28.55...31.55 (31 rows). It is
# NOT a symmetric half-cell tolerance: 28.45 is excluded but 31.55 is kept.
# The rule that reproduces it is  min < centre < max + RES.
#
# This matters because full granules are clipped here while subsets arrive
# pre-clipped. Any other rule gives the two sources different grids, and every
# full granule then fails as a grid mismatch.
EPS = 1e-6
RAW_GPM = ROOT / CFG["paths"]["raw_gpm"]
OUT = ROOT / CFG["paths"]["processed"]
QUARANTINE = RAW_GPM / "_quarantine"

FILL_THRESHOLD = -9000.0   # IMERG _FillValue is -9999.9

# 3B-HHR.MS.MRG.3IMERG.20240601-S000000-E002959.0000.V07B.HDF5[.nc4]
_NAME_RE = re.compile(r"3IMERG\.(\d{8})-S(\d{6})-E\d{6}")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def timestamp_from_name(fpath: Path) -> datetime:
    """
    Granule start time taken from the filename.

    Preferred over the `time` variable: unambiguous, needs no epoch
    assumption, and present even when a subset drops the time array.
    """
    m = _NAME_RE.search(fpath.name)
    if not m:
        raise ValueError(f"cannot parse IMERG timestamp from filename: {fpath.name}")
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _axis_order(dset, n_lat: int, n_lon: int) -> str:
    """
    Decide whether the array is lat-major or lon-major.

    Trusts DimensionNames when present, otherwise matches the array shape
    against the coordinate lengths.
    """
    dims = dset.attrs.get("DimensionNames")
    if dims is not None:
        if isinstance(dims, bytes):
            dims = dims.decode()
        dims = str(dims).replace(" ", "").lower()
        if dims == "time,lat,lon":
            return "lat_major"
        if dims == "time,lon,lat":
            return "lon_major"

    shape = tuple(s for s in dset.shape if s != 1)
    if shape == (n_lat, n_lon):
        return "lat_major"
    if shape == (n_lon, n_lat):
        return "lon_major"
    if n_lat == n_lon:
        return "lon_major"   # square subset, no attribute: IMERG native order
    raise ValueError(f"shape {dset.shape} matches neither "
                     f"({n_lat},{n_lon}) nor ({n_lon},{n_lat})")


def read_imerg(fpath: Path):
    """
    Read one IMERG granule, full or OPeNDAP subset.

    Returns (precip[lat, lon] in mm/hr with NaN fill, lats ascending,
    lons ascending), subset to the configured bbox.
    """
    import h5py

    with h5py.File(fpath, "r") as hf:
        grp = hf["Grid"] if "Grid" in hf else hf   # flattened DAP2 has no Grid

        dset = None
        for name in ("precipitation", "precipitationCal"):
            if name in grp:
                dset = grp[name]
                break
        if dset is None:
            raise KeyError(f"no precipitation variable; keys = {list(grp.keys())}")

        # Round to the 0.1-deg grid. Subsets store coordinates as float32
        # (81.05 comes back as 81.049995 or 81.050003 depending on the file),
        # and full granules as float64. Without rounding, the same physical
        # cell gets two different keys and groupby splits it in two.
        lats = np.round(np.asarray(grp["lat"][:], dtype=np.float64), 3)
        lons = np.round(np.asarray(grp["lon"][:], dtype=np.float64), 3)

        order = _axis_order(dset, lats.size, lons.size)
        arr = np.squeeze(np.asarray(dset[:], dtype=np.float32))
        if order == "lon_major":
            arr = arr.T                             # -> (lat, lon)

    arr = np.array(arr, dtype=np.float32, copy=True)
    arr[arr < FILL_THRESHOLD] = np.nan

    # Subset to bbox. Full granules are global; OPeNDAP subsets are already
    # clipped, so this is a no-op for them.
    lat_sel = np.where((lats > BBOX["lat_min"] + EPS)
                       & (lats < BBOX["lat_max"] + RES - EPS))[0]
    lon_sel = np.where((lons > BBOX["lon_min"] + EPS)
                       & (lons < BBOX["lon_max"] + RES - EPS))[0]
    if lat_sel.size == 0 or lon_sel.size == 0:
        raise ValueError("no grid points inside configured bbox")

    arr = arr[np.ix_(lat_sel, lon_sel)]
    lats, lons = lats[lat_sel], lons[lon_sel]

    # Normalise orientation so every file stacks identically.
    if lats.size > 1 and lats[0] > lats[-1]:
        lats, arr = lats[::-1], arr[::-1, :]
    if lons.size > 1 and lons[0] > lons[-1]:
        lons, arr = lons[::-1], arr[:, ::-1]

    return arr, lats, lons


def quarantine(fpath: Path, reason: str) -> None:
    """Move an unreadable file aside. Raw data is never deleted."""
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(fpath), str(QUARANTINE / fpath.name))
        log.warning("  Quarantined %s (%s)", fpath.name, reason)
    except Exception as exc:
        log.error("  Could not quarantine %s: %s", fpath.name, exc)


def build_environmental_grid_from_gpm(gpm_3h: pd.DataFrame) -> pd.DataFrame:
    """
    Produce environmental_grid.parquet compatible with build_features.py.
    Weather columns stay NaN until ERA5 is merged in.
    """
    df = gpm_3h.copy()
    df["temperature_2m_c"] = np.nan
    df["humidity_pct"] = np.nan
    df["pressure_hpa"] = np.nan
    df["soil_moisture_m3m3"] = np.nan
    df["data_source"] = "GPM_IMERG_V07"
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def collect_files():
    nc4 = sorted(RAW_GPM.rglob("*.HDF5.nc4"))
    full = sorted(RAW_GPM.rglob("*.HDF5"))
    full_only = [f for f in full if not (f.parent / (f.name + ".nc4")).exists()]
    files = [f for f in nc4 + full_only if QUARANTINE not in f.parents]
    return sorted(files, key=lambda p: p.name), len(nc4), len(full_only)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="inspect a sample of files and exit without writing")
    ap.add_argument("--all-months", action="store_true",
                    help="process every month, not just the monsoon months")
    args = ap.parse_args(argv)

    log.info("=== process_gpm.py ===")
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        import h5py  # noqa: F401
    except ImportError:
        log.error("h5py not installed. Run: pip install h5py")
        return 1

    all_files, n_nc4, n_full = collect_files()
    if not all_files:
        log.error("No GPM files in %s. Run download_data.py --source gpm first.", RAW_GPM)
        return 1
    log.info("Found %d files (%d OPeNDAP subsets, %d full granules)",
             len(all_files), n_nc4, n_full)

    if not args.all_months:
        kept = []
        for f in all_files:
            try:
                if timestamp_from_name(f).month in MONSOON_MONTHS:
                    kept.append(f)
            except ValueError:
                kept.append(f)     # unparseable name: let the reader judge it
        log.info("Monsoon filter (months %s): %d of %d files",
                 sorted(MONSOON_MONTHS), len(kept), len(all_files))
        all_files = kept

    if args.dry_run:
        log.info("--- DRY RUN: inspecting up to 5 files, writing nothing ---")
        for f in all_files[:5]:
            try:
                arr, lats, lons = read_imerg(f)
                log.info("  OK   %s | ts=%s | grid=%dx%d | valid=%.1f%% | max=%.2f mm/hr",
                         f.name, timestamp_from_name(f), lats.size, lons.size,
                         100 * np.isfinite(arr).mean(), float(np.nanmax(arr)))
            except Exception as exc:
                log.error("  FAIL %s: %s", f.name, exc)
        return 0

    # ---- Pass 1: read granules, accumulate directly into 3h bins ----------
    # Accumulating here rather than collecting one DataFrame per file keeps
    # peak memory at tens of MB instead of several GB across ~47k files.
    sums, counts = {}, {}
    ref_lats = ref_lons = None
    n_lat = n_lon = 0
    n_ok = n_bad = n_mismatch = 0

    for i, fpath in enumerate(all_files):
        if i % 2000 == 0:
            log.info("  [%d/%d] %s", i + 1, len(all_files), fpath.name)
        try:
            ts = timestamp_from_name(fpath)
            arr, lats, lons = read_imerg(fpath)
        except Exception as exc:
            log.error("  Failed to parse %s: %s", fpath.name, exc)
            quarantine(fpath, str(exc)[:80])
            n_bad += 1
            continue

        if ref_lats is None:
            ref_lats, ref_lons = lats, lons
            n_lat, n_lon = lats.size, lons.size
            log.info("  Reference grid: %d lat x %d lon | lat %.2f-%.2f | lon %.2f-%.2f",
                     n_lat, n_lon, lats[0], lats[-1], lons[0], lons[-1])
        elif arr.shape != (n_lat, n_lon):
            log.error("  Grid mismatch in %s: %s vs (%d,%d)",
                      fpath.name, arr.shape, n_lat, n_lon)
            quarantine(fpath, "grid mismatch")
            n_mismatch += 1
            continue

        bin_ts = pd.Timestamp(ts).floor("3h")
        if bin_ts not in sums:
            sums[bin_ts] = np.zeros((n_lat, n_lon), dtype=np.float64)
            counts[bin_ts] = np.zeros((n_lat, n_lon), dtype=np.int16)
        valid = np.isfinite(arr)
        sums[bin_ts][valid] += arr[valid] * 0.5      # mm/hr over 30 min -> mm
        counts[bin_ts][valid] += 1
        n_ok += 1

    log.info("Parsed %d granules OK | %d unreadable | %d grid mismatch",
             n_ok, n_bad, n_mismatch)
    if n_bad or n_mismatch:
        log.warning("Quarantined files are in %s. Inspect them; "
                    "do not assume they are corrupt.", QUARANTINE)
    if not sums:
        log.error("No granules could be read. Nothing written.")
        return 1

    # ---- Pass 2: bins -> long DataFrame ----------------------------------
    bins = sorted(sums)
    log.info("Building 3h frame: %d bins x %d cells = %d rows",
             len(bins), n_lat * n_lon, len(bins) * n_lat * n_lon)

    stack = np.empty((len(bins), n_lat, n_lon), dtype=np.float32)
    obs = np.empty((len(bins), n_lat, n_lon), dtype=np.int16)
    for k, b in enumerate(bins):
        c = counts[b]
        v = sums[b].astype(np.float32)
        v[c == 0] = np.nan          # no observations -> NaN, not a fake zero
        stack[k], obs[k] = v, c

    lat_grid, lon_grid = np.meshgrid(ref_lats, ref_lons, indexing="ij")
    n_cells = n_lat * n_lon
    df_3h = pd.DataFrame({
        "timestamp": np.repeat(np.array(bins, dtype="datetime64[ns]"), n_cells),
        "latitude": np.tile(lat_grid.ravel().astype(np.float32), len(bins)),
        "longitude": np.tile(lon_grid.ravel().astype(np.float32), len(bins)),
        "precip_3h_mm": stack.reshape(-1),
        "n_obs_3h": obs.reshape(-1),
    })
    df_3h["precip_3h_mm"] = df_3h["precip_3h_mm"].clip(lower=0)

    gpm_out = OUT / "gpm_rainfall.parquet"
    df_3h.to_parquet(gpm_out, index=False)
    log.info("  Saved %s (%.1f MB)", gpm_out.name, gpm_out.stat().st_size / 1e6)

    env_out = OUT / "environmental_grid.parquet"
    build_environmental_grid_from_gpm(df_3h).to_parquet(env_out, index=False)
    log.info("  Saved %s (%.1f MB, %d rows)",
             env_out.name, env_out.stat().st_size / 1e6, len(df_3h))

    # ---- Verification -----------------------------------------------------
    log.info("=== GPM DATA VERIFICATION ===")
    log.info("  Time range : %s to %s", df_3h["timestamp"].min(), df_3h["timestamp"].max())
    log.info("  Lat range  : %.2f to %.2f",
             float(df_3h["latitude"].min()), float(df_3h["latitude"].max()))
    log.info("  Lon range  : %.2f to %.2f",
             float(df_3h["longitude"].min()), float(df_3h["longitude"].max()))
    log.info("  Precip mm/3h min/mean/max : %.3f / %.3f / %.3f",
             float(df_3h["precip_3h_mm"].min()),
             float(df_3h["precip_3h_mm"].mean()),
             float(df_3h["precip_3h_mm"].max()))
    log.info("  Bins with all 6 granules  : %.1f%%",
             100 * (df_3h["n_obs_3h"] == 6).mean())
    log.info("  NaN precip rows           : %d (%.2f%%)",
             int(df_3h["precip_3h_mm"].isna().sum()),
             100 * df_3h["precip_3h_mm"].isna().mean())

    # Seasonal total per cell is the check that catches a units error.
    season = df_3h.dropna(subset=["precip_3h_mm"]).copy()
    season["year"] = season["timestamp"].dt.year
    per_cell = season.groupby("year")["precip_3h_mm"].sum() / n_cells
    log.info("  Seasonal rainfall per cell (expect ~1000-1200 mm for a full "
             "Jun-Sep monsoon):")
    for year, total in per_cell.items():
        log.info("    %d : %7.1f mm", year, total)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
