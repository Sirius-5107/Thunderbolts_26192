"""
download_data.py
----------------
Downloads real datasets for flash-flood prediction in Uttarakhand.

Supported sources:
  gpm    -- NASA GPM IMERG Final Run V07 (half-hourly HDF5)
  era5   -- ERA5-Land + ERA5 single-level (NetCDF4, via CDS API)
  srtm   -- SRTM DEM 30m via OpenTopography or 90m CGIAR-CSI
  events -- IMD flood event catalog (already committed)

Usage:
  python src/data/download_data.py --source gpm [--start 2023-08-01] [--end 2023-08-31]

GPM credentials: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars.
  Register free: https://urs.earthdata.nasa.gov/
  Required app:  https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg
"""

import argparse
import datetime
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "data_config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GPM IMERG Final Run V07  (GPM_3IMERGHH.07, half-hourly, 0.1 deg)
# ---------------------------------------------------------------------------
# Access method: NASA GES DISC HTTPS data server with Earthdata cookie auth.
# Auth flow:
#   1. POST credentials to URS to get a session cookie
#   2. All subsequent GES DISC requests carry that cookie automatically
#   3. Files redirect through URS; requests.Session follows redirects
#
# Product:  GPM_3IMERGHH.07  (Final Run, latency ~3.5 months)
# Files:    3B-HHR.MS.MRG.3IMERG.<date>-S<HH><MM>00-E<HH><MM>59.<HHMM>.V07B.HDF5
# Size:     ~3-4 MB per file, 48 files/day
# Docs:     https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary

GESDISC_BASE = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"
URS_LOGIN    = "https://urs.earthdata.nasa.gov"


def _build_earthdata_session(user: str, pwd: str) -> requests.Session:
    """
    Create an authenticated requests.Session for NASA GES DISC.
    Uses cookie-based auth: posts to URS, then all redirects are followed.
    """
    session = requests.Session()
    session.auth = (user, pwd)
    # requests follows redirects automatically; Earthdata sets a cookie after
    # the first authenticated redirect. We prime it here.
    r = session.get(URS_LOGIN, timeout=20, allow_redirects=True)
    if r.status_code not in (200, 302):
        raise RuntimeError(f"URS login page returned HTTP {r.status_code}")
    log.info("GPM: Earthdata session initialised (HTTP %d)", r.status_code)
    return session


def _list_day_files(session: requests.Session, date: datetime.date) -> list:
    """Return list of HDF5 filenames available for a given date on GES DISC."""
    doy = date.strftime("%j")
    url = f"{GESDISC_BASE}/{date.year}/{doy}/"
    r = session.get(url, timeout=30)
    if r.status_code == 401:
        raise RuntimeError(
            "HTTP 401 Unauthorized. Check credentials and ensure the GES DISC app is approved:\n"
            "  https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg"
        )
    if r.status_code == 404:
        log.warning("GPM: No directory for %s (404) — data may not be available yet", date)
        return []
    if r.status_code != 200:
        raise RuntimeError(f"GES DISC listing HTTP {r.status_code} for {url}")
    # Extract HDF5 filenames from HTML directory listing
    files = re.findall(r'href="(3B-HHR\.MS\.MRG\.3IMERG\.[^"]+\.HDF5)"', r.text)
    return sorted(set(files))


def _download_file(session: requests.Session, date: datetime.date,
                   fname: str, outdir: Path) -> Path:
    """Download one HDF5 file; skip if already present and non-zero."""
    fpath = outdir / fname
    if fpath.exists() and fpath.stat().st_size > 100_000:
        return fpath  # already downloaded

    doy = date.strftime("%j")
    url = f"{GESDISC_BASE}/{date.year}/{doy}/{fname}"
    r = session.get(url, stream=True, timeout=120, allow_redirects=True)
    if r.status_code == 401:
        raise RuntimeError(
            f"HTTP 401 downloading {fname}. "
            "Approve the GES DISC app at:\n"
            "  https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg"
        )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} downloading {fname}")

    with open(fpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    size_mb = fpath.stat().st_size / 1e6
    if size_mb < 0.1:
        fpath.unlink()
        raise RuntimeError(f"Downloaded file too small ({size_mb:.2f} MB) — likely an error page")
    return fpath


def download_gpm(cfg, start_override: str = None, end_override: str = None) -> bool:
    """
    Download GPM IMERG Final Run V07 HDF5 files for the Uttarakhand bbox period.

    Date range defaults to one month (Aug 2023) unless overridden via CLI.
    Files saved to: data/raw/gpm/YYYY/DOY/

    Required env vars:
      EARTHDATA_USERNAME
      EARTHDATA_PASSWORD
    """
    user = os.environ.get("EARTHDATA_USERNAME", "").strip()
    pwd  = os.environ.get("EARTHDATA_PASSWORD", "").strip()
    if not user or not pwd:
        log.error(
            "GPM: EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set.\n"
            "  Register at https://urs.earthdata.nasa.gov/ and set env vars."
        )
        return False

    # Date range — default to Aug 2023 test window
    start_str = start_override or "2023-08-01"
    end_str   = end_override   or "2023-08-31"
    start_dt  = datetime.date.fromisoformat(start_str)
    end_dt    = datetime.date.fromisoformat(end_str)

    outdir = ROOT / cfg["paths"]["raw_gpm"]
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("GPM IMERG: %s to %s", start_str, end_str)
    log.info("GPM IMERG: bbox lat %.1f-%.1f lon %.1f-%.1f",
             cfg["region"]["bbox"]["lat_min"], cfg["region"]["bbox"]["lat_max"],
             cfg["region"]["bbox"]["lon_min"], cfg["region"]["bbox"]["lon_max"])
    log.info("GPM IMERG: Output dir: %s", outdir)

    # Authenticate
    try:
        session = _build_earthdata_session(user, pwd)
    except Exception as e:
        log.error("GPM: Authentication failed: %s", e)
        return False

    # Verify with one listing before committing to full download
    log.info("GPM: Verifying access with listing for %s ...", start_dt)
    try:
        test_files = _list_day_files(session, start_dt)
    except RuntimeError as e:
        log.error("GPM: Access check failed: %s", e)
        return False

    if not test_files:
        log.error("GPM: No files found for %s — check date range and product availability", start_dt)
        return False

    log.info("GPM: Access verified. Found %d files for %s. Starting download...", len(test_files), start_dt)

    # Download all days
    total_files = 0
    total_bytes = 0
    current = start_dt

    while current <= end_dt:
        day_dir = outdir / str(current.year) / current.strftime("%j")
        day_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = _list_day_files(session, current)
        except RuntimeError as e:
            log.error("GPM: Listing failed for %s: %s", current, e)
            current += datetime.timedelta(days=1)
            continue

        if not files:
            log.warning("GPM: No files for %s, skipping", current)
            current += datetime.timedelta(days=1)
            continue

        day_count = 0
        for fname in files:
            try:
                fpath = _download_file(session, current, fname, day_dir)
                total_bytes += fpath.stat().st_size
                day_count += 1
                total_files += 1
            except RuntimeError as e:
                log.error("GPM: Download failed for %s: %s", fname, e)
                return False
            time.sleep(0.1)  # polite rate limit

        log.info("GPM: %s — %d files (total so far: %d, %.1f MB)",
                 current, day_count, total_files, total_bytes / 1e6)
        current += datetime.timedelta(days=1)

    log.info("GPM IMERG: Download complete — %d files, %.1f MB total",
             total_files, total_bytes / 1e6)
    return True


# ---------------------------------------------------------------------------
# ERA5 via CDS API
# ---------------------------------------------------------------------------

def download_era5(cfg) -> bool:
    try:
        import cdsapi
    except ImportError:
        log.warning("ERA5: cdsapi not installed. Run: pip install cdsapi")
        return False

    cdsrc = Path.home() / ".cdsapirc"
    if not cdsrc.exists():
        log.warning("ERA5: ~/.cdsapirc not found. Register at https://cds.climate.copernicus.eu/")
        return False

    bbox = cfg["region"]["bbox"]
    area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
    months = [str(m).zfill(2) for m in cfg["time"]["monsoon_months"]]

    outdir = ROOT / cfg["paths"]["raw_era5"]
    outdir.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    era5land_out = outdir / "era5land_precipitation_monsoon.nc"
    if not era5land_out.exists():
        years = [str(y) for y in range(2019, 2024)]
        log.info("ERA5-Land: Requesting precipitation 2019-2023, monsoon months")
        c.retrieve("reanalysis-era5-land", {
            "variable": ["total_precipitation", "volumetric_soil_water_layer_1"],
            "year": years, "month": months,
            "day": [str(d).zfill(2) for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area, "format": "netcdf",
        }, str(era5land_out))
    else:
        log.info("ERA5-Land: Already exists.")

    era5_out = outdir / "era5_weather_monsoon.nc"
    if not era5_out.exists():
        years = [str(y) for y in range(2019, 2024)]
        log.info("ERA5: Requesting weather variables")
        c.retrieve("reanalysis-era5-single-levels", {
            "variable": ["2m_temperature", "2m_dewpoint_temperature",
                         "surface_pressure", "10m_u_component_of_wind",
                         "10m_v_component_of_wind"],
            "product_type": "reanalysis",
            "year": years, "month": months,
            "day": [str(d).zfill(2) for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area, "format": "netcdf",
        }, str(era5_out))
    else:
        log.info("ERA5: Already exists.")

    return True


# ---------------------------------------------------------------------------
# SRTM DEM
# ---------------------------------------------------------------------------

def download_srtm(cfg) -> bool:
    import math
    bbox   = cfg["region"]["bbox"]
    outdir = ROOT / cfg["paths"]["raw_srtm"]
    outdir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENTOPO_API_KEY", "")

    if api_key:
        url = (f"https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1"
               f"&south={bbox['lat_min']}&north={bbox['lat_max']}"
               f"&west={bbox['lon_min']}&east={bbox['lon_max']}"
               f"&outputFormat=GTiff&API_Key={api_key}")
        outfile = outdir / "srtm_uttarakhand_30m.tif"
        if not outfile.exists():
            r = requests.get(url, stream=True, timeout=300)
            if r.status_code == 200:
                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                log.info("SRTM GL1: %.1f MB", outfile.stat().st_size / 1e6)
            else:
                log.error("SRTM GL1: HTTP %d", r.status_code)
                return False
        return True

    # Fallback: CGIAR-CSI 90m tiles (open)
    def tile_num(lon, lat):
        return math.floor((lon + 180) / 5) + 1, math.floor((60 - lat) / 5) + 1

    corners = [(bbox["lon_min"], bbox["lat_max"]), (bbox["lon_max"], bbox["lat_max"]),
                (bbox["lon_min"], bbox["lat_min"]), (bbox["lon_max"], bbox["lat_min"])]
    tiles = set(tile_num(lon, lat) for lon, lat in corners)
    for tx, ty in tiles:
        fname = f"srtm_{tx:02d}_{ty:02d}.zip"
        outfile = outdir / fname
        if outfile.exists():
            continue
        url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{fname}"
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            with open(outfile, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            log.info("SRTM 90m: %s %.1f MB", fname, outfile.stat().st_size / 1e6)
        else:
            log.error("SRTM 90m: HTTP %d for %s", r.status_code, url)
    return True


# ---------------------------------------------------------------------------
# Flood Events
# ---------------------------------------------------------------------------

def check_events(cfg) -> bool:
    p = ROOT / cfg["paths"]["raw_events"] / "floods_india.xlsx"
    if p.exists():
        log.info("Flood events: %s (%.0f KB)", p.name, p.stat().st_size / 1e3)
        return True
    log.warning("Flood events file not found: %s", p)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download flash flood datasets")
    parser.add_argument("--source", default="all",
                        choices=["all", "gpm", "era5", "srtm", "events"])
    parser.add_argument("--start", default=None,
                        help="GPM start date YYYY-MM-DD (default: 2023-08-01)")
    parser.add_argument("--end", default=None,
                        help="GPM end date YYYY-MM-DD (default: 2023-08-31)")
    args = parser.parse_args()
    cfg = load_config()

    results = {}
    if args.source in ("all", "gpm"):
        results["gpm"] = download_gpm(cfg, args.start, args.end)
    if args.source in ("all", "era5"):
        results["era5"] = download_era5(cfg)
    if args.source in ("all", "srtm"):
        results["srtm"] = download_srtm(cfg)
    if args.source in ("all", "events"):
        results["events"] = check_events(cfg)

    failed = [k for k, v in results.items() if not v]
    log.info("Summary: %s", results)
    if failed:
        log.warning("Failed/skipped: %s", failed)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
