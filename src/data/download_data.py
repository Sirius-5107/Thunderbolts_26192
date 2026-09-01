"""
download_data.py
----------------
Downloads real datasets for flash-flood prediction in Uttarakhand.

GPM IMERG auth: NASA Earthdata Bearer token via URS OAuth endpoint.
This is the NASA GES DISC-documented Python download method.
See: https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Download%20Data%20Files%20from%20HTTPS%20Service%20with%20Python

Usage:
  python src/data/download_data.py --source gpm [--start 2023-08-01] [--end 2023-08-31]
  python src/data/download_data.py --source gpm --test-one   # download ONE file only

Credentials: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD env vars.
  Register: https://urs.earthdata.nasa.gov/
  Required app approval: https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg
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
# NASA Earthdata authentication — Bearer token method
# ---------------------------------------------------------------------------
# GES DISC file downloads follow this redirect chain:
#   GET file.HDF5  ->  302 to URS OAuth  ->  302 back with cookie  ->  200 file
#
# requests.Session with Basic Auth FAILS because:
#   - requests strips Authorization header on cross-domain redirects (RFC 7235)
#   - URS receives no credentials on the redirect -> returns login HTML (0 bytes data)
#
# NASA-documented fix: obtain a Bearer token from URS *before* any file request,
# then send "Authorization: Bearer <token>" directly — no redirect auth needed.
# Source: https://disc.gsfc.nasa.gov/information/howto (Python download guide)

URS_TOKEN_URL = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token"
URS_LOGIN     = "https://urs.earthdata.nasa.gov"
GESDISC_BASE    = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07"
GESDISC_OPENDAP = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07"


def _get_bearer_token(user: str, pwd: str) -> str:
    """
    Obtain a long-lived Bearer token from NASA Earthdata URS.
    This token is sent as 'Authorization: Bearer <token>' on all GES DISC requests,
    bypassing the redirect-auth stripping issue entirely.
    """
    r = requests.post(
        URS_TOKEN_URL,
        auth=(user, pwd),
        timeout=30,
    )
    if r.status_code == 401:
        raise RuntimeError(
            "URS authentication failed (HTTP 401). "
            "Check EARTHDATA_USERNAME / EARTHDATA_PASSWORD."
        )
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"URS token endpoint returned HTTP {r.status_code}: {r.text[:300]}"
        )
    data = r.json()
    token = data.get("access_token") or data.get("token", {}).get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in URS response: {data}")
    log.info("GPM: Earthdata Bearer token obtained (expires: %s)",
             data.get("expiration_date", "unknown"))
    return token


def _build_session(token: str) -> requests.Session:
    """Build a requests.Session that sends Bearer auth on every request."""
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    # Disable automatic Basic Auth — we use Bearer only
    session.auth = None
    return session


# ---------------------------------------------------------------------------
# GES DISC listing and download
# ---------------------------------------------------------------------------

def _list_day_files(session: requests.Session, date: datetime.date) -> list:
    """Return sorted list of HDF5 filenames for one day from GES DISC."""
    doy = date.strftime("%j")
    url = f"{GESDISC_BASE}/{date.year}/{doy}/"
    r = session.get(url, timeout=30)
    if r.status_code == 401:
        raise RuntimeError(
            "HTTP 401 on directory listing. "
            "Approve the GES DISC app: "
            "https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg"
        )
    if r.status_code == 404:
        log.warning("GPM: 404 for %s — data not yet available", date)
        return []
    if r.status_code != 200:
        raise RuntimeError(f"GES DISC listing HTTP {r.status_code} for {url}")
    files = re.findall(r'href="(3B-HHR\.MS\.MRG\.3IMERG\.[^"]+\.HDF5)"', r.text)
    return sorted(set(files))


def _opendap_subset_url(date: datetime.date, fname: str,
                         lat_min: float, lat_max: float,
                         lon_min: float, lon_max: float) -> str:
    """
    Build an OPeNDAP URL that downloads ONLY the Uttarakhand bbox subset.

    GPM IMERG V07B global grid: lat 1800 pts (-89.95 to 89.95, S->N, 0.1 deg)
                                 lon 3600 pts (-179.95 to 179.95, W->E, 0.1 deg)
    Array axis order: [time, lon, lat]  (V07B)

    OPeNDAP slice notation: variable[time_start:time_end][lon_start:lon_end][lat_start:lat_end]
    Returns a .nc4 NetCDF4 file with only the requested subset (~50-200 KB vs 8.5 MB full file).
    """
    doy = date.strftime("%j")
    # Convert geographic coords to array indices
    lat_i0 = round((lat_min + 90.0) / 0.1)
    lat_i1 = round((lat_max + 90.0) / 0.1)
    lon_i0 = round((lon_min + 180.0) / 0.1)
    lon_i1 = round((lon_max + 180.0) / 0.1)
    base = f"{GESDISC_OPENDAP}/{date.year}/{doy}/{fname}.nc4"
    query = (
        f"?precipitation[0:0][{lon_i0}:{lon_i1}][{lat_i0}:{lat_i1}]"
        f",time[0:0],lon[{lon_i0}:{lon_i1}],lat[{lat_i0}:{lat_i1}]"
    )
    return base + query


def _download_one_file(session: requests.Session, date: datetime.date,
                       fname: str, outdir: Path,
                       bbox: dict = None,
                       max_retries: int = 3) -> Path:
    """
    Download one GPM IMERG file via OPeNDAP spatial subset (bbox only).

    OPeNDAP returns a NetCDF4 (.nc4) file containing only the requested
    lat/lon bbox — ~50-200 KB instead of the 8.5 MB global HDF5 file.
    Saved with .nc4 extension; process_gpm.py handles both HDF5 and NC4.

    Falls back to full HDF5 download if bbox is None.
    """
    # Use .nc4 extension for OPeNDAP subset files
    use_opendap = bbox is not None
    out_name = fname + ".nc4" if use_opendap else fname
    fpath = outdir / out_name
    min_size = 10_000 if use_opendap else 5_000_000  # subsets are small
    if fpath.exists() and fpath.stat().st_size > min_size:
        return fpath

    doy = date.strftime("%j")
    if use_opendap:
        url = _opendap_subset_url(date, fname,
                                   bbox["lat_min"], bbox["lat_max"],
                                   bbox["lon_min"], bbox["lon_max"])
    else:
        url = f"{GESDISC_BASE}/{date.year}/{doy}/{fname}"

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait = 5 * attempt
            log.info("  Retry %d/%d for %s (waiting %ds)...", attempt, max_retries, fname, wait)
            time.sleep(wait)

        # Remove partial file from previous attempt
        if fpath.exists():
            fpath.unlink()

        log.info("  Downloading: %s (attempt %d)", fname, attempt)
        try:
            r = session.get(url, stream=True, timeout=180, allow_redirects=True)

            if r.status_code == 401:
                raise RuntimeError(
                    f"HTTP 401 downloading {fname}. "
                    "Approve GES DISC app: "
                    "https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg"
                )
            if r.status_code != 200:
                body_preview = r.content[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"HTTP {r.status_code} downloading {fname}.\n"
                    f"Response preview: {body_preview}"
                )

            ctype = r.headers.get("Content-Type", "")
            if "html" in ctype.lower():
                body = r.content[:1000].decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"GES DISC returned HTML (Content-Type: {ctype}). "
                    f"Approve app at: https://urs.earthdata.nasa.gov/approve_app?client_id=ijpRZvb9qeKCK5ctsn75Tg\n"
                    f"Preview: {body[:300]}"
                )

            with open(fpath, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            log.warning("  Connection error on attempt %d: %s", attempt, e)
            if attempt == max_retries:
                if fpath.exists():
                    fpath.unlink()
                raise RuntimeError(
                    f"Failed to download {fname} after {max_retries} attempts: {e}"
                )
            continue  # retry

        # Validate downloaded file
        size_mb = fpath.stat().st_size / 1e6
        log.info("  Written: %.2f MB", size_mb)

        # OPeNDAP bbox subsets are legitimately small (~20-50 KB)
        size_threshold_mb = 0.005 if use_opendap else 0.5
        if size_mb < size_threshold_mb:
            content = fpath.read_bytes()
            fpath.unlink()
            if attempt == max_retries:
                raise RuntimeError(
                    f"File too small ({size_mb:.3f} MB) after {max_retries} attempts. "
                    f"First 200 bytes: {content[:200]}"
                )
            log.warning("  File too small (%.3f MB), retrying...", size_mb)
            continue  # retry

        # Verify file type
        header = fpath.read_bytes()[:8]
        if use_opendap:
            # NetCDF4 magic: starts with \x89HDF (same as HDF5) or CDF\x01/CDF\x02
            valid = header[:4] in (b"\x89HDF", b"CDF\x01", b"CDF\x02")
        else:
            valid = header == b"\x89HDF\r\n\x1a\n"
        if not valid:
            fpath.unlink()
            raise RuntimeError(
                f"Invalid file (bad header: {header!r}). "
                "Likely an HTML error page."
            )

        log.info("  ✓ %s (%.1f KB)", out_name, fpath.stat().st_size / 1e3)
        return fpath  # success

    # Should not reach here
    raise RuntimeError(f"Download failed for {fname} after {max_retries} attempts")


# ---------------------------------------------------------------------------
# GPM IMERG main download function
# ---------------------------------------------------------------------------

def download_gpm(cfg, start_override=None, end_override=None,
                 test_one=False) -> bool:
    """
    Download GPM IMERG Final Run V07 half-hourly data via OPeNDAP spatial subset.

    Uses OPeNDAP to download ONLY the Uttarakhand bbox (~50-200 KB per file)
    instead of full global HDF5 files (~8.5 MB each). This reduces storage
    from ~600 GB (5 years full) to ~3-5 GB (3 monsoon seasons subsetted).

    Default scope: 3 monsoon seasons (2021-2023, Jun-Sep) — sufficient for ML.
    Override with --start / --end for custom ranges.

    Files saved as: data/raw/gpm/YYYY/DOY/<filename>.HDF5.nc4
    """
    user = os.environ.get("EARTHDATA_USERNAME", "").strip()
    pwd  = os.environ.get("EARTHDATA_PASSWORD", "").strip()
    if not user or not pwd:
        log.error(
            "EARTHDATA_USERNAME or EARTHDATA_PASSWORD not set.\n"
            "Register at https://urs.earthdata.nasa.gov/"
        )
        return False

    bbox = cfg["region"]["bbox"]

    # Default: 3 monsoon seasons 2021-2023 (Jun-Sep only)
    if start_override:
        start_dt = datetime.date.fromisoformat(start_override)
        end_dt   = datetime.date.fromisoformat(end_override or start_override)
        date_ranges = [(start_dt, end_dt)]
    else:
        monsoon_months = set(cfg["time"]["monsoon_months"])
        date_ranges = []
        for year in [2021, 2022, 2023]:
            for month in sorted(monsoon_months):
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                date_ranges.append((
                    datetime.date(year, month, 1),
                    datetime.date(year, month, last_day),
                ))
        log.info("GPM: Default scope = 3 monsoon seasons (2021-2023, Jun-Sep)")
        log.info("GPM: Using OPeNDAP spatial subset — bbox only, ~50-200 KB per file")

    outdir = ROOT / cfg["paths"]["raw_gpm"]
    outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: obtain Bearer token
    log.info("GPM: Obtaining Earthdata Bearer token for user '%s'...", user)
    try:
        token = _get_bearer_token(user, pwd)
    except RuntimeError as e:
        log.error("GPM: Token fetch failed: %s", e)
        return False

    session = _build_session(token)

    # Step 2: verify listing for first date range start
    first_start = date_ranges[0][0]
    log.info("GPM: Verifying directory listing for %s...", first_start)
    try:
        files = _list_day_files(session, first_start)
    except RuntimeError as e:
        log.error("GPM: Listing failed: %s", e)
        return False
    if not files:
        log.error("GPM: No HDF5 files listed for %s.", first_start)
        return False
    log.info("GPM: Access verified. Found %d files for %s.", len(files), first_start)

    # Step 3: test-one mode
    if test_one:
        fname   = files[0]
        day_dir = outdir / str(first_start.year) / first_start.strftime("%j")
        day_dir.mkdir(parents=True, exist_ok=True)
        try:
            fpath = _download_one_file(session, first_start, fname, day_dir, bbox=bbox)
            log.info("GPM TEST-ONE: SUCCESS — %s (%.1f KB)",
                     fpath.name, fpath.stat().st_size / 1e3)
            return True
        except RuntimeError as e:
            log.error("GPM TEST-ONE: FAILED — %s", e)
            return False

    # Step 4: download all date ranges
    total_files = 0
    total_bytes = 0

    for start_dt, end_dt in date_ranges:
        current = start_dt
        while current <= end_dt:
            day_dir = outdir / str(current.year) / current.strftime("%j")
            day_dir.mkdir(parents=True, exist_ok=True)

            try:
                day_files = _list_day_files(session, current)
            except RuntimeError as e:
                log.error("GPM: Listing failed for %s: %s", current, e)
                current += datetime.timedelta(days=1)
                continue

            if not day_files:
                current += datetime.timedelta(days=1)
                continue

            day_bytes = 0
            for fname in day_files:
                try:
                    fpath = _download_one_file(session, current, fname,
                                               day_dir, bbox=bbox)
                    day_bytes   += fpath.stat().st_size
                    total_files += 1
                    total_bytes += fpath.stat().st_size
                except RuntimeError as e:
                    log.error("GPM: %s", e)
                    return False
                time.sleep(0.05)

            log.info("GPM: %s — %d files, %.0f KB (total: %d files, %.1f MB)",
                     current, len(day_files), day_bytes / 1e3,
                     total_files, total_bytes / 1e6)
            current += datetime.timedelta(days=1)

    log.info("GPM: Download complete — %d files, %.1f MB total", total_files, total_bytes / 1e6)
    return True


def _verify_hdf5_content(fpath: Path):
    """
    Open the file with h5py and print key metadata to confirm genuine GPM IMERG V07B data.

    V07B structure (changed from V06):
      Dataset name : /Grid/precipitation  (V06 used precipitationCal)
      Axis order   : (time, lat, lon)     (V06 used (time, lon, lat))
    """
    try:
        import h5py
    except ImportError:
        log.warning("h5py not installed — skipping deep HDF5 verification. Run: pip install h5py")
        return

    with h5py.File(fpath, "r") as hf:
        grp   = hf["Grid"]
        lats  = grp["lat"][:]
        lons  = grp["lon"][:]
        t_sec = int(grp["time"][0])

        # V07B uses "precipitation"; V06 used "precipitationCal" — try both
        if "precipitation" in grp:
            precip = grp["precipitation"][:]
            ds_name = "precipitation"
        elif "precipitationCal" in grp:
            precip = grp["precipitationCal"][:]
            ds_name = "precipitationCal"
        else:
            available = list(grp.keys())
            log.warning("  Neither precipitation nor precipitationCal found. Keys: %s", available)
            return

    import datetime as dt_mod
    GPS_EPOCH = dt_mod.datetime(1980, 1, 6, 0, 0, 0)
    ts = GPS_EPOCH + dt_mod.timedelta(seconds=t_sec)
    valid = precip[precip > -9000]
    log.info("  HDF5 content verification:")
    log.info("    Timestamp   : %s UTC", ts)
    log.info("    Dataset     : %s", ds_name)
    log.info("    Global lat  : %.1f to %.1f  (%d pts)", lats.min(), lats.max(), len(lats))
    log.info("    Global lon  : %.1f to %.1f  (%d pts)", lons.min(), lons.max(), len(lons))
    log.info("    Precip shape: %s  dtype=%s", precip.shape, precip.dtype)
    log.info("    Precip range: %.3f to %.3f mm/hr (valid pixels: %d)",
             float(valid.min()) if valid.size else 0,
             float(valid.max()) if valid.size else 0,
             valid.size)
    log.info("  ✓ Genuine GPM IMERG V07B HDF5 confirmed")


# ---------------------------------------------------------------------------
# ERA5, SRTM, Events — unchanged
# ---------------------------------------------------------------------------

def download_era5(cfg, start_year_override=None, end_year_override=None) -> bool:
    """
    Download ERA5-Land and ERA5 single-level reanalysis data.

    Requests are split by year to stay within CDS size limits.
    Files saved to: data/raw/era5/
      era5land_<YYYY>.nc   -- total_precipitation, soil_moisture
      era5_<YYYY>.nc       -- temperature, dewpoint, pressure, wind
    """
    try:
        import cdsapi
    except ImportError:
        log.warning("ERA5: cdsapi not installed. Run: pip install cdsapi")
        return False
    cdsrc = Path.home() / ".cdsapirc"
    if not cdsrc.exists():
        log.warning("ERA5: ~/.cdsapirc not found. Register at https://cds.climate.copernicus.eu/")
        return False

    bbox   = cfg["region"]["bbox"]
    area   = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
    months = [str(m).zfill(2) for m in cfg["time"]["monsoon_months"]]
    outdir = ROOT / cfg["paths"]["raw_era5"]
    outdir.mkdir(parents=True, exist_ok=True)

    start_year = start_year_override or int(cfg["time"]["start_date"][:4])
    end_year   = end_year_override   or int(cfg["time"]["end_date"][:4])
    # Default floor: 2018 (earliest practical ERA5 for this project)
    start_year = max(int(start_year), 2018)

    c = cdsapi.Client()

    datasets = [
        ("era5land",   "reanalysis-era5-land",
         ["total_precipitation", "volumetric_soil_water_layer_1"], {}),
        ("era5",       "reanalysis-era5-single-levels",
         ["2m_temperature", "2m_dewpoint_temperature",
          "surface_pressure", "10m_u_component_of_wind",
          "10m_v_component_of_wind"],
         {"product_type": "reanalysis"}),
    ]

    for prefix, dataset, variables, extra in datasets:
        for year in range(start_year, end_year + 1):
            out = outdir / f"{prefix}_{year}.nc"
            if out.exists():
                log.info("ERA5: %s already exists, skipping.", out.name)
                continue
            log.info("ERA5: Requesting %s for year %d...", prefix, year)
            req = {
                "variable": variables,
                "year":     [str(year)],
                "month":    months,
                "day":      [str(d).zfill(2) for d in range(1, 32)],
                "time":     [f"{h:02d}:00" for h in range(0, 24, 3)],  # 3-hourly
                "area":     area,
                "format":   "netcdf",
                **extra,
            }
            try:
                c.retrieve(dataset, req, str(out))
                log.info("ERA5: Saved %s (%.0f MB)", out.name, out.stat().st_size / 1e6)
            except Exception as e:
                log.error("ERA5: Failed for %s year %d: %s", prefix, year, e)
                if out.exists():
                    out.unlink()
                return False

    return True


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
        out = outdir / "srtm_uttarakhand_30m.tif"
        if not out.exists():
            r = requests.get(url, stream=True, timeout=300)
            if r.status_code == 200:
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                log.info("SRTM GL1: %.1f MB", out.stat().st_size / 1e6)
            else:
                log.error("SRTM GL1: HTTP %d", r.status_code)
                return False
        return True
    def tile_num(lon, lat):
        return math.floor((lon + 180) / 5) + 1, math.floor((60 - lat) / 5) + 1
    corners = [(bbox["lon_min"], bbox["lat_max"]), (bbox["lon_max"], bbox["lat_max"]),
               (bbox["lon_min"], bbox["lat_min"]), (bbox["lon_max"], bbox["lat_min"])]
    for tx, ty in set(tile_num(lon, lat) for lon, lat in corners):
        fname  = f"srtm_{tx:02d}_{ty:02d}.zip"
        out    = outdir / fname
        if out.exists():
            continue
        url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/{fname}"
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 200:
            with open(out, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
            log.info("SRTM 90m: %s %.1f MB", fname, out.stat().st_size / 1e6)
        else:
            log.error("SRTM 90m: HTTP %d for %s", r.status_code, url)
    return True


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
    parser.add_argument("--test-one", action="store_true",
                        help="Download only the first file of start_date to verify auth+download")
    args = parser.parse_args()
    cfg = load_config()

    results = {}
    if args.source in ("all", "gpm"):
        results["gpm"] = download_gpm(cfg, args.start, args.end,
                                       test_one=args.test_one)
    if args.source in ("all", "era5"):
        sy = int(args.start[:4]) if args.start else None
        ey = int(args.end[:4])   if args.end   else None
        results["era5"] = download_era5(cfg, sy, ey)
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
