import argparse
import calendar
import csv
import gzip
import io
import math
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
META = PROJECT / "data" / "external" / "source_manifests"
RESULTS = PROJECT / "results" / "data_downloads"

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_STATION_URLS = [
    "https://mesonet.agron.iastate.edu/sites/networks.php?special=allasos&format=csv&nohtml",
    "https://www.mesonet.agron.iastate.edu/sites/networks.php?special=allasos&format=csv&nohtml",
]
NOAA_STORM_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
FAA_LIST_URL = "https://www.fly.faa.gov/adv/adv_list"

BASE_AIRPORTS = ["ATL", "CLT", "DEN", "DFW", "EWR", "JFK", "LAX", "LGA", "ORD", "SFO"]
EXTENDED_AIRPORTS = [
    "BOS",
    "BWI",
    "DCA",
    "DTW",
    "FLL",
    "HNL",
    "IAD",
    "IAH",
    "LAS",
    "MCO",
    "MDW",
    "MIA",
    "MSP",
    "PHL",
    "PHX",
    "RDU",
    "SAN",
    "SEA",
    "SLC",
    "TPA",
]
IEM_STATION_OVERRIDES = {
    "HNL": "PHNL",
}


def ensure_dirs() -> None:
    for rel in [
        "bts_on_time",
        "iem_asos",
        "faa_atcscc_advisories",
        "noaa_storm_events",
        "iem_station_metadata",
        "iem_asos_neighbors",
    ]:
        (RAW / rel).mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def request_get(
    url: str,
    *,
    params=None,
    timeout: int = 90,
    stream: bool = False,
    retries: int = 4,
    headers: dict | None = None,
) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            request_headers = {"User-Agent": "open-aviation-fusion-research/0.2"}
            if headers:
                request_headers.update(headers)
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                stream=stream,
                headers=request_headers,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"{response.status_code} retryable response", response=response)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(8 * attempt)
    raise RuntimeError(last_error)


def download_file(url: str, path: Path, *, timeout: int = 300, retries: int = 4, resume: bool = True) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return {"artifact": str(path), "url": url, "bytes": path.stat().st_size, "status": "cached"}
    tmp = path.with_suffix(path.suffix + ".part")
    headers = {}
    mode = "wb"
    existing = tmp.stat().st_size if tmp.exists() else 0
    if resume and existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    elif tmp.exists():
        tmp.unlink()
    with request_get(url, timeout=timeout, stream=True, retries=retries, headers=headers) as response:
        if existing > 0 and response.status_code != 206:
            tmp.unlink(missing_ok=True)
            mode = "wb"
        with tmp.open(mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    return {"artifact": str(path), "url": response.url, "bytes": path.stat().st_size, "status": "downloaded"}


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_months(year: int) -> list[tuple[int, int]]:
    return [(year, month) for month in range(1, 13)]


def parse_month_numbers(text: str | None) -> list[int]:
    if not text:
        return list(range(1, 13))
    months = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            months.extend(range(int(start), int(end) + 1))
        else:
            months.append(int(part))
    return sorted({m for m in months if 1 <= m <= 12})


def iter_month_dates(year: int, month: int):
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        yield date(year, month, day)


def next_month(year: int, month: int) -> tuple[int, int, int]:
    if month == 12:
        return year + 1, 1, 1
    return year, month + 1, 1


def download_bts_month(year: int, month: int) -> dict:
    url = BTS_URL.format(year=year, month=month)
    path = RAW / "bts_on_time" / f"bts_on_time_{year}_{month:02d}.zip"
    row = download_file(url, path, timeout=300)
    row.update({"dataset": "BTS On-Time Performance", "year": year, "month": month})
    return row


def iem_params(station: str, year: int, month: int) -> list[tuple[str, str]]:
    y2, m2, d2 = next_month(year, month)
    return [
        ("station", station),
        ("data", "tmpf"),
        ("data", "dwpf"),
        ("data", "sknt"),
        ("data", "drct"),
        ("data", "vsby"),
        ("data", "skyc1"),
        ("data", "skyl1"),
        ("year1", str(year)),
        ("month1", str(month)),
        ("day1", "1"),
        ("year2", str(y2)),
        ("month2", str(m2)),
        ("day2", str(d2)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "yes"),
        ("report_type", "3"),
    ]


def download_iem_airport_month(airport: str, year: int, month: int, *, station: str | None = None, out_dir: str = "iem_asos") -> dict:
    station = station or IEM_STATION_OVERRIDES.get(airport, airport)
    path = RAW / out_dir / f"iem_asos_{year}_{month:02d}_{airport}.csv"
    if out_dir == "iem_asos_neighbors":
        path = RAW / out_dir / f"iem_asos_{year}_{month:02d}_{airport}_{station}.csv"
    if path.exists() and path.stat().st_size > 1024:
        return {
            "dataset": "IEM ASOS",
            "airport": airport,
            "station": station,
            "year": year,
            "month": month,
            "artifact": str(path),
            "bytes": path.stat().st_size,
            "status": "cached",
            "url": IEM_ASOS_URL,
        }
    response_url = IEM_ASOS_URL
    tmp = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    last_error = ""
    for attempt in range(1, 6):
        try:
            with request_get(IEM_ASOS_URL, params=iem_params(station, year, month), timeout=240, stream=True, retries=1) as response:
                response_url = response.url
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if tmp.exists():
                tmp.unlink()
            if attempt == 5:
                return {
                    "dataset": "IEM ASOS",
                    "airport": airport,
                    "station": station,
                    "year": year,
                    "month": month,
                    "artifact": str(path),
                    "bytes": 0,
                    "status": f"failed: {last_error}",
                    "url": response_url,
                }
            time.sleep(20 * attempt)
    tmp.replace(path)
    return {
        "dataset": "IEM ASOS",
        "airport": airport,
        "station": station,
        "year": year,
        "month": month,
        "artifact": str(path),
        "bytes": path.stat().st_size,
        "status": "downloaded",
        "url": response_url,
    }


def faa_params(d: date) -> dict:
    return {
        "whichAdvisories": "ATCSCC",
        "advisoryCategory": "NotAll",
        "date": d.isoformat(),
        "gStop": "true",
        "_gStop": "on",
        "gDelay": "true",
        "_gDelay": "on",
        "_airflow": "on",
        "_ctop": "on",
        "_route": "on",
        "_other": "on",
    }


def parse_period(base: date, token: str, default_day: int | None = None) -> datetime | None:
    token = str(token).strip().upper().replace("Z", "")
    m = re.match(r"\s*(?:(\d{2})/)?(\d{4})\s*", token)
    if not m:
        return None
    day = int(m.group(1)) if m.group(1) else (default_day or base.day)
    hhmm = m.group(2)
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    candidates = [
        (base.year, base.month),
        (base.year + (1 if base.month == 12 else 0), 1 if base.month == 12 else base.month + 1),
        (base.year - (1 if base.month == 1 else 0), 12 if base.month == 1 else base.month - 1),
    ]
    for year, month in candidates:
        try:
            return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("UTC"))
        except ValueError:
            continue
    return None


def find_period(upper: str, labels: list[str], adv_date: date) -> tuple[datetime, datetime] | None:
    time_token = r"(?:\d{2}/)?\d{4}Z?"
    for label in labels:
        pattern = rf"{label}:\s*({time_token})\s*-\s*({time_token})"
        match = re.search(pattern, upper)
        if not match:
            continue
        start_utc = parse_period(adv_date, match.group(1))
        if start_utc is None:
            continue
        end_utc = parse_period(adv_date, match.group(2), default_day=start_utc.day)
        if end_utc is None:
            continue
        if end_utc <= start_utc:
            end_utc += timedelta(days=1)
        return start_utc, end_utc
    return None


def parse_advisory(text: str, adv_date: date, source_url: str) -> dict | None:
    clean = re.sub(r"[ \t]+", " ", text)
    upper = clean.upper()
    title_match = re.search(
        r"(ATCSCC\s+ADVZY\s+\d+\s+[A-Z0-9/]+\s+\d{2}/\d{2}/\d{4}\s+.*?)(?:\s+MESSAGE:|\s+CTL ELEMENT:|\s+EVENT TIME:)",
        upper,
    )
    title = title_match.group(1) if title_match else upper[:300]
    if re.search(r"\b(CNX|CANCEL|CANCELED|CANCELLED|CANCELLATION|PROPOSED)\b", title):
        return None
    if "GROUND DELAY PROGRAM" in title:
        tmi_type = "GDP"
        period = find_period(
            upper,
            ["CUMULATIVE PROGRAM PERIOD", "ARRIVALS ESTIMATED FOR", "GDP PERIOD", "EVENT TIME"],
            adv_date,
        )
    elif "GROUND STOP" in title:
        tmi_type = "GS"
        period = find_period(upper, ["GROUND STOP PERIOD", "EVENT TIME"], adv_date)
    else:
        return None
    airport_match = re.search(r"CTL ELEMENT:\s*([A-Z0-9]{3,4})", upper)
    title_airport_match = re.search(r"ADVZY\s+\d+\s+([A-Z0-9]{3,4})/", title)
    elem_match = re.search(r"ELEMENT TYPE:\s*([A-Z]+)", upper)
    airport = airport_match.group(1) if airport_match else (title_airport_match.group(1) if title_airport_match else "")
    if not airport or period is None:
        return None
    if elem_match and elem_match.group(1) != "APT":
        return None
    start_utc, end_utc = period
    duration_min = (end_utc - start_utc).total_seconds() / 60.0
    if duration_min <= 0 or duration_min > 48 * 60:
        return None
    reason = ""
    reason_match = re.search(r"(?:REASON|IMPACTING CONDITION):\s*([A-Z0-9 /._-]+)", upper)
    if reason_match:
        reason = reason_match.group(1).strip()
    return {
        "advisory_date": adv_date.isoformat(),
        "airport": airport,
        "tmi_type": tmi_type,
        "start_utc": start_utc.isoformat(),
        "end_utc": end_utc.isoformat(),
        "duration_min": round(duration_min, 2),
        "reason": reason,
        "source_url": source_url,
    }


def fetch_faa_day(d: date, *, force: bool = False) -> tuple[dict, list[dict], list[dict]]:
    day_dir = RAW / "faa_atcscc_advisories" / d.strftime("%Y_%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = day_dir / f"parsed_{d.strftime('%Y%m%d')}.csv"
    error_path = day_dir / f"errors_{d.strftime('%Y%m%d')}.csv"
    if parsed_path.exists() and not force:
        parsed = pd.read_csv(parsed_path).to_dict("records") if parsed_path.stat().st_size > 0 else []
        errors = pd.read_csv(error_path).to_dict("records") if error_path.exists() and error_path.stat().st_size > 0 else []
        manifest = {
            "dataset": "FAA ATCSCC advisories",
            "date": d.isoformat(),
            "artifact": str(parsed_path),
            "bytes": parsed_path.stat().st_size,
            "records": len(parsed),
            "status": "cached",
            "url": FAA_LIST_URL,
        }
        return manifest, parsed, errors

    list_path = day_dir / f"list_{d.isoformat()}.html"
    try:
        if list_path.exists() and not force:
            html = list_path.read_text(encoding="iso-8859-1", errors="ignore")
        else:
            response = request_get(FAA_LIST_URL, params=faa_params(d), timeout=75)
            html = response.text
            list_path.write_text(html, encoding="iso-8859-1", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        links = []
        seen = set()
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            if "adv_otherdis" not in href:
                continue
            detail_url = "https://www.fly.faa.gov" + href if href.startswith("/") else href
            if detail_url in seen:
                continue
            seen.add(detail_url)
            advn = href.split("advn=", 1)[1].split("&", 1)[0] if "advn=" in href else str(len(links) + 1)
            links.append((advn, detail_url))

        rows = []
        errors = []
        for advn, detail_url in links:
            detail_path = day_dir / f"detail_{d.strftime('%Y%m%d')}_{advn}.html"
            try:
                if detail_path.exists() and not force:
                    detail_html = detail_path.read_text(encoding="iso-8859-1", errors="ignore")
                else:
                    detail_response = request_get(detail_url, timeout=75)
                    detail_html = detail_response.text
                    detail_path.write_text(detail_html, encoding="iso-8859-1", errors="ignore")
                detail_text = BeautifulSoup(detail_html, "html.parser").get_text("\n", strip=True)
                parsed = parse_advisory(detail_text, d, detail_url)
                if parsed:
                    rows.append(parsed)
            except Exception as exc:
                errors.append(
                    {
                        "date": d.isoformat(),
                        "advn": advn,
                        "detail_url": detail_url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:
        rows = []
        errors = [{"date": d.isoformat(), "advn": "", "detail_url": "", "error": f"{type(exc).__name__}: {exc}"}]

    pd.DataFrame(rows, columns=["advisory_date", "airport", "duration_min", "end_utc", "reason", "source_url", "start_utc", "tmi_type"]).to_csv(parsed_path, index=False)
    pd.DataFrame(errors, columns=["date", "advn", "detail_url", "error"]).to_csv(error_path, index=False)
    manifest = {
        "dataset": "FAA ATCSCC advisories",
        "date": d.isoformat(),
        "artifact": str(parsed_path),
        "bytes": parsed_path.stat().st_size if parsed_path.exists() else 0,
        "records": len(rows),
        "status": "downloaded",
        "url": FAA_LIST_URL,
    }
    return manifest, rows, errors


def combine_faa_year(year: int, rows: list[dict] | None = None) -> tuple[Path, Path]:
    if rows is None:
        frames = []
        for month in range(1, 13):
            day_dir = RAW / "faa_atcscc_advisories" / f"{year}_{month:02d}"
            for d in iter_month_dates(year, month):
                path = day_dir / f"parsed_{d.strftime('%Y%m%d')}.csv"
                if path.exists() and path.stat().st_size > 0:
                    frames.append(pd.read_csv(path))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"])
        df["start_utc_dt"] = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
        df["end_utc_dt"] = pd.to_datetime(df["end_utc"], utc=True, errors="coerce")
        duration = (df["end_utc_dt"] - df["start_utc_dt"]).dt.total_seconds() / 60.0
        df = df[(duration > 0) & (duration <= 48 * 60)].copy()
        df = df.drop(columns=["start_utc_dt", "end_utc_dt"])
    full_path = RAW / "faa_atcscc_advisories" / f"faa_atcscc_gdp_gs_reparsed_{year}_v2.csv"
    df.to_csv(full_path, index=False)

    error_frames = []
    for month in range(1, 13):
        day_dir = RAW / "faa_atcscc_advisories" / f"{year}_{month:02d}"
        for d in iter_month_dates(year, month):
            path = day_dir / f"errors_{d.strftime('%Y%m%d')}.csv"
            if path.exists() and path.stat().st_size > 0:
                err = pd.read_csv(path)
                if not err.empty:
                    error_frames.append(err)
    err_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame(columns=["date", "advn", "detail_url", "error"])
    error_path = RAW / "faa_atcscc_advisories" / f"faa_atcscc_errors_reparsed_{year}_v2.csv"
    err_df.to_csv(error_path, index=False)
    return full_path, error_path


def noaa_latest_file(year: int, kind: str) -> str:
    html = request_get(NOAA_STORM_DIR, timeout=90).text
    pattern = rf"StormEvents_{kind}-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    names = sorted(set(re.findall(pattern, html)))
    if not names:
        raise RuntimeError(f"No NOAA Storm Events {kind} file found for {year}")
    return names[-1]


def download_noaa_storm(year: int, kind: str) -> dict:
    name = noaa_latest_file(year, kind)
    url = NOAA_STORM_DIR + name
    target_kind = "details" if kind == "details" else "locations"
    path = RAW / "noaa_storm_events" / f"StormEvents_{target_kind}_{year}.csv.gz"
    row = download_file(url, path, timeout=300)
    row.update({"dataset": "NOAA Storm Events", "year": year, "kind": target_kind, "source_file": name})
    return row


def download_station_metadata() -> dict:
    path = RAW / "iem_station_metadata" / "iem_stations_all.csv"
    if path.exists() and path.stat().st_size > 1024:
        with path.open("rb") as f:
            head = f.read(64).lstrip()
        if not head.startswith(b"<"):
            return {"dataset": "IEM station metadata", "artifact": str(path), "bytes": path.stat().st_size, "status": "cached", "url": IEM_STATION_URLS[0]}
        path.unlink()
    last_error = ""
    for url in IEM_STATION_URLS:
        try:
            row = download_file(url, path, timeout=120)
            row.update({"dataset": "IEM station metadata"})
            return row
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return {"dataset": "IEM station metadata", "artifact": str(path), "bytes": 0, "status": f"failed: {last_error}", "url": IEM_STATION_URLS[0]}


def read_airport_locations(airports: list[str]) -> pd.DataFrame:
    path = RAW / "ourairports" / "airports.csv"
    df = pd.read_csv(path, low_memory=False)
    use = df[df["iata_code"].isin(airports)].copy()
    return use[["iata_code", "latitude_deg", "longitude_deg"]].rename(columns={"iata_code": "airport", "latitude_deg": "lat", "longitude_deg": "lon"})


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def select_neighbor_stations() -> pd.DataFrame:
    meta_path = RAW / "iem_station_metadata" / "iem_stations_all.csv"
    stations = pd.read_csv(meta_path, low_memory=False)
    lower_map = {c.lower(): c for c in stations.columns}
    id_col = lower_map.get("stid") or lower_map.get("id") or lower_map.get("station")
    lat_col = lower_map.get("lat") or lower_map.get("latitude")
    lon_col = lower_map.get("lon") or lower_map.get("longitude")
    network_col = lower_map.get("network")
    if not id_col or not lat_col or not lon_col:
        raise RuntimeError(f"Unexpected IEM station metadata columns: {stations.columns.tolist()}")
    stations = stations[[id_col, lat_col, lon_col] + ([network_col] if network_col else [])].copy()
    stations = stations.rename(columns={id_col: "station", lat_col: "lat", lon_col: "lon", network_col: "network" if network_col else "network"})
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    stations = stations.dropna(subset=["station", "lat", "lon"])
    if "network" in stations.columns:
        keep = stations["network"].astype(str).str.contains("ASOS|AWOS", case=False, na=False)
        if keep.any():
            stations = stations[keep].copy()
    airports = read_airport_locations(BASE_AIRPORTS)
    rows = []
    for ap in airports.itertuples(index=False):
        candidates = []
        for st in stations.itertuples(index=False):
            station_id = str(st.station).strip()
            if station_id.upper() == ap.airport:
                continue
            dist = haversine_km(ap.lat, ap.lon, st.lat, st.lon)
            if dist <= 80:
                candidates.append((station_id, dist, st.lat, st.lon, getattr(st, "network", "")))
        for rank, (station, dist, lat, lon, network) in enumerate(sorted(candidates, key=lambda x: x[1])[:3], start=1):
            rows.append(
                {
                    "airport": ap.airport,
                    "neighbor_rank": rank,
                    "station": station,
                    "distance_km": round(dist, 3),
                    "lat": lat,
                    "lon": lon,
                    "network": network,
                }
            )
    out = RAW / "iem_station_metadata" / "iem_neighbor_stations_2025_base10.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def bts_readability(path: Path) -> tuple[bool, str]:
    needed = {"FlightDate", "Origin", "Dest", "CRSDepTime", "CRSArrTime", "ArrDelayMinutes", "Cancelled", "Diverted"}
    try:
        with zipfile.ZipFile(path) as zf:
            csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            with zf.open(csv_name) as f:
                cols = set(pd.read_csv(f, nrows=0).columns)
        missing = sorted(needed - cols)
        return not missing, "" if not missing else f"missing {missing}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def csv_columns(path: Path, compression: str | None = None) -> set[str]:
    return set(pd.read_csv(path, nrows=0, compression=compression).columns)


def audit_downloads() -> pd.DataFrame:
    rows = []
    bts_files = [RAW / "bts_on_time" / f"bts_on_time_2024_{m:02d}.zip" for m in range(1, 13)]
    bts_ok = 0
    bts_issues = []
    for path in bts_files:
        ok, issue = bts_readability(path)
        bts_ok += int(path.exists() and ok)
        if not path.exists():
            bts_issues.append(path.name)
        elif not ok:
            bts_issues.append(f"{path.name}: {issue}")
    rows.append({"dataset": "2024 BTS", "expected": 12, "available": bts_ok, "issues": "; ".join(bts_issues)})

    iem_2024 = [RAW / "iem_asos" / f"iem_asos_2024_{m:02d}_{ap}.csv" for ap in BASE_AIRPORTS for m in range(1, 13)]
    iem_cols = {"valid", "tmpf", "sknt", "drct", "vsby", "skyc1", "skyl1"}
    iem_ok = 0
    iem_issues = []
    for path in iem_2024:
        if path.exists() and path.stat().st_size > 1024:
            cols = csv_columns(path)
            missing = sorted(iem_cols - cols)
            if not missing:
                iem_ok += 1
            else:
                iem_issues.append(f"{path.name}: missing {missing}")
        else:
            iem_issues.append(path.name)
    rows.append({"dataset": "2024 IEM ASOS base10", "expected": 120, "available": iem_ok, "issues": "; ".join(iem_issues[:20])})

    faa_days = []
    for month in range(1, 13):
        for d in iter_month_dates(2024, month):
            faa_days.append(RAW / "faa_atcscc_advisories" / f"2024_{month:02d}" / f"parsed_{d.strftime('%Y%m%d')}.csv")
    faa_available = sum(1 for p in faa_days if p.exists())
    faa_full = RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_reparsed_2024_v2.csv"
    faa_issue = ""
    if faa_full.exists():
        cols = csv_columns(faa_full)
        missing = sorted({"airport", "tmi_type", "start_utc", "end_utc"} - cols)
        faa_issue = f"combined missing {missing}" if missing else ""
    else:
        faa_issue = "combined missing"
    rows.append({"dataset": "2024 FAA ATCSCC daily parsed", "expected": 366, "available": faa_available, "issues": faa_issue})

    iem_2025_ext = [RAW / "iem_asos" / f"iem_asos_2025_{m:02d}_{ap}.csv" for ap in EXTENDED_AIRPORTS for m in range(1, 13)]
    ext_ok = 0
    ext_issues = []
    for path in iem_2025_ext:
        if path.exists() and path.stat().st_size > 1024:
            cols = csv_columns(path)
            missing = sorted(iem_cols - cols)
            if not missing:
                ext_ok += 1
            else:
                ext_issues.append(f"{path.name}: missing {missing}")
        else:
            ext_issues.append(path.name)
    rows.append({"dataset": "2025 IEM ASOS extended20", "expected": 240, "available": ext_ok, "issues": "; ".join(ext_issues[:20])})

    noaa_paths = [
        RAW / "noaa_storm_events" / f"StormEvents_{kind}_{year}.csv.gz"
        for year in [2024, 2025]
        for kind in ["details", "locations"]
    ]
    noaa_ok = 0
    noaa_issues = []
    for path in noaa_paths:
        if path.exists() and path.stat().st_size > 1024:
            cols = csv_columns(path, compression="gzip")
            lower = {c.upper() for c in cols}
            if "EVENT_TYPE" in lower or "LOCATION_INDEX" in lower:
                noaa_ok += 1
            else:
                noaa_issues.append(f"{path.name}: unexpected columns")
        else:
            noaa_issues.append(path.name)
    rows.append({"dataset": "NOAA Storm Events 2024-2025", "expected": 4, "available": noaa_ok, "issues": "; ".join(noaa_issues)})

    meta_path = RAW / "iem_station_metadata" / "iem_stations_all.csv"
    rows.append(
        {
            "dataset": "IEM station metadata",
            "expected": 1,
            "available": int(meta_path.exists() and meta_path.stat().st_size > 1024),
            "issues": "" if meta_path.exists() and meta_path.stat().st_size > 1024 else "missing",
        }
    )

    neighbor_index = RAW / "iem_station_metadata" / "iem_neighbor_stations_2025_base10.csv"
    if neighbor_index.exists() and neighbor_index.stat().st_size > 0:
        nidx = pd.read_csv(neighbor_index)
        expected_neighbor = len(nidx) * 12
        neighbor_paths = [
            RAW / "iem_asos_neighbors" / f"iem_asos_2025_{m:02d}_{row.airport}_{row.station}.csv"
            for row in nidx.itertuples(index=False)
            for m in range(1, 13)
        ]
        neighbor_ok = sum(1 for p in neighbor_paths if p.exists() and p.stat().st_size > 1024)
        neighbor_issue = "" if neighbor_ok == expected_neighbor else f"neighbor files expected {expected_neighbor}, got {neighbor_ok}"
    else:
        expected_neighbor = 360
        neighbor_ok = 0
        neighbor_issue = "neighbor index missing"
    rows.append({"dataset": "2025 neighbor IEM ASOS", "expected": expected_neighbor, "available": neighbor_ok, "issues": neighbor_issue})

    audit = pd.DataFrame(rows)
    audit.to_csv(RESULTS / "supplemental_source_data_audit.csv", index=False)
    lines = ["# Supplemental source-data audit", ""]
    for row in audit.itertuples(index=False):
        lines.append(f"- {row.dataset}: {row.available}/{row.expected}. {row.issues}".rstrip())
    (RESULTS / "supplemental_source_data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def download_bts_all(years: list[int], workers: int) -> list[dict]:
    jobs = [(year, month) for year in years for month in range(1, 13)]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_bts_month, year, month): (year, month) for year, month in jobs}
        for fut in as_completed(futures):
            year, month = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"dataset": "BTS On-Time Performance", "year": year, "month": month, "status": f"failed: {type(exc).__name__}: {exc}", "bytes": 0, "artifact": "", "url": BTS_URL.format(year=year, month=month)}
            rows.append(row)
            print(f"BTS {year}-{month:02d}: {row['status']} {row.get('bytes', 0)} bytes", flush=True)
    return rows


def download_iem_batch(jobs: list[tuple[str, int, int, str | None, str]], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_iem_airport_month, airport, year, month, station=station, out_dir=out_dir): (airport, year, month, station, out_dir)
            for airport, year, month, station, out_dir in jobs
        }
        for fut in as_completed(futures):
            airport, year, month, station, out_dir = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"dataset": "IEM ASOS", "airport": airport, "station": station or airport, "year": year, "month": month, "status": f"failed: {type(exc).__name__}: {exc}", "bytes": 0, "artifact": "", "url": IEM_ASOS_URL}
            rows.append(row)
            print(f"IEM {airport}/{station or airport} {year}-{month:02d}: {row['status']} {row.get('bytes', 0)} bytes", flush=True)
    return rows


def download_faa_year(year: int, workers: int) -> list[dict]:
    return download_faa_months(year, list(range(1, 13)), workers)


def download_faa_months(year: int, months: list[int], workers: int) -> list[dict]:
    jobs = [d for month in months for d in iter_month_dates(year, month)]
    manifest_rows = []
    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_faa_day, d): d for d in jobs}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                manifest, rows, errors = fut.result()
            except Exception as exc:
                manifest, rows = (
                    {
                        "dataset": "FAA ATCSCC advisories",
                        "date": d.isoformat(),
                        "status": f"failed: {type(exc).__name__}: {exc}",
                        "bytes": 0,
                        "records": 0,
                        "artifact": "",
                        "url": FAA_LIST_URL,
                    },
                    [],
                )
            manifest_rows.append(manifest)
            all_rows.extend(rows)
            print(f"FAA {d.isoformat()}: {manifest['status']} parsed={len(rows)}", flush=True)
    full_path, error_path = combine_faa_year(year)
    manifest_rows.append(
        {
            "dataset": "FAA ATCSCC advisories combined",
            "year": year,
            "artifact": str(full_path),
            "bytes": full_path.stat().st_size if full_path.exists() else 0,
            "records": len(pd.read_csv(full_path)) if full_path.exists() and full_path.stat().st_size > 0 else 0,
            "status": "parsed",
            "url": FAA_LIST_URL,
        }
    )
    manifest_rows.append(
        {
            "dataset": "FAA ATCSCC advisory parse errors",
            "year": year,
            "artifact": str(error_path),
            "bytes": error_path.stat().st_size if error_path.exists() else 0,
            "records": len(pd.read_csv(error_path)) if error_path.exists() and error_path.stat().st_size > 0 else 0,
            "status": "parsed",
            "url": FAA_LIST_URL,
        }
    )
    return manifest_rows


def download_noaa_all(years: list[int]) -> list[dict]:
    rows = []
    for year in years:
        for kind in ["details", "locations"]:
            try:
                row = download_noaa_storm(year, kind)
            except Exception as exc:
                row = {"dataset": "NOAA Storm Events", "year": year, "kind": kind, "status": f"failed: {type(exc).__name__}: {exc}", "bytes": 0, "artifact": "", "url": NOAA_STORM_DIR}
            rows.append(row)
            print(f"NOAA Storm Events {year} {kind}: {row['status']} {row.get('bytes', 0)} bytes", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "audit"], default="full")
    parser.add_argument("--bts-workers", type=int, default=4)
    parser.add_argument("--iem-workers", type=int, default=4)
    parser.add_argument("--faa-workers", type=int, default=8)
    parser.add_argument("--months", default="", help="Month numbers to process, e.g. 2 or 2,3 or 2-4. Applies to full mode.")
    args = parser.parse_args()

    ensure_dirs()
    all_manifest_rows = []
    months_to_process = parse_month_numbers(args.months)

    if args.mode in {"smoke", "full"}:
        if args.mode == "smoke":
            bts_years = [2024]
            bts_jobs = [(2024, 1)]
            bts_rows = []
            for year, month in bts_jobs:
                bts_rows.append(download_bts_month(year, month))
                print(f"BTS {year}-{month:02d}: {bts_rows[-1]['status']} {bts_rows[-1].get('bytes', 0)} bytes", flush=True)
        else:
            bts_rows = []
            with ThreadPoolExecutor(max_workers=args.bts_workers) as ex:
                futures = {ex.submit(download_bts_month, 2024, month): month for month in months_to_process}
                for fut in as_completed(futures):
                    month = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"dataset": "BTS On-Time Performance", "year": 2024, "month": month, "status": f"failed: {type(exc).__name__}: {exc}", "bytes": 0, "artifact": "", "url": BTS_URL.format(year=2024, month=month)}
                    bts_rows.append(row)
                    print(f"BTS 2024-{month:02d}: {row['status']} {row.get('bytes', 0)} bytes", flush=True)
        all_manifest_rows.extend(bts_rows)
        month_label = "all" if not args.months else "_".join(f"{m:02d}" for m in months_to_process)
        write_manifest(META / f"supplemental_bts_manifest_{args.mode}_{month_label}.csv", bts_rows)

        if args.mode == "smoke":
            iem_jobs = [(airport, 2024, 1, None, "iem_asos") for airport in BASE_AIRPORTS]
            iem_jobs += [(airport, 2025, 1, None, "iem_asos") for airport in ["BOS", "MCO", "SEA", "PHX", "IAH"]]
        else:
            iem_jobs = [(airport, 2024, month, None, "iem_asos") for airport in BASE_AIRPORTS for month in months_to_process]
            iem_jobs += [(airport, 2025, month, None, "iem_asos") for airport in EXTENDED_AIRPORTS for month in months_to_process]
        iem_rows = download_iem_batch(iem_jobs, args.iem_workers)
        all_manifest_rows.extend(iem_rows)
        write_manifest(META / f"supplemental_iem_asos_manifest_{args.mode}_{month_label}.csv", iem_rows)

        if args.mode == "smoke":
            faa_manifest = []
            for d in iter_month_dates(2024, 1):
                manifest, rows, errors = fetch_faa_day(d)
                faa_manifest.append(manifest)
                print(f"FAA {d.isoformat()}: {manifest['status']} parsed={len(rows)}", flush=True)
            full_path, error_path = combine_faa_year(2024)
            faa_manifest.append({"dataset": "FAA ATCSCC advisories combined", "year": 2024, "artifact": str(full_path), "bytes": full_path.stat().st_size if full_path.exists() else 0, "status": "parsed", "url": FAA_LIST_URL})
        else:
            faa_manifest = download_faa_months(2024, months_to_process, args.faa_workers)
        all_manifest_rows.extend(faa_manifest)
        write_manifest(META / f"supplemental_faa_atcscc_manifest_{args.mode}_{month_label}.csv", faa_manifest)

        if args.mode == "smoke":
            noaa_rows = []
            try:
                noaa_rows.append(download_noaa_storm(2025, "details"))
            except Exception as exc:
                noaa_rows.append({"dataset": "NOAA Storm Events", "year": 2025, "kind": "details", "status": f"failed: {type(exc).__name__}: {exc}", "bytes": 0, "artifact": "", "url": NOAA_STORM_DIR})
            print(f"NOAA Storm Events 2025 details: {noaa_rows[-1]['status']} {noaa_rows[-1].get('bytes', 0)} bytes", flush=True)
        else:
            noaa_rows = download_noaa_all([2024, 2025])
        all_manifest_rows.extend(noaa_rows)
        write_manifest(META / f"supplemental_noaa_storm_events_manifest_{args.mode}_{month_label}.csv", noaa_rows)

        if args.mode == "full":
            station_row = download_station_metadata()
            all_manifest_rows.append(station_row)
            print(f"IEM station metadata: {station_row['status']} {station_row.get('bytes', 0)} bytes", flush=True)
            neighbors = select_neighbor_stations()
            neighbor_jobs = [
                (row.airport, 2025, month, row.station, "iem_asos_neighbors")
                for row in neighbors.itertuples(index=False)
                for month in months_to_process
            ]
            neighbor_rows = download_iem_batch(neighbor_jobs, args.iem_workers)
            all_manifest_rows.extend(neighbor_rows)
            write_manifest(META / f"supplemental_iem_asos_neighbors_manifest_full_{month_label}.csv", neighbor_rows)
        write_manifest(META / f"supplemental_source_data_manifest_{args.mode}_{month_label}.csv", all_manifest_rows)

    audit = audit_downloads()
    print(audit.to_string(index=False), flush=True)
    print(RESULTS / "supplemental_source_data_audit.md", flush=True)


if __name__ == "__main__":
    main()
