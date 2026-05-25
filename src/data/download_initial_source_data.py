import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
META = PROJECT / "data" / "external" / "source_manifests"

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
OURAIRPORTS = {
    "airports": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "runways": "https://davidmegginson.github.io/ourairports-data/runways.csv",
}
NOAA_GLOBAL_HOURLY = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
FAA_LIST_URL = "https://www.fly.faa.gov/adv/adv_list"

AIRPORT_STATIONS = {
    "ATL": "72219013874",
    "CLT": "72314013881",
    "DEN": "72565003017",
    "DFW": "72259003927",
    "EWR": "72502014734",
    "JFK": "74486094789",
    "LAX": "72295023174",
    "LGA": "72503014732",
    "ORD": "72530094846",
    "SFO": "72494023234",
}


def download_file(url: str, path: Path, timeout: int = 180) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return {"artifact": str(path), "url": url, "bytes": path.stat().st_size, "status": "cached"}
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    return {"artifact": str(path), "url": url, "bytes": path.stat().st_size, "status": "downloaded"}


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


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
    try:
        return datetime(base.year, base.month, day, hour, minute, tzinfo=ZoneInfo("UTC"))
    except ValueError:
        next_month = base.month + 1
        next_year = base.year
        if next_month == 13:
            next_month = 1
        next_year += 1
    return datetime(next_year, next_month, day, hour, minute, tzinfo=ZoneInfo("UTC"))


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
        "duration_min": round((end_utc - start_utc).total_seconds() / 60, 2),
        "reason": reason,
        "source_url": source_url,
    }


def fetch_faa_day(d: date) -> tuple[dict, list[dict]]:
    day_dir = RAW / "faa_atcscc_advisories" / d.strftime("%Y_%m")
    day_dir.mkdir(parents=True, exist_ok=True)
    list_path = day_dir / f"list_{d.isoformat()}.html"
    if list_path.exists():
        html = list_path.read_text(encoding="iso-8859-1", errors="ignore")
    else:
        r = requests.get(FAA_LIST_URL, params=faa_params(d), timeout=60)
        r.raise_for_status()
        html = r.text
        list_path.write_text(html, encoding="iso-8859-1", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    detail_count = 0
    for a in soup.find_all("a"):
        href = a.get("href") or ""
        if "adv_otherdis" not in href:
            continue
        detail_url = "https://www.fly.faa.gov" + href if href.startswith("/") else href
        advn = ""
        if "advn=" in href:
            advn = href.split("advn=", 1)[1].split("&", 1)[0]
        if not advn:
            advn = str(detail_count + 1)
        detail_path = day_dir / f"detail_{d.strftime('%Y%m%d')}_{advn}.html"
        if detail_path.exists():
            detail_html = detail_path.read_text(encoding="iso-8859-1", errors="ignore")
        else:
            r = requests.get(detail_url, timeout=60)
            r.raise_for_status()
            detail_html = r.text
            detail_path.write_text(detail_html, encoding="iso-8859-1", errors="ignore")
        detail_count += 1
        text = BeautifulSoup(detail_html, "html.parser").get_text("\n", strip=True)
        parsed = parse_advisory(text, d, detail_url)
        if parsed:
            rows.append(parsed)
    manifest = {
        "artifact": str(list_path),
        "url": FAA_LIST_URL,
        "bytes": list_path.stat().st_size,
        "status": "downloaded",
        "records": detail_count,
        "date": d.isoformat(),
    }
    return manifest, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    static_jobs = []
    for name, url in OURAIRPORTS.items():
        static_jobs.append((url, RAW / "ourairports" / f"{name}.csv"))
    for year, month in [(2024, 7), (2025, 7), (2025, 12)]:
        static_jobs.append((BTS_URL.format(year=year, month=month), RAW / "bts_on_time" / f"bts_on_time_{year}_{month:02d}.zip"))
    for airport, station in AIRPORT_STATIONS.items():
        url = NOAA_GLOBAL_HOURLY.format(year=2025, station=station)
        static_jobs.append((url, RAW / "noaa_global_hourly" / f"noaa_global_hourly_2025_{airport}_{station}.csv"))

    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(download_file, url, path): (url, path) for url, path in static_jobs}
        for fut in as_completed(futures):
            row = fut.result()
            manifest_rows.append(row)
            print(f"{row['status']}: {row['artifact']} ({row['bytes']} bytes)", flush=True)

    faa_rows = []
    faa_manifest = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_faa_day, d): d for d in iter_dates(date(2025, 7, 1), date(2025, 7, 7))}
        for fut in as_completed(futures):
            manifest, rows = fut.result()
            faa_manifest.append(manifest)
            faa_rows.extend(rows)
            print(f"faa: {manifest['date']} details={manifest['records']} parsed={len(rows)}", flush=True)

    faa_csv = RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_2025_07_01_07.csv"
    write_csv(faa_csv, faa_rows)
    manifest_rows.extend(faa_manifest)
    manifest_rows.append(
        {
            "artifact": str(faa_csv),
            "url": FAA_LIST_URL,
            "bytes": faa_csv.stat().st_size if faa_csv.exists() else 0,
            "status": "parsed",
            "records": len(faa_rows),
        }
    )

    manifest_path = META / "initial_source_data_download_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
