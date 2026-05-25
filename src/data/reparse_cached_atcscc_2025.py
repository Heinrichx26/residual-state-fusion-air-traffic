import argparse
import calendar
import csv
from datetime import date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from download_initial_source_data import RAW, parse_advisory


FIELDS = ["advisory_date", "airport", "duration_min", "end_utc", "reason", "source_url", "start_utc", "tmi_type"]
ERROR_FIELDS = ["date", "detail_file", "error"]


def iter_dates(year: int):
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1] + 1):
            yield date(year, month, day)


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_url_for_detail(path: Path, d: date) -> str:
    advn = path.stem.split("_")[-1]
    return f"https://www.fly.faa.gov/adv/adv_otherdis?adv_date={d:%m%d%Y}&advn={advn}"


def reparse_day(d: date, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_dir = RAW / "faa_atcscc_advisories" / d.strftime("%Y_%m")
    rows = []
    errors = []
    for path in sorted(day_dir.glob(f"detail_{d:%Y%m%d}_*.html")):
        try:
            html = path.read_text(encoding="iso-8859-1", errors="ignore")
            text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
            parsed = parse_advisory(text, d, source_url_for_detail(path, d))
            if parsed:
                rows.append(parsed)
        except Exception as exc:
            errors.append({"date": d.isoformat(), "detail_file": str(path), "error": f"{type(exc).__name__}: {exc}"})
    parsed_path = day_dir / f"parsed_{d:%Y%m%d}{suffix}.csv"
    error_path = day_dir / f"errors_{d:%Y%m%d}{suffix}.csv"
    write_rows(parsed_path, rows, FIELDS)
    write_rows(error_path, errors, ERROR_FIELDS)
    return pd.DataFrame(rows, columns=FIELDS), pd.DataFrame(errors, columns=ERROR_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--suffix", default="_v2")
    args = parser.parse_args()

    all_rows = []
    all_errors = []
    month_rows = {}
    for d in iter_dates(args.year):
        parsed, errors = reparse_day(d, args.suffix)
        print(f"{d.isoformat()}: parsed={len(parsed)} errors={len(errors)}", flush=True)
        if not parsed.empty:
            all_rows.append(parsed)
            month_rows.setdefault(d.month, []).append(parsed)
        if not errors.empty:
            all_errors.append(errors)

    base = RAW / "faa_atcscc_advisories"
    for month, frames in month_rows.items():
        month_df = pd.concat(frames, ignore_index=True)
        month_df = month_df.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"])
        month_df.to_csv(base / f"faa_atcscc_gdp_gs_2025_{month:02d}{args.suffix}.csv", index=False)

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=FIELDS)
    if not full.empty:
        full = full.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"])
    full_path = base / f"faa_atcscc_gdp_gs_reparsed_{args.year}{args.suffix}.csv"
    full.to_csv(full_path, index=False)
    if all_errors:
        pd.concat(all_errors, ignore_index=True).to_csv(base / f"faa_atcscc_errors_reparsed_{args.year}{args.suffix}.csv", index=False)
    print(f"full={full_path} rows={len(full)}", flush=True)
    if not full.empty:
        print(full.groupby("tmi_type").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
