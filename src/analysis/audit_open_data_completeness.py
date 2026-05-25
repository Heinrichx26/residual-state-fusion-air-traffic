import calendar
from datetime import date
from pathlib import Path

import pandas as pd

from smoke_open_fusion_topics import AIRPORT_TZ


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
ATCSCC_REPARSED = RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_reparsed_2025_v2.csv"


def expected_dates(year: int = 2025) -> list[date]:
    return [
        date(year, month, day)
        for month in range(1, 13)
        for day in range(1, calendar.monthrange(year, month)[1] + 1)
    ]


def count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return len(pd.read_csv(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    bts_expected = [RAW / "bts_on_time" / f"bts_on_time_2025_{month:02d}.zip" for month in range(1, 13)]
    rows.append(
        {
            "dataset": "BTS On-Time Performance",
            "expected_units": len(bts_expected),
            "available_units": sum(path.exists() and path.stat().st_size > 1024 for path in bts_expected),
            "missing_units": ";".join(path.name for path in bts_expected if not path.exists() or path.stat().st_size <= 1024),
            "issue_rows": 0,
        }
    )

    weather_expected = [
        RAW / "iem_asos" / f"iem_asos_2025_{month:02d}_{airport}.csv"
        for month in range(1, 13)
        for airport in AIRPORT_TZ
    ]
    rows.append(
        {
            "dataset": "IEM ASOS hourly weather",
            "expected_units": len(weather_expected),
            "available_units": sum(path.exists() and path.stat().st_size > 1024 for path in weather_expected),
            "missing_units": ";".join(path.name for path in weather_expected if not path.exists() or path.stat().st_size <= 1024),
            "issue_rows": 0,
        }
    )

    dates = expected_dates()
    parsed_expected = [RAW / "faa_atcscc_advisories" / d.strftime("%Y_%m") / f"parsed_{d:%Y%m%d}_v2.csv" for d in dates]
    errors_expected = [RAW / "faa_atcscc_advisories" / d.strftime("%Y_%m") / f"errors_{d:%Y%m%d}_v2.csv" for d in dates]
    error_rows = sum(count_csv_rows(path) for path in errors_expected)
    rows.append(
        {
            "dataset": "FAA ATCSCC parsed daily advisories",
            "expected_units": len(parsed_expected),
            "available_units": sum(path.exists() for path in parsed_expected),
            "missing_units": ";".join(path.name for path in parsed_expected if not path.exists()),
            "issue_rows": error_rows,
        }
    )

    rows.append(
        {
            "dataset": "FAA ATCSCC full-year combined advisories",
            "expected_units": 1,
            "available_units": int(ATCSCC_REPARSED.exists() and ATCSCC_REPARSED.stat().st_size > 0),
            "missing_units": "" if ATCSCC_REPARSED.exists() and ATCSCC_REPARSED.stat().st_size > 0 else ATCSCC_REPARSED.name,
            "issue_rows": 0,
        }
    )

    audit = pd.DataFrame(rows)
    audit["complete"] = (audit["expected_units"] == audit["available_units"]) & (audit["issue_rows"] == 0)
    audit.to_csv(OUT / "open_data_completeness_audit.csv", index=False)

    type_lines = []
    if ATCSCC_REPARSED.exists():
        events = pd.read_csv(ATCSCC_REPARSED)
        type_month = (
            events.assign(month=pd.to_datetime(events["advisory_date"]).dt.month)
            .groupby(["month", "tmi_type"], as_index=False)
            .size()
            .rename(columns={"size": "rows"})
        )
        type_month.to_csv(OUT / "atcscc_reparsed_month_type_counts.csv", index=False)
        type_counts = events.groupby("tmi_type").size().to_dict()
        type_lines = [f"- {key}: {value}" for key, value in sorted(type_counts.items())]

    lines = [
        "# Open data completeness audit",
        "",
        f"BTS monthly files: {int(audit.loc[audit['dataset'].eq('BTS On-Time Performance'), 'available_units'].iloc[0])}/12.",
        f"IEM airport-month weather files: {int(audit.loc[audit['dataset'].eq('IEM ASOS hourly weather'), 'available_units'].iloc[0])}/120.",
        f"ATCSCC daily parsed files: {int(audit.loc[audit['dataset'].eq('FAA ATCSCC parsed daily advisories'), 'available_units'].iloc[0])}/365.",
        f"ATCSCC parser issue rows: {error_rows}.",
        "",
        "ATCSCC full-year rows by type:",
        *type_lines,
        "",
        "Decision: data coverage is sufficient for the full-year window experiment." if audit["complete"].all() else "Decision: data coverage needs another repair pass.",
    ]
    (OUT / "open_data_completeness_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "open_data_completeness_audit.md", flush=True)


if __name__ == "__main__":
    main()
