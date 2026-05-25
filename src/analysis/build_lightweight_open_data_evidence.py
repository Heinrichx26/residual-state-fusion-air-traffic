from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "experiments" / "supplemental_validation" / "open_data_evidence"

MAIN_10 = ["ATL", "CLT", "DEN", "DFW", "EWR", "JFK", "LAX", "LGA", "ORD", "SFO"]
EXTENDED_30 = [
    "ATL", "BOS", "BWI", "CLT", "DCA", "DEN", "DFW", "DTW", "EWR", "FLL",
    "HNL", "IAD", "IAH", "JFK", "LAS", "LAX", "LGA", "MCO", "MDW", "MIA",
    "MSP", "ORD", "PHL", "PHX", "RDU", "SAN", "SEA", "SFO", "SLC", "TPA",
]

STORM_FILTER_TYPES = {
    "THUNDERSTORM WIND",
    "HAIL",
    "TORNADO",
    "FLASH FLOOD",
    "HEAVY RAIN",
    "WINTER STORM",
    "WINTER WEATHER",
    "BLIZZARD",
    "DENSE FOG",
}


def read_bts_month(path: Path) -> pd.DataFrame:
    cols = ["Year", "Month", "Dest", "Cancelled", "ArrDelay"]
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        return pd.read_csv(zf.open(name), usecols=cols)


def build_bts_consistency() -> None:
    rows = []
    for year in [2024, 2025]:
        for month in range(1, 13):
            path = RAW / "bts_on_time" / f"bts_on_time_{year}_{month:02d}.zip"
            df = read_bts_month(path)
            df = df[df["Dest"].isin(EXTENDED_30)].copy()
            df["scheduled_arrivals"] = 1
            df["cancelled_flag"] = df["Cancelled"].fillna(0).astype(float)
            df["long_delay60"] = ((df["ArrDelay"].fillna(-999) >= 60) & (df["cancelled_flag"] == 0)).astype(int)
            grouped = (
                df.groupby(["Year", "Month", "Dest"], as_index=False)
                .agg(
                    scheduled_arrivals=("scheduled_arrivals", "sum"),
                    cancellations=("cancelled_flag", "sum"),
                    long_delay60=("long_delay60", "sum"),
                    mean_arr_delay=("ArrDelay", "mean"),
                )
                .rename(columns={"Year": "year", "Month": "month", "Dest": "airport"})
            )
            rows.append(grouped)
    airport_month = pd.concat(rows, ignore_index=True)
    airport_month["scope"] = airport_month["airport"].apply(lambda a: "main10" if a in MAIN_10 else "extended30")
    airport_month["cancellation_rate"] = airport_month["cancellations"] / airport_month["scheduled_arrivals"]
    airport_month["long_delay60_rate"] = airport_month["long_delay60"] / airport_month["scheduled_arrivals"]
    airport_month.to_csv(OUT / "bts_airport_month_traffic_2024_2025.csv", index=False)

    summary_rows = []
    for scope_name, airports in [("main10", MAIN_10), ("extended30", EXTENDED_30)]:
        sub = airport_month[airport_month["airport"].isin(airports)]
        for year, g in sub.groupby("year"):
            summary_rows.append(
                {
                    "scope": scope_name,
                    "year": int(year),
                    "airports": len(airports),
                    "airport_months": int(g.shape[0]),
                    "scheduled_arrivals": int(g["scheduled_arrivals"].sum()),
                    "cancellations": int(g["cancellations"].sum()),
                    "long_delay60": int(g["long_delay60"].sum()),
                    "cancellation_rate": g["cancellations"].sum() / g["scheduled_arrivals"].sum(),
                    "long_delay60_rate": g["long_delay60"].sum() / g["scheduled_arrivals"].sum(),
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT / "bts_traffic_consistency_summary.csv", index=False)


def build_noaa_distribution() -> None:
    rows = []
    summary_rows = []
    for year in [2024, 2025]:
        path = RAW / "noaa_storm_events" / f"StormEvents_details_{year}.csv.gz"
        df = pd.read_csv(path, compression="gzip", usecols=["EVENT_TYPE", "STATE"])
        df["EVENT_TYPE"] = df["EVENT_TYPE"].fillna("").str.upper().str.strip()
        counts = df["EVENT_TYPE"].value_counts().rename_axis("event_type").reset_index(name="events")
        counts["year"] = year
        counts["share"] = counts["events"] / counts["events"].sum()
        counts["used_in_filter"] = counts["event_type"].isin(STORM_FILTER_TYPES)
        rows.append(counts)
        summary_rows.append(
            {
                "year": year,
                "total_events": int(len(df)),
                "filter_type_events": int(counts.loc[counts["used_in_filter"], "events"].sum()),
                "filter_type_share": float(counts.loc[counts["used_in_filter"], "events"].sum() / len(df)),
                "states": int(df["STATE"].nunique()),
            }
        )
    dist = pd.concat(rows, ignore_index=True)
    dist.to_csv(OUT / "noaa_storm_event_type_distribution_2024_2025.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "noaa_storm_event_summary_2024_2025.csv", index=False)


def reason_category(reason: str) -> str:
    text = str(reason or "").upper().strip()
    if not text:
        return "MISSING"
    if any(k in text for k in ["THUNDER", "WEATHER", "WIND", "CEILING", "VIS", "SNOW", "ICE", "FOG", "RAIN", "TORNADO", "HURRICANE"]):
        return "WEATHER_RELATED"
    if "VOLUME" in text or "DEMAND" in text or "ROUTES" in text:
        return "VOLUME_OR_DEMAND"
    if "STAFF" in text:
        return "STAFFING"
    if any(k in text for k in ["EQUIPMENT", "OUTAGE", "OCL", "IT ISSUES"]):
        return "EQUIPMENT_OR_OUTAGE"
    if any(k in text for k in ["RWY", "RUNWAY", "TAXI", "CONSTRUCTION", "MAINTENANCE", "DISABLED", "OBSTRUCTION"]):
        return "RUNWAY_OR_SURFACE"
    if "REQUEST" in text:
        return "REQUEST"
    return "OTHER"


def build_atcscc_reason_diagnostic() -> None:
    path = RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_reparsed_2025_v2.csv"
    df = pd.read_csv(path)
    df["reason_category"] = df["reason"].apply(reason_category)
    summary = (
        df.groupby(["tmi_type", "reason_category"], as_index=False)
        .size()
        .rename(columns={"size": "records"})
    )
    summary["share_within_type"] = summary["records"] / summary.groupby("tmi_type")["records"].transform("sum")
    summary.to_csv(OUT / "atcscc_reason_category_distribution_2025.csv", index=False)

    raw_reasons = (
        df["reason"].fillna("MISSING").value_counts().rename_axis("reason").reset_index(name="records")
    )
    raw_reasons.to_csv(OUT / "atcscc_reason_raw_top_2025.csv", index=False)

    assessment = [
        "# ATCSCC reason diagnostic assessment",
        "",
        "The reason field was parsed successfully for the 2025 retained GDP/GS advisory records.",
        "The distribution is useful as an internal parsing diagnostic, but it is not added to the manuscript or Supplementary material at this stage.",
        "Many reasons are broad weather-related labels. Those labels may refer to terminal-area, enroute, forecast, or traffic-management contexts and require additional interpretation relative to airport-station ASOS conditions.",
        "Keeping the diagnostic outside the paper preserves the current focused argument: local mild weather plus a strong public advisory predicts realized disruption.",
        "",
        f"Rows classified: {len(df)}.",
        f"Missing reason rows: {int((df['reason_category'] == 'MISSING').sum())}.",
    ]
    (OUT / "atcscc_reason_diagnostic_assessment.md").write_text("\n".join(assessment), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_bts_consistency()
    build_noaa_distribution()
    build_atcscc_reason_diagnostic()
    print(f"Wrote lightweight evidence outputs to {OUT}")


if __name__ == "__main__":
    main()
