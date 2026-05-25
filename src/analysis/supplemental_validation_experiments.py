import argparse
import calendar
import math
import re
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from smoke_open_fusion_topics import AIRPORT_TZ, read_bts_month, weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
MAIN_OUT = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
OUT = PROJECT / "results" / "experiments" / "supplemental_validation"

BASE_AIRPORTS = list(AIRPORT_TZ)
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

EXTRA_AIRPORT_TZ = {
    "BOS": "America/New_York",
    "BWI": "America/New_York",
    "DCA": "America/New_York",
    "DTW": "America/Detroit",
    "FLL": "America/New_York",
    "HNL": "Pacific/Honolulu",
    "IAD": "America/New_York",
    "IAH": "America/Chicago",
    "LAS": "America/Los_Angeles",
    "MCO": "America/New_York",
    "MDW": "America/Chicago",
    "MIA": "America/New_York",
    "MSP": "America/Chicago",
    "PHL": "America/New_York",
    "PHX": "America/Phoenix",
    "RDU": "America/New_York",
    "SAN": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SLC": "America/Denver",
    "TPA": "America/New_York",
}
AIRPORT_TZ_ALL = AIRPORT_TZ | EXTRA_AIRPORT_TZ

WINDOWS = {
    "active": (0, 0),
    "post_3h": (0, 3),
}
SENSITIVITY_WINDOWS = ["active", "post_1h", "post_2h", "post_3h", "extended_6h"]

SIGNIFICANT_STORM_TYPES = {
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


def parse_months(text: str) -> list[int]:
    if not text:
        return list(range(1, 13))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    return sorted({m for m in out if 1 <= m <= 12})


def parse_hhmm_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    values = pd.to_numeric(series, errors="coerce")
    hours = (values // 100).astype("float")
    minutes = (values % 100).astype("float")
    hours = hours.mask(hours == 24, 0)
    valid = values.notna() & hours.between(0, 23) & minutes.between(0, 59)
    return hours.where(valid), minutes.where(valid)


def scheduled_utc_hour(df: pd.DataFrame, airport_col: str, time_col: str, add_day: pd.Series) -> pd.Series:
    hours, minutes = parse_hhmm_series(df[time_col])
    base = pd.to_datetime(df["FlightDate"], errors="coerce") + pd.to_timedelta(add_day.astype(int), unit="D")
    local_naive = base + pd.to_timedelta(hours.fillna(0), unit="h") + pd.to_timedelta(minutes.fillna(0), unit="m")
    local_naive = local_naive.where(hours.notna() & minutes.notna())
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for airport, tz in AIRPORT_TZ_ALL.items():
        mask = df[airport_col].eq(airport) & local_naive.notna()
        if not mask.any():
            continue
        localized = local_naive.loc[mask].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        out.loc[mask] = localized.dt.tz_convert("UTC").dt.floor("h").dt.tz_localize(None)
    return out


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("M", np.nan), errors="coerce")


def build_month_arrival_panel(year: int, month: int, airports: list[str]) -> pd.DataFrame:
    usecols = [
        "FlightDate",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    bts = read_bts_month(year, month, usecols)
    bts = bts[bts["Dest"].isin(airports)].copy()
    dep_int = pd.to_numeric(bts["CRSDepTime"], errors="coerce")
    arr_int = pd.to_numeric(bts["CRSArrTime"], errors="coerce")
    add_day = (arr_int < dep_int).fillna(False)
    bts["utc_hour"] = scheduled_utc_hour(bts, "Dest", "CRSArrTime", add_day)
    bts = bts.dropna(subset=["utc_hour"])
    bts["arr_delay_min"] = pd.to_numeric(bts["ArrDelayMinutes"], errors="coerce")
    bts["arr_delay60"] = (bts["arr_delay_min"] >= 60).astype(float)
    bts["cancelled"] = pd.to_numeric(bts["Cancelled"], errors="coerce").fillna(0.0)
    bts["diverted"] = pd.to_numeric(bts["Diverted"], errors="coerce").fillna(0.0)
    panel = (
        bts.groupby(["Dest", "utc_hour"], as_index=False)
        .agg(
            arrivals=("Dest", "size"),
            arr_delay60_count=("arr_delay60", "sum"),
            cancel_count=("cancelled", "sum"),
            divert_count=("diverted", "sum"),
            mean_arr_delay=("arr_delay_min", "mean"),
            p90_arr_delay=("arr_delay_min", lambda s: float(np.nanpercentile(s, 90)) if s.notna().any() else np.nan),
        )
        .rename(columns={"Dest": "airport"})
    )
    panel["arr_delay60_rate"] = panel["arr_delay60_count"] / panel["arrivals"]
    panel["cancel_rate"] = panel["cancel_count"] / panel["arrivals"]
    panel["divert_rate"] = panel["divert_count"] / panel["arrivals"]
    panel["month"] = month
    return panel


def complete_airport_hour_grid(panel: pd.DataFrame, year: int, months: list[int], airports: list[str]) -> pd.DataFrame:
    grids = []
    for month in months:
        start = pd.Timestamp(year=year, month=month, day=1)
        days = calendar.monthrange(year, month)[1]
        end = start + pd.Timedelta(days=days)
        idx = pd.MultiIndex.from_product(
            [airports, pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")],
            names=["airport", "utc_hour"],
        )
        grid = pd.DataFrame(index=idx).reset_index()
        grid["month"] = month
        grids.append(grid)
    merged = pd.concat(grids, ignore_index=True).merge(panel, on=["airport", "utc_hour", "month"], how="left")
    count_cols = ["arrivals", "arr_delay60_count", "cancel_count", "divert_count"]
    merged[count_cols] = merged[count_cols].fillna(0)
    for col in ["arr_delay60_rate", "cancel_rate", "divert_rate"]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def read_iem_weather(year: int, months: list[int], airports: list[str], source_dir: str = "iem_asos") -> pd.DataFrame:
    frames = []
    for month in months:
        for airport in airports:
            path = RAW / source_dir / f"iem_asos_{year}_{month:02d}_{airport}.csv"
            if not path.exists() and source_dir == "iem_asos_neighbors":
                matches = sorted((RAW / source_dir).glob(f"iem_asos_{year}_{month:02d}_{airport}_*.csv"))
            else:
                matches = [path]
            for item in matches:
                if not item.exists():
                    continue
                df = pd.read_csv(item, low_memory=False)
                if "valid" not in df.columns:
                    continue
                df["airport"] = airport
                df["station_file"] = item.name
                df["utc_hour"] = pd.to_datetime(df["valid"], utc=True, errors="coerce").dt.floor("h").dt.tz_localize(None)
                df["wind_speed_mps"] = to_float(df["sknt"]) * 0.514444
                df["visibility_km"] = to_float(df["vsby"]) * 1.60934
                df["ceiling_m"] = to_float(df["skyl1"]) * 0.3048
                df["temperature_c"] = (to_float(df["tmpf"]) - 32.0) * 5.0 / 9.0
                frames.append(
                    df[
                        [
                            "airport",
                            "station_file",
                            "utc_hour",
                            "wind_speed_mps",
                            "visibility_km",
                            "ceiling_m",
                            "temperature_c",
                        ]
                    ]
                )
    if not frames:
        return pd.DataFrame()
    weather = pd.concat(frames, ignore_index=True)
    return (
        weather.groupby(["airport", "utc_hour"], as_index=False)
        .agg(
            stations=("station_file", "nunique"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            visibility_km=("visibility_km", "mean"),
            ceiling_m=("ceiling_m", "min"),
            temperature_c=("temperature_c", "mean"),
        )
    )


def add_weather(panel: pd.DataFrame, year: int, months: list[int], airports: list[str]) -> pd.DataFrame:
    weather = read_iem_weather(year, months, airports)
    panel = panel.merge(weather.drop(columns=["stations"], errors="ignore"), on=["airport", "utc_hour"], how="left")
    wind_sd = panel["wind_speed_mps"].std(ddof=0)
    wind_sd = wind_sd if pd.notna(wind_sd) and wind_sd > 0 else 1.0
    panel["weather_score"] = (
        panel["wind_speed_mps"].fillna(panel["wind_speed_mps"].median()) / wind_sd
        + (20 - panel["visibility_km"].fillna(panel["visibility_km"].median())).clip(lower=0) / 20
        + (1000 - panel["ceiling_m"].fillna(panel["ceiling_m"].median())).clip(lower=0) / 1000
    )
    panel["mild_weather_abs"] = (
        (panel["visibility_km"] >= 8)
        & (panel["wind_speed_mps"] <= 7)
        & ((panel["ceiling_m"] >= 1000) | panel["ceiling_m"].isna())
    ).astype(float)
    panel["day_of_week"] = panel["utc_hour"].dt.dayofweek.astype(str)
    panel["local_hour"] = [
        pd.Timestamp(row.utc_hour, tz="UTC").tz_convert(AIRPORT_TZ_ALL[row.airport]).hour
        for row in panel.itertuples(index=False)
    ]
    return panel


def read_events(year: int, airports: list[str]) -> pd.DataFrame:
    path = RAW / "faa_atcscc_advisories" / f"faa_atcscc_gdp_gs_reparsed_{year}_v2.csv"
    events = pd.read_csv(path)
    events = events[events["airport"].isin(airports)].copy()
    events = events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"])
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    duration_min = (events["end_utc"] - events["start_utc"]).dt.total_seconds() / 60.0
    events = events[events["start_utc"].dt.year.eq(year) & (duration_min > 0) & (duration_min <= 48 * 60)].copy()
    return events


def add_window_minutes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    hour_start = panel["utc_hour"]
    hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
    for name in WINDOWS:
        panel[f"{name}_minutes"] = 0.0
        panel[f"{name}_count"] = 0
    for ev in events.itertuples(index=False):
        airport_mask = panel["airport"].eq(ev.airport)
        for name, (pre_h, post_h) in WINDOWS.items():
            start = ev.start_utc - pd.Timedelta(hours=pre_h)
            end = ev.end_utc + pd.Timedelta(hours=post_h)
            mask = airport_mask & (hour_start < end) & (hour_end > start)
            if not mask.any():
                continue
            overlap_start = hour_start[mask].map(lambda x: max(x, start))
            overlap_end = hour_end[mask].map(lambda x: min(x, end))
            minutes = (overlap_end - overlap_start).dt.total_seconds() / 60.0
            panel.loc[mask, f"{name}_minutes"] += minutes.to_numpy()
            panel.loc[mask, f"{name}_count"] += 1
    for name in WINDOWS:
        panel[f"{name}_strong"] = (panel[f"{name}_minutes"] >= 45).astype(float)
    return panel


def add_extended_6h_minutes(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    if "extended_6h_minutes" in panel.columns:
        return panel
    panel["extended_6h_minutes"] = 0.0
    hour_start = panel["utc_hour"]
    hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
    for ev in events.itertuples(index=False):
        start = ev.start_utc
        end = ev.end_utc + pd.Timedelta(hours=6)
        mask = panel["airport"].eq(ev.airport) & (hour_start < end) & (hour_end > start)
        if not mask.any():
            continue
        overlap_start = hour_start[mask].map(lambda x: max(x, start))
        overlap_end = hour_end[mask].map(lambda x: min(x, end))
        panel.loc[mask, "extended_6h_minutes"] += ((overlap_end - overlap_start).dt.total_seconds() / 60.0).to_numpy()
    return panel


def ensure_window_flags(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    for name in WINDOWS:
        strong_col = f"{name}_strong"
        minutes_col = f"{name}_minutes"
        if strong_col not in panel.columns and minutes_col in panel.columns:
            panel[strong_col] = (panel[minutes_col] >= 45).astype(float)
    return panel


def state_delta(panel: pd.DataFrame, window: str, label: str, group_cols: list[str] | None = None) -> pd.DataFrame:
    panel = ensure_window_flags(panel)
    rows = []
    if group_cols:
        grouped = panel.groupby(group_cols, observed=False)
    else:
        grouped = [((), panel)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        conflict = group[(group["mild_weather_abs"] == 1.0) & (group[f"{window}_strong"] == 1.0)]
        baseline = group[(group["mild_weather_abs"] == 1.0) & (group[f"{window}_strong"] == 0.0)]
        row = {
            "label": label,
            "window": window,
            "conflict_airport_hours": int(len(conflict)),
            "baseline_airport_hours": int(len(baseline)),
            "conflict_arrivals": int(conflict["arrivals"].sum()),
            "baseline_arrivals": int(baseline["arrivals"].sum()),
            "conflict_arr_delay60_rate": weighted_mean(conflict, "arr_delay60_rate", "arrivals"),
            "baseline_arr_delay60_rate": weighted_mean(baseline, "arr_delay60_rate", "arrivals"),
            "conflict_cancel_rate": weighted_mean(conflict, "cancel_rate", "arrivals"),
            "baseline_cancel_rate": weighted_mean(baseline, "cancel_rate", "arrivals"),
            "conflict_mean_arr_delay": weighted_mean(conflict, "mean_arr_delay", "arrivals"),
            "baseline_mean_arr_delay": weighted_mean(baseline, "mean_arr_delay", "arrivals"),
            "conflict_p90_arr_delay": weighted_mean(conflict, "p90_arr_delay", "arrivals"),
            "baseline_p90_arr_delay": weighted_mean(baseline, "p90_arr_delay", "arrivals"),
        }
        row["delta_arr_delay60_rate"] = row["conflict_arr_delay60_rate"] - row["baseline_arr_delay60_rate"]
        row["delta_cancel_rate"] = row["conflict_cancel_rate"] - row["baseline_cancel_rate"]
        row["delta_mean_arr_delay"] = row["conflict_mean_arr_delay"] - row["baseline_mean_arr_delay"]
        row["delta_p90_arr_delay"] = row["conflict_p90_arr_delay"] - row["baseline_p90_arr_delay"]
        for col, value in zip(group_cols or [], key):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_panel(year: int, months: list[int], airports: list[str]) -> pd.DataFrame:
    panels = []
    for month in months:
        print(f"building BTS arrival panel {year}-{month:02d}", flush=True)
        panels.append(build_month_arrival_panel(year, month, airports))
    panel = pd.concat(panels, ignore_index=True)
    panel = complete_airport_hour_grid(panel, year, months, airports)
    panel = add_weather(panel, year, months, airports)
    events = read_events(year, airports)
    return add_window_minutes(panel, events)


def write_panel_outputs(panel: pd.DataFrame, out_dir: Path, label: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / f"{label}_airport_hour_panel.csv", index=False)
    score = pd.concat([state_delta(panel, window, label) for window in WINDOWS], ignore_index=True)
    month = pd.concat([state_delta(panel, window, label, ["month"]) for window in WINDOWS], ignore_index=True)
    airport = pd.concat([state_delta(panel, window, label, ["airport"]) for window in WINDOWS], ignore_index=True)
    score.to_csv(out_dir / f"{label}_scorecard.csv", index=False)
    month.to_csv(out_dir / f"{label}_month_delta.csv", index=False)
    airport.to_csv(out_dir / f"{label}_airport_delta.csv", index=False)
    return score


def airport_lat_lon() -> dict[str, tuple[float, float]]:
    stations = pd.read_csv(RAW / "iem_station_metadata" / "iem_stations_all.csv", low_memory=False)
    out = {}
    for airport in BASE_AIRPORTS + EXTENDED_AIRPORTS:
        sub = stations[stations["stid"].eq(airport)]
        if sub.empty and airport == "HNL":
            sub = stations[stations["stid"].eq("PHNL")]
        if not sub.empty:
            out[airport] = (float(sub.iloc[0]["lat"]), float(sub.iloc[0]["lon"]))
    return out


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    radius = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = math.radians(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def storm_local_to_utc(time_text: pd.Series, zone_text: pd.Series) -> pd.Series:
    naive = pd.to_datetime(time_text, format="%d-%b-%y %H:%M:%S", errors="coerce")
    offsets = zone_text.astype(str).str.extract(r"([+-]\d+)")[0].astype(float)
    return naive - pd.to_timedelta(offsets, unit="h")


def add_storm_event_flag(panel: pd.DataFrame, year: int, radius_km: float = 75.0, buffer_hours: int = 3) -> pd.DataFrame:
    details_path = RAW / "noaa_storm_events" / f"StormEvents_details_{year}.csv.gz"
    cols = [
        "EVENT_TYPE",
        "BEGIN_DATE_TIME",
        "END_DATE_TIME",
        "CZ_TIMEZONE",
        "BEGIN_LAT",
        "BEGIN_LON",
        "END_LAT",
        "END_LON",
    ]
    details = pd.read_csv(details_path, usecols=lambda c: c in cols, low_memory=False)
    details["EVENT_TYPE"] = details["EVENT_TYPE"].astype(str).str.upper()
    details = details[details["EVENT_TYPE"].isin(SIGNIFICANT_STORM_TYPES)].copy()
    details["lat"] = pd.to_numeric(details["BEGIN_LAT"], errors="coerce").fillna(pd.to_numeric(details["END_LAT"], errors="coerce"))
    details["lon"] = pd.to_numeric(details["BEGIN_LON"], errors="coerce").fillna(pd.to_numeric(details["END_LON"], errors="coerce"))
    details = details.dropna(subset=["lat", "lon"])
    details["begin_utc"] = storm_local_to_utc(details["BEGIN_DATE_TIME"], details["CZ_TIMEZONE"])
    details["end_utc"] = storm_local_to_utc(details["END_DATE_TIME"], details["CZ_TIMEZONE"])
    details["end_utc"] = details["end_utc"].where(details["end_utc"].notna(), details["begin_utc"])
    details = details.dropna(subset=["begin_utc", "end_utc"])
    coords = airport_lat_lon()
    panel = panel.copy()
    panel["nearby_storm_event"] = 0.0
    for airport, (lat, lon) in coords.items():
        if airport not in set(panel["airport"]):
            continue
        dist = haversine_km(details["lat"].to_numpy(float), details["lon"].to_numpy(float), lat, lon)
        near = details[dist <= radius_km]
        if near.empty:
            continue
        airport_mask = panel["airport"].eq(airport)
        hour_start = panel["utc_hour"]
        hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
        for event in near.itertuples(index=False):
            start = event.begin_utc - pd.Timedelta(hours=buffer_hours)
            end = event.end_utc + pd.Timedelta(hours=buffer_hours)
            mask = airport_mask & (hour_start < end) & (hour_end > start)
            panel.loc[mask, "nearby_storm_event"] = 1.0
    return panel


def run_2024(months: list[int], label: str) -> None:
    panel = build_panel(2024, months, BASE_AIRPORTS)
    score = write_panel_outputs(panel, OUT / "cross_year_2024", label)
    print(score.to_string(index=False), flush=True)


def run_extended(months: list[int], airports: list[str], label: str) -> None:
    panel = build_panel(2025, months, airports)
    score = write_panel_outputs(panel, OUT / "extended_airports_2025", label)
    print(score.to_string(index=False), flush=True)


def run_noaa_filter() -> None:
    panel = ensure_window_flags(pd.read_csv(MAIN_OUT / "airport_hour_panel_with_windows.csv", parse_dates=["utc_hour"]))
    panel = add_storm_event_flag(panel, 2025)
    out_dir = OUT / "noaa_storm_filter"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "noaa_storm_filter" / "base10_2025_panel_with_storm_flag.csv", index=False)
    clean = panel[panel["nearby_storm_event"] == 0.0].copy()
    score = pd.concat([state_delta(clean, window, "noaa_clean_2025") for window in WINDOWS], ignore_index=True)
    month = pd.concat([state_delta(clean, window, "noaa_clean_2025", ["month"]) for window in WINDOWS], ignore_index=True)
    score.to_csv(out_dir / "noaa_clean_scorecard.csv", index=False)
    month.to_csv(out_dir / "noaa_clean_month_delta.csv", index=False)
    print(score.to_string(index=False), flush=True)


def run_neighbor_weather() -> None:
    panel = ensure_window_flags(pd.read_csv(MAIN_OUT / "airport_hour_panel_with_windows.csv", parse_dates=["utc_hour"]))
    neigh = read_iem_weather(2025, list(range(1, 13)), BASE_AIRPORTS, source_dir="iem_asos_neighbors")
    neigh = neigh.rename(
        columns={
            "stations": "neighbor_stations",
            "wind_speed_mps": "neighbor_wind_speed_mps",
            "visibility_km": "neighbor_visibility_km",
            "ceiling_m": "neighbor_ceiling_m",
            "temperature_c": "neighbor_temperature_c",
        }
    )
    panel = panel.merge(neigh, on=["airport", "utc_hour"], how="left")
    panel["neighbor_mild_weather_abs"] = (
        (panel["neighbor_visibility_km"] >= 8)
        & (panel["neighbor_wind_speed_mps"] <= 7)
        & ((panel["neighbor_ceiling_m"] >= 1000) | panel["neighbor_ceiling_m"].isna())
    ).astype(float)
    panel["mild_agree"] = (panel["mild_weather_abs"] == panel["neighbor_mild_weather_abs"]).astype(float)
    panel["mild_weather_abs_original"] = panel["mild_weather_abs"]
    panel["mild_weather_abs"] = panel["neighbor_mild_weather_abs"]
    out_dir = OUT / "neighbor_weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    score = pd.concat([state_delta(panel, window, "neighbor_weather_2025") for window in WINDOWS], ignore_index=True)
    month = pd.concat([state_delta(panel, window, "neighbor_weather_2025", ["month"]) for window in WINDOWS], ignore_index=True)
    agree = (
        panel.groupby("airport", as_index=False)
        .agg(
            airport_hours=("airport", "size"),
            neighbor_stations=("neighbor_stations", "median"),
            mild_agreement=("mild_agree", "mean"),
            original_mild_share=("mild_weather_abs_original", "mean"),
            neighbor_mild_share=("neighbor_mild_weather_abs", "mean"),
        )
    )
    panel.to_csv(out_dir / "base10_2025_panel_with_neighbor_weather.csv", index=False)
    score.to_csv(out_dir / "neighbor_weather_scorecard.csv", index=False)
    month.to_csv(out_dir / "neighbor_weather_month_delta.csv", index=False)
    agree.to_csv(out_dir / "neighbor_weather_agreement.csv", index=False)
    print(score.to_string(index=False), flush=True)
    print(agree.to_string(index=False), flush=True)


def sensitivity_mild(panel: pd.DataFrame, weather_rule: str) -> pd.Series:
    if weather_rule == "base":
        vis, wind, ceil = 8.0, 7.0, 1000.0
    elif weather_rule == "loose":
        vis, wind, ceil = 6.0, 8.0, 800.0
    elif weather_rule == "strict":
        vis, wind, ceil = 10.0, 6.0, 1200.0
    else:
        raise ValueError(weather_rule)
    return (
        (panel["visibility_km"] >= vis)
        & (panel["wind_speed_mps"] <= wind)
        & ((panel["ceiling_m"] >= ceil) | panel["ceiling_m"].isna())
    ).astype(float)


def run_threshold_sensitivity() -> None:
    panel = pd.read_csv(MAIN_OUT / "airport_hour_panel_with_windows.csv", parse_dates=["utc_hour"])
    events = read_events(2025, BASE_AIRPORTS)
    panel = add_extended_6h_minutes(panel, events)
    out_dir = OUT / "threshold_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    month_rows = []
    for threshold in [30, 45, 60]:
        for weather_rule in ["loose", "base", "strict"]:
            use = panel.copy()
            use["mild_weather_abs"] = sensitivity_mild(use, weather_rule)
            for window in SENSITIVITY_WINDOWS:
                minutes_col = f"{window}_minutes"
                if minutes_col not in use.columns:
                    continue
                use[f"{window}_strong"] = (use[minutes_col] >= threshold).astype(float)
                score = state_delta(use, window, f"{weather_rule}_{threshold}min")
                score["weather_rule"] = weather_rule
                score["strong_minutes_threshold"] = threshold
                rows.append(score)
                month = state_delta(use, window, f"{weather_rule}_{threshold}min", ["month"])
                month["weather_rule"] = weather_rule
                month["strong_minutes_threshold"] = threshold
                month_rows.append(month)
    scorecard = pd.concat(rows, ignore_index=True)
    month_delta = pd.concat(month_rows, ignore_index=True)
    stability = (
        month_delta.assign(positive_delay_delta=month_delta["delta_arr_delay60_rate"] > 0)
        .groupby(["weather_rule", "strong_minutes_threshold", "window"], as_index=False)
        .agg(
            months_positive_delta=("positive_delay_delta", "sum"),
            months_with_200_conflict_arrivals=("conflict_arrivals", lambda s: int((s >= 200).sum())),
        )
    )
    scorecard = scorecard.merge(stability, on=["weather_rule", "strong_minutes_threshold", "window"], how="left")
    scorecard.to_csv(out_dir / "threshold_sensitivity_scorecard.csv", index=False)
    month_delta.to_csv(out_dir / "threshold_sensitivity_month_delta.csv", index=False)
    print(scorecard[["weather_rule", "strong_minutes_threshold", "window", "conflict_arrivals", "delta_arr_delay60_rate", "months_positive_delta"]].to_string(index=False), flush=True)


def weighted_delta_for_exposure(panel: pd.DataFrame, exposure_col: str) -> float:
    conflict = panel[(panel["mild_weather_abs"] == 1.0) & (panel[exposure_col] == 1.0)]
    baseline = panel[(panel["mild_weather_abs"] == 1.0) & (panel[exposure_col] == 0.0)]
    return weighted_mean(conflict, "arr_delay60_rate", "arrivals") - weighted_mean(baseline, "arr_delay60_rate", "arrivals")


def run_permutation_test(iterations: int = 100) -> None:
    panel = ensure_window_flags(pd.read_csv(MAIN_OUT / "airport_hour_panel_with_windows.csv", parse_dates=["utc_hour"]))
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy().reset_index(drop=True)
    rng = np.random.default_rng(20260513)
    out_dir = OUT / "permutation_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for window in ["active", "post_3h"]:
        exposure_col = f"{window}_strong"
        observed = weighted_delta_for_exposure(panel, exposure_col)
        work = panel[["airport", "month", "local_hour", "mild_weather_abs", "arrivals", "arr_delay60_rate", exposure_col]].copy()
        groups = [idx.to_numpy() for _, idx in work.groupby(["airport", "month", "local_hour"], observed=False).groups.items()]
        values = work[exposure_col].to_numpy(float)
        for i in range(iterations):
            permuted = values.copy()
            for idx in groups:
                if len(idx) > 1:
                    permuted[idx] = rng.permutation(permuted[idx])
            work["permuted_exposure"] = permuted
            rows.append(
                {
                    "window": window,
                    "iteration": i + 1,
                    "observed_delta": observed,
                    "permuted_delta": weighted_delta_for_exposure(work, "permuted_exposure"),
                }
            )
    perm = pd.DataFrame(rows)
    summary = (
        perm.groupby("window", as_index=False)
        .agg(
            observed_delta=("observed_delta", "first"),
            permutation_mean=("permuted_delta", "mean"),
            permutation_p95=("permuted_delta", lambda s: float(np.nanquantile(s, 0.95))),
            permutation_max=("permuted_delta", "max"),
        )
    )
    summary["observed_above_p95"] = summary["observed_delta"] > summary["permutation_p95"]
    perm.to_csv(out_dir / "permutation_draws.csv", index=False)
    summary.to_csv(out_dir / "permutation_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def write_summary() -> None:
    lines = ["# Supplemental validation summary", ""]
    checks = [
        ("2024 smoke", OUT / "cross_year_2024" / "smoke_2024_01_07_12_scorecard.csv"),
        ("2024 full", OUT / "cross_year_2024" / "full_2024_scorecard.csv"),
        ("2025 extended smoke", OUT / "extended_airports_2025" / "smoke_extended_2025_scorecard.csv"),
        ("2025 30-airport extension", OUT / "extended_airports_2025" / "full_30_airports_2025_scorecard.csv"),
        ("NOAA clean", OUT / "noaa_storm_filter" / "noaa_clean_scorecard.csv"),
        ("Neighbor weather", OUT / "neighbor_weather" / "neighbor_weather_scorecard.csv"),
    ]
    for name, path in checks:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        lines.append(f"## {name}")
        for row in df.itertuples(index=False):
            lines.append(
                f"- {row.window}: conflict arrivals {int(row.conflict_arrivals)}, "
                f"delay delta {row.delta_arr_delay60_rate:+.3f}, cancellation delta {row.delta_cancel_rate:+.3f}."
            )
        lines.append("")
    threshold_path = OUT / "threshold_sensitivity" / "threshold_sensitivity_scorecard.csv"
    if threshold_path.exists():
        df = pd.read_csv(threshold_path)
        core = df[(df["weather_rule"].eq("base")) & (df["window"].isin(["active", "post_3h"]))]
        lines.append("## Threshold sensitivity")
        for row in core.itertuples(index=False):
            lines.append(
                f"- {row.window}, {int(row.strong_minutes_threshold)} min: "
                f"delay delta {row.delta_arr_delay60_rate:+.3f}, positive months {int(row.months_positive_delta)}/12."
            )
        lines.append("")
    permutation_path = OUT / "permutation_test" / "permutation_summary.csv"
    if permutation_path.exists():
        df = pd.read_csv(permutation_path)
        lines.append("## Permutation test")
        for row in df.itertuples(index=False):
            verdict = "above" if bool(row.observed_above_p95) else "within"
            lines.append(
                f"- {row.window}: observed {row.observed_delta:+.3f}, "
                f"permutation 95th percentile {row.permutation_p95:+.3f}, observed is {verdict} the 95th percentile."
            )
        lines.append("")
    (OUT / "supplemental_validation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "supplemental_validation_summary.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["2024", "extended", "noaa", "neighbor", "threshold", "permutation", "summary"],
        required=True,
    )
    parser.add_argument("--months", default="")
    parser.add_argument("--airports", default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    if args.mode == "2024":
        label = args.label or ("full_2024" if months == list(range(1, 13)) else "smoke_2024")
        run_2024(months, label)
    elif args.mode == "extended":
        airports = [x.strip().upper() for x in args.airports.split(",") if x.strip()] or EXTENDED_AIRPORTS
        label = args.label or "extended_2025"
        run_extended(months, airports, label)
    elif args.mode == "noaa":
        run_noaa_filter()
    elif args.mode == "neighbor":
        run_neighbor_weather()
    elif args.mode == "threshold":
        run_threshold_sensitivity()
    elif args.mode == "permutation":
        run_permutation_test(args.iterations)
    elif args.mode == "summary":
        write_summary()


if __name__ == "__main__":
    main()
