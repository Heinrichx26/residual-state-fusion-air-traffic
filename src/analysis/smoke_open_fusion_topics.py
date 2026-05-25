import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
RESULTS = PROJECT / "results" / "smoke_tests"

AIRPORT_TZ = {
    "ATL": "America/New_York",
    "CLT": "America/New_York",
    "DEN": "America/Denver",
    "DFW": "America/Chicago",
    "EWR": "America/New_York",
    "JFK": "America/New_York",
    "LAX": "America/Los_Angeles",
    "LGA": "America/New_York",
    "ORD": "America/Chicago",
    "SFO": "America/Los_Angeles",
}

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


def parse_hhmm(value) -> tuple[int, int] | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace('"', "")
    if not text:
        return None
    try:
        if "." in text:
            text = str(int(float(text)))
        text = text.zfill(4)
        hour = int(text[:2])
        minute = int(text[2:])
    except ValueError:
        return None
    if hour == 24:
        hour = 0
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def local_schedule_to_utc_hour(date_value, hhmm, airport: str, add_day: bool = False):
    parsed = parse_hhmm(hhmm)
    if parsed is None or airport not in AIRPORT_TZ:
        return pd.NaT
    base = pd.to_datetime(date_value).to_pydatetime()
    if add_day:
        base = base + timedelta(days=1)
    hour, minute = parsed
    local_dt = datetime(base.year, base.month, base.day, hour, minute, tzinfo=ZoneInfo(AIRPORT_TZ[airport]))
    return pd.Timestamp(local_dt.astimezone(ZoneInfo("UTC"))).floor("h")


def read_bts_month(year: int, month: int, usecols: list[str]) -> pd.DataFrame:
    path = RAW / "bts_on_time" / f"bts_on_time_{year}_{month:02d}.zip"
    with zipfile.ZipFile(path) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            return pd.read_csv(f, usecols=usecols, low_memory=False)


def parse_noaa_component(value: str, index: int, missing: set[str]) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(",")
    if len(parts) <= index:
        return np.nan
    item = parts[index].strip()
    if item in missing or item == "":
        return np.nan
    try:
        return float(item)
    except ValueError:
        return np.nan


def read_weather_2025(airports: list[str]) -> pd.DataFrame:
    frames = []
    for airport in airports:
        station = AIRPORT_STATIONS[airport]
        path = RAW / "noaa_global_hourly" / f"noaa_global_hourly_2025_{airport}_{station}.csv"
        df = pd.read_csv(path, usecols=lambda c: c in {"DATE", "WND", "VIS", "CIG", "TMP"}, low_memory=False)
        df["airport"] = airport
        df["utc_hour"] = pd.to_datetime(df["DATE"], utc=True).dt.floor("h").dt.tz_localize(None)
        df["wind_dir_deg"] = df["WND"].map(lambda x: parse_noaa_component(x, 0, {"999"}))
        df["wind_speed_mps"] = df["WND"].map(lambda x: parse_noaa_component(x, 3, {"9999"})) / 10.0
        df["visibility_km"] = df["VIS"].map(lambda x: parse_noaa_component(x, 0, {"999999"})) / 1000.0
        df["ceiling_m"] = df["CIG"].map(lambda x: parse_noaa_component(x, 0, {"99999"}))
        df["temperature_c"] = df["TMP"].map(lambda x: parse_noaa_component(x, 0, {"+9999", "9999"})) / 10.0
        frames.append(
            df[
                [
                    "airport",
                    "utc_hour",
                    "wind_dir_deg",
                    "wind_speed_mps",
                    "visibility_km",
                    "ceiling_m",
                    "temperature_c",
                ]
            ]
        )
    weather = pd.concat(frames, ignore_index=True)
    return (
        weather.groupby(["airport", "utc_hour"], as_index=False)
        .agg(
            wind_dir_deg=("wind_dir_deg", "mean"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            visibility_km=("visibility_km", "mean"),
            ceiling_m=("ceiling_m", "mean"),
            temperature_c=("temperature_c", "mean"),
        )
    )


def weighted_mean(df: pd.DataFrame, value: str, weight: str) -> float:
    use = df[[value, weight]].replace([np.inf, -np.inf], np.nan).dropna()
    if use.empty or use[weight].sum() <= 0:
        return np.nan
    return float((use[value] * use[weight]).sum() / use[weight].sum())


def weighted_lstsq(df: pd.DataFrame, y_col: str, x_cols: list[str], weight_col: str) -> pd.DataFrame:
    model_df = df[[y_col, weight_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    model_df = model_df[model_df[weight_col] > 0].copy()
    y = model_df[y_col].to_numpy(float)
    w = np.sqrt(model_df[weight_col].to_numpy(float))
    x_parts = [np.ones((len(model_df), 1))]
    names = ["Intercept"]
    for col in x_cols:
        if model_df[col].dtype == "object" or str(model_df[col].dtype).startswith("category"):
            dummies = pd.get_dummies(model_df[col], prefix=col, drop_first=True, dtype=float)
            x_parts.append(dummies.to_numpy(float))
            names.extend(dummies.columns.tolist())
        else:
            x_parts.append(model_df[[col]].to_numpy(float))
            names.append(col)
    x = np.hstack(x_parts)
    coef, _, _, _ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    pred = x @ coef
    rss = float(((y - pred) ** 2 * model_df[weight_col].to_numpy(float)).sum())
    tss = float(((y - weighted_mean(model_df, y_col, weight_col)) ** 2 * model_df[weight_col].to_numpy(float)).sum())
    rsq = 1 - rss / tss if tss > 0 else np.nan
    return pd.DataFrame(
        {
            "term": names,
            "coef": coef,
            "weighted_r2": rsq,
            "n": len(model_df),
            "weight_sum": model_df[weight_col].sum(),
        }
    )


def build_arrival_hour_panel() -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    usecols = [
        "FlightDate",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    bts = read_bts_month(2025, 7, usecols)
    bts = bts[bts["Dest"].isin(airports)].copy()
    dep_int = pd.to_numeric(bts["CRSDepTime"], errors="coerce")
    arr_int = pd.to_numeric(bts["CRSArrTime"], errors="coerce")
    add_day = (arr_int < dep_int).fillna(False)
    bts["utc_hour"] = [
        local_schedule_to_utc_hour(row.FlightDate, row.CRSArrTime, row.Dest, bool(next_day)).tz_localize(None)
        if pd.notna(local_schedule_to_utc_hour(row.FlightDate, row.CRSArrTime, row.Dest, bool(next_day)))
        else pd.NaT
        for row, next_day in zip(bts.itertuples(index=False), add_day)
    ]
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
    return panel


def add_atcscc_events(panel: pd.DataFrame) -> pd.DataFrame:
    events = pd.read_csv(RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_2025_07_01_07.csv")
    events = events[events["airport"].isin(AIRPORT_TZ)].copy()
    events = events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc"])
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True).dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True).dt.tz_localize(None)
    panel = panel.copy()
    panel["event_minutes"] = 0.0
    panel["event_count"] = 0
    panel["gs_minutes"] = 0.0
    panel["gdp_minutes"] = 0.0
    for ev in events.itertuples(index=False):
        hour_start = panel["utc_hour"]
        hour_end = panel["utc_hour"] + pd.Timedelta(hours=1)
        mask = (panel["airport"] == ev.airport) & (hour_start < ev.end_utc) & (hour_end > ev.start_utc)
        if not mask.any():
            continue
        overlap_start = hour_start[mask].map(lambda x: max(x, ev.start_utc))
        overlap_end = hour_end[mask].map(lambda x: min(x, ev.end_utc))
        minutes = (overlap_end - overlap_start).dt.total_seconds() / 60.0
        panel.loc[mask, "event_minutes"] += minutes.to_numpy()
        panel.loc[mask, "event_count"] += 1
        if ev.tmi_type == "GS":
            panel.loc[mask, "gs_minutes"] += minutes.to_numpy()
        elif ev.tmi_type == "GDP":
            panel.loc[mask, "gdp_minutes"] += minutes.to_numpy()
    panel["event_any"] = (panel["event_minutes"] > 0).astype(float)
    return panel


def run_atcscc_signal_smoke() -> None:
    out = RESULTS / "atcscc_signal_value"
    out.mkdir(parents=True, exist_ok=True)
    airports = list(AIRPORT_TZ)
    panel = build_arrival_hour_panel()
    start = pd.Timestamp("2025-07-01T00:00:00")
    end = pd.Timestamp("2025-07-08T00:00:00")
    panel = panel[(panel["utc_hour"] >= start) & (panel["utc_hour"] < end)].copy()
    full_index = pd.MultiIndex.from_product(
        [airports, pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")], names=["airport", "utc_hour"]
    )
    panel = panel.set_index(["airport", "utc_hour"]).reindex(full_index).reset_index()
    count_cols = ["arrivals", "arr_delay60_count", "cancel_count", "divert_count"]
    panel[count_cols] = panel[count_cols].fillna(0)
    panel[["arr_delay60_rate", "cancel_rate", "divert_rate"]] = panel[
        ["arr_delay60_rate", "cancel_rate", "divert_rate"]
    ].fillna(0)
    weather = read_weather_2025(airports)
    panel = panel.merge(weather, on=["airport", "utc_hour"], how="left")
    panel = add_atcscc_events(panel)
    panel["low_visibility"] = (panel["visibility_km"] < 8).astype(float)
    panel["weather_score"] = (
        panel["wind_speed_mps"].fillna(panel["wind_speed_mps"].median()) / panel["wind_speed_mps"].std(ddof=0)
        + (20 - panel["visibility_km"].fillna(panel["visibility_km"].median())).clip(lower=0) / 20
        + (1000 - panel["ceiling_m"].fillna(panel["ceiling_m"].median())).clip(lower=0) / 1000
    )
    panel["utc_hour_key"] = panel["utc_hour"].dt.strftime("%Y-%m-%d %H")
    panel.to_csv(out / "airport_hour_panel.csv", index=False)

    contrast_rows = []
    for label, sub in [("all", panel), ("has_arrivals", panel[panel["arrivals"] > 0])]:
        for event_value in [0.0, 1.0]:
            g = sub[sub["event_any"] == event_value]
            contrast_rows.append(
                {
                    "sample": label,
                    "event_any": int(event_value),
                    "airport_hours": len(g),
                    "arrivals": int(g["arrivals"].sum()),
                    "event_minutes": round(g["event_minutes"].sum(), 1),
                    "arr_delay60_rate": weighted_mean(g, "arr_delay60_rate", "arrivals"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
                    "mean_arr_delay": weighted_mean(g, "mean_arr_delay", "arrivals"),
                    "p90_arr_delay": weighted_mean(g, "p90_arr_delay", "arrivals"),
                    "weather_score": weighted_mean(g, "weather_score", "arrivals"),
                    "wind_speed_mps": weighted_mean(g, "wind_speed_mps", "arrivals"),
                    "visibility_km": weighted_mean(g, "visibility_km", "arrivals"),
                }
            )
    contrast = pd.DataFrame(contrast_rows)
    contrast.to_csv(out / "event_contrast_summary.csv", index=False)

    bins = panel[panel["arrivals"] > 0].copy()
    bins["weather_tercile"] = pd.qcut(bins["weather_score"], 3, labels=["low", "mid", "high"], duplicates="drop")
    weather_bin = (
        bins.groupby(["weather_tercile", "event_any"], observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "arrivals": g["arrivals"].sum(),
                    "arr_delay60_rate": weighted_mean(g, "arr_delay60_rate", "arrivals"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
                    "mean_arr_delay": weighted_mean(g, "mean_arr_delay", "arrivals"),
                    "weather_score": weighted_mean(g, "weather_score", "arrivals"),
                }
            )
        )
        .reset_index()
    )
    weather_bin.to_csv(out / "event_weather_bin_summary.csv", index=False)

    model_base = panel[(panel["arrivals"] >= 3) & panel["weather_score"].notna()].copy()
    model_base["airport"] = model_base["airport"].astype("category")
    model_base["utc_hour_key"] = model_base["utc_hour_key"].astype("category")
    regs = []
    for y in ["arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]:
        reg = weighted_lstsq(model_base, y, ["event_any", "weather_score", "airport", "utc_hour_key"], "arrivals")
        reg.insert(0, "outcome", y)
        regs.append(reg[reg["term"].isin(["event_any", "weather_score"])])
    reg_summary = pd.concat(regs, ignore_index=True)
    reg_summary.to_csv(out / "event_regression_summary.csv", index=False)

    airport_summary = (
        panel.groupby("airport")
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "event_hours": int((g["event_any"] > 0).sum()),
                    "event_minutes": round(g["event_minutes"].sum(), 1),
                    "arrivals": int(g["arrivals"].sum()),
                    "event_arr_delay60_rate": weighted_mean(g[g["event_any"] > 0], "arr_delay60_rate", "arrivals"),
                    "non_event_arr_delay60_rate": weighted_mean(g[g["event_any"] == 0], "arr_delay60_rate", "arrivals"),
                    "event_cancel_rate": weighted_mean(g[g["event_any"] > 0], "cancel_rate", "arrivals"),
                    "non_event_cancel_rate": weighted_mean(g[g["event_any"] == 0], "cancel_rate", "arrivals"),
                }
            )
        )
        .reset_index()
    )
    airport_summary.to_csv(out / "airport_event_summary.csv", index=False)

    event_coef = reg_summary[(reg_summary["outcome"] == "arr_delay60_rate") & (reg_summary["term"] == "event_any")][
        "coef"
    ].iloc[0]
    cancel_coef = reg_summary[(reg_summary["outcome"] == "cancel_rate") & (reg_summary["term"] == "event_any")][
        "coef"
    ].iloc[0]
    raw_event = contrast[(contrast["sample"] == "has_arrivals") & (contrast["event_any"] == 1)]["arr_delay60_rate"].iloc[0]
    raw_none = contrast[(contrast["sample"] == "has_arrivals") & (contrast["event_any"] == 0)]["arr_delay60_rate"].iloc[0]
    verdict = "usable" if event_coef > 0.03 and raw_event > raw_none else "weak"
    (out / "smoke_assessment.md").write_text(
        "\n".join(
            [
                "# ATCSCC signal-value smoke assessment",
                "",
                f"Verdict: {verdict}.",
                f"Event-hour weighted long-arrival-delay rate: {raw_event:.3f}.",
                f"Non-event weighted long-arrival-delay rate: {raw_none:.3f}.",
                f"Two-way fixed-effect event coefficient on long-arrival-delay rate: {event_coef:.3f}.",
                f"Two-way fixed-effect event coefficient on cancellation rate: {cancel_coef:.3f}.",
                "",
                "Interpretation: a positive event coefficient after airport and UTC-hour controls means the public traffic-management signal carries information beyond common time shocks and airport baseline differences.",
            ]
        ),
        encoding="utf-8",
    )


def runway_headings_for_airports(airports: list[str]) -> dict[str, list[float]]:
    airport_table = pd.read_csv(RAW / "ourairports" / "airports.csv", low_memory=False)
    runway_table = pd.read_csv(RAW / "ourairports" / "runways.csv", low_memory=False)
    us = airport_table[airport_table["iata_code"].isin(airports)].copy()
    ident_by_iata = dict(zip(us["iata_code"], us["ident"]))
    headings = {}
    for iata, ident in ident_by_iata.items():
        sub = runway_table[(runway_table["airport_ident"] == ident) & (runway_table["closed"].fillna(0).astype(int) == 0)]
        vals = []
        for col in ["le_heading_degT", "he_heading_degT"]:
            vals.extend(pd.to_numeric(sub[col], errors="coerce").dropna().tolist())
        headings[iata] = sorted({round(float(v) % 360, 1) for v in vals})
    return headings


def min_angle_to_runway(wind_dir: float, headings: list[float]) -> float:
    if pd.isna(wind_dir) or not headings:
        return np.nan
    diffs = [abs(((wind_dir - h + 180) % 360) - 180) for h in headings]
    return min(diffs)


def build_departure_hour_panel(year: int, month: int) -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    usecols = [
        "FlightDate",
        "Origin",
        "CRSDepTime",
        "DepDelayMinutes",
        "TaxiOut",
        "Cancelled",
        "Diverted",
    ]
    bts = read_bts_month(year, month, usecols)
    bts = bts[bts["Origin"].isin(airports)].copy()
    bts["utc_hour"] = [
        local_schedule_to_utc_hour(row.FlightDate, row.CRSDepTime, row.Origin, False).tz_localize(None)
        if pd.notna(local_schedule_to_utc_hour(row.FlightDate, row.CRSDepTime, row.Origin, False))
        else pd.NaT
        for row in bts.itertuples(index=False)
    ]
    bts = bts.dropna(subset=["utc_hour"])
    bts["dep_delay_min"] = pd.to_numeric(bts["DepDelayMinutes"], errors="coerce")
    bts["taxi_out"] = pd.to_numeric(bts["TaxiOut"], errors="coerce")
    bts["dep_delay60"] = (bts["dep_delay_min"] >= 60).astype(float)
    bts["cancelled"] = pd.to_numeric(bts["Cancelled"], errors="coerce").fillna(0.0)
    bts["diverted"] = pd.to_numeric(bts["Diverted"], errors="coerce").fillna(0.0)
    panel = (
        bts.groupby(["Origin", "utc_hour"], as_index=False)
        .agg(
            departures=("Origin", "size"),
            dep_delay60_count=("dep_delay60", "sum"),
            cancel_count=("cancelled", "sum"),
            mean_dep_delay=("dep_delay_min", "mean"),
            p90_dep_delay=("dep_delay_min", lambda s: float(np.nanpercentile(s, 90)) if s.notna().any() else np.nan),
            mean_taxi_out=("taxi_out", "mean"),
            p90_taxi_out=("taxi_out", lambda s: float(np.nanpercentile(s, 90)) if s.notna().any() else np.nan),
        )
        .rename(columns={"Origin": "airport"})
    )
    panel["dep_delay60_rate"] = panel["dep_delay60_count"] / panel["departures"]
    panel["cancel_rate"] = panel["cancel_count"] / panel["departures"]
    panel["month"] = month
    return panel


def run_runway_wind_smoke() -> None:
    out = RESULTS / "runway_wind_mismatch"
    out.mkdir(parents=True, exist_ok=True)
    airports = list(AIRPORT_TZ)
    panel = pd.concat([build_departure_hour_panel(2025, 7), build_departure_hour_panel(2025, 12)], ignore_index=True)
    weather = read_weather_2025(airports)
    panel = panel.merge(weather, on=["airport", "utc_hour"], how="left")
    headings = runway_headings_for_airports(airports)
    panel["runway_headings"] = panel["airport"].map(lambda a: ";".join(map(str, headings.get(a, []))))
    panel["runway_count"] = panel["airport"].map(lambda a: max(1, len(headings.get(a, [])) // 2))
    panel["min_wind_runway_angle"] = [
        min_angle_to_runway(row.wind_dir_deg, headings.get(row.airport, [])) for row in panel.itertuples(index=False)
    ]
    panel["crosswind_mps"] = panel["wind_speed_mps"] * np.sin(np.deg2rad(panel["min_wind_runway_angle"]))
    panel["aligned_wind_mps"] = panel["wind_speed_mps"] * np.cos(np.deg2rad(panel["min_wind_runway_angle"]))
    panel["low_visibility"] = (panel["visibility_km"] < 8).astype(float)
    panel["local_hour"] = [
        pd.Timestamp(row.utc_hour, tz="UTC").tz_convert(AIRPORT_TZ[row.airport]).hour for row in panel.itertuples(index=False)
    ]
    panel.to_csv(out / "airport_hour_panel.csv", index=False)

    use = panel[(panel["departures"] >= 3) & panel["crosswind_mps"].notna()].copy()
    use["crosswind_quartile"] = pd.qcut(use["crosswind_mps"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    quartile = (
        use.groupby("crosswind_quartile", observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "departures": g["departures"].sum(),
                    "crosswind_mps": weighted_mean(g, "crosswind_mps", "departures"),
                    "dep_delay60_rate": weighted_mean(g, "dep_delay60_rate", "departures"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "departures"),
                    "mean_dep_delay": weighted_mean(g, "mean_dep_delay", "departures"),
                    "p90_taxi_out": weighted_mean(g, "p90_taxi_out", "departures"),
                    "visibility_km": weighted_mean(g, "visibility_km", "departures"),
                }
            )
        )
        .reset_index()
    )
    quartile.to_csv(out / "crosswind_quartile_summary.csv", index=False)

    airport = (
        use.groupby("airport")
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "departures": g["departures"].sum(),
                    "runway_count": g["runway_count"].max(),
                    "median_crosswind_mps": g["crosswind_mps"].median(),
                    "p90_crosswind_mps": g["crosswind_mps"].quantile(0.9),
                    "dep_delay60_rate": weighted_mean(g, "dep_delay60_rate", "departures"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "departures"),
                    "p90_taxi_out": weighted_mean(g, "p90_taxi_out", "departures"),
                }
            )
        )
        .reset_index()
    )
    airport.to_csv(out / "airport_crosswind_summary.csv", index=False)

    use["airport"] = use["airport"].astype("category")
    use["month"] = use["month"].astype("category")
    use["local_hour"] = use["local_hour"].astype("category")
    regs = []
    for y in ["dep_delay60_rate", "cancel_rate", "mean_dep_delay", "p90_taxi_out"]:
        reg = weighted_lstsq(
            use,
            y,
            ["crosswind_mps", "visibility_km", "wind_speed_mps", "airport", "month", "local_hour"],
            "departures",
        )
        reg.insert(0, "outcome", y)
        regs.append(reg[reg["term"].isin(["crosswind_mps", "visibility_km", "wind_speed_mps"])])
    reg_summary = pd.concat(regs, ignore_index=True)
    reg_summary.to_csv(out / "crosswind_regression_summary.csv", index=False)

    q_low = quartile[quartile["crosswind_quartile"] == "Q1_low"]["dep_delay60_rate"].iloc[0]
    q_high = quartile[quartile["crosswind_quartile"] == "Q4_high"]["dep_delay60_rate"].iloc[0]
    xcoef = reg_summary[(reg_summary["outcome"] == "dep_delay60_rate") & (reg_summary["term"] == "crosswind_mps")][
        "coef"
    ].iloc[0]
    verdict = "usable" if (q_high - q_low) > 0.02 and xcoef > 0 else "weak"
    (out / "smoke_assessment.md").write_text(
        "\n".join(
            [
                "# Runway-wind mismatch smoke assessment",
                "",
                f"Verdict: {verdict}.",
                f"Low-crosswind weighted long-departure-delay rate: {q_low:.3f}.",
                f"High-crosswind weighted long-departure-delay rate: {q_high:.3f}.",
                f"Fixed-effect crosswind coefficient on long-departure-delay rate: {xcoef:.4f} per m/s.",
                "",
                "Interpretation: a positive coefficient after airport, month, local-hour, visibility, and total wind controls means runway-relative wind direction carries information beyond generic wind severity.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    run_atcscc_signal_smoke()
    run_runway_wind_smoke()
    print("Smoke tests complete.")
    print(RESULTS / "atcscc_signal_value" / "smoke_assessment.md")
    print(RESULTS / "runway_wind_mismatch" / "smoke_assessment.md")


if __name__ == "__main__":
    main()
