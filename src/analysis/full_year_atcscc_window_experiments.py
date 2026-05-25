from pathlib import Path

import numpy as np
import pandas as pd

from smoke_open_fusion_topics import AIRPORT_TZ, read_bts_month, weighted_lstsq, weighted_mean


PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"
OUT = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
EVENT_FILE = RAW / "faa_atcscc_advisories" / "faa_atcscc_gdp_gs_reparsed_2025_v2.csv"
MONTHS = list(range(1, 13))
WINDOWS = {
    "active": (0, 0),
    "post_1h": (0, 1),
    "post_2h": (0, 2),
    "post_3h": (0, 3),
    "around_1h": (1, 1),
    "around_2h": (2, 2),
}


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("M", np.nan), errors="coerce")


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
    for airport, tz in AIRPORT_TZ.items():
        mask = df[airport_col].eq(airport) & local_naive.notna()
        if not mask.any():
            continue
        localized = local_naive.loc[mask].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        out.loc[mask] = localized.dt.tz_convert("UTC").dt.floor("h").dt.tz_localize(None)
    return out


def build_month_arrival_panel(month: int) -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    usecols = [
        "FlightDate",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "ArrDelayMinutes",
        "Cancelled",
        "Diverted",
    ]
    bts = read_bts_month(2025, month, usecols)
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


def complete_airport_hour_grid(panel: pd.DataFrame) -> pd.DataFrame:
    airports = list(AIRPORT_TZ)
    grids = []
    for month in MONTHS:
        start = pd.Timestamp(year=2025, month=month, day=1)
        end = start + pd.offsets.MonthBegin(1)
        idx = pd.MultiIndex.from_product(
            [airports, pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")],
            names=["airport", "utc_hour"],
        )
        g = pd.DataFrame(index=idx).reset_index()
        g["month"] = month
        grids.append(g)
    grid = pd.concat(grids, ignore_index=True)
    merged = grid.merge(panel, on=["airport", "utc_hour", "month"], how="left")
    count_cols = ["arrivals", "arr_delay60_count", "cancel_count", "divert_count"]
    merged[count_cols] = merged[count_cols].fillna(0)
    for col in ["arr_delay60_rate", "cancel_rate", "divert_rate"]:
        merged[col] = merged[col].fillna(0.0)
    return merged


def read_iem_weather() -> pd.DataFrame:
    frames = []
    for month in MONTHS:
        for airport in AIRPORT_TZ:
            path = RAW / "iem_asos" / f"iem_asos_2025_{month:02d}_{airport}.csv"
            df = pd.read_csv(path, low_memory=False)
            df["airport"] = airport
            df["utc_hour"] = pd.to_datetime(df["valid"], utc=True).dt.floor("h").dt.tz_localize(None)
            df["wind_dir_deg"] = to_float(df["drct"])
            df["wind_speed_mps"] = to_float(df["sknt"]) * 0.514444
            df["visibility_km"] = to_float(df["vsby"]) * 1.60934
            df["ceiling_m"] = to_float(df["skyl1"]) * 0.3048
            df["temperature_c"] = (to_float(df["tmpf"]) - 32.0) * 5.0 / 9.0
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
            ceiling_m=("ceiling_m", "min"),
            temperature_c=("temperature_c", "mean"),
        )
    )


def add_weather(panel: pd.DataFrame) -> pd.DataFrame:
    weather = read_iem_weather()
    panel = panel.merge(weather, on=["airport", "utc_hour"], how="left")
    wind_sd = panel["wind_speed_mps"].std(ddof=0)
    wind_sd = wind_sd if wind_sd > 0 else 1.0
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
        pd.Timestamp(row.utc_hour, tz="UTC").tz_convert(AIRPORT_TZ[row.airport]).hour
        for row in panel.itertuples(index=False)
    ]
    return panel


def read_events() -> pd.DataFrame:
    events = pd.read_csv(EVENT_FILE)
    events = events[events["airport"].isin(AIRPORT_TZ)].copy()
    events = events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "source_url"])
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True).dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True).dt.tz_localize(None)
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
    return panel


def summarize_by_state(panel: pd.DataFrame, window: str, group_cols: list[str]) -> pd.DataFrame:
    strong_col = f"{window}_strong"
    state_col = f"{window}_state"
    use = panel.copy()
    use[state_col] = np.select(
        [
            (use["mild_weather_abs"] == 1.0) & (use[strong_col] == 1.0),
            (use["mild_weather_abs"] == 1.0) & (use[strong_col] == 0.0),
            (use["mild_weather_abs"] == 0.0) & (use[strong_col] == 1.0),
        ],
        ["mild_weather_strong_advisory", "mild_weather_no_strong_advisory", "nonmild_weather_strong_advisory"],
        default="nonmild_weather_no_strong_advisory",
    )
    return (
        use.groupby(group_cols + [state_col], observed=False)
        .apply(
            lambda g: pd.Series(
                {
                    "airport_hours": len(g),
                    "hours_with_arrivals": int((g["arrivals"] > 0).sum()),
                    "arrivals": int(g["arrivals"].sum()),
                    "window_minutes": round(g[f"{window}_minutes"].sum(), 1),
                    "arr_delay60_rate": weighted_mean(g, "arr_delay60_rate", "arrivals"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
                    "mean_arr_delay": weighted_mean(g, "mean_arr_delay", "arrivals"),
                    "p90_arr_delay": weighted_mean(g, "p90_arr_delay", "arrivals"),
                    "weather_score": weighted_mean(g, "weather_score", "arrivals"),
                    "visibility_km": weighted_mean(g, "visibility_km", "arrivals"),
                    "wind_speed_mps": weighted_mean(g, "wind_speed_mps", "arrivals"),
                    "ceiling_m": weighted_mean(g, "ceiling_m", "arrivals"),
                }
            )
        )
        .reset_index()
        .rename(columns={state_col: "state"})
    )


def window_regression(panel: pd.DataFrame, window: str) -> pd.DataFrame:
    model = panel[(panel["arrivals"] >= 3) & panel["weather_score"].notna()].copy()
    strong_col = f"{window}_strong"
    conflict_col = f"{window}_mild_strong_conflict"
    for col in ["airport", "month", "local_hour", "day_of_week"]:
        model[col] = model[col].astype("category")
    regs = []
    for y in ["arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]:
        base = weighted_lstsq(
            model,
            y,
            ["weather_score", "mild_weather_abs", "airport", "month", "local_hour", "day_of_week"],
            "arrivals",
        )
        full = weighted_lstsq(
            model,
            y,
            [
                strong_col,
                conflict_col,
                "weather_score",
                "mild_weather_abs",
                "airport",
                "month",
                "local_hour",
                "day_of_week",
            ],
            "arrivals",
        )
        keep_terms = [strong_col, conflict_col, "weather_score", "mild_weather_abs"]
        keep = full[full["term"].isin(keep_terms)].copy()
        keep.insert(0, "window", window)
        keep.insert(1, "outcome", y)
        keep["baseline_weighted_r2"] = base["weighted_r2"].iloc[0]
        keep["with_advisory_weighted_r2"] = full["weighted_r2"].iloc[0]
        keep["r2_gain"] = keep["with_advisory_weighted_r2"] - keep["baseline_weighted_r2"]
        regs.append(keep)
    return pd.concat(regs, ignore_index=True)


def run_windows(panel: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    window_score_rows = []
    all_state = []
    all_month = []
    all_delta = []
    all_regs = []
    for window in WINDOWS:
        strong_col = f"{window}_strong"
        conflict_col = f"{window}_mild_strong_conflict"
        panel[strong_col] = (panel[f"{window}_minutes"] >= 45).astype(float)
        panel[conflict_col] = ((panel["mild_weather_abs"] == 1.0) & (panel[strong_col] == 1.0)).astype(float)

        state = summarize_by_state(panel, window, [])
        state.insert(0, "window", window)
        all_state.append(state)
        month = summarize_by_state(panel, window, ["month"])
        month.insert(0, "window", window)
        all_month.append(month)

        conflict = state[state["state"] == "mild_weather_strong_advisory"].iloc[0]
        mild_no = state[state["state"] == "mild_weather_no_strong_advisory"].iloc[0]
        delta_rate = conflict["arr_delay60_rate"] - mild_no["arr_delay60_rate"]
        delta_cancel = conflict["cancel_rate"] - mild_no["cancel_rate"]

        month_conflict = month[month["state"] == "mild_weather_strong_advisory"].copy()
        month_mild_no = month[month["state"] == "mild_weather_no_strong_advisory"][
            ["month", "arr_delay60_rate", "cancel_rate", "mean_arr_delay", "p90_arr_delay"]
        ].rename(
            columns={
                "arr_delay60_rate": "mild_no_arr_delay60_rate",
                "cancel_rate": "mild_no_cancel_rate",
                "mean_arr_delay": "mild_no_mean_arr_delay",
                "p90_arr_delay": "mild_no_p90_arr_delay",
            }
        )
        delta = month_conflict.merge(month_mild_no, on="month", how="left")
        delta["delta_arr_delay60_rate"] = delta["arr_delay60_rate"] - delta["mild_no_arr_delay60_rate"]
        delta["delta_cancel_rate"] = delta["cancel_rate"] - delta["mild_no_cancel_rate"]
        delta["delta_mean_arr_delay"] = delta["mean_arr_delay"] - delta["mild_no_mean_arr_delay"]
        delta["delta_p90_arr_delay"] = delta["p90_arr_delay"] - delta["mild_no_p90_arr_delay"]
        all_delta.append(delta)

        reg = window_regression(panel, window)
        all_regs.append(reg)
        strong_coef = reg[(reg["outcome"] == "arr_delay60_rate") & (reg["term"] == strong_col)]["coef"].iloc[0]
        interaction_coef = reg[(reg["outcome"] == "arr_delay60_rate") & (reg["term"] == conflict_col)]["coef"].iloc[0]
        r2_gain = reg[(reg["outcome"] == "arr_delay60_rate") & (reg["term"] == strong_col)]["r2_gain"].iloc[0]
        month_positive = int((delta["delta_arr_delay60_rate"] > 0).sum())
        month_usable = int((delta["arrivals"] >= 200).sum())
        window_score_rows.append(
            {
                "window": window,
                "conflict_airport_hours": int(conflict["airport_hours"]),
                "conflict_arrivals": int(conflict["arrivals"]),
                "conflict_arr_delay60_rate": conflict["arr_delay60_rate"],
                "mild_no_arr_delay60_rate": mild_no["arr_delay60_rate"],
                "delta_arr_delay60_rate": delta_rate,
                "conflict_cancel_rate": conflict["cancel_rate"],
                "mild_no_cancel_rate": mild_no["cancel_rate"],
                "delta_cancel_rate": delta_cancel,
                "months_positive_delta": month_positive,
                "months_with_200_conflict_arrivals": month_usable,
                "strong_coef": strong_coef,
                "interaction_coef": interaction_coef,
                "combined_mild_strong_effect": strong_coef + interaction_coef,
                "r2_gain_delay60": r2_gain,
            }
        )

    pd.concat(all_state, ignore_index=True).to_csv(OUT / "window_state_summary.csv", index=False)
    pd.concat(all_month, ignore_index=True).to_csv(OUT / "window_month_state_summary.csv", index=False)
    pd.concat(all_delta, ignore_index=True).to_csv(OUT / "window_month_mild_conflict_delta.csv", index=False)
    pd.concat(all_regs, ignore_index=True).to_csv(OUT / "window_regression_summary.csv", index=False)
    score = pd.DataFrame(window_score_rows).sort_values(
        ["months_positive_delta", "combined_mild_strong_effect", "conflict_arrivals"], ascending=False
    )
    score.to_csv(OUT / "window_scorecard.csv", index=False)

    best = score.iloc[0]
    lines = [
        "# Full-year ATCSCC window experiment",
        "",
        f"Best window by persistence and combined effect: {best['window']}.",
        f"Conflict arrivals: {int(best['conflict_arrivals'])}.",
        f"Conflict long-arrival-delay rate: {best['conflict_arr_delay60_rate']:.3f}.",
        f"Mild-weather no-strong-advisory long-arrival-delay rate: {best['mild_no_arr_delay60_rate']:.3f}.",
        f"Difference: {best['delta_arr_delay60_rate']:.3f}.",
        f"Conflict cancellation rate: {best['conflict_cancel_rate']:.3f}.",
        f"Mild-weather no-strong-advisory cancellation rate: {best['mild_no_cancel_rate']:.3f}.",
        f"Months with positive conflict delay delta: {int(best['months_positive_delta'])}/12.",
        f"Months with at least 200 conflict arrivals: {int(best['months_with_200_conflict_arrivals'])}/12.",
        f"Combined mild-weather strong-advisory effect: {best['combined_mild_strong_effect']:.3f}.",
        f"Weighted R2 gain for long-arrival-delay rate: {best['r2_gain_delay60']:.3f}.",
        "",
        "The scorecard compares active and longer advisory windows. A longer post-advisory window is useful when disruption materializes after the formal advisory start and persists after the formal end.",
    ]
    (OUT / "full_year_window_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []
    for month in MONTHS:
        print(f"building BTS arrival panel 2025-{month:02d}", flush=True)
        panels.append(build_month_arrival_panel(month))
    panel = pd.concat(panels, ignore_index=True)
    panel = complete_airport_hour_grid(panel)
    panel = add_weather(panel)
    events = read_events()
    panel = add_window_minutes(panel, events)
    panel.to_csv(OUT / "airport_hour_panel_with_windows.csv", index=False)
    run_windows(panel)
    print(OUT / "full_year_window_assessment.md", flush=True)


if __name__ == "__main__":
    main()
