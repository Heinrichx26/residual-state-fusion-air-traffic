from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from full_year_atcscc_window_experiments import scheduled_utc_hour
from smoke_open_fusion_topics import AIRPORT_TZ, read_bts_month


PROJECT = Path(os.environ.get("DCSI_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
MAIN = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
ROOT_OUT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
PANEL_FILE = MAIN / "airport_hour_panel_with_windows.csv"
AIRPORTS = list(AIRPORT_TZ)


def hhi(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    total = counts.sum()
    if total <= 0:
        return np.nan
    shares = counts / total
    return float((shares * shares).sum())


def read_bts_schedule(year: int, month: int) -> pd.DataFrame:
    usecols = [
        "FlightDate",
        "Origin",
        "Dest",
        "CRSDepTime",
        "CRSArrTime",
        "Reporting_Airline",
        "Cancelled",
        "ArrDelayMinutes",
    ]
    return read_bts_month(year, month, usecols)


def bts_demand_month(year: int, month: int) -> pd.DataFrame:
    bts = read_bts_schedule(year, month)
    dep_int = pd.to_numeric(bts["CRSDepTime"], errors="coerce")
    arr_int = pd.to_numeric(bts["CRSArrTime"], errors="coerce")
    add_arr_day = (arr_int < dep_int).fillna(False)

    arr = bts[bts["Dest"].isin(AIRPORTS)].copy()
    arr["utc_hour"] = scheduled_utc_hour(arr, "Dest", "CRSArrTime", add_arr_day.loc[arr.index])
    arr = arr.dropna(subset=["utc_hour"])
    arr_g = (
        arr.groupby(["Dest", "utc_hour"], as_index=False)
        .agg(scheduled_arrivals=("Dest", "size"), arrival_carrier_hhi=("Reporting_Airline", hhi))
        .rename(columns={"Dest": "airport"})
    )

    dep = bts[bts["Origin"].isin(AIRPORTS)].copy()
    dep["utc_hour"] = scheduled_utc_hour(dep, "Origin", "CRSDepTime", pd.Series(False, index=dep.index))
    dep = dep.dropna(subset=["utc_hour"])
    dep_g = (
        dep.groupby(["Origin", "utc_hour"], as_index=False)
        .agg(scheduled_departures=("Origin", "size"), departure_carrier_hhi=("Reporting_Airline", hhi))
        .rename(columns={"Origin": "airport"})
    )

    demand = arr_g.merge(dep_g, on=["airport", "utc_hour"], how="outer")
    demand["month"] = month
    return demand


def add_demand_features(panel: pd.DataFrame, year: int, months: list[int]) -> pd.DataFrame:
    demand = pd.concat([bts_demand_month(year, m) for m in months], ignore_index=True)
    panel = panel.merge(demand, on=["airport", "utc_hour", "month"], how="left")
    for col in ["scheduled_arrivals", "scheduled_departures", "arrival_carrier_hhi", "departure_carrier_hhi"]:
        panel[col] = panel[col].fillna(0.0)
    panel["arrival_bank_intensity"] = bank_intensity(panel, "scheduled_arrivals")
    panel["departure_bank_intensity"] = bank_intensity(panel, "scheduled_departures")
    panel = panel.sort_values(["airport", "utc_hour"]).copy()
    panel["prior_hour_delay60_rate"] = panel.groupby("airport")["arr_delay60_rate"].shift(1).fillna(0.0)
    panel["prior_hour_cancel_rate"] = panel.groupby("airport")["cancel_rate"].shift(1).fillna(0.0)
    panel["prior_hour_arrivals"] = panel.groupby("airport")["arrivals"].shift(1).fillna(0.0)
    return panel


def bank_intensity(panel: pd.DataFrame, col: str) -> pd.Series:
    mean_col = panel.groupby(["airport", "month"])[col].transform("mean").replace(0, np.nan)
    return (panel[col] / mean_col).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def prepare_base_panel(year: int, months: list[int]) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months) & (panel["arrivals"] > 0)].copy()
    panel = panel[panel["weather_score"].notna()].copy()
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["active_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 1.0)
    ).astype(float)
    panel["post_3h_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 1.0)
    ).astype(float)
    panel["active_hours_capped"] = (panel["active_minutes"] / 60).clip(0, 8)
    panel["post_3h_hours_capped"] = (panel["post_3h_minutes"] / 60).clip(0, 8)
    angle = 2 * np.pi * (panel["month"].astype(float) - 1) / 12.0
    panel["month_sin"] = np.sin(angle)
    panel["month_cos"] = np.cos(angle)
    for col in ["airport", "local_hour", "day_of_week"]:
        panel[col] = panel[col].astype(str)
    return add_demand_features(panel, year, months)


def write_demand_audit(panel: pd.DataFrame, path: Path) -> None:
    audit_cols = [
        "scheduled_arrivals",
        "scheduled_departures",
        "arrival_bank_intensity",
        "departure_bank_intensity",
        "arrival_carrier_hhi",
        "departure_carrier_hhi",
    ]
    panel.groupby("month")[audit_cols].agg(["count", "mean", "max"]).to_csv(path)
