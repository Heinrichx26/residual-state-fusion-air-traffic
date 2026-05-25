from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from math import log
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from build_lightweight_open_data_evidence import reason_category
from fusion_prediction_increment import evaluate, feature_levels, fit_grouped_logit, make_design, sigmoid, train_stats
from fusion_strengthening_common import ROOT_OUT
from smoke_open_fusion_topics import weighted_mean


PROJECT = Path(__file__).resolve().parents[2]


def set_project_root(project_root: str | Path | None) -> Path:
    """Point analysis scripts at a reconstructed project root when data live outside this release package."""
    global PROJECT, ROOT_OUT, RAW_ADVISORY, MAIN_2025_PANEL, MAIN_2024_PANEL, EXTENDED_2025_PANEL
    root = project_root or os.environ.get("DCSI_PROJECT_ROOT")
    PROJECT = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
    ROOT_OUT = PROJECT / "results" / "experiments" / "fusion_framework_strengthening"
    RAW_ADVISORY = PROJECT / "data" / "raw" / "faa_atcscc_advisories"
    MAIN_2025_PANEL = PROJECT / "results" / "experiments" / "atcscc_full_year_windows" / "airport_hour_panel_with_windows.csv"
    MAIN_2024_PANEL = (
        PROJECT
        / "results"
        / "experiments"
        / "supplemental_validation"
        / "cross_year_2024"
        / "full_2024_airport_hour_panel.csv"
    )
    EXTENDED_2025_PANEL = (
        PROJECT
        / "results"
        / "experiments"
        / "supplemental_validation"
        / "extended_airports_2025"
        / "full_30_airports_2025_airport_hour_panel.csv"
    )
    return PROJECT


set_project_root(None)

MAIN_10 = ["ATL", "CLT", "DEN", "DFW", "EWR", "JFK", "LAX", "LGA", "ORD", "SFO"]
EXTENDED_30 = [
    "ATL",
    "BOS",
    "BWI",
    "CLT",
    "DCA",
    "DEN",
    "DFW",
    "DTW",
    "EWR",
    "FLL",
    "HNL",
    "IAD",
    "IAH",
    "JFK",
    "LAS",
    "LAX",
    "LGA",
    "MCO",
    "MDW",
    "MIA",
    "MSP",
    "ORD",
    "PHL",
    "PHX",
    "RDU",
    "SAN",
    "SEA",
    "SFO",
    "SLC",
    "TPA",
]

TARGETS = {"long_arrival_delay": "arr_delay60_count", "cancellation": "cancel_count"}
RHO_GRID_DEFAULT = [0.0, 0.25, 0.50, 0.70, 0.85, 0.93, 0.97]
PLACEBO_SHIFTS = [-336, -168, 168, 336]

CALENDAR = ["month_sin", "month_cos"]
WEATHER = ["weather_score", "mild_weather_abs", "wind_speed_mps", "visibility_km", "ceiling_m", "temperature_c"]
WORKLOAD = ["scheduled_arrivals_proxy", "arrival_bank_intensity"]
FIXED_WINDOW = ["active_hours_capped", "post_3h_hours_capped", "active_mild_strong", "post_3h_mild_strong"]
CONSTRAINT_STATE = [
    "dynamic_constraint_state",
    "dynamic_recovery_state",
    "dynamic_mild_constraint",
    "dynamic_state_delta",
]
BASE_NUMERIC = CALENDAR + WEATHER + WORKLOAD
FIXED_NUMERIC = BASE_NUMERIC + FIXED_WINDOW
DYNAMIC_NUMERIC = FIXED_NUMERIC + CONSTRAINT_STATE
MODEL_NUMERIC = {
    "baseline": BASE_NUMERIC,
    "fixed_window": FIXED_NUMERIC,
    "dynamic_state": DYNAMIC_NUMERIC,
}
MAIN_CATS = ["airport", "local_hour", "day_of_week"]
TRANSFER_CATS = ["local_hour", "day_of_week"]

CONSTRAINT_MAP = {
    "WEATHER_RELATED": "weather constraint",
    "VOLUME_OR_DEMAND": "demand constraint",
    "RUNWAY_OR_SURFACE": "runway or surface constraint",
    "EQUIPMENT_OR_OUTAGE": "facility constraint",
    "STAFFING": "facility constraint",
    "REQUEST": "other constraint",
    "OTHER": "other constraint",
    "MISSING": "other constraint",
}


@dataclass(frozen=True)
class PanelBundle:
    name: str
    year: int
    panel: pd.DataFrame


def parse_months(text: str) -> list[int]:
    months: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            months.extend(range(start, end + 1))
        else:
            months.append(int(part))
    return sorted({m for m in months if 1 <= m <= 12})


def parse_airports(text: str) -> list[str] | None:
    if text.strip().upper() in {"ALL", "*"}:
        return None
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def parse_float_grid(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted({max(0.0, min(0.995, v)) for v in values})


def load_panel(path: Path, year: int, months: list[int], airports: list[str] | None = None) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months)].copy()
    if airports is not None:
        panel = panel[panel["airport"].isin(airports)].copy()
    panel["year"] = year
    panel["row_id"] = np.arange(len(panel), dtype=int)
    panel["active_minutes"] = pd.to_numeric(panel["active_minutes"], errors="coerce").fillna(0.0)
    panel["post_3h_minutes"] = pd.to_numeric(panel["post_3h_minutes"], errors="coerce").fillna(0.0)
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["active_hours_capped"] = (panel["active_minutes"] / 60.0).clip(0, 8)
    panel["post_3h_hours_capped"] = (panel["post_3h_minutes"] / 60.0).clip(0, 8)
    panel["active_mild_strong"] = ((panel["active_strong"] == 1.0) & (panel["mild_weather_abs"] == 1.0)).astype(float)
    panel["post_3h_mild_strong"] = ((panel["post_3h_strong"] == 1.0) & (panel["mild_weather_abs"] == 1.0)).astype(float)
    angle = 2 * np.pi * (panel["month"].astype(float) - 1) / 12.0
    panel["month_sin"] = np.sin(angle)
    panel["month_cos"] = np.cos(angle)
    panel["scheduled_arrivals_proxy"] = pd.to_numeric(panel["arrivals"], errors="coerce").fillna(0.0)
    denom = panel.groupby(["airport", "month"])["scheduled_arrivals_proxy"].transform("mean").replace(0, np.nan)
    panel["arrival_bank_intensity"] = (panel["scheduled_arrivals_proxy"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in ["airport", "local_hour", "day_of_week"]:
        panel[col] = panel[col].astype(str)
    for col in WEATHER:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel[panel["weather_score"].notna()].copy()
    panel["row_id"] = np.arange(len(panel), dtype=int)
    return panel


def model_rows(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()


def dynamic_state_features(panel: pd.DataFrame, rho: float, shift_hours: int = 0, prefix: str = "dynamic") -> pd.DataFrame:
    work = panel[["row_id", "airport", "utc_hour", "active_minutes", "mild_weather_abs"]].copy()
    impulses = work[["airport", "utc_hour", "active_minutes"]].copy()
    impulses["utc_hour"] = impulses["utc_hour"] + pd.to_timedelta(shift_hours, unit="h")
    impulses["active_impulse"] = (impulses["active_minutes"] / 60.0).clip(0, 1)
    impulses = impulses.groupby(["airport", "utc_hour"], as_index=False)["active_impulse"].sum()
    output_frames = []
    for airport, g in work.groupby("airport", sort=False):
        start = g["utc_hour"].min()
        end = g["utc_hour"].max()
        idx = pd.DataFrame({"utc_hour": pd.date_range(start, end, freq="h")})
        idx["airport"] = airport
        idx = idx.merge(impulses[impulses["airport"].eq(airport)], on=["airport", "utc_hour"], how="left")
        impulse_values = idx["active_impulse"].fillna(0.0).clip(0, 1).to_numpy(float)
        state_values = np.zeros(len(idx), dtype=float)
        state = 0.0
        for i, impulse in enumerate(impulse_values):
            state = rho * state + impulse
            state_values[i] = min(state, 24.0)
        idx[f"{prefix}_constraint_state"] = state_values
        idx[f"{prefix}_state_delta"] = np.diff(np.r_[0.0, state_values])
        idx[f"{prefix}_current_impulse"] = impulse_values
        output_frames.append(idx[["airport", "utc_hour", f"{prefix}_constraint_state", f"{prefix}_state_delta", f"{prefix}_current_impulse"]])
    states = pd.concat(output_frames, ignore_index=True)
    out = work.merge(states, on=["airport", "utc_hour"], how="left")
    out[f"{prefix}_constraint_state"] = out[f"{prefix}_constraint_state"].fillna(0.0)
    out[f"{prefix}_state_delta"] = out[f"{prefix}_state_delta"].fillna(0.0)
    out[f"{prefix}_current_impulse"] = out[f"{prefix}_current_impulse"].fillna(0.0)
    out[f"{prefix}_recovery_state"] = (out[f"{prefix}_constraint_state"] - out[f"{prefix}_current_impulse"]).clip(lower=0.0)
    out[f"{prefix}_mild_constraint"] = out[f"{prefix}_constraint_state"] * out["mild_weather_abs"].fillna(0.0)
    cols = [
        "row_id",
        f"{prefix}_constraint_state",
        f"{prefix}_recovery_state",
        f"{prefix}_mild_constraint",
        f"{prefix}_state_delta",
    ]
    return out[cols]


def attach_dynamic(panel: pd.DataFrame, rho: float, shift_hours: int = 0) -> pd.DataFrame:
    features = dynamic_state_features(panel, rho=rho, shift_hours=shift_hours, prefix="dynamic")
    out = panel.merge(features, on="row_id", how="left")
    for col in CONSTRAINT_STATE:
        out[col] = out[col].fillna(0.0)
    return out


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    success_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, dict]:
    levels = feature_levels(train, categorical_cols)
    stats = train_stats(train, numeric_cols)
    x_train = make_design(train, numeric_cols, categorical_cols, stats, levels)
    x_test = make_design(test, numeric_cols, categorical_cols, stats, levels)
    beta = fit_grouped_logit(x_train, train[success_col].to_numpy(float), train["arrivals"].to_numpy(float))
    prob = sigmoid(x_test @ beta)
    pred = test[["row_id", "airport", "utc_hour", "month", "arrivals", success_col]].copy()
    pred["pred_prob"] = prob
    metrics = evaluate(pred[success_col].to_numpy(float), pred["arrivals"].to_numpy(float), prob)
    return pred, metrics


def cv_metric_for_rho(panel_by_rho: dict[float, pd.DataFrame], train_months: list[int], rho: float, cats: list[str]) -> float:
    panel = model_rows(panel_by_rho[rho])
    aucs = []
    for month in train_months:
        inner_train = panel[panel["month"] != month].copy()
        inner_test = panel[panel["month"] == month].copy()
        if inner_train.empty or inner_test.empty:
            continue
        for success_col in TARGETS.values():
            _, metrics = fit_predict(inner_train, inner_test, success_col, DYNAMIC_NUMERIC, cats)
            if np.isfinite(metrics["auc"]):
                aucs.append(metrics["auc"])
    return float(np.mean(aucs)) if aucs else -np.inf


def select_rho_for_outer_fold(
    panel_by_rho: dict[float, pd.DataFrame],
    train_months: list[int],
    rho_grid: list[float],
    cats: list[str],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for rho in rho_grid:
        score = cv_metric_for_rho(panel_by_rho, train_months, rho, cats)
        rows.append({"rho": rho, "inner_mean_auc": score})
    table = pd.DataFrame(rows).sort_values(["inner_mean_auc", "rho"], ascending=[False, False])
    return float(table.iloc[0]["rho"]), table


def evaluate_leave_month(panel: pd.DataFrame, months: list[int], rho_grid: list[float], cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_by_rho = {rho: attach_dynamic(panel, rho) for rho in rho_grid}
    metric_rows = []
    prediction_rows = []
    rho_rows = []
    for outer_month in months:
        train_months = [m for m in months if m != outer_month]
        selected_rho, rho_table = select_rho_for_outer_fold(panel_by_rho, train_months, rho_grid, cats)
        rho_table["outer_month"] = outer_month
        rho_rows.append(rho_table)
        base_panel = model_rows(panel)
        dyn_panel = model_rows(panel_by_rho[selected_rho])
        for target, success_col in TARGETS.items():
            for model_name, numeric_cols in [("baseline", BASE_NUMERIC), ("fixed_window", FIXED_NUMERIC)]:
                train = base_panel[base_panel["month"] != outer_month].copy()
                test = base_panel[base_panel["month"] == outer_month].copy()
                pred, metrics = fit_predict(train, test, success_col, numeric_cols, cats)
                metric_rows.append(metrics | {"target": target, "model": model_name, "fold_month": outer_month, "rho": ""})
                pred["target"] = target
                pred["model"] = model_name
                pred["rho"] = ""
                prediction_rows.append(pred)
            train = dyn_panel[dyn_panel["month"] != outer_month].copy()
            test = dyn_panel[dyn_panel["month"] == outer_month].copy()
            pred, metrics = fit_predict(train, test, success_col, DYNAMIC_NUMERIC, cats)
            metric_rows.append(metrics | {"target": target, "model": "dynamic_state", "fold_month": outer_month, "rho": selected_rho})
            pred["target"] = target
            pred["model"] = "dynamic_state"
            pred["rho"] = selected_rho
            prediction_rows.append(pred)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    rho_selection = pd.concat(rho_rows, ignore_index=True)
    overall_rows = []
    for (target, model), g in predictions.groupby(["target", "model"]):
        success_col = TARGETS[target]
        overall_rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": model, "fold_month": "all", "rho": "fold_selected"}
        )
    metrics = pd.concat([metrics, pd.DataFrame(overall_rows)], ignore_index=True)
    return metrics, predictions, rho_selection


def build_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["fold_month"].astype(str).eq("all")].copy()
    base = overall[overall["model"].eq("baseline")][["target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "baseline_auc", "brier": "baseline_brier", "log_loss": "baseline_log_loss"}
    )
    fixed = overall[overall["model"].eq("fixed_window")][["target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "fixed_auc", "brier": "fixed_brier", "log_loss": "fixed_log_loss"}
    )
    out = overall.merge(base, on="target", how="left").merge(fixed, on="target", how="left")
    out["auc_gain_vs_baseline"] = out["auc"] - out["baseline_auc"]
    out["auc_gain_vs_fixed"] = out["auc"] - out["fixed_auc"]
    out["brier_gain_vs_baseline"] = out["baseline_brier"] - out["brier"]
    out["brier_gain_vs_fixed"] = out["fixed_brier"] - out["brier"]
    out["log_loss_gain_vs_baseline"] = out["baseline_log_loss"] - out["log_loss"]
    out["log_loss_gain_vs_fixed"] = out["fixed_log_loss"] - out["log_loss"]
    return out


def decile_table(panel: pd.DataFrame, rho: float) -> pd.DataFrame:
    scored = model_rows(attach_dynamic(panel, rho))
    scored["state_decile"] = pd.qcut(scored["dynamic_constraint_state"].rank(method="first"), 10, labels=False) + 1
    rows = []
    for decile, g in scored.groupby("state_decile", sort=True):
        rows.append(
            {
                "rho": rho,
                "state_decile": int(decile),
                "airport_hours": len(g),
                "arrivals": int(g["arrivals"].sum()),
                "mean_state": weighted_mean(g, "dynamic_constraint_state", "arrivals"),
                "mean_recovery_state": weighted_mean(g, "dynamic_recovery_state", "arrivals"),
                "delay_rate": weighted_mean(g, "arr_delay60_rate", "arrivals"),
                "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
            }
        )
    return pd.DataFrame(rows)


def placebo_metrics(panel: pd.DataFrame, months: list[int], fold_rho: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
    selected = fold_rho.sort_values(["outer_month", "inner_mean_auc"], ascending=[True, False]).drop_duplicates("outer_month")
    rows = []
    for shift in PLACEBO_SHIFTS:
        panel_by_rho_shift = {
            float(rho): model_rows(attach_dynamic(panel, float(rho), shift_hours=shift))
            for rho in sorted(selected["rho"].unique())
        }
        for outer_month, rho in selected[["outer_month", "rho"]].itertuples(index=False):
            dyn_panel = panel_by_rho_shift[float(rho)]
            for target, success_col in TARGETS.items():
                train = dyn_panel[dyn_panel["month"] != int(outer_month)].copy()
                test = dyn_panel[dyn_panel["month"] == int(outer_month)].copy()
                _, metrics = fit_predict(train, test, success_col, DYNAMIC_NUMERIC, cats)
                rows.append(metrics | {"target": target, "shift_hours": int(shift), "fold_month": int(outer_month), "rho": float(rho)})
    fold_metrics = pd.DataFrame(rows)
    overall = []
    for (target, shift), g in fold_metrics.groupby(["target", "shift_hours"]):
        # Fold-level rows are averaged for placebo because predictions are not retained.
        overall.append(
            {
                "target": target,
                "shift_hours": int(shift),
                "mean_fold_auc": float(g["auc"].mean()),
                "mean_fold_brier": float(g["brier"].mean()),
                "folds": int(g.shape[0]),
            }
        )
    return pd.DataFrame(overall)


def global_rho_from_selection(rho_selection: pd.DataFrame) -> float:
    best = rho_selection.sort_values(["outer_month", "inner_mean_auc"], ascending=[True, False]).drop_duplicates("outer_month")
    return float(best["rho"].median())


def train_test_transfer(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    rho_grid: list[float],
    cats: list[str],
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_months = sorted(train_panel["month"].unique().tolist())
    train_by_rho = {rho: attach_dynamic(train_panel, rho) for rho in rho_grid}
    selected_rho, rho_table = select_rho_for_outer_fold(train_by_rho, train_months, rho_grid, cats)
    train_base = model_rows(train_panel)
    test_base = model_rows(test_panel)
    train_dyn = model_rows(train_by_rho[selected_rho])
    test_dyn = model_rows(attach_dynamic(test_panel, selected_rho))
    rows = []
    for target, success_col in TARGETS.items():
        for model_name, numeric_cols, train_use, test_use in [
            ("baseline", BASE_NUMERIC, train_base, test_base),
            ("fixed_window", FIXED_NUMERIC, train_base, test_base),
            ("dynamic_state", DYNAMIC_NUMERIC, train_dyn, test_dyn),
        ]:
            _, metrics = fit_predict(train_use, test_use, success_col, numeric_cols, cats)
            rows.append(metrics | {"transfer": label, "target": target, "model": model_name, "rho": selected_rho if model_name == "dynamic_state" else ""})
    return pd.DataFrame(rows), rho_table.assign(transfer=label)


def transfer_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    base = metrics[metrics["model"].eq("baseline")][["transfer", "target", "auc", "brier"]].rename(
        columns={"auc": "baseline_auc", "brier": "baseline_brier"}
    )
    fixed = metrics[metrics["model"].eq("fixed_window")][["transfer", "target", "auc", "brier"]].rename(
        columns={"auc": "fixed_auc", "brier": "fixed_brier"}
    )
    out = metrics.merge(base, on=["transfer", "target"], how="left").merge(fixed, on=["transfer", "target"], how="left")
    out["auc_gain_vs_baseline"] = out["auc"] - out["baseline_auc"]
    out["auc_gain_vs_fixed"] = out["auc"] - out["fixed_auc"]
    out["brier_gain_vs_baseline"] = out["baseline_brier"] - out["brier"]
    out["brier_gain_vs_fixed"] = out["fixed_brier"] - out["brier"]
    return out


def load_events(year: int, months: list[int], airports: list[str]) -> pd.DataFrame:
    path = RAW_ADVISORY / f"faa_atcscc_gdp_gs_reparsed_{year}_v2.csv"
    events = pd.read_csv(path)
    events = events[events["airport"].isin(airports)].copy()
    events["start_utc"] = pd.to_datetime(events["start_utc"], utc=True).dt.tz_localize(None)
    events["end_utc"] = pd.to_datetime(events["end_utc"], utc=True).dt.tz_localize(None)
    events = events[events["start_utc"].dt.month.isin(months)].copy()
    events = events.dropna(subset=["start_utc", "end_utc"])
    events["reason_category"] = events["reason"].apply(reason_category)
    events["constraint_type"] = events["reason_category"].map(CONSTRAINT_MAP).fillna("other constraint")
    events = events.drop_duplicates(subset=["airport", "tmi_type", "start_utc", "end_utc", "constraint_type"])
    events["event_id"] = np.arange(len(events), dtype=int)
    return events


def event_peak_tables(panel: pd.DataFrame, events: pd.DataFrame, rho: float, rel_start: int = -6, rel_end: int = 18) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = model_rows(attach_dynamic(panel, rho))
    scored = scored.set_index(["airport", "utc_hour"]).sort_index()
    rows = []
    peak_rows = []
    for ev in events.itertuples(index=False):
        event_rows = []
        for rel in range(rel_start, rel_end + 1):
            hour = pd.Timestamp(ev.start_utc).floor("h") + pd.Timedelta(hours=rel)
            key = (ev.airport, hour)
            if key not in scored.index:
                continue
            rec = scored.loc[key]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            row = {
                "event_id": int(ev.event_id),
                "airport": ev.airport,
                "tmi_type": ev.tmi_type,
                "constraint_type": ev.constraint_type,
                "rel_hour": rel,
                "arrivals": float(rec["arrivals"]),
                "state": float(rec["dynamic_constraint_state"]),
                "recovery_state": float(rec["dynamic_recovery_state"]),
                "delay_rate": float(rec["arr_delay60_rate"]),
                "cancel_rate": float(rec["cancel_rate"]),
            }
            rows.append(row)
            event_rows.append(row)
        if event_rows:
            eg = pd.DataFrame(event_rows)
            peak_rows.append(
                {
                    "event_id": int(ev.event_id),
                    "airport": ev.airport,
                    "tmi_type": ev.tmi_type,
                    "constraint_type": ev.constraint_type,
                    "state_peak_hour": int(eg.loc[eg["state"].idxmax(), "rel_hour"]),
                    "delay_peak_hour": int(eg.loc[eg["delay_rate"].idxmax(), "rel_hour"]),
                    "cancel_peak_hour": int(eg.loc[eg["cancel_rate"].idxmax(), "rel_hour"]),
                    "state_peak": float(eg["state"].max()),
                    "delay_peak": float(eg["delay_rate"].max()),
                    "cancel_peak": float(eg["cancel_rate"].max()),
                    "arrivals": int(eg["arrivals"].sum()),
                }
            )
    event_hour = pd.DataFrame(rows)
    peaks = pd.DataFrame(peak_rows)
    curve_rows = []
    if not event_hour.empty:
        for rel, g in event_hour.groupby("rel_hour"):
            curve_rows.append(
                {
                    "rel_hour": int(rel),
                    "events": int(g["event_id"].nunique()),
                    "arrivals": int(g["arrivals"].sum()),
                    "mean_state": weighted_mean(g, "state", "arrivals"),
                    "mean_recovery_state": weighted_mean(g, "recovery_state", "arrivals"),
                    "delay_rate": weighted_mean(g, "delay_rate", "arrivals"),
                    "cancel_rate": weighted_mean(g, "cancel_rate", "arrivals"),
                }
            )
    return pd.DataFrame(curve_rows), peaks


def type_state_features(panel: pd.DataFrame, events: pd.DataFrame, rho: float) -> pd.DataFrame:
    base = panel[["row_id", "airport", "utc_hour", "mild_weather_abs"]].copy()
    frames = []
    hour_start = base["utc_hour"]
    hour_end = hour_start + pd.Timedelta(hours=1)
    for constraint_type, evs in events.groupby("constraint_type"):
        impulse = pd.DataFrame({"row_id": base["row_id"], "active_minutes": 0.0})
        for ev in evs.itertuples(index=False):
            mask = base["airport"].eq(ev.airport) & (hour_start < ev.end_utc) & (hour_end > ev.start_utc)
            if not mask.any():
                continue
            overlap_start = hour_start[mask].map(lambda x: max(x, ev.start_utc))
            overlap_end = hour_end[mask].map(lambda x: min(x, ev.end_utc))
            impulse.loc[mask.to_numpy(), "active_minutes"] += ((overlap_end - overlap_start).dt.total_seconds() / 60.0).to_numpy()
        tmp = base.merge(impulse, on="row_id", how="left")
        tmp["active_minutes"] = tmp["active_minutes"].fillna(0.0).clip(0, 60)
        tmp = tmp.rename(columns={"active_minutes": "active_minutes_type"})
        state_panel = panel[["row_id", "airport", "utc_hour", "mild_weather_abs"]].merge(
            tmp[["row_id", "active_minutes_type"]], on="row_id", how="left"
        )
        state_panel = state_panel.rename(columns={"active_minutes_type": "active_minutes"})
        features = dynamic_state_features(state_panel, rho=rho, shift_hours=0, prefix="type")
        features["constraint_type"] = constraint_type
        frames.append(features)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reason_memory_table(panel: pd.DataFrame, events: pd.DataFrame, rho: float) -> pd.DataFrame:
    features = type_state_features(panel, events, rho)
    if features.empty:
        return pd.DataFrame()
    scored = model_rows(panel).merge(features, on="row_id", how="inner")
    rows = []
    for constraint_type, g in scored.groupby("constraint_type"):
        positive = g[g["type_constraint_state"] > 0].copy()
        if positive.empty:
            continue
        threshold = positive["type_constraint_state"].quantile(0.9)
        top = positive[positive["type_constraint_state"] >= threshold]
        rows.append(
            {
                "constraint_type": constraint_type,
                "rho": rho,
                "implied_half_life_hours": log(0.5) / log(rho) if 0 < rho < 1 else 0.0,
                "positive_airport_hours": int(len(positive)),
                "top_state_airport_hours": int(len(top)),
                "top_state_arrivals": int(top["arrivals"].sum()),
                "mean_top_state": weighted_mean(top, "type_constraint_state", "arrivals"),
                "delay_rate_top_state": weighted_mean(top, "arr_delay60_rate", "arrivals"),
                "cancel_rate_top_state": weighted_mean(top, "cancel_rate", "arrivals"),
                "recovery_state_area": float(positive["type_recovery_state"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("recovery_state_area", ascending=False)


def write_assessment(out: Path, gains: pd.DataFrame, deciles: pd.DataFrame, placebo: pd.DataFrame, transfer: pd.DataFrame, peaks: pd.DataFrame, reason: pd.DataFrame) -> None:
    dyn_delay = gains[(gains["model"].eq("dynamic_state")) & (gains["target"].eq("long_arrival_delay"))].iloc[0]
    dyn_cancel = gains[(gains["model"].eq("dynamic_state")) & (gains["target"].eq("cancellation"))].iloc[0]
    low = deciles[deciles["state_decile"].eq(1)].iloc[0]
    high = deciles[deciles["state_decile"].eq(10)].iloc[0]
    delay_ratio = float(high["delay_rate"] / low["delay_rate"]) if float(low["delay_rate"]) > 0 else np.nan
    cancel_ratio = float(high["cancel_rate"] / low["cancel_rate"]) if float(low["cancel_rate"]) > 0 else np.nan
    strongest_transfer = (
        transfer[(transfer["model"].eq("dynamic_state"))].copy()
        if not transfer.empty and "model" in transfer.columns
        else pd.DataFrame()
    )
    top_reason = reason.head(1).iloc[0] if not reason.empty else None
    peak_summary = ""
    if not peaks.empty:
        peak_summary = (
            f"- Median peak timing: state {peaks['state_peak_hour'].median():.1f} h, "
            f"delay {peaks['delay_peak_hour'].median():.1f} h, cancellation {peaks['cancel_peak_hour'].median():.1f} h."
        )
    lines = [
        "# Dynamic constraint-state inversion assessment",
        "",
        "The experiment treats traffic-management advisories as observed action trajectories and estimates an airport-hour constraint state from action persistence.",
        "",
        f"- Dynamic state AUC gain over fixed windows: delay {float(dyn_delay['auc_gain_vs_fixed']):+.3f}; cancellation {float(dyn_cancel['auc_gain_vs_fixed']):+.3f}.",
        f"- Dynamic state AUC gain over baseline: delay {float(dyn_delay['auc_gain_vs_baseline']):+.3f}; cancellation {float(dyn_cancel['auc_gain_vs_baseline']):+.3f}.",
        f"- Top/bottom state decile ratios: delay {delay_ratio:.2f}; cancellation {cancel_ratio:.2f}.",
        f"- Placebo mean AUC range: {float(placebo['mean_fold_auc'].min()):.3f} to {float(placebo['mean_fold_auc'].max()):.3f}.",
        peak_summary,
    ]
    for row in strongest_transfer.itertuples(index=False):
        lines.append(
            f"- Transfer {row.transfer}, {row.target}: dynamic AUC gain over fixed {float(row.auc_gain_vs_fixed):+.3f}; over baseline {float(row.auc_gain_vs_baseline):+.3f}."
        )
    if top_reason is not None:
        lines.append(
            f"- Largest semantic recovery area: {top_reason['constraint_type']} with {float(top_reason['recovery_state_area']):.1f} state-hours."
        )
    out.joinpath("dcsi_assessment.md").write_text("\n".join([line for line in lines if line]) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    months = parse_months(args.months)
    airports = parse_airports(args.airports) or MAIN_10
    rho_grid = parse_float_grid(args.rho_grid)
    out = ROOT_OUT / args.output_name
    out.mkdir(parents=True, exist_ok=True)

    main_2025 = load_panel(MAIN_2025_PANEL, 2025, months, airports)
    metrics, predictions, rho_selection = evaluate_leave_month(main_2025, months, rho_grid, MAIN_CATS)
    gains = build_gain_table(metrics)
    rho_global = global_rho_from_selection(rho_selection)
    deciles = decile_table(main_2025, rho_global)
    placebo = placebo_metrics(main_2025, months, rho_selection, MAIN_CATS)

    transfer_metrics = []
    transfer_rho = []
    if args.include_transfer:
        use_months_2024 = months
        main_2024 = load_panel(MAIN_2024_PANEL, 2024, use_months_2024, airports)
        metrics_2425, rho_2425 = train_test_transfer(main_2024, main_2025, rho_grid, MAIN_CATS, "train2024_test2025")
        transfer_metrics.append(metrics_2425)
        transfer_rho.append(rho_2425)
        extended = load_panel(EXTENDED_2025_PANEL, 2025, months, EXTENDED_30)
        extension_20 = extended[~extended["airport"].isin(MAIN_10)].copy()
        metrics_ext, rho_ext = train_test_transfer(main_2025, extension_20, rho_grid, TRANSFER_CATS, "main10_to_extension20")
        transfer_metrics.append(metrics_ext)
        transfer_rho.append(rho_ext)
    transfer_all = pd.concat(transfer_metrics, ignore_index=True) if transfer_metrics else pd.DataFrame()
    transfer_gain = transfer_gains(transfer_all) if not transfer_all.empty else pd.DataFrame()
    transfer_rho_all = pd.concat(transfer_rho, ignore_index=True) if transfer_rho else pd.DataFrame()

    events = load_events(2025, months, airports)
    event_curve, event_peaks = event_peak_tables(main_2025, events, rho_global)
    reason_memory = reason_memory_table(main_2025, events, rho_global)

    metrics.to_csv(out / "dcsi_cv_metrics.csv", index=False)
    predictions.to_csv(out / "dcsi_cv_predictions.csv", index=False)
    rho_selection.to_csv(out / "dcsi_rho_selection.csv", index=False)
    gains.to_csv(out / "dcsi_cv_gains.csv", index=False)
    deciles.to_csv(out / "dcsi_deciles.csv", index=False)
    placebo.to_csv(out / "dcsi_placebo_shift.csv", index=False)
    transfer_all.to_csv(out / "dcsi_transfer_metrics.csv", index=False)
    transfer_gain.to_csv(out / "dcsi_transfer_gains.csv", index=False)
    transfer_rho_all.to_csv(out / "dcsi_transfer_rho_selection.csv", index=False)
    event_curve.to_csv(out / "dcsi_event_peak_curve.csv", index=False)
    event_peaks.to_csv(out / "dcsi_event_peak_summary.csv", index=False)
    reason_memory.to_csv(out / "dcsi_reason_memory.csv", index=False)
    write_assessment(out, gains, deciles, placebo, transfer_gain, event_peaks, reason_memory)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=None, help="Project root containing reconstructed data/ and results/ directories.")
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ATL,ORD")
    parser.add_argument("--rho-grid", default=",".join(str(x) for x in RHO_GRID_DEFAULT))
    parser.add_argument("--output-name", default="dynamic_constraint_state_inversion_smoke")
    parser.add_argument("--include-transfer", action="store_true")
    args = parser.parse_args()
    set_project_root(args.project_root)
    run(args)


if __name__ == "__main__":
    main()
