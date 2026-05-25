from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dynamic_constraint_state_inversion import MAIN_10, MAIN_2025_PANEL, parse_float_grid, parse_months
from fusion_prediction_increment import evaluate, feature_levels, fit_grouped_logit, make_design, sigmoid, train_stats
from fusion_strengthening_common import ROOT_OUT


PROJECT = Path(__file__).resolve().parents[2]
ISSUE = ROOT_OUT / "issue_time_full_2025" / "advisory_issue_times.csv"
OUT_ROOT = ROOT_OUT / "dcsi_online_lead_validation"

TARGETS = {
    "long_arrival_delay": "target_arr_delay60_count",
    "cancellation": "target_cancel_count",
}

WEATHER = [
    "weather_score",
    "mild_weather_abs",
    "wind_speed_mps",
    "visibility_km",
    "ceiling_m",
    "temperature_c",
]
BASE_NUMERIC = [
    "target_month_sin",
    "target_month_cos",
    "target_arrivals_proxy",
    "target_arrival_bank_intensity",
] + WEATHER
START_NUMERIC = BASE_NUMERIC + [
    "known_start_impulse",
    "known_start_mild",
    "issued_future_start",
    "issued_future_start_mild",
]
ONLINE_STATE = [
    "online_constraint_state",
    "online_recovery_state",
    "online_mild_constraint",
    "online_state_delta",
]
RELATIONS = [
    "hours_since_known_start",
    "recent_start_clock",
    "issued_future_start",
    "issued_future_start_mild",
]
FIXED_DECAY_COLS = [
    "fixed090_constraint_state",
    "fixed090_recovery_state",
    "fixed090_mild_constraint",
    "fixed090_state_delta",
    "fixed095_constraint_state",
    "fixed095_recovery_state",
    "fixed095_mild_constraint",
    "fixed095_state_delta",
]
CATS = ["airport", "target_local_hour", "target_day_of_week"]


@dataclass(frozen=True)
class LeadSpec:
    horizon: int
    months: list[int]
    airports: list[str] | None
    rho_grid: list[float]


def parse_airports(text: str) -> list[str] | None:
    if text.strip().upper() in {"ALL", "*"}:
        return None
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def ceil_hour(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).tz_localize(None) if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts)
    floor = ts.floor("h")
    if ts == floor:
        return floor
    return floor + pd.Timedelta(hours=1)


def load_base_panel(months: list[int], airports: list[str] | None) -> pd.DataFrame:
    panel = pd.read_csv(MAIN_2025_PANEL, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months)].copy()
    if airports is not None:
        panel = panel[panel["airport"].isin(airports)].copy()
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    panel = panel.sort_values(["airport", "utc_hour"]).reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel), dtype=int)
    angle = 2 * np.pi * (panel["month"].astype(float) - 1) / 12.0
    panel["month_sin"] = np.sin(angle)
    panel["month_cos"] = np.cos(angle)
    panel["target_arrivals_proxy"] = pd.to_numeric(panel["arrivals"], errors="coerce").fillna(0.0)
    denom = panel.groupby(["airport", "month"])["target_arrivals_proxy"].transform("mean").replace(0, np.nan)
    panel["arrival_bank_intensity"] = (panel["target_arrivals_proxy"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in ["airport", "local_hour", "day_of_week"]:
        panel[col] = panel[col].astype(str)
    for col in WEATHER:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0.0)
    return panel


def load_issue_events(airports: list[str] | None) -> pd.DataFrame:
    events = pd.read_csv(ISSUE)
    if airports is not None:
        events = events[events["airport"].isin(airports)].copy()
    for col in ["issue_utc", "effective_start_utc"]:
        events[col] = pd.to_datetime(events[col], utc=True, errors="coerce").dt.tz_localize(None)
    events = events.dropna(subset=["airport", "issue_utc", "effective_start_utc"]).copy()
    events["availability_hour"] = events.apply(
        lambda row: ceil_hour(max(row["issue_utc"], row["effective_start_utc"])),
        axis=1,
    )
    events = events.drop_duplicates(subset=["airport", "tmi_type", "issue_utc", "effective_start_utc", "availability_hour"])
    return events


def attach_online_action_fields(panel: pd.DataFrame, events: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = panel.copy()
    for col in ["known_start_impulse", "issued_future_start"]:
        out[col] = 0.0
    indexed = out.set_index(["airport", "utc_hour"])
    start_counts = events.groupby(["airport", "availability_hour"]).size()
    for key, value in start_counts.items():
        if key in indexed.index:
            indexed.loc[key, "known_start_impulse"] += float(value)
    out = indexed.reset_index()
    out["known_start_impulse"] = out["known_start_impulse"].clip(0, 1)

    hour_start = out["utc_hour"]
    for ev in events.itertuples(index=False):
        mask = (
            out["airport"].eq(ev.airport)
            & (ev.issue_utc <= hour_start)
            & (ev.effective_start_utc > hour_start)
            & (ev.effective_start_utc <= hour_start + pd.to_timedelta(horizon, unit="h"))
        )
        if mask.any():
            out.loc[mask, "issued_future_start"] += 1.0
    out["issued_future_start"] = out["issued_future_start"].clip(0, 1)
    out["known_start_mild"] = out["known_start_impulse"] * out["mild_weather_abs"].fillna(0.0)
    out["issued_future_start_mild"] = out["issued_future_start"] * out["mild_weather_abs"].fillna(0.0)
    return out


def add_target_shift(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    targets = panel[
        [
            "airport",
            "utc_hour",
            "arrivals",
            "arr_delay60_count",
            "cancel_count",
            "local_hour",
            "day_of_week",
            "month",
            "month_sin",
            "month_cos",
            "arrival_bank_intensity",
        ]
    ].copy()
    targets["utc_hour"] = targets["utc_hour"] - pd.Timedelta(hours=horizon)
    targets = targets.rename(
        columns={
            "arrivals": "target_arrivals",
            "arr_delay60_count": "target_arr_delay60_count",
            "cancel_count": "target_cancel_count",
            "local_hour": "target_local_hour",
            "day_of_week": "target_day_of_week",
            "month": "target_month",
            "month_sin": "target_month_sin",
            "month_cos": "target_month_cos",
            "arrival_bank_intensity": "target_arrival_bank_intensity",
        }
    )
    out = panel.merge(targets, on=["airport", "utc_hour"], how="inner")
    out["target_arrivals_proxy"] = pd.to_numeric(out["target_arrivals"], errors="coerce").fillna(0.0)
    out = out[out["target_arrivals"] > 0].copy()
    for col in ["target_local_hour", "target_day_of_week"]:
        out[col] = out[col].astype(str)
    return out.reset_index(drop=True)


def attach_online_state(panel: pd.DataFrame, rho: float, prefix: str = "online") -> pd.DataFrame:
    out_frames = []
    work = panel.sort_values(["airport", "utc_hour"]).copy()
    for airport, g in work.groupby("airport", sort=False):
        g = g.copy()
        impulses = pd.to_numeric(g["known_start_impulse"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(float)
        state_values = np.zeros(len(g), dtype=float)
        delta_values = np.zeros(len(g), dtype=float)
        last_start_idx = -10_000
        since = np.full(len(g), 48.0, dtype=float)
        state = 0.0
        for i, impulse in enumerate(impulses):
            previous = state
            state = min(rho * state + impulse, 24.0)
            state_values[i] = state
            delta_values[i] = state - previous
            if impulse > 0:
                last_start_idx = i
            since[i] = min(48.0, float(i - last_start_idx)) if last_start_idx >= 0 else 48.0
        g[f"{prefix}_constraint_state"] = state_values
        g[f"{prefix}_state_delta"] = delta_values
        g[f"{prefix}_recovery_state"] = (state_values - impulses).clip(min=0.0)
        g[f"{prefix}_mild_constraint"] = state_values * g["mild_weather_abs"].fillna(0.0).to_numpy(float)
        g["hours_since_known_start"] = since
        g["recent_start_clock"] = np.where(since < 48.0, 1.0 / (1.0 + since), 0.0)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def add_fixed_decay_states(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for rho, prefix in [(0.90, "fixed090"), (0.95, "fixed095")]:
        states = attach_online_state(panel, rho, prefix=prefix)[
            [
                "row_id",
                f"{prefix}_constraint_state",
                f"{prefix}_recovery_state",
                f"{prefix}_mild_constraint",
                f"{prefix}_state_delta",
            ]
        ]
        out = out.merge(states, on="row_id", how="left")
    for col in FIXED_DECAY_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
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
    beta = fit_grouped_logit(x_train, train[success_col].to_numpy(float), train["target_arrivals"].to_numpy(float))
    prob = sigmoid(x_test @ beta)
    pred = test[["row_id", "airport", "utc_hour", "month", "target_arrivals", success_col]].copy()
    pred["pred_prob"] = prob
    metrics = evaluate(test[success_col].to_numpy(float), test["target_arrivals"].to_numpy(float), prob)
    return pred, metrics


def cv_score_for_rho(panel_by_rho: dict[float, pd.DataFrame], train_months: list[int], rho: float) -> float:
    panel = panel_by_rho[rho]
    aucs = []
    numeric = START_NUMERIC + ONLINE_STATE
    for month in train_months:
        train = panel[panel["month"] != month].copy()
        test = panel[panel["month"] == month].copy()
        if train.empty or test.empty:
            continue
        for success_col in TARGETS.values():
            _, metrics = fit_predict(train, test, success_col, numeric, CATS)
            if np.isfinite(metrics["auc"]):
                aucs.append(metrics["auc"])
    return float(np.mean(aucs)) if aucs else -np.inf


def select_rho(panel_by_rho: dict[float, pd.DataFrame], train_months: list[int], rho_grid: list[float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for rho in rho_grid:
        rows.append({"rho": rho, "inner_mean_auc": cv_score_for_rho(panel_by_rho, train_months, rho)})
    table = pd.DataFrame(rows).sort_values(["inner_mean_auc", "rho"], ascending=[False, False])
    return float(table.iloc[0]["rho"]), table


def evaluate_specs(base_panel: pd.DataFrame, months: list[int], rho_grid: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_panel = add_fixed_decay_states(base_panel)
    panel_by_rho = {rho: attach_online_state(base_panel, rho, prefix="online") for rho in rho_grid}
    specs = {
        "online_baseline": (BASE_NUMERIC, fixed_panel),
        "known_start_window": (START_NUMERIC, fixed_panel),
        "online_fixed_decay": (START_NUMERIC + FIXED_DECAY_COLS, fixed_panel),
    }
    metric_rows = []
    prediction_rows = []
    rho_rows = []
    for outer_month in months:
        train_months = [m for m in months if m != outer_month]
        selected, rho_table = select_rho(panel_by_rho, train_months, rho_grid)
        rho_table["outer_month"] = outer_month
        rho_rows.append(rho_table)
        dyn_panel = panel_by_rho[selected]
        dynamic_specs = {
            "online_DCSI": (START_NUMERIC + ONLINE_STATE, dyn_panel),
            "online_relation_DCSI": (START_NUMERIC + ONLINE_STATE + RELATIONS, dyn_panel),
        }
        for target, success_col in TARGETS.items():
            for name, (numeric, source_panel) in {**specs, **dynamic_specs}.items():
                train = source_panel[source_panel["month"] != outer_month].copy()
                test = source_panel[source_panel["month"] == outer_month].copy()
                pred, metrics = fit_predict(train, test, success_col, numeric, CATS)
                metric_rows.append(
                    metrics
                    | {
                        "target": target,
                        "model": name,
                        "fold_month": outer_month,
                        "rho": selected if name.startswith("online_DCSI") or name.startswith("online_relation") else "",
                    }
                )
                pred["target"] = target
                pred["model"] = name
                pred["rho"] = selected if name.startswith("online_DCSI") or name.startswith("online_relation") else ""
                prediction_rows.append(pred)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    overall = []
    for (target, model), g in predictions.groupby(["target", "model"]):
        success_col = TARGETS[target]
        overall.append(
            evaluate(g[success_col].to_numpy(float), g["target_arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": model, "fold_month": "all", "rho": "fold_selected" if "DCSI" in model else ""}
        )
    metrics = pd.concat([pd.DataFrame(metric_rows), pd.DataFrame(overall)], ignore_index=True)
    return metrics, predictions, pd.concat(rho_rows, ignore_index=True)


def build_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["fold_month"].astype(str).eq("all")].copy()
    base = overall[overall["model"].eq("online_baseline")][["horizon", "target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "base_auc", "brier": "base_brier", "log_loss": "base_log_loss"}
    )
    fixed = overall[overall["model"].eq("known_start_window")][["horizon", "target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "fixed_auc", "brier": "fixed_brier", "log_loss": "fixed_log_loss"}
    )
    decay = overall[overall["model"].eq("online_fixed_decay")][["horizon", "target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "decay_auc", "brier": "decay_brier", "log_loss": "decay_log_loss"}
    )
    out = (
        overall.merge(base, on=["horizon", "target"], how="left")
        .merge(fixed, on=["horizon", "target"], how="left")
        .merge(decay, on=["horizon", "target"], how="left")
    )
    out["auc_gain_vs_base"] = out["auc"] - out["base_auc"]
    out["auc_gain_vs_start"] = out["auc"] - out["fixed_auc"]
    out["auc_gain_vs_decay"] = out["auc"] - out["decay_auc"]
    out["brier_gain_vs_base"] = out["base_brier"] - out["brier"]
    out["brier_gain_vs_start"] = out["fixed_brier"] - out["brier"]
    out["brier_gain_vs_decay"] = out["decay_brier"] - out["brier"]
    return out


def state_diffs(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, h in panel.groupby("horizon"):
        for state_col in ["known_start_impulse", "online_constraint_state"]:
            top = h[h[state_col] >= h[state_col].quantile(0.9)].copy()
            low = h[h[state_col] <= h[state_col].quantile(0.1)].copy()
            if top.empty or low.empty:
                continue
            rows.append(
                {
                    "horizon": int(horizon),
                    "state_field": state_col,
                    "top_arrivals": int(top["target_arrivals"].sum()),
                    "low_arrivals": int(low["target_arrivals"].sum()),
                    "top_delay_rate": float(top["target_arr_delay60_count"].sum() / top["target_arrivals"].sum()),
                    "low_delay_rate": float(low["target_arr_delay60_count"].sum() / low["target_arrivals"].sum()),
                    "top_cancel_rate": float(top["target_cancel_count"].sum() / top["target_arrivals"].sum()),
                    "low_cancel_rate": float(low["target_cancel_count"].sum() / low["target_arrivals"].sum()),
                }
            )
    return pd.DataFrame(rows)


def write_assessment(out: Path, gains: pd.DataFrame, rho_selection: pd.DataFrame) -> None:
    rows = ["# Online lead-time DCSI assessment", ""]
    for row in gains[gains["model"].isin(["online_DCSI", "online_relation_DCSI"])].itertuples(index=False):
        rows.append(
            f"- Horizon {row.horizon} h, {row.target}, {row.model}: AUC {row.auc:.3f}, "
            f"gain vs baseline {row.auc_gain_vs_base:+.3f}, gain vs start-window {row.auc_gain_vs_start:+.3f}, "
            f"gain vs fixed decay {row.auc_gain_vs_decay:+.3f}."
        )
    selected = rho_selection.sort_values(["horizon", "outer_month", "inner_mean_auc"], ascending=[True, True, False]).drop_duplicates(["horizon", "outer_month"])
    for horizon, g in selected.groupby("horizon"):
        rows.append(f"- Horizon {horizon} h selected rho values: {', '.join(f'{x:.2f}' for x in g['rho'].tolist())}.")
    (out / "online_lead_assessment.md").write_text("\n".join(rows), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    months = parse_months(args.months)
    airports = parse_airports(args.airports)
    rho_grid = parse_float_grid(args.rho_grid)
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    base = load_base_panel(months, airports)
    events = load_issue_events(airports)
    all_metrics = []
    all_predictions = []
    all_rho = []
    lead_panels = []
    for horizon in horizons:
        lead = attach_online_action_fields(base, events, horizon)
        lead = add_target_shift(lead, horizon)
        lead["horizon"] = horizon
        metrics, predictions, rho_selection = evaluate_specs(lead, months, rho_grid)
        metrics["horizon"] = horizon
        predictions["horizon"] = horizon
        rho_selection["horizon"] = horizon
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_rho.append(rho_selection)
        selected = rho_selection.sort_values(["outer_month", "inner_mean_auc"], ascending=[True, False]).drop_duplicates("outer_month")["rho"].median()
        lead_panels.append(attach_online_state(lead, float(selected), prefix="online"))
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    rho_selection = pd.concat(all_rho, ignore_index=True)
    gains = build_gains(metrics)
    lead_panel = pd.concat(lead_panels, ignore_index=True)
    diffs = state_diffs(lead_panel)
    metrics.to_csv(out / "online_lead_metrics.csv", index=False)
    gains.to_csv(out / "online_lead_gains.csv", index=False)
    predictions.to_csv(out / "online_lead_predictions.csv", index=False)
    rho_selection.to_csv(out / "online_lead_rho_selection.csv", index=False)
    diffs.to_csv(out / "online_lead_state_diffs.csv", index=False)
    write_assessment(out, gains, rho_selection)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ATL,ORD")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--rho-grid", default="0,0.25,0.50,0.70,0.85,0.90,0.93,0.95,0.97")
    parser.add_argument("--output-name", default="smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
