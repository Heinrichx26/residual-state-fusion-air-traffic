from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fusion_prediction_increment import (
    evaluate,
    feature_levels,
    fit_grouped_logit,
    make_design,
    sigmoid,
    train_stats,
)
from fusion_strengthening_common import ROOT_OUT, prepare_base_panel, write_demand_audit
from smoke_open_fusion_topics import weighted_mean


OUT = ROOT_OUT / "demand_residual_smoke"
YEAR = 2025
MONTHS = [1, 7, 12]
TARGETS = {"long_arrival_delay": "arr_delay60_count", "cancellation": "cancel_count"}

CALENDAR = ["month_sin", "month_cos"]
WEATHER = ["weather_score", "mild_weather_abs", "wind_speed_mps", "visibility_km", "ceiling_m", "temperature_c"]
SCHEDULE_DEMAND = [
    "scheduled_arrivals",
    "scheduled_departures",
    "arrival_bank_intensity",
    "departure_bank_intensity",
    "arrival_carrier_hhi",
    "departure_carrier_hhi",
]
PRIOR_PRESSURE = [
    "prior_hour_delay60_rate",
    "prior_hour_cancel_rate",
    "prior_hour_arrivals",
]
DEMAND = SCHEDULE_DEMAND + PRIOR_PRESSURE
ADVISORY = ["active_strong", "post_3h_strong", "active_hours_capped", "post_3h_hours_capped"]
FUSED = ["active_mild_strong_conflict", "post_3h_mild_strong_conflict"]
CATS = ["airport", "local_hour", "day_of_week"]

MODEL_SPECS = {
    "calendar_only": {"numeric": CALENDAR, "categorical": CATS},
    "weather_only": {"numeric": WEATHER, "categorical": CATS},
    "schedule_demand_only": {"numeric": SCHEDULE_DEMAND, "categorical": CATS},
    "full_demand_only": {"numeric": DEMAND, "categorical": CATS},
    "calendar_weather_schedule_demand": {
        "numeric": CALENDAR + WEATHER + SCHEDULE_DEMAND,
        "categorical": CATS,
    },
    "calendar_weather_full_demand": {"numeric": CALENDAR + WEATHER + DEMAND, "categorical": CATS},
    "advisory_only": {"numeric": ADVISORY, "categorical": CATS},
    "schedule_demand_advisory": {
        "numeric": CALENDAR + WEATHER + SCHEDULE_DEMAND + ADVISORY,
        "categorical": CATS,
    },
    "full_demand_advisory": {"numeric": CALENDAR + WEATHER + DEMAND + ADVISORY, "categorical": CATS},
    "weather_advisory": {"numeric": WEATHER + ADVISORY, "categorical": CATS},
    "weather_advisory_interaction": {
        "numeric": WEATHER + ADVISORY + FUSED,
        "categorical": CATS,
    },
    "schedule_fused_state": {
        "numeric": CALENDAR + WEATHER + SCHEDULE_DEMAND + ADVISORY + FUSED,
        "categorical": CATS,
    },
    "full_fused_state": {
        "numeric": CALENDAR + WEATHER + DEMAND + ADVISORY + FUSED,
        "categorical": CATS,
    },
}
PRIMARY_BASELINE = "calendar_weather_schedule_demand"
DIRECT_ABLATION_MODELS = [
    "weather_only",
    "advisory_only",
    "weather_advisory",
    "schedule_demand_advisory",
    "schedule_fused_state",
]


def fit_predict(
    panel: pd.DataFrame,
    target: str,
    success_col: str,
    months: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, pred_frames = [], []
    for model_name, spec in MODEL_SPECS.items():
        levels = feature_levels(panel, spec["categorical"])
        folds = []
        for month in months:
            train = panel[panel["month"] != month].copy()
            test = panel[panel["month"] == month].copy()
            stats = train_stats(train, spec["numeric"])
            x_train = make_design(train, spec["numeric"], spec["categorical"], stats, levels)
            x_test = make_design(test, spec["numeric"], spec["categorical"], stats, levels)
            beta = fit_grouped_logit(
                x_train,
                train[success_col].to_numpy(float),
                train["arrivals"].to_numpy(float),
            )
            prob = sigmoid(x_test @ beta)
            fold = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
            fold["target"] = target
            fold["model"] = model_name
            fold["pred_prob"] = prob
            folds.append(fold)
            metric_rows.append(
                evaluate(fold[success_col].to_numpy(float), fold["arrivals"].to_numpy(float), prob)
                | {"target": target, "model": model_name, "fold_month": month}
            )
        all_pred = pd.concat(folds, ignore_index=True)
        pred_frames.append(all_pred)
        metric_rows.append(
            evaluate(
                all_pred[success_col].to_numpy(float),
                all_pred["arrivals"].to_numpy(float),
                all_pred["pred_prob"].to_numpy(float),
            )
            | {"target": target, "model": model_name, "fold_month": "all"}
        )
    return pd.DataFrame(metric_rows), pd.concat(pred_frames, ignore_index=True)


def residual_validation(panel: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = predictions[predictions["model"] == PRIMARY_BASELINE].copy()
    base = base[["airport", "utc_hour", "month", "target", "pred_prob"]]
    merged = panel.merge(base, on=["airport", "utc_hour", "month"], how="inner")
    for target, success_col in TARGETS.items():
        use = merged[merged["target"] == target].copy()
        use["obs_rate"] = use[success_col] / use["arrivals"]
        use["residual"] = use["obs_rate"] - use["pred_prob"]
        for window in ["active", "post_3h"]:
            conflict_col = f"{window}_mild_strong_conflict"
            conflict = use[(use["mild_weather_abs"] == 1.0) & (use[conflict_col] == 1.0)]
            baseline = use[(use["mild_weather_abs"] == 1.0) & (use[conflict_col] == 0.0)]
            c_res = weighted_mean(conflict, "residual", "arrivals")
            b_res = weighted_mean(baseline, "residual", "arrivals")
            rows.append(
                {
                    "target": target,
                    "window": window,
                    "conflict_arrivals": int(conflict["arrivals"].sum()),
                    "baseline_arrivals": int(baseline["arrivals"].sum()),
                    "conflict_residual": c_res,
                    "baseline_residual": b_res,
                    "residual_diff": c_res - b_res,
                }
            )
    return pd.DataFrame(rows)


def metric_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    all_rows = metrics[metrics["fold_month"].astype(str) == "all"].copy()
    base = all_rows[all_rows["model"] == PRIMARY_BASELINE]
    out = all_rows.merge(base[["target", "auc", "log_loss", "brier"]], on="target", suffixes=("", "_base"))
    out["auc_gain_vs_primary"] = out["auc"] - out["auc_base"]
    out["log_loss_gain_vs_primary"] = out["log_loss_base"] - out["log_loss"]
    out["brier_gain_vs_primary"] = out["brier_base"] - out["brier"]
    return out.drop(columns=["auc_base", "log_loss_base", "brier_base"])


def direct_ablation_table(metrics: pd.DataFrame) -> pd.DataFrame:
    all_rows = metrics[metrics["fold_month"].astype(str) == "all"].copy()
    out = all_rows[all_rows["model"].isin(DIRECT_ABLATION_MODELS)].copy()
    order = {model: i for i, model in enumerate(DIRECT_ABLATION_MODELS)}
    out["ablation_order"] = out["model"].map(order)
    base = out[out["model"] == "weather_only"][["target", "auc", "log_loss", "brier"]]
    out = out.merge(base, on="target", suffixes=("", "_weather"))
    out["auc_gain_vs_weather"] = out["auc"] - out["auc_weather"]
    out["log_loss_gain_vs_weather"] = out["log_loss_weather"] - out["log_loss"]
    out["brier_gain_vs_weather"] = out["brier_weather"] - out["brier"]
    return out.drop(columns=["auc_weather", "log_loss_weather", "brier_weather"]).sort_values(
        ["target", "ablation_order"]
    )


def write_assessment(gains: pd.DataFrame, residuals: pd.DataFrame, path, year: int, months: list[int]) -> None:
    chosen = gains[gains["model"] == "schedule_fused_state"].copy()
    delay_res = residuals[residuals["target"] == "long_arrival_delay"]["residual_diff"].min()
    cancel_res = residuals[residuals["target"] == "cancellation"]["residual_diff"].min()
    residual_status = "passes" if delay_res >= 0.10 and cancel_res > 0 else "needs review"
    lines = [
        "# Demand residual smoke assessment",
        "",
        f"Scope: {year} months {','.join(str(m) for m in months)}; original 10 airports.",
        "",
        "Primary model: schedule_fused_state versus calendar_weather_schedule_demand.",
        "",
    ]
    for row in chosen.itertuples(index=False):
        lines.append(
            f"- {row.target}: AUC gain {float(row.auc_gain_vs_primary):+.3f}; "
            f"log-loss gain {float(row.log_loss_gain_vs_primary):+.3f}; "
            f"Brier gain {float(row.brier_gain_vs_primary):+.3f}."
        )
    lines.extend(["", "Residual validation against the schedule-demand baseline:"])
    for row in residuals.itertuples(index=False):
        lines.append(
            f"- {row.target}, {row.window}: residual difference "
            f"{float(row.residual_diff):+.3f} over {int(row.conflict_arrivals)} conflict arrivals."
        )
    lines.extend(
        [
            "",
            "Assessment: prediction increment passes the smoke gate for both outcomes. "
            f"Residual validation status: {residual_status}. "
            "Move this direction into the manuscript only when the selected run meets the predefined gates.",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_months(text: str) -> list[int]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--output-name", default="demand_residual_smoke")
    args = parser.parse_args()

    months = parse_months(args.months)
    out_dir = ROOT_OUT / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = prepare_base_panel(args.year, months)
    write_demand_audit(panel, out_dir / "demand_feature_audit.csv")

    metric_frames, pred_frames = [], []
    for target, success_col in TARGETS.items():
        metrics, preds = fit_predict(panel, target, success_col, months)
        metric_frames.append(metrics)
        pred_frames.append(preds)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    gains = metric_gain_table(metrics)
    ablation = direct_ablation_table(metrics)
    residuals = residual_validation(panel, predictions)
    metrics.to_csv(out_dir / "baseline_ladder_metrics.csv", index=False)
    gains.to_csv(out_dir / "baseline_ladder_gain_summary.csv", index=False)
    ablation.to_csv(out_dir / "direct_ablation_matrix.csv", index=False)
    predictions.to_csv(out_dir / "baseline_ladder_predictions.csv", index=False)
    residuals.to_csv(out_dir / "residual_state_validation.csv", index=False)
    write_assessment(gains, residuals, out_dir / "smoke_assessment.md", args.year, months)
    panel.to_csv(out_dir / "panel_with_demand.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
