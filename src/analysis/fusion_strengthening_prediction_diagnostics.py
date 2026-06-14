from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion_prediction_increment import (
    evaluate,
    feature_levels,
    fit_grouped_logit,
    make_design,
    sigmoid,
    train_stats,
)
from fusion_strengthening_common import ROOT_OUT
from fusion_strengthening_demand_residual import MODEL_SPECS, PRIMARY_BASELINE, TARGETS


DEMAND = ROOT_OUT / "demand_residual_full_2025"
EVENT = ROOT_OUT / "event_response_full_2025_30airports"
DEFAULT_MODELS = [PRIMARY_BASELINE, "schedule_fused_state"]


def grouped_pr_auc(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    positives = successes.astype(float)
    negatives = totals.astype(float) - positives
    total_pos = positives.sum()
    if total_pos <= 0:
        return np.nan
    frame = pd.DataFrame({"prob": prob, "pos": positives, "neg": negatives})
    grouped = frame.groupby("prob", as_index=False)[["pos", "neg"]].sum().sort_values("prob", ascending=False)
    cum_pos = cum_total = prev_recall = ap = 0.0
    for row in grouped.itertuples(index=False):
        cum_pos += row.pos
        cum_total += row.pos + row.neg
        recall = cum_pos / total_pos
        precision = cum_pos / cum_total if cum_total else 0.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def expected_calibration_error(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"success": successes, "total": totals, "prob": prob})
    frame["bin"] = pd.cut(frame["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    terms = []
    for _, group in frame.groupby("bin", observed=True):
        weight = group["total"].sum()
        if weight <= 0:
            continue
        obs = group["success"].sum() / weight
        pred = np.average(group["prob"], weights=group["total"])
        terms.append(weight * abs(obs - pred))
    return float(sum(terms) / totals.sum())


def top_decile_lift(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-prob)
    successes, totals = successes[order], totals[order]
    cutoff = totals.sum() * 0.10
    seen = top_success = top_total = 0.0
    for success, total in zip(successes, totals):
        if seen >= cutoff:
            break
        take = min(total, cutoff - seen)
        frac = take / total if total else 0.0
        top_success += success * frac
        top_total += take
        seen += take
    top_rate = top_success / top_total if top_total else np.nan
    base_rate = successes.sum() / totals.sum()
    return float(top_rate), float(top_rate / base_rate if base_rate else np.nan)


def calibration_slope(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(prob, 1e-5, 1 - 1e-5)
    logit = np.log(clipped / (1 - clipped))
    x = np.column_stack([np.ones(len(logit)), logit])
    beta = fit_grouped_logit(x, successes.astype(float), totals.astype(float), ridge=1e-6)
    return float(beta[0]), float(beta[1])


def metric_row(target: str, model: str, fold: str, data: pd.DataFrame, success_col: str) -> dict[str, object]:
    successes = data[success_col].to_numpy(float)
    totals = data["arrivals"].to_numpy(float)
    prob = data["pred_prob"].to_numpy(float)
    base = evaluate(successes, totals, prob)
    top_rate, top_lift = top_decile_lift(successes, totals, prob)
    intercept, slope = calibration_slope(successes, totals, prob)
    return {
        "target": target,
        "model": model,
        "fold": fold,
        **base,
        "pr_auc": grouped_pr_auc(successes, totals, prob),
        "expected_calibration_error": expected_calibration_error(successes, totals, prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "top_decile_precision": top_rate,
        "top_decile_lift": top_lift,
    }


def prediction_diagnostics(out: Path) -> None:
    predictions = pd.read_csv(DEMAND / "baseline_ladder_predictions.csv")
    rows = []
    for target, success_col in TARGETS.items():
        use = predictions[predictions["target"] == target].dropna(subset=[success_col])
        for model, group in use.groupby("model"):
            rows.append(metric_row(target, model, "leave_one_month_all", group, success_col))
    metrics = pd.DataFrame(rows)
    primary = metrics[metrics["model"] == PRIMARY_BASELINE][
        ["target", "auc", "pr_auc", "brier", "expected_calibration_error"]
    ]
    gains = metrics.merge(primary, on="target", suffixes=("", "_primary"))
    gains["auc_gain_vs_primary"] = gains["auc"] - gains["auc_primary"]
    gains["pr_auc_gain_vs_primary"] = gains["pr_auc"] - gains["pr_auc_primary"]
    gains["brier_gain_vs_primary"] = gains["brier_primary"] - gains["brier"]
    gains["ece_gain_vs_primary"] = gains["expected_calibration_error_primary"] - gains["expected_calibration_error"]
    metrics.to_csv(out / "pr_auc_calibration_metrics.csv", index=False)
    gains.to_csv(out / "pr_auc_calibration_gains.csv", index=False)


def leave_one_airport(out: Path, airports: list[str]) -> None:
    panel = pd.read_csv(DEMAND / "panel_with_demand.csv", parse_dates=["utc_hour"])
    panel = panel[panel["airport"].isin(airports)].copy() if airports else panel
    rows = []
    for airport in sorted(panel["airport"].unique()):
        train = panel[panel["airport"] != airport].copy()
        test = panel[panel["airport"] == airport].copy()
        if train.empty or test.empty:
            continue
        for target, success_col in TARGETS.items():
            for model in DEFAULT_MODELS:
                spec = MODEL_SPECS[model]
                levels = feature_levels(train, spec["categorical"])
                stats = train_stats(train, spec["numeric"])
                x_train = make_design(train, spec["numeric"], spec["categorical"], stats, levels)
                x_test = make_design(test, spec["numeric"], spec["categorical"], stats, levels)
                beta = fit_grouped_logit(x_train, train[success_col].to_numpy(float), train["arrivals"].to_numpy(float))
                prob = sigmoid(x_test @ beta)
                fold = test[["arrivals", success_col]].copy()
                fold["pred_prob"] = prob
                rows.append(metric_row(target, model, airport, fold, success_col))
    metrics = pd.DataFrame(rows)
    base = metrics[metrics["model"] == PRIMARY_BASELINE][["target", "fold", "auc", "pr_auc", "brier"]]
    gains = metrics.merge(base, on=["target", "fold"], suffixes=("", "_primary"))
    gains["auc_gain_vs_primary"] = gains["auc"] - gains["auc_primary"]
    gains["pr_auc_gain_vs_primary"] = gains["pr_auc"] - gains["pr_auc_primary"]
    gains["brier_gain_vs_primary"] = gains["brier_primary"] - gains["brier"]
    metrics.to_csv(out / "leave_one_airport_metrics.csv", index=False)
    gains[gains["model"] == "schedule_fused_state"].to_csv(out / "leave_one_airport_gain_summary.csv", index=False)


def event_pretrend(out: Path) -> None:
    curve = pd.read_csv(EVENT / "event_response_curve.csv")
    pre_mean = curve[curve["window"].isin(["pre_3h", "pre_1h"])]["delay_diff"].mean()
    curve["delay_diff_minus_pre_mean"] = curve["delay_diff"] - pre_mean
    curve["pre_mean_delay_diff"] = pre_mean
    curve.to_csv(out / "event_pretrend_contrast.csv", index=False)


def parse_airports(value: str) -> list[str]:
    if value.strip().upper() in {"ALL", "*"}:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", default="prediction_diagnostics_smoke")
    parser.add_argument("--airports", default="ATL,ORD")
    args = parser.parse_args()
    out = ROOT_OUT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    prediction_diagnostics(out)
    leave_one_airport(out, parse_airports(args.airports))
    event_pretrend(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
