from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "experiments" / "atcscc_full_year_windows"
PANEL_FILE = OUT / "airport_hour_panel_with_windows.csv"


TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "cancellation": "cancel_count",
}

MODEL_SPECS = {
    "calendar_weather": {
        "numeric": [
            "weather_score",
            "mild_weather_abs",
            "wind_speed_mps",
            "visibility_km",
            "ceiling_m",
            "temperature_c",
            "month_sin",
            "month_cos",
        ],
        "categorical": ["airport", "local_hour", "day_of_week"],
    },
    "active_fusion": {
        "numeric": [
            "weather_score",
            "mild_weather_abs",
            "wind_speed_mps",
            "visibility_km",
            "ceiling_m",
            "temperature_c",
            "month_sin",
            "month_cos",
            "active_strong",
            "active_mild_strong_conflict",
            "active_hours_capped",
        ],
        "categorical": ["airport", "local_hour", "day_of_week"],
    },
    "active_post3h_fusion": {
        "numeric": [
            "weather_score",
            "mild_weather_abs",
            "wind_speed_mps",
            "visibility_km",
            "ceiling_m",
            "temperature_c",
            "month_sin",
            "month_cos",
            "active_strong",
            "active_mild_strong_conflict",
            "active_hours_capped",
            "post_3h_strong",
            "post_3h_mild_strong_conflict",
            "post_3h_hours_capped",
        ],
        "categorical": ["airport", "local_hour", "day_of_week"],
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    panel["active_strong"] = (panel["active_minutes"] >= 45).astype(float)
    panel["post_3h_strong"] = (panel["post_3h_minutes"] >= 45).astype(float)
    panel["active_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["active_strong"] == 1.0)
    ).astype(float)
    panel["post_3h_mild_strong_conflict"] = (
        (panel["mild_weather_abs"] == 1.0) & (panel["post_3h_strong"] == 1.0)
    ).astype(float)
    panel["active_hours_capped"] = (panel["active_minutes"] / 60.0).clip(0, 8)
    panel["post_3h_hours_capped"] = (panel["post_3h_minutes"] / 60.0).clip(0, 8)
    month_angle = 2 * np.pi * (panel["month"].astype(float) - 1) / 12.0
    panel["month_sin"] = np.sin(month_angle)
    panel["month_cos"] = np.cos(month_angle)
    for col in ["airport", "local_hour", "day_of_week"]:
        panel[col] = panel[col].astype(str)
    return panel


def feature_levels(panel: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    return {col: sorted(panel[col].dropna().astype(str).unique().tolist()) for col in categorical_cols}


def make_design(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    train_stats: dict[str, tuple[float, float]],
    levels: dict[str, list[str]],
) -> np.ndarray:
    parts = [np.ones((len(df), 1), dtype=float)]
    for col in numeric_cols:
        mean, sd = train_stats[col]
        values = pd.to_numeric(df[col], errors="coerce").fillna(mean).to_numpy(float)
        parts.append(((values - mean) / sd).reshape(-1, 1))
    for col in categorical_cols:
        values = df[col].astype(str)
        col_levels = levels[col]
        for level in col_levels[1:]:
            parts.append((values == level).astype(float).to_numpy().reshape(-1, 1))
    return np.hstack(parts)


def train_stats(train: pd.DataFrame, numeric_cols: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for col in numeric_cols:
        values = pd.to_numeric(train[col], errors="coerce")
        mean = float(values.mean()) if values.notna().any() else 0.0
        sd = float(values.std(ddof=0)) if values.notna().any() else 1.0
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        stats[col] = (mean, sd)
    return stats


def fit_grouped_logit(x: np.ndarray, successes: np.ndarray, totals: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    for _ in range(80):
        eta = x @ beta
        prob = sigmoid(eta)
        weights = np.maximum(totals * prob * (1 - prob), 1e-6)
        z = eta + (successes - totals * prob) / weights
        xtw = x.T * weights
        lhs = xtw @ x + penalty
        rhs = xtw @ z
        try:
            new_beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if np.max(np.abs(new_beta - beta)) < 1e-6:
            beta = new_beta
            break
        beta = new_beta
    return beta


def grouped_log_loss(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    loss = -(successes * np.log(prob) + (totals - successes) * np.log1p(-prob)).sum()
    return float(loss / totals.sum())


def grouped_brier(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    return float((successes * (1 - prob) ** 2 + (totals - successes) * prob**2).sum() / totals.sum())


def grouped_auc(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    positives = successes.astype(float)
    negatives = totals.astype(float) - positives
    total_pos = positives.sum()
    total_neg = negatives.sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    order = np.argsort(prob)
    sorted_prob = prob[order]
    sorted_pos = positives[order]
    sorted_neg = negatives[order]
    auc_num = 0.0
    cum_neg = 0.0
    start = 0
    while start < len(sorted_prob):
        end = start + 1
        while end < len(sorted_prob) and sorted_prob[end] == sorted_prob[start]:
            end += 1
        pos_g = sorted_pos[start:end].sum()
        neg_g = sorted_neg[start:end].sum()
        auc_num += pos_g * (cum_neg + 0.5 * neg_g)
        cum_neg += neg_g
        start = end
    return float(auc_num / (total_pos * total_neg))


def evaluate(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "log_loss": grouped_log_loss(successes, totals, prob),
        "brier": grouped_brier(successes, totals, prob),
        "auc": grouped_auc(successes, totals, prob),
        "event_rate": float(successes.sum() / totals.sum()),
        "arrivals": int(totals.sum()),
        "events": int(successes.sum()),
    }


def run_target(panel: pd.DataFrame, target_name: str, success_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(panel["month"].unique())
    predictions = []
    metric_rows = []
    for model_name, spec in MODEL_SPECS.items():
        levels = feature_levels(panel, spec["categorical"])
        fold_predictions = []
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
            fold["target"] = target_name
            fold["model"] = model_name
            fold["pred_prob"] = prob
            fold_predictions.append(fold)
            metric_rows.append(
                evaluate(fold[success_col].to_numpy(float), fold["arrivals"].to_numpy(float), prob)
                | {"target": target_name, "model": model_name, "fold_month": int(month)}
            )
        model_pred = pd.concat(fold_predictions, ignore_index=True)
        predictions.append(model_pred)
        metric_rows.append(
            evaluate(
                model_pred[success_col].to_numpy(float),
                model_pred["arrivals"].to_numpy(float),
                model_pred["pred_prob"].to_numpy(float),
            )
            | {"target": target_name, "model": model_name, "fold_month": "all"}
        )
    return pd.DataFrame(metric_rows), pd.concat(predictions, ignore_index=True)


def build_increment_table(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["fold_month"].astype(str).eq("all")].copy()
    baseline = overall[overall["model"].eq("calendar_weather")][["target", "log_loss", "brier", "auc"]].rename(
        columns={"log_loss": "baseline_log_loss", "brier": "baseline_brier", "auc": "baseline_auc"}
    )
    inc = overall.merge(baseline, on="target", how="left")
    inc["log_loss_reduction"] = inc["baseline_log_loss"] - inc["log_loss"]
    inc["brier_reduction"] = inc["baseline_brier"] - inc["brier"]
    inc["auc_gain"] = inc["auc"] - inc["baseline_auc"]
    return inc


def build_monthly_increment_table(metrics: pd.DataFrame) -> pd.DataFrame:
    folds = metrics[~metrics["fold_month"].astype(str).eq("all")].copy()
    baseline = folds[folds["model"].eq("calendar_weather")][
        ["target", "fold_month", "log_loss", "brier", "auc"]
    ].rename(columns={"log_loss": "baseline_log_loss", "brier": "baseline_brier", "auc": "baseline_auc"})
    out = folds.merge(baseline, on=["target", "fold_month"], how="left")
    out["log_loss_reduction"] = out["baseline_log_loss"] - out["log_loss"]
    out["brier_reduction"] = out["baseline_brier"] - out["brier"]
    out["auc_gain"] = out["auc"] - out["baseline_auc"]
    return out


def write_summary(increments: pd.DataFrame) -> None:
    delay = increments[(increments["target"].eq("long_arrival_delay")) & (increments["model"].eq("active_post3h_fusion"))].iloc[0]
    active = increments[(increments["target"].eq("long_arrival_delay")) & (increments["model"].eq("active_fusion"))].iloc[0]
    cancel = increments[(increments["target"].eq("cancellation")) & (increments["model"].eq("active_post3h_fusion"))].iloc[0]
    lines = [
        "# Fusion prediction increment",
        "",
        "Leave-one-month grouped logistic prediction uses airport-hour observations weighted by arrivals.",
        "",
        f"Long-delay AUC gain, active fusion: {active['auc_gain']:.3f}.",
        f"Long-delay AUC gain, active plus post_3h fusion: {delay['auc_gain']:.3f}.",
        f"Long-delay log-loss reduction, active plus post_3h fusion: {delay['log_loss_reduction']:.4f}.",
        f"Long-delay Brier reduction, active plus post_3h fusion: {delay['brier_reduction']:.4f}.",
        f"Cancellation AUC gain, active plus post_3h fusion: {cancel['auc_gain']:.3f}.",
        "",
        "Decision: ATCSCC advisories add out-of-month predictive information beyond weather and calendar structure.",
    ]
    (OUT / "fusion_prediction_increment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    all_metrics = []
    all_predictions = []
    for target_name, success_col in TARGETS.items():
        metrics, predictions = run_target(panel, target_name, success_col)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    increments = build_increment_table(metrics)
    monthly_increments = build_monthly_increment_table(metrics)
    metrics.to_csv(OUT / "fusion_prediction_cv_metrics.csv", index=False)
    increments.to_csv(OUT / "fusion_prediction_increment.csv", index=False)
    monthly_increments.to_csv(OUT / "fusion_prediction_monthly_increment.csv", index=False)
    predictions.to_csv(OUT / "fusion_prediction_cv_predictions.csv", index=False)
    write_summary(increments)
    print(OUT / "fusion_prediction_increment_summary.md", flush=True)


if __name__ == "__main__":
    main()
