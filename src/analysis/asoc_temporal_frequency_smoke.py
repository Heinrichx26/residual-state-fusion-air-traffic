from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from asoc_soft_computing_smoke import (
    OUT_ROOT,
    TARGETS,
    capture_at_fraction,
    ece,
    expanded_binomial_frame,
    metric_row,
    parse_airports,
    parse_int_list,
)


PROJECT = Path(__file__).resolve().parents[2]
DCSI_PREDICTIONS = (
    PROJECT
    / "results"
    / "experiments"
    / "fusion_framework_strengthening"
    / "dcsi_online_lead_validation"
    / "online_lead_full_2025"
    / "online_lead_predictions.csv"
)

LAG_HOURS = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]


def direct_success_col(target: str) -> str:
    return "target_arr_delay60_count" if target == "long_arrival_delay" else "target_cancel_count"


def load_prediction_panel(months: list[int], airports: list[str] | None, horizons: list[int]) -> pd.DataFrame:
    pred = pd.read_csv(DCSI_PREDICTIONS, parse_dates=["utc_hour"])
    pred = pred[
        pred["month"].isin(months)
        & pred["horizon"].isin(horizons)
        & pred["model"].isin(["online_baseline", "online_fixed_decay", "online_relation_DCSI"])
    ].copy()
    if airports:
        pred = pred[pred["airport"].isin(airports)].copy()
    pred["target_arr_delay60_count"] = pd.to_numeric(
        pred["target_arr_delay60_count"], errors="coerce"
    ).fillna(0.0)
    pred["target_cancel_count"] = pd.to_numeric(
        pred["target_cancel_count"], errors="coerce"
    ).fillna(0.0)
    pivot = pred.pivot_table(
        index=[
            "target",
            "horizon",
            "airport",
            "utc_hour",
            "month",
            "target_arrivals",
            "target_arr_delay60_count",
            "target_cancel_count",
        ],
        columns="model",
        values="pred_prob",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "online_baseline": "baseline_score",
            "online_fixed_decay": "fixed_decay_score",
            "online_relation_DCSI": "relation_score",
        }
    )
    pivot["target_arrivals"] = pivot["target_arrivals"].astype(float)
    pivot = pivot[pivot["target_arrivals"] > 0].copy()
    for col in ["baseline_score", "fixed_decay_score", "relation_score"]:
        pivot[col] = pd.to_numeric(pivot[col], errors="coerce").fillna(0.0)
    return pivot.sort_values(["target", "horizon", "airport", "utc_hour"]).reset_index(drop=True)


def fft_ratio(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 3:
        return 0.0, 0.0
    centered = values - np.nanmean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    if len(spectrum) <= 2:
        return float(spectrum.sum()), 0.0
    low = float(spectrum[1 : min(3, len(spectrum))].sum())
    high = float(spectrum[min(3, len(spectrum)) :].sum())
    total = low + high
    if total <= 1e-12:
        return 0.0, 0.0
    return low / total, high / total


def add_temporal_frequency_features(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("utc_hour").copy()
    score = out["relation_score"].astype(float)
    margin = out["relation_score"].astype(float) - out["baseline_score"].astype(float)
    fixed_margin = out["relation_score"].astype(float) - out["fixed_decay_score"].astype(float)
    for lag in LAG_HOURS:
        out[f"rel_lag_{lag}h"] = score.shift(lag).fillna(score.expanding().mean())
        out[f"margin_lag_{lag}h"] = margin.shift(lag).fillna(0.0)
    shifted_score = score.shift(1)
    shifted_margin = margin.shift(1)
    for window in WINDOWS:
        roll_score = shifted_score.rolling(window=window, min_periods=2)
        roll_margin = shifted_margin.rolling(window=window, min_periods=2)
        out[f"rel_roll_mean_{window}h"] = roll_score.mean()
        out[f"rel_roll_std_{window}h"] = roll_score.std()
        out[f"rel_roll_max_{window}h"] = roll_score.max()
        out[f"margin_roll_mean_{window}h"] = roll_margin.mean()
        out[f"rel_minus_roll_mean_{window}h"] = score - out[f"rel_roll_mean_{window}h"]
        low_vals = []
        high_vals = []
        prior_values = shifted_score.to_numpy(float)
        for idx in range(len(out)):
            start = max(0, idx - window)
            segment = prior_values[start:idx]
            segment = segment[np.isfinite(segment)]
            low, high = fft_ratio(segment)
            low_vals.append(low)
            high_vals.append(high)
        out[f"rel_fft_low_share_{window}h"] = low_vals
        out[f"rel_fft_high_share_{window}h"] = high_vals
    out["relation_margin"] = margin
    out["relation_fixed_margin"] = fixed_margin
    out["relation_acceleration_1_3"] = out["rel_lag_1h"] - out["rel_lag_3h"]
    out["relation_acceleration_3_12"] = out["rel_lag_3h"] - out["rel_lag_12h"]
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "baseline_score",
        "fixed_decay_score",
        "relation_score",
        "relation_margin",
        "relation_fixed_margin",
        "relation_acceleration_1_3",
        "relation_acceleration_3_12",
    ]
    cols.extend([c for c in df.columns if c.startswith("rel_lag_")])
    cols.extend([c for c in df.columns if c.startswith("margin_lag_")])
    cols.extend([c for c in df.columns if c.startswith("rel_roll_")])
    cols.extend([c for c in df.columns if c.startswith("margin_roll_")])
    cols.extend([c for c in df.columns if c.startswith("rel_minus_roll_")])
    cols.extend([c for c in df.columns if c.startswith("rel_fft_")])
    return sorted(set(cols))


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in panel.groupby(["target", "horizon", "airport"], sort=False):
        parts.append(add_temporal_frequency_features(group))
    out = pd.concat(parts, ignore_index=True)
    cols = feature_columns(out)
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median()).fillna(0.0)
    return out


def isotonic_calibrate(train_prob: np.ndarray, successes: np.ndarray, totals: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    y = successes / np.maximum(totals, 1.0)
    order = np.argsort(train_prob)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
    cal.fit(train_prob[order], y[order], sample_weight=totals[order])
    return np.clip(cal.predict(test_prob), 1e-5, 1 - 1e-5)


def fit_predict_tf_gbdt(train: pd.DataFrame, test: pd.DataFrame, success_col: str, features: list[str]) -> np.ndarray:
    x_train_raw = train[features].copy()
    x_test_raw = test[features].copy()
    x_train, y_train, w_train = expanded_binomial_frame(train.rename(columns={"target_arrivals": "arrivals"}), x_train_raw, success_col)
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.035,
        max_leaf_nodes=18,
        min_samples_leaf=80,
        l2_regularization=0.10,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    train_prob = np.clip(model.predict_proba(x_train_raw)[:, 1], 1e-5, 1 - 1e-5)
    test_prob = np.clip(model.predict_proba(x_test_raw)[:, 1], 1e-5, 1 - 1e-5)
    return isotonic_calibrate(
        train_prob,
        train[success_col].to_numpy(float),
        train["target_arrivals"].to_numpy(float),
        test_prob,
    )


def metric_from_prediction(target: str, model: str, df: pd.DataFrame, success_col: str, prob: np.ndarray) -> dict[str, object]:
    pred = df[["airport", "utc_hour", "month", "target_arrivals", success_col]].copy()
    pred = pred.rename(columns={"target_arrivals": "arrivals"})
    pred["pred_prob"] = np.clip(prob, 1e-5, 1 - 1e-5)
    return metric_row(target, model, pred, success_col)


def run(months: list[int], airports: list[str] | None, horizons: list[int], output_name: str) -> None:
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    panel = add_features(load_prediction_panel(months, airports, horizons))
    features = feature_columns(panel)
    rows = []
    predictions = []
    for (target, horizon), group in panel.groupby(["target", "horizon"], sort=True):
        success_col = direct_success_col(target)
        rows.append(metric_from_prediction(target, f"Relation-DCSI h{horizon}", group, success_col, group["relation_score"].to_numpy(float)))
        fold_parts = []
        for test_month in sorted(group["month"].unique()):
            train = group[group["month"] != test_month].copy()
            test = group[group["month"] == test_month].copy()
            prob = fit_predict_tf_gbdt(train, test, success_col, features)
            fold = test[["target", "horizon", "airport", "utc_hour", "month", "target_arrivals", success_col]].copy()
            fold["pred_prob"] = prob
            fold["test_month"] = test_month
            fold_parts.append(fold)
        pred = pd.concat(fold_parts, ignore_index=True)
        rows.append(metric_from_prediction(target, f"TF-DCSI calibrated encoder h{horizon}", pred, success_col, pred["pred_prob"].to_numpy(float)))
        predictions.append(pred.assign(model=f"TF-DCSI calibrated encoder h{horizon}"))
    metrics = pd.DataFrame(rows)
    gains = add_reference_gains(metrics)
    metrics.to_csv(out / "tf_dcsi_smoke_metrics.csv", index=False)
    gains.to_csv(out / "tf_dcsi_smoke_gains_vs_relation.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "tf_dcsi_smoke_predictions.csv", index=False)
    write_assessment(out, gains, months, airports, horizons)
    print(f"wrote {out}")


def add_reference_gains(metrics: pd.DataFrame) -> pd.DataFrame:
    work = metrics.copy()
    work["horizon"] = work["model"].str.extract(r"h(\d+)")[0].astype(int)
    work["family"] = np.where(work["model"].str.startswith("Relation-DCSI"), "reference", "candidate")
    ref = work[work["family"].eq("reference")][
        ["target", "horizon", "auc", "pr_auc", "brier", "ece_10", "top10_capture", "top_decile_lift"]
    ].rename(
        columns={
            "auc": "relation_auc",
            "pr_auc": "relation_pr_auc",
            "brier": "relation_brier",
            "ece_10": "relation_ece_10",
            "top10_capture": "relation_top10_capture",
            "top_decile_lift": "relation_top_decile_lift",
        }
    )
    out = work.merge(ref, on=["target", "horizon"], how="left")
    out["auc_gain_vs_relation"] = out["auc"] - out["relation_auc"]
    out["pr_auc_gain_vs_relation"] = out["pr_auc"] - out["relation_pr_auc"]
    out["brier_gain_vs_relation"] = out["relation_brier"] - out["brier"]
    out["ece10_gain_vs_relation"] = out["relation_ece_10"] - out["ece_10"]
    out["top10_capture_gain_vs_relation"] = out["top10_capture"] - out["relation_top10_capture"]
    out["top_lift_gain_vs_relation"] = out["top_decile_lift"] - out["relation_top_decile_lift"]
    return out


def write_assessment(out: Path, gains: pd.DataFrame, months: list[int], airports: list[str] | None, horizons: list[int]) -> None:
    lines = [
        "# Temporal-frequency DCSI smoke assessment",
        "",
        f"Scope: months {','.join(str(m) for m in months)}; airports {','.join(airports) if airports else 'ALL'}; horizons {','.join(str(h) for h in horizons)} h.",
        "",
        "Gate: continue only if TF-DCSI beats Relation-DCSI on at least two metrics for a target-horizon pair, including either AUC, PR-AUC, or top-10 capture.",
        "",
    ]
    cand = gains[gains["family"].eq("candidate")].copy()
    for _, row in cand.sort_values(["target", "horizon"]).iterrows():
        positives = sum(
            [
                row["auc_gain_vs_relation"] > 0.005,
                row["pr_auc_gain_vs_relation"] > 0.005,
                row["brier_gain_vs_relation"] > 0.0005,
                row["ece10_gain_vs_relation"] > 0.005,
                row["top10_capture_gain_vs_relation"] > 0.01,
            ]
        )
        core = (
            row["auc_gain_vs_relation"] > 0.005
            or row["pr_auc_gain_vs_relation"] > 0.005
            or row["top10_capture_gain_vs_relation"] > 0.01
        )
        verdict = "promising" if positives >= 2 and core else "weak"
        lines.append(f"## {row['target']} h{int(row['horizon'])}")
        lines.append(
            f"- Verdict: {verdict}. AUC gain {row['auc_gain_vs_relation']:+.3f}, "
            f"PR-AUC gain {row['pr_auc_gain_vs_relation']:+.3f}, Brier gain {row['brier_gain_vs_relation']:+.4f}, "
            f"ECE gain {row['ece10_gain_vs_relation']:+.4f}, top-10 capture gain {row['top10_capture_gain_vs_relation']:+.3f}."
        )
        lines.append("")
    (out / "tf_dcsi_smoke_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ALL")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--output-name", default="tf_dcsi_smoke_10airports_3months")
    args = parser.parse_args()
    run(
        parse_int_list(args.months),
        parse_airports(args.airports),
        parse_int_list(args.horizons),
        args.output_name,
    )


if __name__ == "__main__":
    main()
