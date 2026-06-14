from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from asoc_soft_computing_smoke import (
    OUT_ROOT,
    align_design,
    expanded_binomial_frame,
    metric_row,
    parse_airports,
    parse_int_list,
)
from asoc_temporal_frequency_smoke import DCSI_PREDICTIONS


PANEL = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "experiments"
    / "fusion_framework_strengthening"
    / "temporal_availability_full_2025"
    / "temporal_availability_panel.csv"
)

TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "cancellation": "cancel_count",
}

CATEGORICAL = ["airport", "local_hour", "day_of_week"]
BASE_NUM = [
    "weather_score",
    "mild_weather_abs",
    "wind_speed_mps",
    "visibility_km",
    "ceiling_m",
    "temperature_c",
    "month_sin",
    "month_cos",
    "scheduled_arrivals",
    "scheduled_departures",
    "arrival_bank_intensity",
    "departure_bank_intensity",
    "arrival_carrier_hhi",
    "departure_carrier_hhi",
]
ADVISORY_NUM = [
    "active_strong",
    "post_3h_strong",
    "active_hours_capped",
    "post_3h_hours_capped",
    "active_before_minutes",
    "post_3h_known_minutes",
]
PRIOR_NUM = [
    "prior_hour_delay60_rate",
    "prior_hour_cancel_rate",
    "prior_hour_arrivals",
]
FUZZY_MEMBERSHIP_NUM = [
    "weather_low_membership",
    "weather_mid_membership",
    "weather_high_membership",
    "arrival_pressure_membership",
    "departure_pressure_membership",
    "arrival_bank_membership",
    "departure_bank_membership",
    "active_memory_membership",
    "post_memory_membership",
    "target_prior_membership",
    "target_prior_residual",
    "soft_pressure_index",
]
RELATION_NUM = [
    "relation_score",
]
INTERACTION_NUM = [
    "relation_active_interaction",
    "relation_post_interaction",
    "relation_weather_interaction",
    "relation_arrival_pressure_interaction",
    "relation_prior_interaction",
]
FUZZY_NUM = FUZZY_MEMBERSHIP_NUM + RELATION_NUM + INTERACTION_NUM


MODEL_FEATURES = {
    "LightGBM CWD+advisory": BASE_NUM + ADVISORY_NUM,
    "LightGBM full-demand advisory": BASE_NUM + ADVISORY_NUM + PRIOR_NUM,
    "AFRE soft evidence": BASE_NUM + ADVISORY_NUM + PRIOR_NUM + FUZZY_NUM,
}
ABLATION_FEATURES = {
    "AFRE no demand memory": BASE_NUM
    + ADVISORY_NUM
    + [
        col
        for col in FUZZY_MEMBERSHIP_NUM + RELATION_NUM + INTERACTION_NUM
        if col
        not in {
            "target_prior_membership",
            "target_prior_residual",
            "relation_prior_interaction",
        }
    ],
    "AFRE no fuzzy membership": BASE_NUM + ADVISORY_NUM + PRIOR_NUM + RELATION_NUM,
    "AFRE no relation score": BASE_NUM + ADVISORY_NUM + PRIOR_NUM + FUZZY_MEMBERSHIP_NUM,
    "AFRE no relation interaction": BASE_NUM + ADVISORY_NUM + PRIOR_NUM + FUZZY_MEMBERSHIP_NUM + RELATION_NUM,
}
AIRPORT_GROUPS = [
    ["ATL", "CLT"],
    ["DEN", "DFW"],
    ["EWR", "JFK", "LGA"],
    ["LAX", "ORD", "SFO"],
]


def ramp(x: pd.Series, low: float, high: float) -> pd.Series:
    if high <= low:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - low) / (high - low)).clip(0.0, 1.0)


def triangle(x: pd.Series, left: float, peak: float, right: float) -> pd.Series:
    up = ramp(x, left, peak)
    down = 1.0 - ramp(x, peak, right)
    return np.minimum(up, down).clip(0.0, 1.0)


def load_panel(months: list[int], airports: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(PANEL, parse_dates=["utc_hour"])
    df = df[df["month"].isin(months)].copy()
    if airports:
        df = df[df["airport"].isin(airports)].copy()
    df = df[(df["arrivals"] > 0) & df["weather_score"].notna()].copy()
    for col in ["active_minutes", "post_3h_minutes", "active_before_minutes", "post_3h_known_minutes"]:
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    df["active_strong"] = (df["active_minutes"] >= 45).astype(float)
    df["post_3h_strong"] = (df["post_3h_minutes"] >= 45).astype(float)
    df["active_hours_capped"] = (df["active_minutes"] / 60.0).clip(0, 8)
    df["post_3h_hours_capped"] = (df["post_3h_minutes"] / 60.0).clip(0, 8)
    for col in BASE_NUM + ADVISORY_NUM + PRIOR_NUM:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True)


def relation_scores(months: list[int], target: str, horizon: int) -> pd.DataFrame:
    rel = pd.read_csv(DCSI_PREDICTIONS, parse_dates=["utc_hour"])
    rel = rel[
        rel["month"].isin(months)
        & rel["target"].eq(target)
        & rel["horizon"].eq(horizon)
        & rel["model"].eq("online_relation_DCSI")
    ].copy()
    return rel[["airport", "utc_hour", "month", "pred_prob"]].rename(columns={"pred_prob": "relation_score"})


def add_soft_features(df: pd.DataFrame, target: str, train_rate: float) -> pd.DataFrame:
    out = df.copy()
    prior_col = "prior_hour_delay60_rate" if target == "long_arrival_delay" else "prior_hour_cancel_rate"
    prior_rate = pd.to_numeric(out[prior_col], errors="coerce").fillna(0.0)
    out["weather_low_membership"] = 1.0 - ramp(out["weather_score"], 1.2, 2.4)
    out["weather_mid_membership"] = triangle(out["weather_score"], 1.0, 2.4, 4.2)
    out["weather_high_membership"] = ramp(out["weather_score"], 3.0, 5.2)
    out["arrival_pressure_membership"] = ramp(out["scheduled_arrivals"], 45.0, 85.0)
    out["departure_pressure_membership"] = ramp(out["scheduled_departures"], 45.0, 85.0)
    out["arrival_bank_membership"] = ramp(out["arrival_bank_intensity"], 1.3, 2.1)
    out["departure_bank_membership"] = ramp(out["departure_bank_intensity"], 1.3, 2.1)
    out["active_memory_membership"] = 1.0 - np.exp(-out["active_minutes"] / 60.0)
    out["post_memory_membership"] = 1.0 - np.exp(-out["post_3h_minutes"] / 90.0)
    out["target_prior_membership"] = ramp(prior_rate, max(train_rate * 0.75, 0.01), max(train_rate * 3.0, 0.05))
    out["target_prior_residual"] = prior_rate - train_rate
    out["relation_active_interaction"] = out["relation_score"] * out["active_memory_membership"]
    out["relation_post_interaction"] = out["relation_score"] * out["post_memory_membership"]
    out["relation_weather_interaction"] = out["relation_score"] * out["weather_high_membership"]
    out["relation_arrival_pressure_interaction"] = out["relation_score"] * out["arrival_pressure_membership"]
    out["relation_prior_interaction"] = out["relation_score"] * out["target_prior_membership"]
    out["soft_pressure_index"] = (
        out["weather_high_membership"]
        + out["arrival_pressure_membership"]
        + out["departure_pressure_membership"]
        + out["target_prior_membership"]
        + out["active_memory_membership"]
        + out["post_memory_membership"]
    ) / 6.0
    return out


def design(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    x = df[numeric].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=float)
    return pd.concat([x, dummies], axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, success_col: str, features: list[str]) -> np.ndarray:
    x_train_raw = design(train, features)
    x_test_raw = design(test, features)
    x_train_raw, x_test_raw = align_design(x_train_raw, x_test_raw)
    x_train, y_train, w_train = expanded_binomial_frame(train, x_train_raw, success_col)
    model = LGBMClassifier(
        n_estimators=260,
        num_leaves=24,
        learning_rate=0.035,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    return np.clip(model.predict_proba(x_test_raw)[:, 1], 1e-5, 1 - 1e-5)


def model_features(include_ablations: bool) -> dict[str, list[str]]:
    if not include_ablations:
        return MODEL_FEATURES
    out = MODEL_FEATURES.copy()
    out.update(ABLATION_FEATURES)
    return out


def validation_folds(panel: pd.DataFrame, validation: str, first_test_month: int, min_train_months: int):
    if validation == "month":
        for test_month in sorted(panel["month"].unique()):
            train = panel[panel["month"] != test_month].copy()
            test = panel[panel["month"] == test_month].copy()
            yield f"month_{int(test_month):02d}", train, test
    elif validation == "rolling":
        months = sorted(panel["month"].unique())
        for test_month in [m for m in months if m >= first_test_month]:
            train_months = [m for m in months if m < test_month]
            if len(train_months) < min_train_months:
                continue
            train = panel[panel["month"].isin(train_months)].copy()
            test = panel[panel["month"] == test_month].copy()
            yield f"rolling_month_{int(test_month):02d}", train, test
    elif validation == "airport_group":
        present = set(panel["airport"].astype(str).unique())
        for group in AIRPORT_GROUPS:
            test_airports = [airport for airport in group if airport in present]
            if not test_airports:
                continue
            train = panel[~panel["airport"].isin(test_airports)].copy()
            test = panel[panel["airport"].isin(test_airports)].copy()
            yield "airport_" + "_".join(test_airports), train, test
    else:
        raise ValueError(f"Unknown validation mode: {validation}")


def run(
    months: list[int],
    airports: list[str] | None,
    horizon: int,
    output_name: str,
    validation: str,
    include_ablations: bool,
    first_test_month: int,
    min_train_months: int,
) -> None:
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    base_panel = load_panel(months, airports)
    specs = model_features(include_ablations)
    rows = []
    fold_rows = []
    prediction_parts = []
    for target, success_col in TARGETS.items():
        panel = base_panel.merge(relation_scores(months, target, horizon), on=["airport", "utc_hour", "month"], how="inner")
        rows.append(metric_row(target, f"Relation-DCSI h{horizon}", panel.assign(pred_prob=panel["relation_score"]), success_col))
        for model_name, features in specs.items():
            fold_parts = []
            for fold_id, raw_train, raw_test in validation_folds(panel, validation, first_test_month, min_train_months):
                if raw_train.empty or raw_test.empty:
                    continue
                train_rate = raw_train[success_col].sum() / max(raw_train["arrivals"].sum(), 1.0)
                train = add_soft_features(raw_train, target, train_rate)
                test = add_soft_features(raw_test, target, train_rate)
                prob = fit_predict(train, test, success_col, features)
                fold = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
                fold["target"] = target
                fold["model"] = model_name
                fold["pred_prob"] = prob
                fold["fold_id"] = fold_id
                fold_parts.append(fold)
                fold_rows.append(metric_row(target, model_name, fold, success_col) | {"fold_id": fold_id})
            pred = pd.concat(fold_parts, ignore_index=True)
            rows.append(metric_row(target, model_name, pred, success_col))
            prediction_parts.append(pred)
    metrics = pd.DataFrame(rows)
    fold_metrics = pd.DataFrame(fold_rows)
    gains = add_gain_table(metrics)
    ablations = add_ablation_summary(gains)
    metrics.to_csv(out / "afre_smoke_metrics.csv", index=False)
    fold_metrics.to_csv(out / "afre_fold_metrics.csv", index=False)
    gains.to_csv(out / "afre_smoke_gains.csv", index=False)
    ablations.to_csv(out / "afre_ablation_summary.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(out / "afre_smoke_predictions.csv", index=False)
    write_assessment(out, gains, months, airports, horizon, validation, include_ablations)
    print(f"wrote {out}")


def add_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    ref = out[out["model"].str.startswith("Relation-DCSI")][
        ["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture", "top_decile_lift"]
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
    full = out[out["model"].eq("LightGBM full-demand advisory")][
        ["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture", "top_decile_lift"]
    ].rename(
        columns={
            "auc": "full_auc",
            "pr_auc": "full_pr_auc",
            "brier": "full_brier",
            "ece_10": "full_ece_10",
            "top10_capture": "full_top10_capture",
            "top_decile_lift": "full_top_decile_lift",
        }
    )
    out = out.merge(ref, on="target", how="left").merge(full, on="target", how="left")
    for prefix in ["relation", "full"]:
        out[f"auc_gain_vs_{prefix}"] = out["auc"] - out[f"{prefix}_auc"]
        out[f"pr_auc_gain_vs_{prefix}"] = out["pr_auc"] - out[f"{prefix}_pr_auc"]
        out[f"brier_gain_vs_{prefix}"] = out[f"{prefix}_brier"] - out["brier"]
        out[f"ece10_gain_vs_{prefix}"] = out[f"{prefix}_ece_10"] - out["ece_10"]
        out[f"top10_capture_gain_vs_{prefix}"] = out["top10_capture"] - out[f"{prefix}_top10_capture"]
        out[f"top_lift_gain_vs_{prefix}"] = out["top_decile_lift"] - out[f"{prefix}_top_decile_lift"]
    return out


def add_ablation_summary(gains: pd.DataFrame) -> pd.DataFrame:
    afre = gains[gains["model"].eq("AFRE soft evidence")][
        ["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture", "top_decile_lift"]
    ].rename(
        columns={
            "auc": "afre_auc",
            "pr_auc": "afre_pr_auc",
            "brier": "afre_brier",
            "ece_10": "afre_ece_10",
            "top10_capture": "afre_top10_capture",
            "top_decile_lift": "afre_top_decile_lift",
        }
    )
    out = gains.merge(afre, on="target", how="left")
    out["auc_loss_vs_afre"] = out["afre_auc"] - out["auc"]
    out["pr_auc_loss_vs_afre"] = out["afre_pr_auc"] - out["pr_auc"]
    out["brier_loss_vs_afre"] = out["brier"] - out["afre_brier"]
    out["ece10_loss_vs_afre"] = out["ece_10"] - out["afre_ece_10"]
    out["top10_capture_loss_vs_afre"] = out["afre_top10_capture"] - out["top10_capture"]
    return out[
        [
            "target",
            "model",
            "auc_loss_vs_afre",
            "pr_auc_loss_vs_afre",
            "brier_loss_vs_afre",
            "ece10_loss_vs_afre",
            "top10_capture_loss_vs_afre",
        ]
    ].sort_values(["target", "auc_loss_vs_afre", "pr_auc_loss_vs_afre"], ascending=[True, False, False])


def write_assessment(
    out: Path,
    gains: pd.DataFrame,
    months: list[int],
    airports: list[str] | None,
    horizon: int,
    validation: str,
    include_ablations: bool,
) -> None:
    lines = [
        "# AFRE smoke assessment",
        "",
        f"Scope: months {','.join(str(m) for m in months)}; airports {','.join(airports) if airports else 'ALL'}; Relation-DCSI horizon {horizon} h; validation {validation}.",
        "",
        "Gate: AFRE should beat Relation-DCSI on AUC and PR-AUC for both outcomes, and should improve at least one ranking metric versus full-demand advisory.",
        "",
    ]
    for target in sorted(gains["target"].unique()):
        afre = gains[(gains["target"].eq(target)) & (gains["model"].eq("AFRE soft evidence"))].iloc[0]
        usable = (
            afre["auc_gain_vs_relation"] > 0
            and afre["pr_auc_gain_vs_relation"] > 0
            and (
                afre["auc_gain_vs_full"] > 0
                or afre["pr_auc_gain_vs_full"] > 0
                or afre["top10_capture_gain_vs_full"] > 0
            )
        )
        verdict = "promising" if usable else "weak"
        lines.append(f"## {target}")
        lines.append(
            f"- Verdict: {verdict}. AFRE vs Relation-DCSI: AUC {afre['auc_gain_vs_relation']:+.3f}, "
            f"PR-AUC {afre['pr_auc_gain_vs_relation']:+.3f}, Brier {afre['brier_gain_vs_relation']:+.4f}, "
            f"top-10 capture {afre['top10_capture_gain_vs_relation']:+.3f}. "
            f"AFRE vs full-demand advisory: AUC {afre['auc_gain_vs_full']:+.3f}, "
            f"PR-AUC {afre['pr_auc_gain_vs_full']:+.3f}, top-10 capture {afre['top10_capture_gain_vs_full']:+.3f}."
        )
        lines.append("")
    if include_ablations:
        lines.append("Ablations are included in afre_ablation_summary.csv.")
        lines.append("")
    (out / "afre_smoke_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ALL")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-name", default="afre_smoke_10airports_3months")
    parser.add_argument("--validation", choices=["month", "rolling", "airport_group"], default="month")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--first-test-month", type=int, default=4)
    parser.add_argument("--min-train-months", type=int, default=3)
    args = parser.parse_args()
    run(
        parse_int_list(args.months),
        parse_airports(args.airports),
        args.horizon,
        args.output_name,
        args.validation,
        args.include_ablations,
        args.first_test_month,
        args.min_train_months,
    )


if __name__ == "__main__":
    main()
