from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fusion_prediction_increment import evaluate
from fusion_strengthening_prediction_diagnostics import grouped_pr_auc, top_decile_lift


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "results" / "experiments"
PANEL_FILE = (
    ROOT
    / "fusion_framework_strengthening"
    / "temporal_availability_full_2025"
    / "temporal_availability_panel.csv"
)
DCSI_PREDICTIONS = (
    ROOT
    / "fusion_framework_strengthening"
    / "dcsi_online_lead_validation"
    / "online_lead_full_2025"
    / "online_lead_predictions.csv"
)
OUT_ROOT = ROOT / "applied_soft_computing_smoke"

TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "cancellation": "cancel_count",
}

BASE_NUMERIC = [
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

ACTION_NUMERIC = [
    "active_hours_capped",
    "post_3h_hours_capped",
    "active_before_minutes",
    "active_within_minutes",
    "post_1h_known_minutes",
    "post_3h_known_minutes",
    "active_before_mild",
    "active_within_mild",
    "post_1h_known_mild",
    "post_3h_known_mild",
]

FUZZY_NUMERIC = [
    "f_weather_stress",
    "f_low_visibility",
    "f_low_ceiling",
    "f_wind_stress",
    "f_arrival_bank",
    "f_departure_bank",
    "f_active_constraint",
    "f_recovery_constraint",
    "f_active_mild_conflict",
    "f_recovery_mild_conflict",
    "f_weather_action_conjunction",
    "f_demand_action_conjunction",
]

DCSI_NUMERIC = [
    "dcsi_online_baseline",
    "dcsi_online_fixed_decay",
    "dcsi_online_relation",
    "dcsi_relation_margin",
    "dcsi_relation_fuzzy_conjunction",
]

CATEGORICAL = ["airport", "local_hour", "day_of_week"]


def parse_int_list(text: str) -> list[int]:
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
    return sorted(set(out))


def parse_airports(text: str) -> list[str] | None:
    if text.strip().upper() in {"ALL", "*"}:
        return None
    return [part.strip().upper() for part in text.split(",") if part.strip()]


def s_curve(values: pd.Series, low: float, high: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if high <= low:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - low) / (high - low)).clip(0, 1).fillna(0.0)


def reverse_s_curve(values: pd.Series, low: float, high: float) -> pd.Series:
    return 1.0 - s_curve(values, low, high)


def add_fuzzy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["f_weather_stress"] = s_curve(out["weather_score"], 1.5, 5.0)
    out["f_low_visibility"] = reverse_s_curve(out["visibility_km"], 3.0, 12.0)
    out["f_low_ceiling"] = reverse_s_curve(out["ceiling_m"].fillna(2000.0), 150.0, 1200.0)
    out["f_wind_stress"] = s_curve(out["wind_speed_mps"], 4.0, 14.0)
    out["f_arrival_bank"] = s_curve(out["arrival_bank_intensity"], 0.8, 1.8)
    out["f_departure_bank"] = s_curve(out["departure_bank_intensity"], 0.8, 1.8)
    out["f_active_constraint"] = s_curve(out["active_within_minutes"], 10.0, 60.0)
    out["f_recovery_constraint"] = s_curve(out["post_3h_known_minutes"], 15.0, 180.0)
    out["f_active_mild_conflict"] = out["f_active_constraint"] * out["mild_weather_abs"].fillna(0.0)
    out["f_recovery_mild_conflict"] = out["f_recovery_constraint"] * out["mild_weather_abs"].fillna(0.0)
    out["f_weather_action_conjunction"] = out["f_weather_stress"] * np.maximum(
        out["f_active_constraint"], out["f_recovery_constraint"]
    )
    out["f_demand_action_conjunction"] = np.maximum(out["f_arrival_bank"], out["f_departure_bank"]) * np.maximum(
        out["f_active_constraint"], out["f_recovery_constraint"]
    )
    return out


def load_panel(months: list[int], airports: list[str] | None) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months)].copy()
    if airports:
        panel = panel[panel["airport"].isin(airports)].copy()
    panel = panel[(panel["arrivals"] > 0) & panel["weather_score"].notna()].copy()
    for col in BASE_NUMERIC + ACTION_NUMERIC:
        panel[col] = pd.to_numeric(panel.get(col, 0.0), errors="coerce")
        panel[col] = panel[col].fillna(panel[col].median())
    for col in CATEGORICAL:
        panel[col] = panel[col].astype(str)
    return add_fuzzy_features(panel).reset_index(drop=True)


def add_dcsi_meta_features(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    pred = pd.read_csv(DCSI_PREDICTIONS, parse_dates=["utc_hour"])
    pred = pred[
        pred["target"].eq(target)
        & pred["horizon"].eq(1)
        & pred["model"].isin(["online_baseline", "online_fixed_decay", "online_relation_DCSI"])
    ].copy()
    pred = pred.pivot_table(
        index=["airport", "utc_hour"],
        columns="model",
        values="pred_prob",
        aggfunc="first",
    ).reset_index()
    pred = pred.rename(
        columns={
            "online_baseline": "dcsi_online_baseline",
            "online_fixed_decay": "dcsi_online_fixed_decay",
            "online_relation_DCSI": "dcsi_online_relation",
        }
    )
    out = panel.merge(pred, on=["airport", "utc_hour"], how="inner")
    out["dcsi_relation_margin"] = out["dcsi_online_relation"] - out["dcsi_online_baseline"]
    out["dcsi_relation_fuzzy_conjunction"] = out["dcsi_online_relation"] * np.maximum(
        out["f_weather_action_conjunction"], out["f_demand_action_conjunction"]
    )
    for col in DCSI_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(out[col].median())
    return out.reset_index(drop=True)


def expanded_binomial_frame(df: pd.DataFrame, features: pd.DataFrame, success_col: str):
    successes = pd.to_numeric(df[success_col], errors="coerce").fillna(0.0).to_numpy(float)
    totals = pd.to_numeric(df["arrivals"], errors="coerce").fillna(0.0).to_numpy(float)
    failures = totals - successes
    x = pd.concat([features, features], ignore_index=True)
    y = np.r_[np.ones(len(df), dtype=int), np.zeros(len(df), dtype=int)]
    weights = np.r_[successes, failures]
    keep = weights > 0
    return x.loc[keep].reset_index(drop=True), y[keep], weights[keep]


def design(df: pd.DataFrame, numeric: list[str], include_categories: bool = True) -> pd.DataFrame:
    x = df[numeric].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    if include_categories:
        dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL, dtype=float)
        x = pd.concat([x, dummies], axis=1)
    return x


def align_design(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0.0)
    return train_x, test_x


def isotonic_calibrate(train_prob: np.ndarray, successes: np.ndarray, totals: np.ndarray, test_prob: np.ndarray) -> np.ndarray:
    y = successes / np.maximum(totals, 1.0)
    order = np.argsort(train_prob)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1 - 1e-5)
    cal.fit(train_prob[order], y[order], sample_weight=totals[order])
    return np.clip(cal.predict(test_prob), 1e-5, 1 - 1e-5)


def ece(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total_weight = totals.sum()
    if total_weight <= 0:
        return np.nan
    out = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if not mask.any():
            continue
        obs = successes[mask].sum() / totals[mask].sum()
        pred = np.average(prob[mask], weights=totals[mask])
        out += totals[mask].sum() / total_weight * abs(obs - pred)
    return float(out)


def capture_at_fraction(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray, fraction: float = 0.10) -> float:
    n = max(1, int(np.ceil(len(prob) * fraction)))
    idx = np.argsort(prob)[::-1][:n]
    total_events = successes.sum()
    if total_events <= 0:
        return np.nan
    return float(successes[idx].sum() / total_events)


def metric_row(target: str, model: str, pred: pd.DataFrame, success_col: str) -> dict[str, object]:
    successes = pred[success_col].to_numpy(float)
    totals = pred["arrivals"].to_numpy(float)
    prob = pred["pred_prob"].to_numpy(float)
    top_rate, top_lift = top_decile_lift(successes, totals, prob)
    return {
        "target": target,
        "model": model,
        **evaluate(successes, totals, prob),
        "pr_auc": grouped_pr_auc(successes, totals, prob),
        "ece_10": ece(successes, totals, prob),
        "top10_capture": capture_at_fraction(successes, totals, prob, 0.10),
        "top_decile_precision": top_rate,
        "top_decile_lift": top_lift,
    }


def fit_predict_logit(train: pd.DataFrame, test: pd.DataFrame, success_col: str, numeric: list[str], calibrated: bool) -> np.ndarray:
    x_train_raw = design(train, numeric)
    x_test_raw = design(test, numeric)
    x_train_raw, x_test_raw = align_design(x_train_raw, x_test_raw)
    x_train, y_train, w_train = expanded_binomial_frame(train, x_train_raw, success_col)
    scaler = StandardScaler()
    scaler.fit(x_train_raw)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test_raw)
    model = LogisticRegression(max_iter=2000, C=0.5, solver="lbfgs")
    model.fit(x_train_scaled, y_train, sample_weight=w_train)
    test_prob = np.clip(model.predict_proba(x_test_scaled)[:, 1], 1e-5, 1 - 1e-5)
    if not calibrated:
        return test_prob
    train_prob = np.clip(model.predict_proba(scaler.transform(x_train_raw))[:, 1], 1e-5, 1 - 1e-5)
    return isotonic_calibrate(
        train_prob,
        train[success_col].to_numpy(float),
        train["arrivals"].to_numpy(float),
        test_prob,
    )


def fit_predict_gbdt(train: pd.DataFrame, test: pd.DataFrame, success_col: str, numeric: list[str], calibrated: bool) -> np.ndarray:
    x_train_raw = design(train, numeric)
    x_test_raw = design(test, numeric)
    x_train_raw, x_test_raw = align_design(x_train_raw, x_test_raw)
    x_train, y_train, w_train = expanded_binomial_frame(train, x_train_raw, success_col)
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.045,
        max_leaf_nodes=18,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    test_prob = np.clip(model.predict_proba(x_test_raw)[:, 1], 1e-5, 1 - 1e-5)
    if not calibrated:
        return test_prob
    train_prob = np.clip(model.predict_proba(x_train_raw)[:, 1], 1e-5, 1 - 1e-5)
    return isotonic_calibrate(
        train_prob,
        train[success_col].to_numpy(float),
        train["arrivals"].to_numpy(float),
        test_prob,
    )


def granular_predict(train: pd.DataFrame, test: pd.DataFrame, success_col: str) -> np.ndarray:
    global_rate = train[success_col].sum() / train["arrivals"].sum()
    cuts = {}
    source_cols = {
        "weather_granule": "f_weather_stress",
        "demand_granule": "f_arrival_bank",
        "action_granule": "f_active_constraint",
        "recovery_granule": "f_recovery_constraint",
    }
    train_g = train.copy()
    test_g = test.copy()
    for label, col in source_cols.items():
        q = train[col].quantile([0.33, 0.67]).to_numpy(float)
        if not np.isfinite(q).all() or q[0] == q[1]:
            q = np.array([0.25, 0.75])
        cuts[label] = q
        bins = [-np.inf, q[0], q[1], np.inf]
        train_g[label] = pd.cut(train_g[col], bins=bins, labels=["low", "mid", "high"], include_lowest=True)
        test_g[label] = pd.cut(test_g[col], bins=bins, labels=["low", "mid", "high"], include_lowest=True)
    keys = list(source_cols)
    table = (
        train_g.groupby(keys, observed=False)
        .agg(events=(success_col, "sum"), arrivals=("arrivals", "sum"))
        .reset_index()
    )
    table["granular_rate"] = (table["events"] + 20 * global_rate) / (table["arrivals"] + 20)
    out = test_g[keys].merge(table[keys + ["granular_rate"]], on=keys, how="left")["granular_rate"]
    return out.fillna(global_rate).clip(1e-5, 1 - 1e-5).to_numpy(float)


def direct_column_predict(test: pd.DataFrame, column: str) -> np.ndarray:
    return test[column].clip(1e-5, 1 - 1e-5).to_numpy(float)


def calibrated_column_predict(train: pd.DataFrame, test: pd.DataFrame, success_col: str, column: str) -> np.ndarray:
    train_prob = direct_column_predict(train, column)
    test_prob = direct_column_predict(test, column)
    return isotonic_calibrate(
        train_prob,
        train[success_col].to_numpy(float),
        train["arrivals"].to_numpy(float),
        test_prob,
    )


def run(months: list[int], airports: list[str] | None, output_name: str) -> None:
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel(months, airports)
    rows = []
    all_pred = []
    for target, success_col in TARGETS.items():
        target_panel = add_dcsi_meta_features(panel, target)
        method_specs = [
            ("CWD logistic", "logit", BASE_NUMERIC, False),
            ("CWD+action logistic", "logit", BASE_NUMERIC + ACTION_NUMERIC, False),
            ("Relation-DCSI reference", "direct:dcsi_online_relation", [], False),
            ("UC-DCSI isotonic calibrated state", "calibrate:dcsi_online_relation", [], False),
            ("UC-FES fuzzy calibrated logit", "logit", BASE_NUMERIC + ACTION_NUMERIC + FUZZY_NUMERIC, True),
            ("DCSI+UC-FES calibrated logit", "logit", DCSI_NUMERIC + FUZZY_NUMERIC, True),
            ("G-DCSI granular state table", "granular", [], False),
            ("Soft-GBDT fuzzy calibrated", "gbdt", BASE_NUMERIC + ACTION_NUMERIC + FUZZY_NUMERIC, True),
            ("DCSI+Soft-GBDT fuzzy calibrated", "gbdt", BASE_NUMERIC + ACTION_NUMERIC + FUZZY_NUMERIC + DCSI_NUMERIC, True),
        ]
        for model_name, family, numeric, calibrated in method_specs:
            pred_parts = []
            for test_month in sorted(target_panel["month"].unique()):
                train = target_panel[target_panel["month"] != test_month].copy()
                test = target_panel[target_panel["month"] == test_month].copy()
                if family == "logit":
                    prob = fit_predict_logit(train, test, success_col, numeric, calibrated)
                elif family == "gbdt":
                    prob = fit_predict_gbdt(train, test, success_col, numeric, calibrated)
                elif family == "granular":
                    prob = granular_predict(train, test, success_col)
                elif family.startswith("direct:"):
                    prob = direct_column_predict(test, family.split(":", 1)[1])
                elif family.startswith("calibrate:"):
                    prob = calibrated_column_predict(train, test, success_col, family.split(":", 1)[1])
                else:
                    raise ValueError(family)
                fold = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
                fold["pred_prob"] = np.clip(prob, 1e-5, 1 - 1e-5)
                fold["test_month"] = test_month
                pred_parts.append(fold)
            pred = pd.concat(pred_parts, ignore_index=True)
            rows.append(metric_row(target, model_name, pred, success_col))
            all_pred.append(pred.assign(target=target, model=model_name))
    metrics = pd.DataFrame(rows)
    baseline = metrics[metrics["model"].eq("CWD+action logistic")][
        ["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture", "top_decile_lift"]
    ].rename(
        columns={
            "auc": "baseline_auc",
            "pr_auc": "baseline_pr_auc",
            "brier": "baseline_brier",
            "ece_10": "baseline_ece_10",
            "top10_capture": "baseline_top10_capture",
            "top_decile_lift": "baseline_top_decile_lift",
        }
    )
    gains = metrics.merge(baseline, on="target", how="left")
    gains["auc_gain"] = gains["auc"] - gains["baseline_auc"]
    gains["pr_auc_gain"] = gains["pr_auc"] - gains["baseline_pr_auc"]
    gains["brier_gain"] = gains["baseline_brier"] - gains["brier"]
    gains["ece10_gain"] = gains["baseline_ece_10"] - gains["ece_10"]
    gains["top10_capture_gain"] = gains["top10_capture"] - gains["baseline_top10_capture"]
    gains["top_lift_gain"] = gains["top_decile_lift"] - gains["baseline_top_decile_lift"]
    relation = metrics[metrics["model"].eq("Relation-DCSI reference")][
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
    gains = gains.merge(relation, on="target", how="left")
    gains["auc_gain_vs_relation"] = gains["auc"] - gains["relation_auc"]
    gains["pr_auc_gain_vs_relation"] = gains["pr_auc"] - gains["relation_pr_auc"]
    gains["brier_gain_vs_relation"] = gains["relation_brier"] - gains["brier"]
    gains["ece10_gain_vs_relation"] = gains["relation_ece_10"] - gains["ece_10"]
    gains["top10_capture_gain_vs_relation"] = gains["top10_capture"] - gains["relation_top10_capture"]
    gains["top_lift_gain_vs_relation"] = gains["top_decile_lift"] - gains["relation_top_decile_lift"]
    metrics.to_csv(out / "asoc_smoke_metrics.csv", index=False)
    gains.to_csv(out / "asoc_smoke_gains_vs_action_logit.csv", index=False)
    pd.concat(all_pred, ignore_index=True).to_csv(out / "asoc_smoke_predictions.csv", index=False)
    write_assessment(out, metrics, gains, months, airports)
    print(f"wrote {out}")


def write_assessment(out: Path, metrics: pd.DataFrame, gains: pd.DataFrame, months: list[int], airports: list[str] | None) -> None:
    lines = [
        "# Applied Soft Computing smoke assessment",
        "",
        f"Scope: months {','.join(str(m) for m in months)}; airports {','.join(airports) if airports else 'ALL'}.",
        "",
        "Gate: a method is worth full expansion when it improves at least two of AUC, PR-AUC, Brier, ECE, and top-10% capture against Relation-DCSI, while keeping the other task from collapsing.",
        "",
    ]
    for target in TARGETS:
        use = gains[gains["target"].eq(target)].sort_values(
            ["auc_gain_vs_relation", "pr_auc_gain_vs_relation", "ece10_gain_vs_relation", "top10_capture_gain_vs_relation"],
            ascending=False,
        )
        best = use.iloc[0]
        lines.append(f"## {target}")
        lines.append(
            f"- Best model vs Relation-DCSI: {best['model']}; AUC {best['auc']:.3f} ({best['auc_gain_vs_relation']:+.3f}), "
            f"PR-AUC {best['pr_auc']:.3f} ({best['pr_auc_gain_vs_relation']:+.3f}), "
            f"Brier gain {best['brier_gain_vs_relation']:+.4f}, ECE gain {best['ece10_gain_vs_relation']:+.4f}, "
            f"top-10 capture gain {best['top10_capture_gain_vs_relation']:+.3f}."
        )
        for _, row in use.iterrows():
            if row["model"] == "Relation-DCSI reference":
                continue
            positive = sum(
                [
                    row["auc_gain_vs_relation"] > 0.005,
                    row["pr_auc_gain_vs_relation"] > 0.005,
                    row["brier_gain_vs_relation"] > 0.0005,
                    row["ece10_gain_vs_relation"] > 0.005,
                    row["top10_capture_gain_vs_relation"] > 0.01,
                ]
            )
            verdict = "promising" if positive >= 2 else "weak"
            lines.append(
                f"- {row['model']}: {verdict}; AUC gain {row['auc_gain_vs_relation']:+.3f}, "
                f"PR-AUC gain {row['pr_auc_gain_vs_relation']:+.3f}, Brier gain {row['brier_gain_vs_relation']:+.4f}, "
                f"ECE gain {row['ece10_gain_vs_relation']:+.4f}, top-10 capture gain {row['top10_capture_gain_vs_relation']:+.3f}."
            )
        lines.append("")
    metrics.to_markdown(out / "asoc_smoke_metrics.md", index=False)
    (out / "asoc_smoke_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ATL,DFW,EWR,ORD")
    parser.add_argument("--output-name", default="fuzzy_granular_smoke_4airports_3months")
    args = parser.parse_args()
    run(parse_int_list(args.months), parse_airports(args.airports), args.output_name)


if __name__ == "__main__":
    main()
