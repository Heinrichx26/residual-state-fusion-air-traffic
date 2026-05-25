from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from dynamic_constraint_state_inversion import (
    BASE_NUMERIC,
    DYNAMIC_NUMERIC,
    FIXED_NUMERIC,
    MAIN_10,
    MAIN_2025_PANEL,
    MAIN_CATS,
    TARGETS,
    TRANSFER_CATS,
    attach_dynamic,
    evaluate_leave_month,
    fit_predict,
    load_panel,
    parse_float_grid,
    parse_months,
)
from fusion_prediction_increment import evaluate
from fusion_strengthening_common import ROOT_OUT


OUT_ROOT = ROOT_OUT / "dcsi_kbs_strengthening"


def parse_airports(text: str) -> list[str] | None:
    if text.strip().upper() in {"ALL", "*"}:
        return None
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def action_lag_features(panel: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, list[str]]:
    out = panel.sort_values(["airport", "utc_hour"]).copy()
    out["active_impulse"] = (out["active_minutes"] / 60.0).clip(0, 1)
    cols = []
    for lag in range(max_lag + 1):
        col = f"active_lag_{lag:02d}"
        out[col] = out.groupby("airport", sort=False)["active_impulse"].shift(lag).fillna(0.0)
        cols.append(col)
    return out, cols


def time_since_action_features(panel: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for _, g in panel.sort_values(["airport", "utc_hour"]).groupby("airport", sort=False):
        g = g.copy()
        active = (g["active_minutes"].to_numpy(float) > 0).astype(int)
        since = np.full(len(g), horizon + 1, dtype=float)
        run = horizon + 1
        for i, val in enumerate(active):
            if val == 1:
                run = 0
            else:
                run = min(horizon + 1, run + 1)
            since[i] = run
        g["hours_since_active"] = np.minimum(since, horizon + 1)
        g["recent_action_clock"] = np.where(g["hours_since_active"] <= horizon, 1.0 / (1.0 + g["hours_since_active"]), 0.0)
        g["within_6h_after_action"] = ((g["hours_since_active"] > 0) & (g["hours_since_active"] <= 6)).astype(float)
        g["within_12h_after_action"] = ((g["hours_since_active"] > 0) & (g["hours_since_active"] <= 12)).astype(float)
        g["within_24h_after_action"] = ((g["hours_since_active"] > 0) & (g["hours_since_active"] <= 24)).astype(float)
        frames.append(g)
    cols = ["hours_since_active", "recent_action_clock", "within_6h_after_action", "within_12h_after_action", "within_24h_after_action"]
    return pd.concat(frames, ignore_index=True), cols


def fixed_decay_features(panel: pd.DataFrame, rho_values: list[float]) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy()
    cols = []
    for rho in rho_values:
        prefix = f"fixed_decay_{str(rho).replace('.', '')}"
        features = attach_dynamic(panel, rho)[
            ["row_id", "dynamic_constraint_state", "dynamic_recovery_state", "dynamic_mild_constraint", "dynamic_state_delta"]
        ].copy()
        rename = {
            "dynamic_constraint_state": f"{prefix}_state",
            "dynamic_recovery_state": f"{prefix}_recovery",
            "dynamic_mild_constraint": f"{prefix}_mild",
            "dynamic_state_delta": f"{prefix}_delta",
        }
        features = features.rename(columns=rename)
        out = out.merge(features, on="row_id", how="left")
        cols.extend(rename.values())
    return out, cols


def make_baseline_panel(panel: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out, lag_cols = action_lag_features(panel, max_lag)
    out, clock_cols = time_since_action_features(out, horizon=max(24, max_lag))
    out, decay_cols = fixed_decay_features(out, [0.90, 0.95])
    for col in lag_cols + clock_cols + decay_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    specs = {
        "calendar_weather_demand": BASE_NUMERIC,
        "fixed_active_post": FIXED_NUMERIC,
        "distributed_lag_24h": BASE_NUMERIC + lag_cols,
        "time_since_action": FIXED_NUMERIC + clock_cols,
        "fixed_decay_090_095": FIXED_NUMERIC + decay_cols,
    }
    return out, specs


def evaluate_logit_specs(panel: pd.DataFrame, months: list[int], specs: dict[str, list[str]], cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    for outer_month in months:
        train = panel[panel["month"] != outer_month].copy()
        test = panel[panel["month"] == outer_month].copy()
        for target, success_col in TARGETS.items():
            for name, cols in specs.items():
                pred, metrics = fit_predict(train, test, success_col, cols, cats)
                rows.append(metrics | {"target": target, "model": name, "fold_month": outer_month})
                pred["target"] = target
                pred["model"] = name
                predictions.append(pred)
    pred_all = pd.concat(predictions, ignore_index=True)
    for (target, model), g in pred_all.groupby(["target", "model"]):
        success_col = TARGETS[target]
        rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": model, "fold_month": "all"}
        )
    return pd.DataFrame(rows), pred_all


def lgbm_design(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_levels = {}
    for col in categorical_cols:
        all_levels[col] = sorted(train[col].astype(str).unique().tolist())
    def build(df: pd.DataFrame) -> pd.DataFrame:
        x = df[numeric_cols].copy()
        for col in numeric_cols:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            med = pd.to_numeric(train[col], errors="coerce").median()
            x[col] = x[col].fillna(med if np.isfinite(med) else 0.0)
        dummies = []
        for col in categorical_cols:
            values = df[col].astype(str)
            for level in all_levels[col][1:]:
                dummies.append(pd.Series((values == level).astype(float).to_numpy(), index=df.index, name=f"{col}_{level}"))
        if dummies:
            x = pd.concat([x] + dummies, axis=1)
        return x
    return build(train), build(test)


def expand_grouped_binary(x: pd.DataFrame, successes: np.ndarray, totals: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pos_w = successes.astype(float)
    neg_w = totals.astype(float) - successes.astype(float)
    x2 = pd.concat([x, x], ignore_index=True)
    y = np.r_[np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)]
    w = np.r_[pos_w, neg_w]
    keep = w > 0
    return x2.loc[keep].reset_index(drop=True), y[keep], w[keep]


def evaluate_lightgbm(panel: pd.DataFrame, months: list[int], numeric_cols: list[str], cats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    params = dict(
        n_estimators=180,
        num_leaves=24,
        learning_rate=0.045,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=42,
        verbosity=-1,
        n_jobs=2,
    )
    for outer_month in months:
        train = panel[panel["month"] != outer_month].copy()
        test = panel[panel["month"] == outer_month].copy()
        x_train_raw, x_test = lgbm_design(train, test, numeric_cols, cats)
        for target, success_col in TARGETS.items():
            x_train, y_train, weights = expand_grouped_binary(
                x_train_raw,
                train[success_col].to_numpy(float),
                train["arrivals"].to_numpy(float),
            )
            clf = LGBMClassifier(**params)
            clf.fit(x_train, y_train, sample_weight=weights)
            prob = np.clip(clf.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
            pred = test[["row_id", "airport", "utc_hour", "month", "arrivals", success_col]].copy()
            pred["pred_prob"] = prob
            pred["target"] = target
            pred["model"] = "lightgbm_tabular_lag"
            predictions.append(pred)
            rows.append(
                evaluate(pred[success_col].to_numpy(float), pred["arrivals"].to_numpy(float), prob)
                | {"target": target, "model": "lightgbm_tabular_lag", "fold_month": outer_month}
            )
    pred_all = pd.concat(predictions, ignore_index=True)
    for target, g in pred_all.groupby("target"):
        success_col = TARGETS[target]
        rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": "lightgbm_tabular_lag", "fold_month": "all"}
        )
    return pd.DataFrame(rows), pred_all


def selected_rho_table(rho_selection: pd.DataFrame) -> pd.DataFrame:
    table = rho_selection.copy()
    table["outer_month"] = table["outer_month"].astype(int)
    table = table.sort_values(["outer_month", "inner_mean_auc", "rho"], ascending=[True, False, False])
    return table.drop_duplicates("outer_month")[["outer_month", "rho"]]


def evaluate_dcsi_relation_model(
    panel: pd.DataFrame,
    months: list[int],
    rho_grid: list[float],
    cats: list[str],
    relation_cols: list[str],
    rho_selection: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rho_selection is None:
        _, _, rho_selection = evaluate_leave_month(panel, months, rho_grid, cats)
    selected = selected_rho_table(rho_selection)
    rows = []
    predictions = []
    numeric_cols = DYNAMIC_NUMERIC + relation_cols
    for outer_month, rho in selected.itertuples(index=False):
        dyn_panel = attach_dynamic(panel, float(rho))
        train = dyn_panel[dyn_panel["month"] != int(outer_month)].copy()
        test = dyn_panel[dyn_panel["month"] == int(outer_month)].copy()
        for target, success_col in TARGETS.items():
            pred, metrics = fit_predict(train, test, success_col, numeric_cols, cats)
            rows.append(metrics | {"target": target, "model": "DCSI_relation_state", "fold_month": int(outer_month), "rho": float(rho)})
            pred["target"] = target
            pred["model"] = "DCSI_relation_state"
            pred["rho"] = float(rho)
            predictions.append(pred)
    pred_all = pd.concat(predictions, ignore_index=True)
    for target, g in pred_all.groupby("target"):
        success_col = TARGETS[target]
        rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": "DCSI_relation_state", "fold_month": "all", "rho": "fold_selected"}
        )
    return pd.DataFrame(rows), pred_all


def build_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    overall = metrics[metrics["fold_month"].astype(str).eq("all")].copy()
    base = overall[overall["model"].eq("calendar_weather_demand")][["target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "base_auc", "brier": "base_brier", "log_loss": "base_log_loss"}
    )
    fixed = overall[overall["model"].eq("fixed_active_post")][["target", "auc", "brier", "log_loss"]].rename(
        columns={"auc": "fixed_auc", "brier": "fixed_brier", "log_loss": "fixed_log_loss"}
    )
    out = overall.merge(base, on="target", how="left").merge(fixed, on="target", how="left")
    out["auc_gain_vs_base"] = out["auc"] - out["base_auc"]
    out["auc_gain_vs_fixed"] = out["auc"] - out["fixed_auc"]
    out["brier_gain_vs_base"] = out["base_brier"] - out["brier"]
    out["brier_gain_vs_fixed"] = out["fixed_brier"] - out["brier"]
    out["log_loss_gain_vs_base"] = out["base_log_loss"] - out["log_loss"]
    out["log_loss_gain_vs_fixed"] = out["fixed_log_loss"] - out["log_loss"]
    return out


def attach_existing_dcsi_predictions(out_predictions: pd.DataFrame, existing_dir: Path, months: list[int]) -> pd.DataFrame:
    pred_path = existing_dir / "dcsi_cv_predictions.csv"
    if not pred_path.exists():
        return out_predictions
    dcsi = pd.read_csv(pred_path, parse_dates=["utc_hour"])
    dcsi = dcsi[dcsi["month"].isin(months)].copy()
    dcsi["model"] = dcsi["model"].replace({"baseline": "calendar_weather_demand", "fixed_window": "fixed_active_post", "dynamic_state": "DCSI"})
    dcsi = dcsi[dcsi["model"].eq("DCSI")].copy()
    keep_cols = ["row_id", "airport", "utc_hour", "month", "arrivals", "target", "model", "pred_prob"]
    target_frames = []
    for target, success_col in TARGETS.items():
        subset = dcsi[dcsi["target"].eq(target)].copy()
        if success_col not in subset.columns:
            subset[success_col] = subset["arr_delay60_count"] if target == "long_arrival_delay" else subset["cancel_count"]
        target_frames.append(subset[keep_cols + [success_col]])
    dcsi = pd.concat(target_frames, ignore_index=True)
    return pd.concat([out_predictions, dcsi], ignore_index=True)


def bootstrap_auc_ci(predictions: pd.DataFrame, comparisons: list[tuple[str, str]], n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_rows = []
    for target, success_col in TARGETS.items():
        target_pred = predictions[predictions["target"].eq(target)].copy()
        target_pred["date"] = pd.to_datetime(target_pred["utc_hour"]).dt.date.astype(str)
        target_pred["block"] = target_pred["airport"].astype(str) + "_" + target_pred["date"]
        for reference, candidate in comparisons:
            ref = target_pred[target_pred["model"].eq(reference)][["airport", "utc_hour", "arrivals", success_col, "pred_prob", "block"]].rename(
                columns={"pred_prob": "ref_prob"}
            )
            cand = target_pred[target_pred["model"].eq(candidate)][["airport", "utc_hour", "pred_prob"]].rename(columns={"pred_prob": "cand_prob"})
            paired = ref.merge(cand, on=["airport", "utc_hour"], how="inner")
            if paired.empty:
                continue
            blocks = paired["block"].drop_duplicates().to_numpy()
            block_map = {block: i for i, block in enumerate(blocks)}
            block_code = paired["block"].map(block_map).to_numpy(int)
            successes = paired[success_col].to_numpy(float)
            totals = paired["arrivals"].to_numpy(float)
            ref_prob = paired["ref_prob"].to_numpy(float)
            cand_prob = paired["cand_prob"].to_numpy(float)
            diffs = []
            for _ in range(n_boot):
                sampled = rng.integers(0, len(blocks), size=len(blocks))
                block_counts = np.bincount(sampled, minlength=len(blocks))
                row_counts = block_counts[block_code].astype(float)
                keep = row_counts > 0
                boot_successes = successes[keep] * row_counts[keep]
                boot_totals = totals[keep] * row_counts[keep]
                ref_auc = evaluate(
                    boot_successes,
                    boot_totals,
                    ref_prob[keep],
                )["auc"]
                cand_auc = evaluate(
                    boot_successes,
                    boot_totals,
                    cand_prob[keep],
                )["auc"]
                if np.isfinite(ref_auc) and np.isfinite(cand_auc):
                    diffs.append(cand_auc - ref_auc)
            diffs = np.asarray(diffs, dtype=float)
            point = evaluate(successes, totals, cand_prob)["auc"] - evaluate(successes, totals, ref_prob)["auc"]
            out_rows.append(
                {
                    "target": target,
                    "reference": reference,
                    "candidate": candidate,
                    "auc_diff": point,
                    "ci_low": float(np.quantile(diffs, 0.025)) if len(diffs) else np.nan,
                    "ci_high": float(np.quantile(diffs, 0.975)) if len(diffs) else np.nan,
                    "p_diff_gt_0": float((diffs > 0).mean()) if len(diffs) else np.nan,
                    "bootstrap_reps": int(len(diffs)),
                }
            )
    return pd.DataFrame(out_rows)


def knowledge_schema_tables(out: Path) -> None:
    entities = pd.DataFrame(
        [
            ("AirportHour", "airport, utc_hour", "Common temporal object that binds all source records."),
            ("PhysicalObservation", "weather variables, mild-weather class", "Observed operating context."),
            ("ManagementAction", "GDP/GS interval, issue/start/end/cancel time", "Observed control trajectory."),
            ("PlannedDemand", "scheduled arrivals, bank intensity", "Planned workload context."),
            ("OutcomeClosure", "long delay, cancellation, mean/tail delay", "Delayed validation evidence."),
            ("ConstraintState", "Z, recovery Z, mild-weather Z, delta Z", "Hidden state reconstructed by DCSI."),
            ("ReasonFamily", "weather, demand, runway/surface, facility, other", "Semantic mechanism layer."),
        ],
        columns=["entity", "key_attributes", "knowledge_role"],
    )
    relations = pd.DataFrame(
        [
            ("PhysicalObservation", "observed_at", "AirportHour", "Weather context attaches to the airport-hour node.", "context feature"),
            ("ManagementAction", "acts_on", "AirportHour", "Advisory overlap creates the action impulse.", "state update"),
            ("PlannedDemand", "scheduled_for", "AirportHour", "Scheduled workload contextualizes expected pressure.", "context feature"),
            ("OutcomeClosure", "closes_with", "AirportHour", "Later BTS outcomes validate the inferred state.", "validation"),
            ("ManagementAction", "has_reason", "ReasonFamily", "Parsed reason text provides semantic constraint family.", "mechanism layer"),
            ("ConstraintState", "recovers_after", "ManagementAction", "State persistence propagates after active action impulses.", "state recursion"),
            ("ConstraintState", "validated_by", "OutcomeClosure", "Outcome closure tests state ranking and event timing.", "closure rule"),
        ],
        columns=["source_entity", "relation", "target_entity", "meaning", "used_by"],
    )
    rules = pd.DataFrame(
        [
            ("R1", "Only ManagementAction may update Z.", "prevents outcome leakage into state construction"),
            ("R2", "PhysicalObservation and PlannedDemand contextualize scoring but do not create action impulses.", "keeps source roles separated"),
            ("R3", "OutcomeClosure is held for validation and scoring.", "keeps delayed outcomes outside state recursion"),
            ("R4", "ReasonFamily partitions action impulses for mechanism memory.", "adds semantic interpretation"),
            ("R5", "Negative controls disturb airport or time relation while preserving broad exposure.", "tests action-link specificity"),
        ],
        columns=["rule_id", "knowledge_rule", "purpose"],
    )
    entities.to_csv(out / "kbs_knowledge_entities.csv", index=False)
    relations.to_csv(out / "kbs_knowledge_relations.csv", index=False)
    rules.to_csv(out / "kbs_knowledge_rules.csv", index=False)


def run(args: argparse.Namespace) -> None:
    months = parse_months(args.months)
    airports = parse_airports(args.airports)
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    rho_grid = parse_float_grid(args.rho_grid)
    panel = load_panel(MAIN_2025_PANEL, 2025, months, airports)
    panel, specs = make_baseline_panel(panel, args.max_lag)
    relation_cols = ["hours_since_active", "recent_action_clock", "within_6h_after_action", "within_12h_after_action", "within_24h_after_action"]
    metrics, predictions = evaluate_logit_specs(panel, months, specs, MAIN_CATS)
    lgbm_metrics, lgbm_predictions = evaluate_lightgbm(panel, months, specs["fixed_decay_090_095"] + [f"active_lag_{i:02d}" for i in range(args.max_lag + 1)], MAIN_CATS)
    metrics = pd.concat([metrics, lgbm_metrics], ignore_index=True)
    predictions = pd.concat([predictions, lgbm_predictions], ignore_index=True)

    dcsi_existing = ROOT_OUT / "dynamic_constraint_state_inversion_full_2025"
    selected_rho_source = None
    if set(months) != set(range(1, 13)) or airports != MAIN_10:
        dcsi_metrics, dcsi_pred, selected_rho_source = evaluate_leave_month(panel, months, rho_grid, MAIN_CATS)
        dcsi_pred["model"] = dcsi_pred["model"].replace({"dynamic_state": "DCSI"})
        dcsi_pred = dcsi_pred[dcsi_pred["model"].eq("DCSI")]
        metrics = pd.concat(
            [
                metrics,
                dcsi_metrics[dcsi_metrics["model"].eq("dynamic_state")].replace({"dynamic_state": "DCSI"}),
            ],
            ignore_index=True,
        )
        predictions = pd.concat([predictions, dcsi_pred], ignore_index=True)
    else:
        predictions = attach_existing_dcsi_predictions(predictions, dcsi_existing, months)
        dcsi_metrics = pd.read_csv(dcsi_existing / "dcsi_cv_metrics.csv")
        dcsi_metrics = dcsi_metrics[dcsi_metrics["model"].eq("dynamic_state")].replace({"dynamic_state": "DCSI"})
        metrics = pd.concat([metrics, dcsi_metrics], ignore_index=True)
        selected_rho_source = pd.read_csv(dcsi_existing / "dcsi_rho_selection.csv")

    relation_metrics, relation_predictions = evaluate_dcsi_relation_model(
        panel,
        months,
        rho_grid,
        MAIN_CATS,
        relation_cols,
        rho_selection=selected_rho_source,
    )
    metrics = pd.concat([metrics, relation_metrics], ignore_index=True)
    predictions = pd.concat([predictions, relation_predictions], ignore_index=True)

    gains = build_gain_table(metrics)
    comparisons = [
        ("fixed_active_post", "DCSI"),
        ("fixed_active_post", "DCSI_relation_state"),
        ("distributed_lag_24h", "DCSI"),
        ("distributed_lag_24h", "DCSI_relation_state"),
        ("time_since_action", "DCSI"),
        ("time_since_action", "DCSI_relation_state"),
        ("fixed_decay_090_095", "DCSI"),
        ("fixed_decay_090_095", "DCSI_relation_state"),
        ("lightgbm_tabular_lag", "DCSI"),
        ("lightgbm_tabular_lag", "DCSI_relation_state"),
    ]
    ci = bootstrap_auc_ci(predictions, comparisons, n_boot=args.bootstrap, seed=args.seed)
    knowledge_schema_tables(out)
    metrics.to_csv(out / "kbs_baseline_metrics.csv", index=False)
    gains.to_csv(out / "kbs_baseline_gains.csv", index=False)
    predictions.to_csv(out / "kbs_baseline_predictions.csv", index=False)
    ci.to_csv(out / "kbs_pair_bootstrap_auc_ci.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ATL,ORD")
    parser.add_argument("--rho-grid", default="0,0.25,0.50,0.70,0.85,0.90,0.93,0.95,0.97,0.985")
    parser.add_argument("--max-lag", type=int, default=24)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
