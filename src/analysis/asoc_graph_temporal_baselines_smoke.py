from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from asoc_fuzzy_residual_evidence_smoke import (
    ADVISORY_NUM,
    BASE_NUM,
    FUZZY_MEMBERSHIP_NUM,
    PRIOR_NUM,
    TARGETS,
    add_soft_features,
    load_panel,
    relation_scores,
    validation_folds,
)
from asoc_soft_computing_smoke import (
    OUT_ROOT,
    align_design,
    expanded_binomial_frame,
    metric_row,
    parse_airports,
    parse_int_list,
)


BASE_FEATURES = BASE_NUM + ADVISORY_NUM + PRIOR_NUM + FUZZY_MEMBERSHIP_NUM
GRAPH_SIGNALS = [
    "weather_score",
    "weather_high_membership",
    "scheduled_arrivals",
    "scheduled_departures",
    "target_prior_membership",
    "active_memory_membership",
    "post_memory_membership",
    "soft_pressure_index",
]
TEMPORAL_SIGNALS = [
    "weather_score",
    "scheduled_arrivals",
    "scheduled_departures",
    "target_prior_membership",
    "active_memory_membership",
    "post_memory_membership",
    "soft_pressure_index",
]
TEMPORAL_WINDOWS = [3, 6, 12, 24]
AIRPORT_METADATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "ourairports" / "airports.csv"


def design(df: pd.DataFrame, numeric: list[str]) -> pd.DataFrame:
    x = df[numeric].copy()
    for col in numeric:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    dummies = pd.get_dummies(
        df[["airport", "local_hour", "day_of_week"]],
        prefix=["airport", "local_hour", "day_of_week"],
        dtype=float,
    )
    return pd.concat([x, dummies], axis=1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, success_col: str, features: list[str]) -> np.ndarray:
    x_train_raw = design(train, features)
    x_test_raw = design(test, features)
    x_train_raw, x_test_raw = align_design(x_train_raw, x_test_raw)
    x_train, y_train, w_train = expanded_binomial_frame(train, x_train_raw, success_col)
    model = LGBMClassifier(
        n_estimators=220,
        num_leaves=20,
        learning_rate=0.04,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    return np.clip(model.predict_proba(x_test_raw)[:, 1], 1e-5, 1 - 1e-5)


def target_correlation_graph_weights(train: pd.DataFrame, success_col: str) -> pd.DataFrame:
    tmp = train[["airport", "utc_hour", "arrivals", success_col]].copy()
    tmp["event_rate"] = tmp[success_col].fillna(0.0) / tmp["arrivals"].replace(0, np.nan)
    pivot = tmp.pivot_table(index="utc_hour", columns="airport", values="event_rate", aggfunc="mean")
    corr = pivot.corr(min_periods=24).clip(lower=0).fillna(0.0)
    airports = list(corr.index)
    for airport in airports:
        corr.loc[airport, airport] = 0.0
    for airport in airports:
        row_sum = corr.loc[airport].sum()
        if row_sum <= 1e-12:
            values = pd.Series(1.0, index=airports)
            values.loc[airport] = 0.0
            corr.loc[airport] = values / values.sum()
        else:
            corr.loc[airport] = corr.loc[airport] / row_sum
    return corr


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def airport_metadata_graph_weights(airports: list[str]) -> pd.DataFrame:
    airports = sorted(set(str(airport) for airport in airports))
    if len(airports) < 2:
        return pd.DataFrame(np.zeros((len(airports), len(airports))), index=airports, columns=airports)
    meta = pd.read_csv(AIRPORT_METADATA)
    meta = meta[meta["iata_code"].isin(airports)].copy()
    meta = meta.set_index("iata_code").reindex(airports)
    lat = pd.to_numeric(meta["latitude_deg"], errors="coerce").to_numpy(float).copy()
    lon = pd.to_numeric(meta["longitude_deg"], errors="coerce").to_numpy(float).copy()
    elev = pd.to_numeric(meta["elevation_ft"], errors="coerce").fillna(0.0).to_numpy(float).copy()
    missing = np.isnan(lat) | np.isnan(lon)
    lat[missing] = np.nanmean(lat[~missing]) if np.any(~missing) else 0.0
    lon[missing] = np.nanmean(lon[~missing]) if np.any(~missing) else 0.0
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    lat2 = lat[None, :]
    lon2 = lon[None, :]
    dist = haversine_km(lat1, lon1, lat2, lon2)
    elev_gap = np.abs(elev[:, None] - elev[None, :])
    weights = np.exp(-dist / 1200.0) * np.exp(-elev_gap / 5000.0)
    np.fill_diagonal(weights, 0.0)
    out = pd.DataFrame(weights, index=airports, columns=airports)
    for airport in airports:
        row_sum = out.loc[airport].sum()
        if row_sum <= 1e-12:
            values = pd.Series(1.0, index=airports)
            values.loc[airport] = 0.0
            out.loc[airport] = values / values.sum()
        else:
            out.loc[airport] = out.loc[airport] / row_sum
    return out


def graph_weights(
    train: pd.DataFrame,
    success_col: str,
    graph_source: str,
    graph_airports: list[str] | None = None,
) -> pd.DataFrame:
    if graph_source == "target_correlation":
        return target_correlation_graph_weights(train, success_col)
    if graph_source == "airport_metadata":
        airports = graph_airports if graph_airports is not None else sorted(train["airport"].astype(str).unique())
        return airport_metadata_graph_weights(airports)
    raise ValueError(f"Unknown graph source: {graph_source}")


def add_graph_features(df: pd.DataFrame, weights: pd.DataFrame, train_medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    airports = list(weights.index)
    for signal in GRAPH_SIGNALS:
        values = pd.to_numeric(out[signal], errors="coerce").fillna(train_medians.get(signal, 0.0))
        out[signal] = values
        out[f"graph_{signal}"] = 0.0
    for _, idx in out.groupby("utc_hour", sort=False).groups.items():
        idx_list = list(idx)
        hour = out.loc[idx_list]
        present = set(hour["airport"].astype(str))
        for row_idx, airport in zip(idx_list, hour["airport"].astype(str)):
            if airport not in weights.index:
                continue
            usable_neighbors = [neighbor for neighbor in airports if neighbor in present and neighbor != airport]
            if not usable_neighbors:
                continue
            raw_w = weights.loc[airport, usable_neighbors].to_numpy(float)
            if raw_w.sum() <= 1e-12:
                raw_w = np.ones(len(usable_neighbors), dtype=float) / len(usable_neighbors)
            else:
                raw_w = raw_w / raw_w.sum()
            neighbor_frame = hour.set_index("airport").loc[usable_neighbors]
            for signal in GRAPH_SIGNALS:
                out.loc[row_idx, f"graph_{signal}"] = float(np.dot(raw_w, neighbor_frame[signal].to_numpy(float)))
    return out


def add_graph_features_fast(df: pd.DataFrame, weights: pd.DataFrame, train_medians: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    airports = list(weights.index)
    weight_matrix = weights.reindex(index=airports, columns=airports).fillna(0.0).to_numpy(float)
    for signal in GRAPH_SIGNALS:
        out[signal] = pd.to_numeric(out[signal], errors="coerce").fillna(train_medians.get(signal, 0.0))
        pivot = out.pivot_table(index="utc_hour", columns="airport", values=signal, aggfunc="first").reindex(columns=airports)
        observed = pivot.notna().to_numpy(float)
        values = pivot.fillna(0.0).to_numpy(float)
        numerator = values @ weight_matrix.T
        denominator = observed @ weight_matrix.T
        graph_values = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 1e-12,
        )
        graph = pd.DataFrame(graph_values, index=pivot.index, columns=airports)
        col = f"graph_{signal}"
        long = graph.stack().rename(col).reset_index()
        long.columns = ["utc_hour", "airport", col]
        out = out.merge(long, on=["utc_hour", "airport"], how="left")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def add_temporal_features(df: pd.DataFrame, train_medians: dict[str, float]) -> pd.DataFrame:
    parts = []
    for _, group in df.sort_values(["airport", "utc_hour"]).groupby("airport", sort=False):
        out = group.copy()
        for signal in TEMPORAL_SIGNALS:
            series = pd.to_numeric(out[signal], errors="coerce").fillna(train_medians.get(signal, 0.0))
            shifted = series.shift(1)
            out[f"temp_lag1_{signal}"] = shifted.fillna(train_medians.get(signal, 0.0))
            for window in TEMPORAL_WINDOWS:
                roll = shifted.rolling(window=window, min_periods=2)
                out[f"temp_mean{window}_{signal}"] = roll.mean()
                out[f"temp_max{window}_{signal}"] = roll.max()
                out[f"temp_std{window}_{signal}"] = roll.std()
        parts.append(out)
    merged = pd.concat(parts, ignore_index=True)
    temp_cols = [col for col in merged.columns if col.startswith("temp_")]
    for col in temp_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(train_medians.get(col, 0.0)).fillna(0.0)
    return merged


def train_medians(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    medians = {}
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce")
        medians[col] = float(values.median()) if values.notna().any() else 0.0
    return medians


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    graph_cols = [col for col in df.columns if col.startswith("graph_")]
    temp_cols = [col for col in df.columns if col.startswith("temp_")]
    return {
        "AFRE no relation reference": BASE_FEATURES,
        "Graph-neighborhood evidence": BASE_FEATURES + graph_cols,
        "Temporal-window evidence": BASE_FEATURES + temp_cols,
        "Graph-temporal evidence": BASE_FEATURES + graph_cols + temp_cols,
    }


def fold_prepare(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    target: str,
    success_col: str,
    graph_source: str = "target_correlation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rate = raw_train[success_col].sum() / max(raw_train["arrivals"].sum(), 1.0)
    train = add_soft_features(raw_train.copy(), target, train_rate)
    test = add_soft_features(raw_test.copy(), target, train_rate)
    medians = train_medians(train, sorted(set(BASE_FEATURES + GRAPH_SIGNALS + TEMPORAL_SIGNALS)))
    graph_airports = sorted(set(train["airport"].astype(str)).union(set(test["airport"].astype(str))))
    weights = graph_weights(train, success_col, graph_source, graph_airports)
    train = add_graph_features_fast(train, weights, medians)
    test = add_graph_features_fast(test, weights, medians)
    train = add_temporal_features(train, medians)
    test = add_temporal_features(test, medians)
    all_cols = [col for col in train.columns if col.startswith("temp_")]
    medians.update(train_medians(train, all_cols))
    for frame in [train, test]:
        for col in all_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(medians.get(col, 0.0)).fillna(0.0)
    return train, test


def add_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    ref = out[out["model"].eq("Relation-DCSI h1")][["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture"]].rename(
        columns={
            "auc": "relation_auc",
            "pr_auc": "relation_pr_auc",
            "brier": "relation_brier",
            "ece_10": "relation_ece_10",
            "top10_capture": "relation_top10_capture",
        }
    )
    afre = out[out["model"].eq("AFRE no relation reference")][
        ["target", "auc", "pr_auc", "brier", "ece_10", "top10_capture"]
    ].rename(
        columns={
            "auc": "afre_auc",
            "pr_auc": "afre_pr_auc",
            "brier": "afre_brier",
            "ece_10": "afre_ece_10",
            "top10_capture": "afre_top10_capture",
        }
    )
    out = out.merge(ref, on="target", how="left").merge(afre, on="target", how="left")
    for prefix in ["relation", "afre"]:
        out[f"auc_gain_vs_{prefix}"] = out["auc"] - out[f"{prefix}_auc"]
        out[f"pr_auc_gain_vs_{prefix}"] = out["pr_auc"] - out[f"{prefix}_pr_auc"]
        out[f"brier_gain_vs_{prefix}"] = out[f"{prefix}_brier"] - out["brier"]
        out[f"ece10_gain_vs_{prefix}"] = out[f"{prefix}_ece_10"] - out["ece_10"]
        out[f"top10_capture_gain_vs_{prefix}"] = out["top10_capture"] - out[f"{prefix}_top10_capture"]
    return out


def run(
    months: list[int],
    airports: list[str] | None,
    output_name: str,
    selected_models: list[str] | None,
    validation: str,
    first_test_month: int,
    min_train_months: int,
    graph_source: str,
) -> None:
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    base_panel = load_panel(months, airports)
    rows = []
    fold_rows = []
    predictions = []
    for target, success_col in TARGETS.items():
        panel = base_panel.merge(relation_scores(months, target, 1), on=["airport", "utc_hour", "month"], how="inner")
        fold_payloads = [
            (fold_id, raw_train, raw_test)
            for fold_id, raw_train, raw_test in validation_folds(panel, validation, first_test_month, min_train_months)
            if not raw_train.empty and not raw_test.empty
        ]
        if not fold_payloads:
            raise ValueError(f"No validation folds produced for target {target} with validation={validation}.")
        relation_eval = pd.concat(
            [
                raw_test[["airport", "utc_hour", "month", "arrivals", success_col, "relation_score"]].copy()
                for _, _, raw_test in fold_payloads
            ],
            ignore_index=True,
        ).rename(columns={"relation_score": "pred_prob"})
        relation_eval["target"] = target
        relation_eval["model"] = "Relation-DCSI h1"
        rows.append(metric_row(target, "Relation-DCSI h1", relation_eval, success_col))
        for fold_id, _, raw_test in fold_payloads:
            relation_fold = raw_test[["airport", "utc_hour", "month", "arrivals", success_col, "relation_score"]].copy()
            relation_fold = relation_fold.rename(columns={"relation_score": "pred_prob"})
            fold_rows.append(metric_row(target, "Relation-DCSI h1", relation_fold, success_col) | {"fold_id": fold_id})
        model_folds: dict[str, list[pd.DataFrame]] = {}
        for fold_id, raw_train, raw_test in fold_payloads:
            train, test = fold_prepare(raw_train, raw_test, target, success_col, graph_source)
            specs = feature_sets(train)
            if selected_models:
                specs = {name: features for name, features in specs.items() if name in selected_models}
            for model_name, features in specs.items():
                prob = fit_predict(train, test, success_col, features)
                fold = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
                fold["target"] = target
                fold["model"] = model_name
                fold["fold_id"] = fold_id
                fold["pred_prob"] = prob
                model_folds.setdefault(model_name, []).append(fold)
                fold_rows.append(metric_row(target, model_name, fold, success_col) | {"fold_id": fold_id})
                print(f"finished {target} {fold_id} {model_name}", flush=True)
        for model_name, fold_parts in model_folds.items():
            pred = pd.concat(fold_parts, ignore_index=True)
            rows.append(metric_row(target, model_name, pred, success_col))
            predictions.append(pred)
    metrics = pd.DataFrame(rows)
    fold_metrics = pd.DataFrame(fold_rows)
    gains = add_gain_table(metrics)
    metrics.to_csv(out / "graph_temporal_metrics.csv", index=False)
    fold_metrics.to_csv(out / "graph_temporal_fold_metrics.csv", index=False)
    gains.to_csv(out / "graph_temporal_gains.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "graph_temporal_predictions.csv", index=False)
    write_assessment(out, gains, months, airports, validation, graph_source)
    print(f"wrote {out}")


def write_assessment(
    out: Path,
    gains: pd.DataFrame,
    months: list[int],
    airports: list[str] | None,
    validation: str,
    graph_source: str,
) -> None:
    lines = [
        "# Graph and temporal-neighborhood baseline smoke assessment",
        "",
        f"Scope: months {','.join(str(m) for m in months)}; airports {','.join(airports) if airports else 'ALL'}; validation {validation}; graph source {graph_source}.",
        "",
        "Gate: continue only if a graph or temporal candidate improves PR-AUC by at least 0.005 versus AFRE no relation without a top-10 capture loss, or improves top-10 capture by at least 0.010 without PR-AUC loss.",
        "",
    ]
    candidates = gains[
        gains["model"].isin(["Graph-neighborhood evidence", "Temporal-window evidence", "Graph-temporal evidence"])
    ].copy()
    for target, group in candidates.groupby("target", sort=True):
        best = group.sort_values(["pr_auc_gain_vs_afre", "top10_capture_gain_vs_afre", "auc_gain_vs_afre"], ascending=False).iloc[0]
        usable = (
            best["pr_auc_gain_vs_afre"] >= 0.005
            and best["top10_capture_gain_vs_afre"] >= 0
        ) or (
            best["top10_capture_gain_vs_afre"] >= 0.010
            and best["pr_auc_gain_vs_afre"] >= 0
        )
        verdict = "promising" if usable else "weak"
        lines.append(f"## {target}")
        lines.append(
            f"- Verdict: {verdict}. Best candidate {best['model']}: "
            f"AUC gain vs AFRE {best['auc_gain_vs_afre']:+.3f}, "
            f"PR-AUC gain vs AFRE {best['pr_auc_gain_vs_afre']:+.3f}, "
            f"Brier gain vs AFRE {best['brier_gain_vs_afre']:+.4f}, "
            f"top-10 capture gain vs AFRE {best['top10_capture_gain_vs_afre']:+.3f}. "
            f"Gain vs Relation-DCSI: AUC {best['auc_gain_vs_relation']:+.3f}, "
            f"PR-AUC {best['pr_auc_gain_vs_relation']:+.3f}, top-10 {best['top10_capture_gain_vs_relation']:+.3f}."
        )
        lines.append("")
    (out / "graph_temporal_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ALL")
    parser.add_argument("--output-name", default="graph_temporal_smoke_10airports_3months")
    parser.add_argument("--models", default="ALL")
    parser.add_argument("--validation", choices=["month", "rolling", "airport_group"], default="month")
    parser.add_argument("--first-test-month", type=int, default=4)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--graph-source", choices=["target_correlation", "airport_metadata"], default="target_correlation")
    args = parser.parse_args()
    selected_models = None if args.models.strip().upper() == "ALL" else [
        item.strip() for item in args.models.split(";") if item.strip()
    ]
    run(
        parse_int_list(args.months),
        parse_airports(args.airports),
        args.output_name,
        selected_models,
        args.validation,
        args.first_test_month,
        args.min_train_months,
        args.graph_source,
    )


if __name__ == "__main__":
    main()
