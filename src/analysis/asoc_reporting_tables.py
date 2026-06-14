from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asoc_fuzzy_residual_evidence_smoke import TARGETS, add_soft_features, load_panel, relation_scores
from asoc_soft_computing_smoke import OUT_ROOT, parse_airports
from asoc_temporal_frequency_smoke import DCSI_PREDICTIONS


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    PROJECT
    / "results"
    / "experiments"
    / "applied_soft_computing_smoke"
    / "tree_family_full_10airports_2025"
    / "tree_family_predictions.csv"
)
DEFAULT_GRAPH_PREDICTIONS = (
    PROJECT
    / "results"
    / "experiments"
    / "applied_soft_computing_smoke"
    / "graph_temporal_full_10airports_2025_rerun"
    / "graph_temporal_predictions.csv"
)
DEFAULT_MODELS = [
    "Relation-DCSI h1",
    "Graph-temporal evidence",
    "LightGBM full-demand advisory",
    "LightGBM AFRE no relation",
    "XGBoost AFRE soft evidence",
]
QUEUE_FRACTIONS = [0.05, 0.10, 0.15, 0.20]


def parse_months(text: str) -> list[int]:
    months: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(piece) for piece in part.split("-", 1)]
            months.extend(range(start, end + 1))
        else:
            months.append(int(part))
    return sorted({month for month in months if 1 <= month <= 12})


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    keep = values.notna() & (weights > 0)
    if not keep.any():
        return np.nan
    return float(np.average(values[keep], weights=weights[keep]))


def relation_prediction_frame(months: list[int], airports: set[str], target: str, keys: pd.DataFrame) -> pd.DataFrame:
    success_col = TARGETS[target]
    rel = pd.read_csv(DCSI_PREDICTIONS, parse_dates=["utc_hour"])
    rel = rel[
        rel["month"].isin(months)
        & rel["airport"].isin(airports)
        & rel["target"].eq(target)
        & rel["horizon"].eq(1)
        & rel["model"].eq("online_relation_DCSI")
    ].copy()
    rel = rel[["airport", "utc_hour", "month", "pred_prob"]].copy()
    rel["target"] = target
    rel["model"] = "Relation-DCSI h1"
    key_cols = ["airport", "utc_hour", "month", "arrivals", success_col]
    return keys[key_cols].drop_duplicates().merge(rel, on=["airport", "utc_hour", "month"], how="inner")


def read_prediction_files(prediction_files: list[Path], months: list[int], airports: list[str] | None, models: list[str]) -> pd.DataFrame:
    frames = []
    tree_models = [model for model in models if model != "Relation-DCSI h1"]
    for prediction_file in prediction_files:
        if not prediction_file.exists():
            continue
        frame = pd.read_csv(prediction_file, parse_dates=["utc_hour"])
        frame = frame[frame["month"].isin(months) & frame["model"].isin(tree_models)].copy()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No prediction files contained the requested models.")
    pred = pd.concat(frames, ignore_index=True)
    if airports:
        pred = pred[pred["airport"].isin(airports)].copy()
    pred = pred.drop_duplicates(["target", "model", "airport", "utc_hour", "month"], keep="first")
    return pred


def load_predictions(months: list[int], airports: list[str] | None, prediction_files: list[Path], models: list[str]) -> pd.DataFrame:
    pred = read_prediction_files(prediction_files, months, airports, models)
    parts = [pred]
    airport_set = set(pred["airport"].astype(str).unique())
    for target in sorted(pred["target"].dropna().unique()):
        success_col = TARGETS[target]
        keys = pred[pred["target"].eq(target)][["airport", "utc_hour", "month", "arrivals", success_col]].copy()
        parts.append(relation_prediction_frame(months, airport_set, target, keys))
    out = pd.concat(parts, ignore_index=True)
    out = out[out["model"].isin(models)].copy()
    out["pred_prob"] = pd.to_numeric(out["pred_prob"], errors="coerce").clip(1e-6, 1 - 1e-6)
    return out.sort_values(["target", "model", "utc_hour", "airport"]).reset_index(drop=True)


def add_risk_decile(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    order = np.argsort(out["pred_prob"].to_numpy(float))
    ranks = np.empty(len(out), dtype=int)
    ranks[order] = np.arange(len(out))
    out["risk_decile"] = np.minimum(10, np.floor(ranks / max(len(out), 1) * 10).astype(int) + 1)
    return out


def calibration_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in pred.groupby(["target", "model"], sort=True):
        success_col = TARGETS[target]
        work = add_risk_decile(group)
        total_events = float(work[success_col].fillna(0.0).sum())
        for decile, dec in work.groupby("risk_decile", sort=True):
            arrivals = float(dec["arrivals"].sum())
            events = float(dec[success_col].fillna(0.0).sum())
            mean_pred = weighted_mean(dec["pred_prob"], dec["arrivals"])
            observed = events / arrivals if arrivals > 0 else np.nan
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "risk_decile": int(decile),
                    "airport_hours": int(len(dec)),
                    "arrivals": int(arrivals),
                    "events": int(events),
                    "mean_pred_prob": mean_pred,
                    "observed_rate": observed,
                    "calibration_gap": observed - mean_pred if np.isfinite(observed) and np.isfinite(mean_pred) else np.nan,
                    "abs_calibration_gap": abs(observed - mean_pred) if np.isfinite(observed) and np.isfinite(mean_pred) else np.nan,
                    "event_capture_share": events / total_events if total_events > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def queue_threshold_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, model), group in pred.groupby(["target", "model"], sort=True):
        success_col = TARGETS[target]
        total_events = float(group[success_col].fillna(0.0).sum())
        total_arrivals = float(group["arrivals"].sum())
        base_rate = total_events / total_arrivals if total_arrivals > 0 else np.nan
        for fraction in QUEUE_FRACTIONS:
            n = max(1, int(np.ceil(len(group) * fraction)))
            selected = group.sort_values("pred_prob", ascending=False).head(n)
            alert_events = float(selected[success_col].fillna(0.0).sum())
            alert_arrivals = float(selected["arrivals"].sum())
            alert_rate = alert_events / alert_arrivals if alert_arrivals > 0 else np.nan
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "queue_fraction": fraction,
                    "alert_airport_hours": int(len(selected)),
                    "alert_arrivals": int(alert_arrivals),
                    "events_captured": int(alert_events),
                    "total_events": int(total_events),
                    "event_capture_rate": alert_events / total_events if total_events > 0 else np.nan,
                    "alert_event_rate": alert_rate,
                    "base_event_rate": base_rate,
                    "lift": alert_rate / base_rate if base_rate and np.isfinite(alert_rate) else np.nan,
                    "min_queue_pred_prob": float(selected["pred_prob"].min()),
                }
            )
    return pd.DataFrame(rows)


def month_train_rates(panel: pd.DataFrame, target: str, success_col: str) -> dict[int, float]:
    rates = {}
    for month in sorted(panel["month"].unique()):
        train = panel[panel["month"] != month].copy()
        rates[int(month)] = float(train[success_col].sum() / max(train["arrivals"].sum(), 1.0))
    return rates


def soft_feature_panel(months: list[int], airports: list[str] | None) -> pd.DataFrame:
    base = load_panel(months, airports)
    parts = []
    for target, success_col in TARGETS.items():
        panel = base.merge(relation_scores(months, target, 1), on=["airport", "utc_hour", "month"], how="inner")
        rates = month_train_rates(panel, target, success_col)
        month_parts = []
        for month, month_frame in panel.groupby("month", sort=True):
            month_parts.append(add_soft_features(month_frame.copy(), target, rates[int(month)]))
        out = pd.concat(month_parts, ignore_index=True)
        out["target"] = target
        out["target_events"] = out[success_col]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def monitoring_records(pred: pd.DataFrame, months: list[int], airports: list[str] | None, model: str, top_n: int) -> pd.DataFrame:
    features = soft_feature_panel(months, airports)
    model_pred = pred[pred["model"].eq(model)].copy()
    records = []
    merge_cols = ["target", "airport", "utc_hour", "month"]
    merged = model_pred.merge(features, on=merge_cols, how="inner", suffixes=("", "_feature"))
    for target, group in merged.groupby("target", sort=True):
        success_col = TARGETS[target]
        use = group.sort_values("pred_prob", ascending=False).head(top_n).copy()
        use["observed_rate"] = use[success_col].fillna(0.0) / use["arrivals"]
        use["queue_flag"] = "top_record"
        records.append(
            use[
                [
                    "target",
                    "airport",
                    "utc_hour",
                    "month",
                    "model",
                    "pred_prob",
                    "queue_flag",
                    "arrivals",
                    success_col,
                    "observed_rate",
                    "weather_high_membership",
                    "arrival_pressure_membership",
                    "departure_pressure_membership",
                    "active_memory_membership",
                    "post_memory_membership",
                    "target_prior_membership",
                    "target_prior_residual",
                    "soft_pressure_index",
                    "active_minutes",
                    "post_3h_minutes",
                ]
            ].rename(columns={success_col: "events"})
        )
    return pd.concat(records, ignore_index=True)


def write_assessment(out: Path, calibration: pd.DataFrame, queue: pd.DataFrame, records: pd.DataFrame) -> None:
    lines = [
        "# ASOC reporting tables assessment",
        "",
        "Generated table-ready calibration, queue-threshold, and monitoring-record files.",
        "",
        "## Queue summary at 10%",
        "",
    ]
    q10 = queue[np.isclose(queue["queue_fraction"], 0.10)].copy()
    for row in q10.sort_values(["target", "event_capture_rate"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"- {row.target}, {row.model}: capture {row.event_capture_rate:.3f}, "
            f"lift {row.lift:.2f}, events {row.events_captured}/{row.total_events}."
        )
    lines.extend(["", "## Calibration summary", ""])
    cal_summary = calibration.groupby(["target", "model"], sort=True).apply(
        lambda d: pd.Series(
            {
                "weighted_abs_gap": np.average(d["abs_calibration_gap"], weights=d["arrivals"]),
                "max_abs_gap": d["abs_calibration_gap"].max(),
            }
        ),
        include_groups=False,
    ).reset_index()
    for row in cal_summary.sort_values(["target", "weighted_abs_gap"]).itertuples(index=False):
        lines.append(
            f"- {row.target}, {row.model}: weighted absolute calibration gap {row.weighted_abs_gap:.4f}, "
            f"max decile gap {row.max_abs_gap:.4f}."
        )
    lines.extend(["", f"Monitoring records: {len(records)} rows."])
    (out / "reporting_tables_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    months = parse_months(args.months)
    airports = parse_airports(args.airports)
    models = [item.strip() for item in args.models.split(";") if item.strip()]
    prediction_files = [Path(args.prediction_file)]
    prediction_files.extend(
        Path(item.strip()) for item in args.extra_prediction_files.split(";") if item.strip()
    )
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(months, airports, prediction_files, models)
    calibration = calibration_table(pred)
    queue = queue_threshold_table(pred)
    records = monitoring_records(pred, months, airports, args.monitoring_model, args.monitoring_top_n)
    pred.to_csv(out / "selected_model_predictions.csv", index=False)
    calibration.to_csv(out / "calibration_decile_table.csv", index=False)
    queue.to_csv(out / "queue_threshold_table.csv", index=False)
    records.to_csv(out / "explainable_monitoring_records.csv", index=False)
    write_assessment(out, calibration, queue, records)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--extra-prediction-files", default=str(DEFAULT_GRAPH_PREDICTIONS))
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--airports", default="ALL")
    parser.add_argument("--models", default=";".join(DEFAULT_MODELS))
    parser.add_argument("--monitoring-model", default="Graph-temporal evidence")
    parser.add_argument("--monitoring-top-n", type=int, default=12)
    parser.add_argument("--output-name", default="asoc_reporting_tables_full_2025")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
