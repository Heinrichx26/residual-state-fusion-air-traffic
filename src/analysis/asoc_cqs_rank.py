from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fusion_prediction_increment import evaluate, grouped_auc
from fusion_strengthening_prediction_diagnostics import grouped_pr_auc, top_decile_lift

from asoc_graph_temporal_baselines_smoke import feature_sets, fold_prepare
from asoc_raeg_rank import (
    GT_AFRE_MODEL,
    MAIN_AIRPORTS,
    MODEL_NAME as RAEG_MODEL,
    OUT_ROOT,
    SMOKE_AIRPORTS,
    TARGETS,
    fit_gt_afre_probability_train_test,
    load_relation_scores,
    raeg_validation_folds,
)
from asoc_soft_computing_smoke import capture_at_fraction, ece, parse_airports, parse_int_list

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


MODEL_NAME = "CQS-Rank"
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]


def numeric_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def prepared_gt_probability(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    target: str,
    success_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train, test = fold_prepare(raw_train, raw_test, target, success_col, graph_source="target_correlation")
    features = feature_sets(train)[GT_AFRE_MODEL]
    train_prob, test_prob = fit_gt_afre_probability_train_test(train, test, success_col, features)
    return train, test, train_prob, test_prob


def cqs_score(prob: np.ndarray, arrivals: np.ndarray) -> np.ndarray:
    return np.maximum(arrivals, 1.0) * np.clip(prob, 0.0, 1.0)


def fit_predict_cqs(raw_train: pd.DataFrame, raw_test: pd.DataFrame, target: str, success_col: str) -> pd.DataFrame:
    _, test, _, test_prob = prepared_gt_probability(
        raw_train,
        raw_test,
        target,
        success_col,
    )
    test_n = np.maximum(numeric_col(test, "arrivals").to_numpy(float), 1.0)
    test_score = cqs_score(test_prob, test_n)
    out_cols = ["airport", "utc_hour", "month", "arrivals", "arr_delay60_count", "arr_delay120_count", "cancel_count", "excess_delay60_minutes"]
    out = test[[col for col in out_cols if col in test.columns]].copy()
    out["target"] = target
    out["model"] = MODEL_NAME
    out["pred_prob"] = test_prob
    out["cqs_rank_score"] = test_score
    return out


def load_reference_predictions(path: Path | None, targets: list[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    pred = pd.read_csv(path)
    keep_models = {GT_AFRE_MODEL, RAEG_MODEL}
    pred = pred[pred["model"].isin(keep_models) & pred["target"].isin(targets)].copy()
    needed = ["airport", "utc_hour", "month", "arrivals", "arr_delay60_count", "arr_delay120_count", "cancel_count", "excess_delay60_minutes", "target", "model", "pred_prob"]
    return pred[[col for col in needed if col in pred.columns]]


def align_common_evaluation_rows(pred: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["target", "airport", "utc_hour", "month"]
    if pred.empty or MODEL_NAME not in set(pred["model"].astype(str)):
        return pred
    aligned_parts = []
    for target, group in pred.groupby("target", sort=True):
        model_keys = {}
        for model, model_group in group.groupby("model", sort=True):
            keys = set(map(tuple, model_group[key_cols].to_numpy(object)))
            if keys:
                model_keys[str(model)] = keys
        if MODEL_NAME not in model_keys:
            aligned_parts.append(group)
            continue
        common = set(model_keys[MODEL_NAME])
        for name, keys in model_keys.items():
            if name != MODEL_NAME:
                common &= keys
        if not common:
            aligned_parts.append(group)
            continue
        key_index = pd.MultiIndex.from_frame(group[key_cols])
        common_index = pd.MultiIndex.from_tuples(list(common), names=key_cols)
        aligned_parts.append(group[key_index.isin(common_index)].copy())
    return pd.concat(aligned_parts, ignore_index=True) if aligned_parts else pred


def metric_row(target: str, model: str, pred: pd.DataFrame, success_col: str, budgets: list[float]) -> dict[str, object]:
    successes = pred[success_col].to_numpy(float)
    totals = pred["arrivals"].to_numpy(float)
    prob = pred["pred_prob"].to_numpy(float)
    rank_score = pred["cqs_rank_score"].to_numpy(float) if model == MODEL_NAME and "cqs_rank_score" in pred.columns else prob
    top_rate, top_lift = top_decile_lift(successes, totals, rank_score)
    row = {
        "target": target,
        "model": model,
        **evaluate(successes, totals, prob),
        "auc": grouped_auc(successes, totals, rank_score),
        "pr_auc": grouped_pr_auc(successes, totals, rank_score),
        "ece_10": ece(successes, totals, prob),
        "top_decile_precision": top_rate,
        "top_decile_lift": top_lift,
    }
    for fraction in budgets:
        row[f"top{int(round(fraction * 100))}_capture"] = capture_at_fraction(successes, totals, rank_score, fraction)
    return row


def evaluate_predictions(pred: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    rows = []
    for (target, model), group in pred.groupby(["target", "model"], sort=True):
        success_col = TARGETS[str(target)]
        if success_col not in group.columns or group[success_col].sum() <= 0:
            continue
        rows.append(metric_row(str(target), str(model), group, success_col, budgets))
    return pd.DataFrame(rows)


def metric_gain(candidate: pd.Series, reference: pd.Series, metric: str) -> float:
    if metric in {"brier", "ece_10", "log_loss"}:
        return float(reference[metric]) - float(candidate[metric])
    return float(candidate[metric]) - float(reference[metric])


def compare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [col for col in ["auc", "pr_auc", "brier", "ece_10", "top5_capture", "top10_capture", "top20_capture"] if col in metrics.columns]
    for target, group in metrics.groupby("target", sort=True):
        cqs = group[group["model"].eq(MODEL_NAME)]
        if cqs.empty:
            continue
        cqs_row = cqs.iloc[0]
        for ref_name in [GT_AFRE_MODEL, RAEG_MODEL]:
            ref = group[group["model"].eq(ref_name)]
            if ref.empty:
                continue
            ref_row = ref.iloc[0]
            out = {"target": target, "candidate": MODEL_NAME, "reference": ref_name}
            for metric in metric_cols:
                out[f"{metric}_gain"] = metric_gain(cqs_row, ref_row, metric)
            rows.append(out)
    return pd.DataFrame(rows)


def write_assessment(out_dir: Path, metrics: pd.DataFrame, gains: pd.DataFrame, scope: str) -> None:
    lines = [
        "# CQS-Rank assessment",
        "",
        f"Scope: {scope}.",
        "",
        "Gate: usable if CQS-Rank improves top-10 capture over GT-AFRE/RAEG-Rank for a delay target and cancellation while keeping Brier and ECE no worse than the calibrated probability anchor.",
        "",
    ]
    usable_targets = set()
    for target, group in metrics.groupby("target", sort=True):
        lines.append(f"## {target}")
        cqs = group[group["model"].eq(MODEL_NAME)]
        if cqs.empty:
            lines.append("- No CQS-Rank metrics available.")
            continue
        c = cqs.iloc[0]
        lines.append(
            f"- CQS-Rank: AUC {c.get('auc', np.nan):.4f}; PR-AUC {c.get('pr_auc', np.nan):.4f}; "
            f"Brier {c.get('brier', np.nan):.5f}; ECE {c.get('ece_10', np.nan):.5f}; "
            f"top-10 {c.get('top10_capture', np.nan):.4f}."
        )
        for row in gains[gains["target"].eq(target)].itertuples(index=False):
            top10 = getattr(row, "top10_capture_gain", np.nan)
            brier = getattr(row, "brier_gain", np.nan)
            ece_gain = getattr(row, "ece_10_gain", np.nan)
            usable = top10 > 0 and brier > -0.0008 and ece_gain > -0.0030
            if usable:
                usable_targets.add(str(target))
            verdict = "usable signal" if usable else "diagnostic only"
            lines.append(
                f"- vs {row.reference}: {verdict}; PR-AUC {getattr(row, 'pr_auc_gain', np.nan):+.4f}; "
                f"top-10 {top10:+.4f}; Brier {brier:+.5f}; ECE {ece_gain:+.5f}."
            )
        lines.append("")
    has_delay = bool({"long_arrival_delay", "severe_arrival_delay"} & usable_targets)
    has_cancel = "cancellation" in usable_targets
    lines.append(f"Overall smoke gate: {'pass' if has_delay and has_cancel else 'hold'}.")
    (out_dir / "cqs_rank_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    panel_path: Path,
    reference_predictions: Path | None,
    output_name: str,
    months: list[int],
    airports: list[str] | None,
    validation: str,
    first_test_month: int,
    min_train_months: int,
    targets: list[str],
) -> Path:
    out_dir = OUT_ROOT / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_panel = pd.read_csv(panel_path)
    base_panel["utc_hour"] = pd.to_datetime(base_panel["utc_hour"])
    base_panel = base_panel[base_panel["month"].isin(months)].copy()
    if airports is not None:
        base_panel = base_panel[base_panel["airport"].isin(airports)].copy()
    base_panel = base_panel[(base_panel["arrivals"] > 0) & base_panel["weather_score"].notna()].copy()
    for col in ["airport", "local_hour", "day_of_week"]:
        if col in base_panel.columns:
            base_panel[col] = base_panel[col].astype(str)
    prediction_parts = []
    for target in targets:
        success_col = TARGETS[target]
        relation = load_relation_scores(months, target, 1)
        relation["airport"] = relation["airport"].astype(str)
        if airports is not None:
            relation = relation[relation["airport"].isin(airports)].copy()
        panel = base_panel.merge(relation, on=["airport", "utc_hour", "month"], how="inner")
        for fold_id, train, test in raeg_validation_folds(panel, validation, first_test_month, min_train_months):
            if train.empty or test.empty or train[success_col].sum() <= 0:
                continue
            fold = fit_predict_cqs(train.copy(), test.copy(), target, success_col)
            fold["fold_id"] = fold_id
            prediction_parts.append(fold)
    if not prediction_parts:
        raise RuntimeError("No CQS-Rank predictions were produced.")
    cqs_pred = pd.concat(prediction_parts, ignore_index=True)
    ref = load_reference_predictions(reference_predictions, targets)
    if not ref.empty:
        ref["utc_hour"] = pd.to_datetime(ref["utc_hour"])
        ref = ref[ref["month"].isin(months)].copy()
        if airports is not None:
            ref = ref[ref["airport"].isin(airports)].copy()
        pred = pd.concat([ref, cqs_pred], ignore_index=True, sort=False)
    else:
        pred = cqs_pred
    pred = align_common_evaluation_rows(pred)
    metrics = evaluate_predictions(pred, BUDGET_FRACTIONS)
    gains = compare_metrics(metrics)
    cqs_pred.to_csv(out_dir / "cqs_rank_predictions.csv", index=False)
    pred.to_csv(out_dir / "cqs_rank_eval_predictions.csv", index=False)
    metrics.to_csv(out_dir / "cqs_rank_metrics.csv", index=False)
    gains.to_csv(out_dir / "cqs_rank_gains.csv", index=False)
    write_assessment(out_dir, metrics, gains, f"months={months}; airports={airports}; validation={validation}")
    return out_dir


def build_arg_parser(default_output: str, default_months: str, default_airports: str, default_panel: str, default_reference: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrated queue-set ranking with closed-form event-mass utility.")
    parser.add_argument("--panel", type=Path, default=Path(default_panel))
    parser.add_argument("--reference-predictions", type=Path, default=Path(default_reference))
    parser.add_argument("--output-name", default=default_output)
    parser.add_argument("--months", default=default_months)
    parser.add_argument("--airports", default=default_airports)
    parser.add_argument("--validation", default="month", choices=["month", "rolling", "rolling_quarter"])
    parser.add_argument("--first-test-month", type=int, default=4)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--targets", default="long_arrival_delay,severe_arrival_delay,cancellation")
    return parser


def run_from_args(args: argparse.Namespace) -> Path:
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    return run(
        panel_path=args.panel,
        reference_predictions=args.reference_predictions,
        output_name=args.output_name,
        months=parse_int_list(args.months),
        airports=parse_airports(args.airports),
        validation=args.validation,
        first_test_month=args.first_test_month,
        min_train_months=args.min_train_months,
        targets=targets,
    )


if __name__ == "__main__":
    default_panel = "results/experiments/applied_soft_computing_smoke/raeg_scope_panel_2025_base10_allmonths_cf_severity/raeg_scope_panel.csv"
    default_ref = "results/experiments/applied_soft_computing_smoke/raeg_rank_base10_2025_cf_severity_residual01_component_ablation_qrolling_e3/raeg_rank_predictions.csv"
    parser = build_arg_parser("cqs_rank_base10_2025", "1-12", MAIN_AIRPORTS, default_panel, default_ref)
    out = run_from_args(parser.parse_args())
    print(f"wrote {out}")
