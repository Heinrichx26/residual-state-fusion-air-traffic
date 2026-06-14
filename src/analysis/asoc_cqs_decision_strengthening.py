from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asoc_cqs_decision_audit import KEY_COLS, TARGETS, key_index, numeric_col, selected_subset, top_keys
from asoc_cqs_robust_certificate import add_calibration_lower_score, calibration_gap_table
from fusion_strengthening_common import PROJECT


OUT_ROOT = PROJECT / "results" / "experiments" / "applied_soft_computing_smoke"
MODEL_NAME = "CQS-Rank"
GT_AFRE_MODEL = "Graph-temporal evidence"
RAEG_MODEL = "RAEG-Rank"
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]
DEFAULT_PRIMARY = OUT_ROOT / "cqs_rank_base10_2025_rolling_quarter" / "cqs_rank_eval_predictions.csv"
DEFAULT_FULL30 = (
    OUT_ROOT
    / "raeg_rank_full30_2025_cf_severity_residual01_rolling_quarter_e4_r50"
    / "raeg_rank_predictions.csv"
)
DEFAULT_CROSS_YEAR = (
    OUT_ROOT
    / "raeg_cross_year_2024_to_2025_base10_cf_severity_residual01_e4"
    / "raeg_cross_year_predictions.csv"
)


SCENARIOS = {
    "primary_10airport_rolling": DEFAULT_PRIMARY,
    "scope_30airport_rolling": DEFAULT_FULL30,
    "cross_year_2024_to_2025": DEFAULT_CROSS_YEAR,
}


COMPARATOR_PROTOCOL = [
    {
        "method": "Graph-temporal evidence",
        "role": "GT-AFRE probability anchor",
        "input_rows": "same airport-hour keys, target closures, scheduled-arrival denominators, public-record timestamp contract",
        "sequence_eligibility": "uses all rows passing the airport-hour availability filter; temporal windows are shifted before scoring",
        "graph_input": "target-correlation graph estimated inside the training fold and applied to held-out rows",
        "training_rule": "fold-local memberships, graph weights, supervised fusion, and calibration",
        "tuning_boundary": "fixed LightGBM budget and fold-local calibration; no held-out target labels used for transformations",
        "output": "scheduled-arrival event probability used by probability queues and CQS event mass",
    },
    {
        "method": "RAEG-Rank",
        "role": "mechanism audit and probability-queue comparator",
        "input_rows": "same airport-hour keys, target closures, scheduled-arrival denominators, public-record timestamp contract",
        "sequence_eligibility": "same rows as the CQS carrier audit after matched-row alignment",
        "graph_input": "target-correlation, metadata, and operating-context graph views fitted or fixed before held-out scoring",
        "training_rule": "reliability masses, action-counterfactual state, residual head, convex score, and calibration fitted in training folds",
        "tuning_boundary": "fixed residual weight and training budget selected before final CQS audits",
        "output": "calibrated probability score used as the main probability-queue reference",
    },
    {
        "method": "STGTN airport-hour adaptation",
        "role": "recent graph-temporal comparator",
        "input_rows": "same sequence-eligible airport-hour rows and target closures as GT-AFRE",
        "sequence_eligibility": "airport histories require the same lagged public-record sequence availability for each compared row",
        "graph_input": "airport graph supplied under the same airport set and validation fold",
        "training_rule": "train on the fold training months and score held-out airport-hours only",
        "tuning_boundary": "shared sequence length, hidden-size range, epoch budget, and early-stopping rule across graph-temporal comparators",
        "output": "airport-hour event score evaluated with the same AUC, PR-AUC, and top-queue capture definitions",
    },
    {
        "method": "GE-STT airport-hour adaptation",
        "role": "recent graph-enhanced spatial-temporal comparator",
        "input_rows": "same sequence-eligible airport-hour rows and target closures as STGTN",
        "sequence_eligibility": "same lagged sequence window and row exclusion rule as STGTN",
        "graph_input": "same airport graph representation as the graph-temporal comparator set",
        "training_rule": "train on fold training months and score held-out airport-hours only",
        "tuning_boundary": "shared neural training budget and sequence configuration with STGTN",
        "output": "airport-hour event score evaluated on matched held-out records",
    },
    {
        "method": "DGS imbalance-learning adaptation",
        "role": "recent imbalance-learning comparator",
        "input_rows": "same airport-hour target rows and scheduled-arrival denominators",
        "sequence_eligibility": "does not require sequence rows; matched comparisons use the common evaluated row set",
        "graph_input": "no graph-specific advantage; uses the same airport-hour evidence fields available to non-sequence scorers",
        "training_rule": "imbalance regions and sampling parameters fitted inside training folds",
        "tuning_boundary": "fixed imbalance weighting and calibration choices selected without held-out target labels",
        "output": "airport-hour event score evaluated with paired queue metrics",
    },
    {
        "method": "Relation-DCSI h1",
        "role": "action-state relation reference",
        "input_rows": "same airport-hour keys with issue-time admissible advisory relation scores",
        "sequence_eligibility": "one-hour lead relation state available before target closure",
        "graph_input": "relation score uses action-state structure, not target-outcome graph learning from held-out rows",
        "training_rule": "relation transforms computed under the same timestamp contract",
        "tuning_boundary": "fixed one-hour relation horizon used across targets",
        "output": "airport-hour relation score evaluated as a probability-queue reference where available",
    },
]


def grouped_log_loss(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    prob = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)
    successes = successes.astype(float)
    totals = totals.astype(float)
    loss = -(successes * np.log(prob) + (totals - successes) * np.log1p(-prob)).sum()
    return float(loss / max(totals.sum(), 1.0))


def grouped_brier(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    successes = successes.astype(float)
    totals = totals.astype(float)
    prob = np.clip(prob.astype(float), 0.0, 1.0)
    loss = successes * (1.0 - prob) ** 2 + (totals - successes) * prob**2
    return float(loss.sum() / max(totals.sum(), 1.0))


def grouped_auc(successes: np.ndarray, totals: np.ndarray, score: np.ndarray) -> float:
    positives = successes.astype(float)
    negatives = totals.astype(float) - positives
    total_pos = positives.sum()
    total_neg = negatives.sum()
    if total_pos <= 0 or total_neg <= 0:
        return np.nan
    frame = pd.DataFrame({"score": score, "pos": positives, "neg": negatives})
    grouped = frame.groupby("score", as_index=False)[["pos", "neg"]].sum().sort_values("score", ascending=True)
    cum_neg = 0.0
    u_stat = 0.0
    for row in grouped.itertuples(index=False):
        u_stat += row.pos * (cum_neg + 0.5 * row.neg)
        cum_neg += row.neg
    return float(u_stat / (total_pos * total_neg))


def grouped_pr_auc(successes: np.ndarray, totals: np.ndarray, score: np.ndarray) -> float:
    positives = successes.astype(float)
    negatives = totals.astype(float) - positives
    total_pos = positives.sum()
    if total_pos <= 0:
        return np.nan
    frame = pd.DataFrame({"score": score, "pos": positives, "neg": negatives})
    grouped = frame.groupby("score", as_index=False)[["pos", "neg"]].sum().sort_values("score", ascending=False)
    cum_pos = 0.0
    cum_total = 0.0
    prev_recall = 0.0
    ap = 0.0
    for row in grouped.itertuples(index=False):
        cum_pos += row.pos
        cum_total += row.pos + row.neg
        recall = cum_pos / total_pos
        precision = cum_pos / cum_total if cum_total else 0.0
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


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
        obs = successes[mask].sum() / max(totals[mask].sum(), 1.0)
        pred = np.average(prob[mask], weights=totals[mask])
        out += totals[mask].sum() / total_weight * abs(obs - pred)
    return float(out)


def capture_at_fraction(successes: np.ndarray, score: np.ndarray, fraction: float) -> float:
    n = max(1, int(np.ceil(len(score) * fraction)))
    idx = np.argsort(score)[::-1][:n]
    total_events = successes.sum()
    if total_events <= 0:
        return np.nan
    return float(successes[idx].sum() / total_events)


def top_decile_lift(successes: np.ndarray, totals: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    n = max(1, int(np.ceil(len(score) * 0.10)))
    idx = np.argsort(score)[::-1][:n]
    top_events = successes[idx].sum()
    top_arrivals = totals[idx].sum()
    top_rate = top_events / top_arrivals if top_arrivals > 0 else np.nan
    base_rate = successes.sum() / totals.sum() if totals.sum() > 0 else np.nan
    return float(top_rate), float(top_rate / base_rate if base_rate else np.nan)


def evaluate_predictions(pred: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    rows = []
    for (target, model), group in pred.groupby(["target", "model"], sort=True):
        success_col = TARGETS[str(target)]
        if success_col not in group.columns:
            continue
        successes = numeric_col(group, success_col).to_numpy(float)
        totals = numeric_col(group, "arrivals").to_numpy(float)
        prob = numeric_col(group, "pred_prob").to_numpy(float)
        score = (
            numeric_col(group, "cqs_rank_score").to_numpy(float)
            if str(model) == MODEL_NAME and "cqs_rank_score" in group.columns
            else prob
        )
        top_rate, top_lift = top_decile_lift(successes, totals, score)
        row = {
            "target": target,
            "model": model,
            "log_loss": grouped_log_loss(successes, totals, prob),
            "brier": grouped_brier(successes, totals, prob),
            "auc": grouped_auc(successes, totals, score),
            "event_rate": float(successes.sum() / max(totals.sum(), 1.0)),
            "arrivals": float(totals.sum()),
            "events": float(successes.sum()),
            "pr_auc": grouped_pr_auc(successes, totals, score),
            "ece_10": ece(successes, totals, prob),
            "top_decile_precision": top_rate,
            "top_decile_lift": top_lift,
        }
        for fraction in budgets:
            row[f"top{int(round(fraction * 100))}_capture"] = capture_at_fraction(successes, score, fraction)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_gain(candidate: pd.Series, reference: pd.Series, metric: str) -> float:
    if metric in {"brier", "ece_10", "log_loss"}:
        return float(reference[metric]) - float(candidate[metric])
    return float(candidate[metric]) - float(reference[metric])


def compare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        col
        for col in ["auc", "pr_auc", "brier", "ece_10", "top5_capture", "top10_capture", "top20_capture"]
        if col in metrics.columns
    ]
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


def read_predictions(path: Path, smoke: bool) -> pd.DataFrame:
    if smoke:
        chunks = pd.read_csv(path, chunksize=80_000, low_memory=False)
        parts = []
        for chunk in chunks:
            parts.append(chunk)
            if sum(len(part) for part in parts) >= 160_000:
                break
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.read_csv(path, low_memory=False)
    if "utc_hour" in frame.columns:
        frame["utc_hour"] = pd.to_datetime(frame["utc_hour"], errors="coerce")
    for col in ["month", "arrivals"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def align_common_models(pred: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    parts = []
    for target, group in pred.groupby("target", sort=True):
        common: set[tuple[object, ...]] | None = None
        for name in model_names:
            model_group = group[group["model"].eq(name)]
            if model_group.empty:
                common = set()
                break
            keys = set(map(tuple, model_group[KEY_COLS].to_numpy(object)))
            common = keys if common is None else common & keys
        if not common:
            continue
        idx = pd.MultiIndex.from_frame(group[KEY_COLS])
        parts.append(group[idx.isin(pd.MultiIndex.from_tuples(list(common), names=KEY_COLS))].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_cqs_eval(pred: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "airport",
        "utc_hour",
        "month",
        "arrivals",
        "arr_delay60_count",
        "arr_delay120_count",
        "cancel_count",
        "excess_delay60_minutes",
        "target",
        "model",
        "pred_prob",
        "cqs_rank_score",
        "fold_id",
    ]
    pred = pred[[col for col in needed if col in pred.columns]].copy()
    existing_cqs = pred[pred["model"].eq(MODEL_NAME)].copy()
    if existing_cqs.empty:
        gt = pred[pred["model"].eq(GT_AFRE_MODEL)].copy()
        if gt.empty:
            raise ValueError(f"No {GT_AFRE_MODEL} rows available for CQS construction.")
        existing_cqs = gt.copy()
        existing_cqs["model"] = MODEL_NAME
        existing_cqs["cqs_rank_score"] = numeric_col(existing_cqs, "arrivals") * numeric_col(existing_cqs, "pred_prob")
        pred = pd.concat([pred, existing_cqs], ignore_index=True, sort=False)
    elif "cqs_rank_score" not in existing_cqs.columns or numeric_col(existing_cqs, "cqs_rank_score").eq(0).all():
        mask = pred["model"].eq(MODEL_NAME)
        pred.loc[mask, "cqs_rank_score"] = numeric_col(pred.loc[mask], "arrivals") * numeric_col(
            pred.loc[mask], "pred_prob"
        )
    model_names = [MODEL_NAME, GT_AFRE_MODEL, RAEG_MODEL]
    aligned = align_common_models(pred[pred["model"].isin(model_names)].copy(), model_names)
    if aligned.empty:
        raise ValueError("No common rows after aligning CQS, GT-AFRE, and RAEG-Rank.")
    return aligned


def selected_events(frame: pd.DataFrame, score_col: str, success_col: str, fraction: float) -> tuple[float, set[tuple[object, ...]]]:
    keys = top_keys(frame, score_col, fraction)
    selected = selected_subset(frame, keys)
    return float(numeric_col(selected, success_col).sum()), keys


def scenario_metrics(eval_pred: pd.DataFrame, out_dir: Path, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = evaluate_predictions(eval_pred, BUDGET_FRACTIONS)
    gains = compare_metrics(metrics)
    metrics.insert(0, "scenario", scenario)
    gains.insert(0, "scenario", scenario)
    metrics.to_csv(out_dir / f"{scenario}_cqs_metrics.csv", index=False)
    gains.to_csv(out_dir / f"{scenario}_cqs_gains.csv", index=False)
    return metrics, gains


def stress_scores(frame: pd.DataFrame, success_col: str, n_bins: int, radius_mode: str) -> pd.DataFrame:
    scored = add_calibration_lower_score(frame, success_col, n_bins)
    if radius_mode == "bin_abs_gap":
        return scored
    if radius_mode == "bin_abs_gap_x1_5":
        scored["calibration_gap"] = scored["calibration_gap"] * 1.5
    elif radius_mode == "bin_abs_gap_x2":
        scored["calibration_gap"] = scored["calibration_gap"] * 2.0
    elif radius_mode == "global_max_gap":
        scored["calibration_gap"] = float(scored["calibration_gap"].max())
    else:
        raise ValueError(radius_mode)
    scored["calibration_lower_prob"] = np.maximum(0.0, numeric_col(scored, "pred_prob") - scored["calibration_gap"])
    scored["calibration_lower_score"] = numeric_col(scored, "arrivals") * scored["calibration_lower_prob"]
    return scored


def calibration_stress(eval_pred: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for target, success_col in TARGETS.items():
        target_frame = eval_pred[eval_pred["target"].eq(target)].copy()
        cqs = target_frame[target_frame["model"].eq(MODEL_NAME)].copy()
        ref = target_frame[target_frame["model"].eq(RAEG_MODEL)].copy()
        if cqs.empty or ref.empty:
            continue
        cqs_keys = top_keys(cqs, "cqs_rank_score", 0.10)
        ref_events, ref_keys = selected_events(ref, "pred_prob", success_col, 0.10)
        total_events = float(numeric_col(cqs, success_col).sum())
        for n_bins in [5, 10, 20]:
            gap_table = calibration_gap_table(cqs, success_col, n_bins)
            for radius_mode in ["bin_abs_gap", "bin_abs_gap_x1_5", "bin_abs_gap_x2", "global_max_gap"]:
                scored = stress_scores(cqs, success_col, n_bins, radius_mode)
                robust_events, robust_keys = selected_events(scored, "calibration_lower_score", success_col, 0.10)
                rows.append(
                    {
                        "scenario": scenario,
                        "target": target,
                        "n_bins": n_bins,
                        "radius_mode": radius_mode,
                        "max_gap": float(gap_table["gap"].max()),
                        "mean_gap": float(gap_table["gap"].mean()),
                        "queue_overlap_with_cqs": len(robust_keys & cqs_keys) / max(len(cqs_keys), 1),
                        "queue_overlap_with_raeg": len(robust_keys & ref_keys) / max(len(ref_keys), 1),
                        "robust_events": robust_events,
                        "reference_events": ref_events,
                        "robust_event_gain_vs_raeg": robust_events - ref_events,
                        "robust_capture_gain_vs_raeg": (robust_events - ref_events) / max(total_events, 1.0),
                    }
                )
    return pd.DataFrame(rows)


def shifted_probability(frame: pd.DataFrame) -> pd.Series:
    out = []
    for _, group in frame.sort_values(["airport", "utc_hour"]).groupby("airport", sort=False):
        shifted = numeric_col(group, "pred_prob").shift(24)
        shifted = shifted.fillna(numeric_col(group, "pred_prob").median())
        out.append(shifted)
    return pd.concat(out).reindex(frame.sort_values(["airport", "utc_hour"]).index).sort_index()


def placebo_checks(eval_pred: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260612)
    for target, success_col in TARGETS.items():
        target_frame = eval_pred[eval_pred["target"].eq(target)].copy()
        cqs = target_frame[target_frame["model"].eq(MODEL_NAME)].copy()
        ref = target_frame[target_frame["model"].eq(RAEG_MODEL)].copy()
        if cqs.empty or ref.empty:
            continue
        total_events = float(numeric_col(cqs, success_col).sum())
        ref_events, _ = selected_events(ref, "pred_prob", success_col, 0.10)
        cqs_events, _ = selected_events(cqs, "cqs_rank_score", success_col, 0.10)
        tests: list[tuple[str, pd.Series]] = []
        perm_arrivals = numeric_col(cqs, "arrivals").to_numpy(float).copy()
        rng.shuffle(perm_arrivals)
        tests.append(("permuted_exposure", pd.Series(perm_arrivals, index=cqs.index) * numeric_col(cqs, "pred_prob")))
        shifted_prob = shifted_probability(cqs)
        tests.append(("shifted_probability_24h", numeric_col(cqs, "arrivals") * shifted_prob))
        tests.append(("probability_only_queue", numeric_col(cqs, "pred_prob")))
        for name, score in tests:
            tmp = cqs.copy()
            tmp["_placebo_score"] = score
            events, _ = selected_events(tmp, "_placebo_score", success_col, 0.10)
            rows.append(
                {
                    "scenario": scenario,
                    "target": target,
                    "placebo": name,
                    "events": events,
                    "capture": events / max(total_events, 1.0),
                    "gain_vs_raeg_top10_events": events - ref_events,
                    "event_loss_vs_cqs": events - cqs_events,
                    "capture_loss_vs_cqs": (events - cqs_events) / max(total_events, 1.0),
                }
            )
    return pd.DataFrame(rows)


def budget_needed(frame: pd.DataFrame, score_col: str, success_col: str, target_events: float) -> float:
    for fraction in np.arange(0.01, 0.301, 0.01):
        events, _ = selected_events(frame, score_col, success_col, float(fraction))
        if events >= target_events:
            return float(fraction)
    return float("nan")


def decision_value(eval_pred: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for target, success_col in TARGETS.items():
        target_frame = eval_pred[eval_pred["target"].eq(target)].copy()
        cqs = target_frame[target_frame["model"].eq(MODEL_NAME)].copy()
        ref = target_frame[target_frame["model"].eq(RAEG_MODEL)].copy()
        if cqs.empty or ref.empty:
            continue
        total_events = float(numeric_col(cqs, success_col).sum())
        cqs_events, cqs_keys = selected_events(cqs, "cqs_rank_score", success_col, 0.10)
        ref_events, ref_keys = selected_events(ref, "pred_prob", success_col, 0.10)
        cqs_selected = selected_subset(cqs, cqs_keys)
        ref_selected_as_cqs = selected_subset(cqs, ref_keys)
        cqs_delay = float(numeric_col(cqs_selected, "excess_delay60_minutes").sum())
        ref_delay = float(numeric_col(ref_selected_as_cqs, "excess_delay60_minutes").sum())
        queue_size = max(len(cqs_keys), 1)
        ref_need_for_cqs = budget_needed(ref, "pred_prob", success_col, cqs_events)
        cqs_need_for_ref = budget_needed(cqs, "cqs_rank_score", success_col, ref_events)
        rows.append(
            {
                "scenario": scenario,
                "target": target,
                "queue_size": queue_size,
                "total_events": total_events,
                "cqs_top10_events": cqs_events,
                "raeg_top10_events": ref_events,
                "event_gain": cqs_events - ref_events,
                "capture_gain": (cqs_events - ref_events) / max(total_events, 1.0),
                "event_gain_per_100_reviewed_hours": (cqs_events - ref_events) / queue_size * 100.0,
                "excess_delay60_gain": cqs_delay - ref_delay,
                "excess_delay60_gain_per_100_reviewed_hours": (cqs_delay - ref_delay) / queue_size * 100.0,
                "raeg_budget_needed_for_cqs_top10_events": ref_need_for_cqs,
                "cqs_budget_needed_for_raeg_top10_events": cqs_need_for_ref,
                "budget_saved_to_match_raeg_top10": 0.10 - cqs_need_for_ref if np.isfinite(cqs_need_for_ref) else np.nan,
                "extra_raeg_budget_needed_to_match_cqs_top10": ref_need_for_cqs - 0.10 if np.isfinite(ref_need_for_cqs) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compact_matrix(metrics: pd.DataFrame, gains: pd.DataFrame, stress: pd.DataFrame, value: pd.DataFrame) -> pd.DataFrame:
    metric_rows = metrics[metrics["model"].eq(MODEL_NAME)].copy()
    gain_rows = gains[gains["reference"].eq(RAEG_MODEL)].copy()
    stress_rows = stress[(stress["n_bins"].eq(10)) & (stress["radius_mode"].eq("bin_abs_gap"))].copy()
    keep = metric_rows[["scenario", "target", "top10_capture", "top20_capture", "brier", "ece_10"]].merge(
        gain_rows[["scenario", "target", "top10_capture_gain", "top20_capture_gain"]],
        on=["scenario", "target"],
        how="left",
    )
    keep = keep.merge(
        stress_rows[["scenario", "target", "queue_overlap_with_cqs", "robust_event_gain_vs_raeg"]],
        on=["scenario", "target"],
        how="left",
    )
    keep = keep.merge(
        value[["scenario", "target", "event_gain", "event_gain_per_100_reviewed_hours", "extra_raeg_budget_needed_to_match_cqs_top10"]],
        on=["scenario", "target"],
        how="left",
    )
    return keep.sort_values(["scenario", "target"]).reset_index(drop=True)


def write_assessment(out_dir: Path, matrix: pd.DataFrame, stress: pd.DataFrame, placebo: pd.DataFrame) -> None:
    lines = [
        "# CQS decision strengthening assessment",
        "",
        "The added audits evaluate CQS under wider operational scope, external-year scoring, calibration-radius changes, and falsification checks.",
        "",
    ]
    for scenario, group in matrix.groupby("scenario", sort=True):
        lines.append(f"## {scenario}")
        positive = int((group["top10_capture_gain"] > 0).sum())
        lines.append(f"Top-10 CQS gains over the RAEG probability queue are positive for {positive}/{len(group)} targets.")
        for row in group.itertuples(index=False):
            lines.append(
                f"- {row.target}: top-10 gain {row.top10_capture_gain:+.4f}; robust event gain "
                f"{row.robust_event_gain_vs_raeg:+.0f}; event gain per 100 reviewed hours "
                f"{row.event_gain_per_100_reviewed_hours:+.1f}."
            )
        lines.append("")
    stress_gate = stress[(stress["radius_mode"].isin(["bin_abs_gap", "bin_abs_gap_x1_5", "bin_abs_gap_x2"]))]
    lines.append(
        "Calibration stress check: "
        f"{int((stress_gate['robust_event_gain_vs_raeg'] > 0).sum())}/{len(stress_gate)} tested target-radius cells keep positive lower-bound event gain."
    )
    placebo_loss = placebo.groupby("placebo")["capture_loss_vs_cqs"].mean().sort_values()
    for name, value in placebo_loss.items():
        lines.append(f"Placebo {name}: mean capture change versus CQS {value:+.4f}.")
    (out_dir / "cqs_decision_strengthening_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out_dir = OUT_ROOT / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_paths = {
        "primary_10airport_rolling": Path(args.primary_predictions),
        "scope_30airport_rolling": Path(args.full30_predictions),
        "cross_year_2024_to_2025": Path(args.cross_year_predictions),
    }
    if args.scenarios != "ALL":
        wanted = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        scenario_paths = {name: path for name, path in scenario_paths.items() if name in wanted}

    all_metrics = []
    all_gains = []
    all_stress = []
    all_placebo = []
    all_value = []
    for scenario, path in scenario_paths.items():
        pred = read_predictions(path, smoke=args.smoke)
        eval_pred = build_cqs_eval(pred)
        metrics, gains = scenario_metrics(eval_pred, out_dir, scenario)
        stress = calibration_stress(eval_pred, scenario)
        placebo = placebo_checks(eval_pred, scenario)
        value = decision_value(eval_pred, scenario)
        stress.to_csv(out_dir / f"{scenario}_calibration_stress.csv", index=False)
        placebo.to_csv(out_dir / f"{scenario}_placebo_checks.csv", index=False)
        value.to_csv(out_dir / f"{scenario}_decision_value.csv", index=False)
        all_metrics.append(metrics)
        all_gains.append(gains)
        all_stress.append(stress)
        all_placebo.append(placebo)
        all_value.append(value)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    gains_all = pd.concat(all_gains, ignore_index=True)
    stress_all = pd.concat(all_stress, ignore_index=True)
    placebo_all = pd.concat(all_placebo, ignore_index=True)
    value_all = pd.concat(all_value, ignore_index=True)
    matrix = compact_matrix(metrics_all, gains_all, stress_all, value_all)
    metrics_all.to_csv(out_dir / "all_scenario_cqs_metrics.csv", index=False)
    gains_all.to_csv(out_dir / "all_scenario_cqs_gains.csv", index=False)
    stress_all.to_csv(out_dir / "all_scenario_calibration_stress.csv", index=False)
    placebo_all.to_csv(out_dir / "all_scenario_placebo_checks.csv", index=False)
    value_all.to_csv(out_dir / "all_scenario_decision_value.csv", index=False)
    matrix.to_csv(out_dir / "cqs_strengthening_matrix.csv", index=False)
    pd.DataFrame(COMPARATOR_PROTOCOL).to_csv(out_dir / "expanded_comparator_protocol.csv", index=False)
    write_assessment(out_dir, matrix, stress_all, placebo_all)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CQS decision-strengthening audits for ASC revision.")
    parser.add_argument("--output-name", default="cqs_decision_strengthening_20260612")
    parser.add_argument("--primary-predictions", default=str(DEFAULT_PRIMARY))
    parser.add_argument("--full30-predictions", default=str(DEFAULT_FULL30))
    parser.add_argument("--cross-year-predictions", default=str(DEFAULT_CROSS_YEAR))
    parser.add_argument("--scenarios", default="ALL")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
