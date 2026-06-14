from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_NAME = "CQS-Rank"
DEFAULT_REFERENCE = "RAEG-Rank"
KEY_COLS = ["airport", "utc_hour", "month"]
TARGETS = {
    "long_arrival_delay": "arr_delay60_count",
    "severe_arrival_delay": "arr_delay120_count",
    "cancellation": "cancel_count",
}


def numeric_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame[KEY_COLS])


def top_keys(frame: pd.DataFrame, score_col: str, fraction: float) -> set[tuple[object, ...]]:
    use = frame.copy()
    use["_score"] = numeric_col(use, score_col, -np.inf)
    use = use.sort_values(["_score", "airport", "utc_hour"], ascending=[False, True, True], kind="mergesort")
    k = int(np.ceil(fraction * len(use)))
    k = min(max(k, 1), len(use))
    return set(map(tuple, use.head(k)[KEY_COLS].to_numpy(object)))


def target_pair(
    predictions: pd.DataFrame,
    target: str,
    reference_model: str = DEFAULT_REFERENCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frame = predictions[predictions["target"].eq(target)].copy()
    candidate = target_frame[target_frame["model"].eq(MODEL_NAME)].copy()
    reference = target_frame[target_frame["model"].eq(reference_model)].copy()
    if candidate.empty:
        raise ValueError(f"No {MODEL_NAME} rows for target {target}")
    if reference.empty:
        raise ValueError(f"No {reference_model} rows for target {target}")
    common = key_index(candidate).intersection(key_index(reference))
    candidate = candidate[key_index(candidate).isin(common)].copy()
    reference = reference[key_index(reference).isin(common)].copy()
    return candidate, reference


def selected_subset(base: pd.DataFrame, keys: set[tuple[object, ...]]) -> pd.DataFrame:
    if not keys:
        return base.iloc[[]].copy()
    idx = key_index(base)
    return base[idx.isin(pd.MultiIndex.from_tuples(list(keys), names=KEY_COLS))].copy()


def subset_stats(frame: pd.DataFrame, success_col: str) -> dict[str, float]:
    events = float(numeric_col(frame, success_col).sum())
    arrivals = numeric_col(frame, "arrivals")
    risk = numeric_col(frame, "pred_prob")
    mass = arrivals * risk
    return {
        "events": events,
        "arrivals_mean": float(arrivals.mean()) if len(frame) else 0.0,
        "calibrated_risk_mean": float(risk.mean()) if len(frame) else 0.0,
        "event_mass_mean": float(mass.mean()) if len(frame) else 0.0,
        "observed_event_rate": float(events / max(arrivals.sum(), 1.0)) if len(frame) else 0.0,
        "excess_delay60_minutes": float(numeric_col(frame, "excess_delay60_minutes").sum()),
    }


def audit_pair(
    predictions: pd.DataFrame,
    target: str,
    success_col: str,
    fraction: float = 0.10,
    reference_model: str = DEFAULT_REFERENCE,
) -> dict[str, object]:
    candidate, reference = target_pair(predictions, target, reference_model)
    cqs_keys = top_keys(candidate, "cqs_rank_score", fraction)
    ref_keys = top_keys(reference, "pred_prob", fraction)
    common = cqs_keys & ref_keys
    cqs_only = cqs_keys - ref_keys
    ref_only = ref_keys - cqs_keys
    cqs_only_frame = selected_subset(candidate, cqs_only)
    ref_only_frame = selected_subset(candidate, ref_only)
    c_stats = subset_stats(cqs_only_frame, success_col)
    r_stats = subset_stats(ref_only_frame, success_col)
    total_events = float(numeric_col(candidate, success_col).sum())
    queue_size = len(cqs_keys)
    row: dict[str, object] = {
        "target": target,
        "reference_model": reference_model,
        "queue_fraction": fraction,
        "airport_hours": int(len(candidate)),
        "queue_size": int(queue_size),
        "overlap_hours": int(len(common)),
        "overlap_share": float(len(common) / max(queue_size, 1)),
        "churn_share": float(len(cqs_only) / max(queue_size, 1)),
        "cqs_only_hours": int(len(cqs_only)),
        "reference_only_hours": int(len(ref_only)),
        "total_events": total_events,
    }
    for prefix, stats in [("cqs_only", c_stats), ("reference_only", r_stats)]:
        for name, value in stats.items():
            row[f"{prefix}_{name}"] = value
    row["event_gain"] = c_stats["events"] - r_stats["events"]
    row["event_capture_gain"] = row["event_gain"] / max(total_events, 1.0)
    row["excess_delay60_gain"] = c_stats["excess_delay60_minutes"] - r_stats["excess_delay60_minutes"]
    return row


def traffic_labels(base: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    arrivals = numeric_col(base, "arrivals")
    q1, q2 = np.quantile(arrivals, [1 / 3, 2 / 3])
    labels = pd.Series("high", index=base.index, dtype=object)
    labels[arrivals <= q1] = "low"
    labels[(arrivals > q1) & (arrivals <= q2)] = "mid"
    return labels, {"low_max": float(q1), "mid_max": float(q2)}


def traffic_stratum_audit(
    predictions: pd.DataFrame,
    target: str,
    success_col: str,
    fraction: float = 0.10,
    reference_model: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    candidate, reference = target_pair(predictions, target, reference_model)
    cqs_keys = top_keys(candidate, "cqs_rank_score", fraction)
    ref_keys = top_keys(reference, "pred_prob", fraction)
    labels, bounds = traffic_labels(candidate)
    work = candidate.copy()
    work["traffic_stratum"] = labels
    work["in_cqs"] = key_index(work).isin(pd.MultiIndex.from_tuples(list(cqs_keys), names=KEY_COLS))
    work["in_reference"] = key_index(work).isin(pd.MultiIndex.from_tuples(list(ref_keys), names=KEY_COLS))
    rows = []
    total_events = float(numeric_col(work, success_col).sum())
    order = ["low", "mid", "high"]
    for stratum in order:
        group = work[work["traffic_stratum"].eq(stratum)]
        cqs = group[group["in_cqs"]]
        ref = group[group["in_reference"]]
        c_events = float(numeric_col(cqs, success_col).sum())
        r_events = float(numeric_col(ref, success_col).sum())
        rows.append(
            {
                "target": target,
                "reference_model": reference_model,
                "queue_fraction": fraction,
                "traffic_stratum": stratum,
                "low_max_arrivals": bounds["low_max"],
                "mid_max_arrivals": bounds["mid_max"],
                "cqs_selected": int(len(cqs)),
                "reference_selected": int(len(ref)),
                "cqs_events": c_events,
                "reference_events": r_events,
                "event_gain": c_events - r_events,
                "event_capture_gain": (c_events - r_events) / max(total_events, 1.0),
                "cqs_excess_delay60_minutes": float(numeric_col(cqs, "excess_delay60_minutes").sum()),
                "reference_excess_delay60_minutes": float(numeric_col(ref, "excess_delay60_minutes").sum()),
            }
        )
    return pd.DataFrame(rows)


def fold_event_audit(
    predictions: pd.DataFrame,
    target: str,
    success_col: str,
    fraction: float = 0.10,
    reference_model: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    candidate, reference = target_pair(predictions, target, reference_model)
    if "fold_id" not in candidate.columns:
        return pd.DataFrame()
    reference = reference.set_index(KEY_COLS)
    rows = []
    for fold_id, fold_candidate in candidate.groupby("fold_id", sort=True):
        fold_keys = key_index(fold_candidate)
        fold_reference = reference[reference.index.isin(fold_keys)].reset_index()
        cqs_keys = top_keys(fold_candidate, "cqs_rank_score", fraction)
        ref_keys = top_keys(fold_reference, "pred_prob", fraction)
        cqs_selected = selected_subset(fold_candidate, cqs_keys)
        ref_selected = selected_subset(fold_candidate, ref_keys)
        c_events = float(numeric_col(cqs_selected, success_col).sum())
        r_events = float(numeric_col(ref_selected, success_col).sum())
        rows.append(
            {
                "target": target,
                "reference_model": reference_model,
                "queue_fraction": fraction,
                "fold_id": fold_id,
                "cqs_events": c_events,
                "reference_events": r_events,
                "event_gain": c_events - r_events,
                "event_capture_gain": (c_events - r_events)
                / max(float(numeric_col(fold_candidate, success_col).sum()), 1.0),
            }
        )
    return pd.DataFrame(rows)


def write_assessment(out_dir: Path, pair: pd.DataFrame, strata: pd.DataFrame, folds: pd.DataFrame) -> None:
    lines = [
        "# CQS-Rank decision audit",
        "",
        "The audit compares the CQS-Rank fixed-budget queue with the RAEG-Rank probability queue on identical airport-hours.",
        "CQS-only rows enter the CQS queue and do not enter the RAEG queue; reference-only rows have the opposite status.",
        "",
    ]
    for row in pair.itertuples(index=False):
        lines.append(f"## {row.target}")
        lines.append(
            f"- Top-{int(row.queue_fraction * 100)} queue overlap {row.overlap_share:.3f}; churn {row.churn_share:.3f}; "
            f"event gain {row.event_gain:+.0f}; capture gain {row.event_capture_gain:+.4f}."
        )
        lines.append(
            f"- CQS-only hours have mean arrivals {row.cqs_only_arrivals_mean:.2f}, mean calibrated risk "
            f"{row.cqs_only_calibrated_risk_mean:.4f}, and mean event mass {row.cqs_only_event_mass_mean:.3f}; "
            f"reference-only hours have mean arrivals {row.reference_only_arrivals_mean:.2f}, mean calibrated risk "
            f"{row.reference_only_calibrated_risk_mean:.4f}, and mean event mass {row.reference_only_event_mass_mean:.3f}."
        )
        fold_rows = folds[folds["target"].eq(row.target)]
        if not fold_rows.empty:
            win_rate = float((fold_rows["event_gain"] > 0).mean())
            lines.append(f"- Fold-level event-gain win rate: {win_rate:.3f} over {len(fold_rows)} rolling-quarter folds.")
        lines.append("")
    high = strata[strata["traffic_stratum"].eq("high")]
    if not high.empty:
        lines.append("## Traffic-stratum evidence")
        for row in high.itertuples(index=False):
            lines.append(
                f"- {row.target}: high-traffic stratum event gain {row.event_gain:+.0f} "
                f"with CQS-selected {row.cqs_selected} and reference-selected {row.reference_selected} airport-hours."
            )
    (out_dir / "cqs_decision_audit_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    input_path: Path,
    out_dir: Path,
    fraction: float,
    reference_model: str,
    targets: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(input_path, parse_dates=["utc_hour"], low_memory=False)
    pair_rows = []
    strata_parts = []
    fold_parts = []
    for target in targets:
        success_col = TARGETS[target]
        pair_rows.append(audit_pair(predictions, target, success_col, fraction, reference_model))
        strata_parts.append(traffic_stratum_audit(predictions, target, success_col, fraction, reference_model))
        fold_part = fold_event_audit(predictions, target, success_col, fraction, reference_model)
        if not fold_part.empty:
            fold_parts.append(fold_part)
    pair = pd.DataFrame(pair_rows)
    strata = pd.concat(strata_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    pair.to_csv(out_dir / "cqs_decision_pair_audit.csv", index=False)
    strata.to_csv(out_dir / "cqs_decision_traffic_strata.csv", index=False)
    if not folds.empty:
        folds.to_csv(out_dir / "cqs_decision_fold_audit.csv", index=False)
    write_assessment(out_dir, pair, strata, folds)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CQS-Rank queue decisions against a probability queue.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/experiments/applied_soft_computing_smoke/"
            "cqs_rank_base10_2025_rolling_quarter/cqs_rank_eval_predictions.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/experiments/applied_soft_computing_smoke/"
            "cqs_rank_base10_2025_rolling_quarter"
        ),
    )
    parser.add_argument("--queue-fraction", type=float, default=0.10)
    parser.add_argument("--reference-model", default=DEFAULT_REFERENCE)
    parser.add_argument("--targets", default="long_arrival_delay,severe_arrival_delay,cancellation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    out = run(args.input, args.out_dir, args.queue_fraction, args.reference_model, targets)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
