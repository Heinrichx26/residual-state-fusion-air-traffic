from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asoc_cqs_decision_audit import DEFAULT_REFERENCE, KEY_COLS, TARGETS, key_index, numeric_col, selected_subset, top_keys


MODEL_NAME = "CQS-Rank"


def risk_bins(frame: pd.DataFrame, n_bins: int) -> pd.Series:
    probs = numeric_col(frame, "pred_prob")
    ranks = probs.rank(method="first")
    bins = pd.qcut(ranks, q=min(n_bins, len(frame)), labels=False, duplicates="drop")
    return pd.Series(bins, index=frame.index, dtype="Int64").fillna(0).astype(int)


def calibration_gap_table(frame: pd.DataFrame, success_col: str, n_bins: int = 10) -> pd.DataFrame:
    work = frame.copy()
    work["_bin"] = risk_bins(work, n_bins)
    work["_arrivals"] = numeric_col(work, "arrivals")
    work["_events"] = numeric_col(work, success_col)
    work["_pred_mass"] = work["_arrivals"] * numeric_col(work, "pred_prob")
    rows = []
    for bin_id, group in work.groupby("_bin", sort=True):
        arrivals = float(group["_arrivals"].sum())
        pred_rate = float(group["_pred_mass"].sum() / max(arrivals, 1.0))
        obs_rate = float(group["_events"].sum() / max(arrivals, 1.0))
        rows.append(
            {
                "bin_id": int(bin_id),
                "count": int(len(group)),
                "arrivals": arrivals,
                "pred_rate": pred_rate,
                "obs_rate": obs_rate,
                "gap": abs(obs_rate - pred_rate),
            }
        )
    return pd.DataFrame(rows)


def add_calibration_lower_score(frame: pd.DataFrame, success_col: str, n_bins: int = 10) -> pd.DataFrame:
    work = frame.copy()
    work["calibration_bin"] = risk_bins(work, n_bins)
    gaps = calibration_gap_table(work, success_col, n_bins).set_index("bin_id")["gap"].to_dict()
    work["calibration_gap"] = work["calibration_bin"].map(gaps).astype(float)
    work["calibration_lower_prob"] = np.maximum(0.0, numeric_col(work, "pred_prob") - work["calibration_gap"])
    work["calibration_lower_score"] = numeric_col(work, "arrivals") * work["calibration_lower_prob"]
    return work


def paired_target_frame(predictions: pd.DataFrame, target: str, reference_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def robust_certificate(
    predictions: pd.DataFrame,
    target: str,
    success_col: str,
    fraction: float = 0.10,
    reference_model: str = DEFAULT_REFERENCE,
    n_bins: int = 10,
) -> dict[str, object]:
    candidate, reference = paired_target_frame(predictions, target, reference_model)
    scored = add_calibration_lower_score(candidate, success_col, n_bins)
    robust_keys = top_keys(scored, "calibration_lower_score", fraction)
    cqs_keys = top_keys(candidate, "cqs_rank_score", fraction)
    ref_keys = top_keys(reference, "pred_prob", fraction)
    robust_selected = selected_subset(scored, robust_keys)
    cqs_selected = selected_subset(candidate, cqs_keys)
    ref_selected = selected_subset(candidate, ref_keys)

    robust_events = float(numeric_col(robust_selected, success_col).sum())
    cqs_events = float(numeric_col(cqs_selected, success_col).sum())
    reference_events = float(numeric_col(ref_selected, success_col).sum())
    total_events = float(numeric_col(candidate, success_col).sum())
    robust_mass = float(numeric_col(robust_selected, "arrivals").mul(numeric_col(robust_selected, "pred_prob")).sum())
    robust_lower_mass = float(numeric_col(robust_selected, "calibration_lower_score").sum())
    return {
        "target": target,
        "reference_model": reference_model,
        "queue_fraction": fraction,
        "n_bins": n_bins,
        "queue_size": int(len(robust_keys)),
        "robust_overlap_with_cqs": float(len(robust_keys & cqs_keys) / max(len(robust_keys), 1)),
        "robust_events": robust_events,
        "cqs_events": cqs_events,
        "reference_events": reference_events,
        "robust_event_gain_vs_reference": robust_events - reference_events,
        "robust_capture_gain_vs_reference": (robust_events - reference_events) / max(total_events, 1.0),
        "robust_event_gap_vs_cqs": robust_events - cqs_events,
        "robust_mean_gap": float(scored["calibration_gap"].mean()),
        "robust_max_gap": float(scored["calibration_gap"].max()),
        "robust_expected_mass": robust_mass,
        "robust_lower_mass": robust_lower_mass,
    }


def run(
    input_path: Path,
    out_dir: Path,
    fraction: float,
    reference_model: str,
    targets: list[str],
    n_bins: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(input_path, parse_dates=["utc_hour"], low_memory=False)
    rows = []
    gap_parts = []
    for target in targets:
        success_col = TARGETS[target]
        candidate, _ = paired_target_frame(predictions, target, reference_model)
        rows.append(robust_certificate(predictions, target, success_col, fraction, reference_model, n_bins))
        gaps = calibration_gap_table(candidate, success_col, n_bins)
        gaps.insert(0, "target", target)
        gap_parts.append(gaps)
    cert = pd.DataFrame(rows)
    gaps = pd.concat(gap_parts, ignore_index=True)
    cert.to_csv(out_dir / "cqs_calibration_robust_certificate.csv", index=False)
    gaps.to_csv(out_dir / "cqs_calibration_gap_bins.csv", index=False)
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a calibration-robust lower event-mass certificate for CQS-Rank.")
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
    parser.add_argument("--n-bins", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    out = run(args.input, args.out_dir, args.queue_fraction, args.reference_model, targets, args.n_bins)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
