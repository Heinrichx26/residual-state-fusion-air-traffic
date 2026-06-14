from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asoc_cqs_rank import GT_AFRE_MODEL, MODEL_NAME, RAEG_MODEL, TARGETS, capture_at_fraction, metric_row


def metric_gain(candidate: dict[str, float], reference: dict[str, float], metric: str) -> float:
    if metric in {"brier", "ece_10", "log_loss"}:
        return float(reference[metric]) - float(candidate[metric])
    return float(candidate[metric]) - float(reference[metric])


def bootstrap(
    predictions: pd.DataFrame,
    reps: int,
    seed: int,
    budgets: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    metrics = ["top5_capture", "top10_capture", "top20_capture", "brier", "ece_10"]
    key_cols = ["airport", "utc_hour", "month"]
    predictions = predictions.copy()
    predictions["utc_hour"] = pd.to_datetime(predictions["utc_hour"])
    predictions["airport_day"] = predictions["airport"].astype(str) + "_" + predictions["utc_hour"].dt.strftime("%Y-%m-%d")
    for target, target_pred in predictions.groupby("target", sort=True):
        success_col = TARGETS[str(target)]
        candidate = target_pred[target_pred["model"].eq(MODEL_NAME)]
        if candidate.empty:
            continue
        candidate_cols = key_cols + [success_col, "airport_day", "arrivals", "pred_prob", "cqs_rank_score"]
        cand = candidate[candidate_cols].rename(columns={"pred_prob": "cand_prob", "cqs_rank_score": "cand_score"})
        for reference_model in [GT_AFRE_MODEL, RAEG_MODEL]:
            reference = target_pred[target_pred["model"].eq(reference_model)]
            if reference.empty:
                continue
            ref = reference[key_cols + [success_col, "airport_day", "arrivals", "pred_prob"]].rename(
                columns={"pred_prob": "ref_prob"}
            )
            aligned = cand.merge(ref, on=key_cols + [success_col, "airport_day", "arrivals"], how="inner", validate="one_to_one")
            if aligned.empty:
                continue
            full_candidate = metric_row(
                str(target),
                MODEL_NAME,
                aligned.rename(columns={"cand_prob": "pred_prob", "cand_score": "cqs_rank_score"}),
                success_col,
                budgets,
            )
            full_reference = metric_row(
                str(target),
                reference_model,
                aligned.rename(columns={"ref_prob": "pred_prob"}),
                success_col,
                budgets,
            )
            observed = {metric: metric_gain(full_candidate, full_reference, metric) for metric in metrics}
            clusters = aligned["airport_day"].drop_duplicates().to_numpy()
            cluster_index = {cluster: np.flatnonzero(aligned["airport_day"].to_numpy() == cluster) for cluster in clusters}
            draws = {metric: [] for metric in metrics}
            for rep in range(reps):
                sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
                sampled_idx = np.concatenate([cluster_index[cluster] for cluster in sampled_clusters])
                sample = aligned.iloc[sampled_idx]
                sample_candidate = metric_row(
                    str(target),
                    MODEL_NAME,
                    sample.rename(columns={"cand_prob": "pred_prob", "cand_score": "cqs_rank_score"}),
                    success_col,
                    budgets,
                )
                sample_reference = metric_row(
                    str(target),
                    reference_model,
                    sample.rename(columns={"ref_prob": "pred_prob"}),
                    success_col,
                    budgets,
                )
                for metric in metrics:
                    gain = metric_gain(sample_candidate, sample_reference, metric)
                    draws[metric].append(gain)
                    draw_rows.append(
                        {
                            "target": target,
                            "candidate_model": MODEL_NAME,
                            "reference_model": reference_model,
                            "metric": metric,
                            "rep": rep,
                            "gain": gain,
                        }
                    )
            for metric, values in draws.items():
                arr = np.asarray(values, dtype=float)
                summary_rows.append(
                    {
                        "target": target,
                        "candidate_model": MODEL_NAME,
                        "reference_model": reference_model,
                        "metric": metric,
                        "observed_gain": observed[metric],
                        "ci_low": float(np.quantile(arr, 0.025)),
                        "ci_high": float(np.quantile(arr, 0.975)),
                        "bootstrap_reps": reps,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/experiments/applied_soft_computing_smoke/cqs_rank_base10_2025_rolling_quarter/cqs_rank_eval_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/experiments/applied_soft_computing_smoke/cqs_rank_base10_2025_rolling_quarter"))
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260612)
    args = parser.parse_args()
    predictions = pd.read_csv(args.input)
    summary, draws = bootstrap(predictions, reps=args.reps, seed=args.seed, budgets=[0.01, 0.05, 0.10, 0.20, 0.30])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "cqs_rank_bootstrap_summary.csv", index=False)
    draws.to_csv(args.out_dir / "cqs_rank_bootstrap_draws.csv", index=False)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
