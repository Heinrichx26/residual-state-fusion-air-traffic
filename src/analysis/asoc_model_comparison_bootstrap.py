from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from asoc_fuzzy_residual_evidence_smoke import TARGETS
from asoc_soft_computing_smoke import OUT_ROOT, capture_at_fraction, ece
from asoc_temporal_frequency_smoke import DCSI_PREDICTIONS
from fusion_prediction_increment import evaluate
from fusion_strengthening_prediction_diagnostics import grouped_pr_auc


PROJECT = Path(__file__).resolve().parents[2]


def metric_values(df: pd.DataFrame, success_col: str) -> dict[str, float]:
    successes = df[success_col].to_numpy(float)
    totals = df["arrivals"].to_numpy(float)
    prob = df["pred_prob"].to_numpy(float)
    base = evaluate(successes, totals, prob)
    return {
        "auc": base["auc"],
        "pr_auc": grouped_pr_auc(successes, totals, prob),
        "brier": base["brier"],
        "ece_10": ece(successes, totals, prob),
        "top10_capture": capture_at_fraction(successes, totals, prob),
    }


def weighted_top_capture(successes: np.ndarray, row_weights: np.ndarray, prob: np.ndarray, fraction: float = 0.10) -> float:
    total_events = float((successes * row_weights).sum())
    total_rows = float(row_weights.sum())
    if total_events <= 0 or total_rows <= 0:
        return np.nan
    cutoff = max(1.0, np.ceil(total_rows * fraction))
    order = np.argsort(prob)[::-1]
    remaining = cutoff
    captured = 0.0
    for idx in order:
        weight = float(row_weights[idx])
        if weight <= 0:
            continue
        take = min(weight, remaining)
        captured += float(successes[idx]) * take
        remaining -= take
        if remaining <= 0:
            break
    return captured / total_events


def weighted_metric_values(frame: pd.DataFrame, success_col: str, cluster_counts: np.ndarray) -> dict[str, float]:
    cluster_idx = frame["_cluster_idx"].to_numpy(int)
    row_weights = cluster_counts[cluster_idx].astype(float)
    keep = row_weights > 0
    if not keep.any():
        return {metric: np.nan for metric in ["auc", "pr_auc", "brier", "ece_10", "top10_capture"]}
    successes = frame[success_col].to_numpy(float)[keep]
    totals = frame["arrivals"].to_numpy(float)[keep]
    prob = frame["pred_prob"].to_numpy(float)[keep]
    row_weights = row_weights[keep]
    weighted_successes = successes * row_weights
    weighted_totals = totals * row_weights
    weighted_failures = np.maximum(weighted_totals - weighted_successes, 0.0)
    y_binary = np.r_[np.ones(len(prob), dtype=int), np.zeros(len(prob), dtype=int)]
    prob_binary = np.r_[prob, prob]
    weights_binary = np.r_[weighted_successes, weighted_failures]
    keep_binary = weights_binary > 0
    if keep_binary.sum() == 0 or len(np.unique(y_binary[keep_binary])) < 2:
        auc = np.nan
        pr_auc = np.nan
    else:
        auc = float(roc_auc_score(y_binary[keep_binary], prob_binary[keep_binary], sample_weight=weights_binary[keep_binary]))
        pr_auc = float(
            average_precision_score(
                y_binary[keep_binary],
                prob_binary[keep_binary],
                sample_weight=weights_binary[keep_binary],
            )
        )
    brier = float(
        (
            weighted_successes * (1.0 - prob) ** 2
            + weighted_failures * prob**2
        ).sum()
        / max(weighted_totals.sum(), 1.0)
    )
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece_10": ece(weighted_successes, weighted_totals, prob),
        "top10_capture": weighted_top_capture(successes, row_weights, prob),
    }


def relation_predictions(months: list[int], airports: set[str], target: str, keys: pd.DataFrame) -> pd.DataFrame:
    success_col = TARGETS[target]
    rel = pd.read_csv(DCSI_PREDICTIONS, parse_dates=["utc_hour"])
    rel = rel[
        rel["month"].isin(months)
        & rel["airport"].isin(airports)
        & rel["target"].eq(target)
        & rel["horizon"].eq(1)
        & rel["model"].eq("online_relation_DCSI")
    ].copy()
    if target == "long_arrival_delay":
        rel[success_col] = pd.to_numeric(rel["target_arr_delay60_count"], errors="coerce").fillna(0.0)
    else:
        rel[success_col] = pd.to_numeric(rel["target_cancel_count"], errors="coerce").fillna(0.0)
    rel = rel.rename(columns={"target_arrivals": "arrivals", "pred_prob": "pred_prob"})
    rel = rel[["airport", "utc_hour", "month", "arrivals", success_col, "pred_prob"]].copy()
    rel["model"] = "Relation-DCSI h1"
    return rel.merge(keys[["airport", "utc_hour", "month"]].drop_duplicates(), on=["airport", "utc_hour", "month"], how="inner")


def load_predictions(prediction_file: Path, months: list[int], model_names: list[str]) -> pd.DataFrame:
    pred = pd.read_csv(prediction_file, parse_dates=["utc_hour"])
    pred = pred[pred["month"].isin(months) & pred["model"].isin(model_names)].copy()
    parts = [pred]
    airports = set(pred["airport"].astype(str).unique())
    for target in sorted(pred["target"].dropna().unique()):
        keys = pred[pred["target"].eq(target)][["airport", "utc_hour", "month"]].copy()
        parts.append(relation_predictions(months, airports, target, keys).assign(target=target))
    out = pd.concat(parts, ignore_index=True)
    out["airport_day"] = out["airport"].astype(str) + "|" + out["utc_hour"].dt.floor("D").astype(str)
    return out


def bootstrap_compare(
    pred: pd.DataFrame,
    pairs: list[tuple[str, str]],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    draws = []
    for target, target_frame in pred.groupby("target", sort=True):
        success_col = TARGETS[target]
        clusters = np.array(sorted(target_frame["airport_day"].unique()))
        cluster_index = {cluster: idx for idx, cluster in enumerate(clusters)}
        target_frame = target_frame.copy()
        target_frame["_cluster_idx"] = target_frame["airport_day"].map(cluster_index).astype(int)
        model_frames = {model: frame.copy() for model, frame in target_frame.groupby("model", sort=False)}
        for candidate, baseline in pairs:
            if candidate not in model_frames or baseline not in model_frames:
                continue
            pair_draws = []
            for rep in range(reps):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                sampled_idx = np.array([cluster_index[item] for item in sampled], dtype=int)
                cluster_counts = np.bincount(sampled_idx, minlength=len(clusters))
                cand_m = weighted_metric_values(model_frames[candidate], success_col, cluster_counts)
                base_m = weighted_metric_values(model_frames[baseline], success_col, cluster_counts)
                row = {
                    "target": target,
                    "candidate": candidate,
                    "baseline": baseline,
                    "rep": rep,
                }
                for metric in ["auc", "pr_auc", "brier", "ece_10", "top10_capture"]:
                    if metric in {"brier", "ece_10"}:
                        row[f"{metric}_gain"] = base_m[metric] - cand_m[metric]
                    else:
                        row[f"{metric}_gain"] = cand_m[metric] - base_m[metric]
                pair_draws.append(row)
            draws.extend(pair_draws)
            draw_frame = pd.DataFrame(pair_draws)
            summary = {
                "target": target,
                "candidate": candidate,
                "baseline": baseline,
                "reps": int(len(draw_frame)),
            }
            for metric in ["auc_gain", "pr_auc_gain", "brier_gain", "ece_10_gain", "top10_capture_gain"]:
                vals = draw_frame[metric].dropna().to_numpy(float)
                summary[metric] = float(np.mean(vals))
                summary[f"{metric}_ci_low"] = float(np.quantile(vals, 0.025))
                summary[f"{metric}_ci_high"] = float(np.quantile(vals, 0.975))
            rows.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(draws)


def parse_months(text: str) -> list[int]:
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
    return sorted({month for month in out if 1 <= month <= 12})


def run(args: argparse.Namespace) -> None:
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    prediction_file = PROJECT / args.prediction_file
    model_names = [item.strip() for item in args.models.split(";") if item.strip()]
    pred = load_predictions(prediction_file, parse_months(args.months), model_names)
    pairs = []
    for item in args.pairs.split(";"):
        item = item.strip()
        if not item:
            continue
        candidate, baseline = [piece.strip() for piece in item.split("::", 1)]
        pairs.append((candidate, baseline))
    summary, draws = bootstrap_compare(pred, pairs, args.reps, args.seed)
    summary.to_csv(out / "bootstrap_comparison_summary.csv", index=False)
    draws.to_csv(out / "bootstrap_comparison_draws.csv", index=False)
    write_assessment(out, summary, args.reps)
    print(f"wrote {out}")


def write_assessment(out: Path, summary: pd.DataFrame, reps: int) -> None:
    lines = [
        "# Bootstrap model-comparison assessment",
        "",
        f"Airport-day cluster bootstrap; requested reps {reps}.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"## {row.target}: {row.candidate} vs {row.baseline}")
        lines.append(
            f"- AUC gain {row.auc_gain:+.3f} [{row.auc_gain_ci_low:+.3f}, {row.auc_gain_ci_high:+.3f}], "
            f"PR-AUC gain {row.pr_auc_gain:+.3f} [{row.pr_auc_gain_ci_low:+.3f}, {row.pr_auc_gain_ci_high:+.3f}], "
            f"top-10 gain {row.top10_capture_gain:+.3f} [{row.top10_capture_gain_ci_low:+.3f}, {row.top10_capture_gain_ci_high:+.3f}]."
        )
        lines.append("")
    (out / "bootstrap_comparison_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", default="results/experiments/applied_soft_computing_smoke/tree_family_full_10airports_2025/tree_family_predictions.csv")
    parser.add_argument("--months", default="1-12")
    parser.add_argument(
        "--models",
        default="LightGBM full-demand advisory;LightGBM AFRE no relation;LightGBM AFRE soft evidence;XGBoost AFRE soft evidence",
    )
    parser.add_argument(
        "--pairs",
        default=(
            "LightGBM AFRE no relation::Relation-DCSI h1;"
            "LightGBM AFRE soft evidence::Relation-DCSI h1;"
            "XGBoost AFRE soft evidence::Relation-DCSI h1;"
            "LightGBM AFRE no relation::LightGBM full-demand advisory;"
            "LightGBM AFRE soft evidence::LightGBM full-demand advisory;"
            "XGBoost AFRE soft evidence::LightGBM full-demand advisory"
        ),
    )
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--output-name", default="bootstrap_tree_family_full_10airports_2025")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
