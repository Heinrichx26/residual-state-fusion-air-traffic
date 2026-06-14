from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from asoc_raeg_rank import (
    BUDGET_FRACTIONS,
    OUT_ROOT,
    RAEGConfig,
    TARGETS,
    add_gain_table,
    extended_metric_row,
    paired_bootstrap,
    parse_float_list,
    queue_budget_curve,
    write_assessment,
)


def infer_source_predictions(source: Path) -> Path:
    if source.is_file():
        return source
    candidate = source / "raeg_rank_predictions.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"No raeg_rank_predictions.csv found in {source}")
    return candidate


def load_source_manifest(source: Path) -> dict:
    manifest_path = source / "raeg_rank_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def build_tables(predictions: pd.DataFrame, budgets: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=True):
        success_col = TARGETS[str(target)]
        metrics_rows.append(extended_metric_row(str(target), str(model), group, success_col, budgets))
        for fold_id, fold in group.groupby("fold_id", sort=True):
            fold_rows.append(extended_metric_row(str(target), str(model), fold, success_col, budgets) | {"fold_id": fold_id})
    metrics = pd.DataFrame(metrics_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    budget_curve = queue_budget_curve(predictions, budgets)
    return metrics, fold_metrics, budget_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Directory containing raeg_rank_predictions.csv or the CSV itself.")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--budget-fractions", default="0.01,0.05,0.10,0.20,0.30")
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260611)
    args = parser.parse_args()

    source_path = Path(args.source)
    prediction_path = infer_source_predictions(source_path)
    source_dir = prediction_path.parent
    output_name = args.output_name or f"{source_dir.name}_extmetrics"
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)

    budgets = parse_float_list(args.budget_fractions) if args.budget_fractions else BUDGET_FRACTIONS
    predictions = pd.read_csv(prediction_path, parse_dates=["utc_hour"])
    metrics, fold_metrics, budget_curve = build_tables(predictions, budgets)
    gains = add_gain_table(metrics)
    bootstrap_summary, bootstrap_draws = paired_bootstrap(
        predictions,
        budgets=budgets,
        reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )

    metrics.to_csv(out / "raeg_rank_metrics.csv", index=False)
    fold_metrics.to_csv(out / "raeg_rank_fold_metrics.csv", index=False)
    gains.to_csv(out / "raeg_rank_gains.csv", index=False)
    gains.to_markdown(out / "raeg_rank_gains.md", index=False)
    budget_curve.to_csv(out / "raeg_rank_budget_curve.csv", index=False)
    if not bootstrap_summary.empty:
        bootstrap_summary.to_csv(out / "raeg_rank_bootstrap_summary.csv", index=False)
        bootstrap_draws.to_csv(out / "raeg_rank_bootstrap_draws.csv", index=False)

    manifest = load_source_manifest(source_dir)
    config = RAEGConfig(**{k: v for k, v in manifest.get("config", {}).items() if k in RAEGConfig.__dataclass_fields__})
    write_assessment(
        out,
        gains,
        bootstrap_summary,
        months=manifest.get("months", []),
        airports=manifest.get("airports") if isinstance(manifest.get("airports"), list) else None,
        validation=manifest.get("validation", "postprocess"),
        config=config,
    )
    post_manifest = {
        "source_predictions": str(prediction_path),
        "source_manifest": str(source_dir / "raeg_rank_manifest.json"),
        "budget_fractions": budgets,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "source": manifest,
    }
    (out / "raeg_rank_postprocess_manifest.json").write_text(json.dumps(post_manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
