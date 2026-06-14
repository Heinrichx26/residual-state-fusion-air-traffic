from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CQS_MODEL = "CQS-Rank"
GT_MODEL = "Graph-temporal evidence"
RAEG_MODEL = "RAEG-Rank"


def build_carrier_audit(metrics: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target, group in metrics.groupby("target", sort=True):
        cqs = group[group["model"].eq(CQS_MODEL)]
        if cqs.empty:
            continue
        cqs_row = cqs.iloc[0]
        target_gains = gains[gains["target"].eq(target)]
        gt = target_gains[target_gains["reference"].eq(GT_MODEL)]
        raeg = target_gains[target_gains["reference"].eq(RAEG_MODEL)]
        if gt.empty or raeg.empty:
            continue
        gt_row = gt.iloc[0]
        raeg_row = raeg.iloc[0]
        gain_gt = float(gt_row["top10_capture_gain"])
        gain_raeg = float(raeg_row["top10_capture_gain"])
        rows.append(
            {
                "target": target,
                "cqs_top10_capture": float(cqs_row["top10_capture"]),
                "gain_vs_gt_probability": gain_gt,
                "gain_vs_raeg_probability": gain_raeg,
                "brier_gain_vs_gt": float(gt_row.get("brier_gain", 0.0)),
                "brier_gain_vs_raeg": float(raeg_row.get("brier_gain", 0.0)),
                "ece_gain_vs_gt": float(gt_row.get("ece_10_gain", 0.0)),
                "ece_gain_vs_raeg": float(raeg_row.get("ece_10_gain", 0.0)),
                "positive_against_both": bool(gain_gt > 0.0 and gain_raeg > 0.0),
            }
        )
    return pd.DataFrame(rows)


def run(input_dir: Path, out_dir: Path) -> Path:
    metrics = pd.read_csv(input_dir / "cqs_rank_metrics.csv")
    gains = pd.read_csv(input_dir / "cqs_rank_gains.csv")
    audit = build_carrier_audit(metrics, gains)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "cqs_queue_carrier_audit.csv", index=False)
    lines = [
        "# CQS-Rank carrier audit",
        "",
        "The audit compares the closed-form CQS queue against two probability queues on the same airport-hours.",
        "Positive top-10 gains against both probability carriers indicate that the queue-set decision layer is not dependent on the RAEG residual layer.",
        "",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"- {row.target}: CQS top-10 {row.cqs_top10_capture:.4f}; "
            f"gain vs GT-AFRE probability {row.gain_vs_gt_probability:+.4f}; "
            f"gain vs RAEG probability {row.gain_vs_raeg_probability:+.4f}."
        )
    (out_dir / "cqs_queue_carrier_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit whether CQS queue gains depend on the RAEG probability carrier.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiments/applied_soft_computing_smoke/cqs_rank_base10_2025_rolling_quarter"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/experiments/applied_soft_computing_smoke/cqs_rank_base10_2025_rolling_quarter"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args.input_dir, args.out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
