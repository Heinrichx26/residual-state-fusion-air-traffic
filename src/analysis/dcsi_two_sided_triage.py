from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("DCSI_PROJECT_ROOT", ROOT))
PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "fusion_framework_strengthening"
    / "dcsi_online_lead_validation"
    / "online_lead_full_2025"
    / "online_lead_predictions.csv"
)
OUT_ROOT = (
    ROOT
    / "results"
    / "experiments"
    / "fusion_framework_strengthening"
    / "dcsi_two_sided_triage"
)


TARGET_EVENT = {
    "long_arrival_delay": "target_arr_delay60_count",
    "cancellation": "target_cancel_count",
}

MODEL_LABEL = {
    "online_baseline": "Online context",
    "online_fixed_decay": "Fixed decay",
    "online_DCSI": "Online DCSI",
    "online_relation_DCSI": "Relation-DCSI",
}


def selected_metrics(df: pd.DataFrame, mask: pd.Series, event_col: str) -> dict[str, float]:
    selected = df.loc[mask]
    total_events = df[event_col].sum()
    total_hours = len(df)
    selected_events = selected[event_col].sum()
    selected_hours = len(selected)
    selected_arrivals = selected["target_arrivals"].sum()
    event_rate = selected_events / selected["target_arrivals"].sum() if selected_arrivals else 0.0
    base_rate = total_events / df["target_arrivals"].sum() if df["target_arrivals"].sum() else 0.0
    return {
        "selected_hours": float(selected_hours),
        "selected_arrivals": float(selected_arrivals),
        "selected_events": float(selected_events),
        "workload_share": selected_hours / total_hours if total_hours else 0.0,
        "event_capture_or_leakage": selected_events / total_events if total_events else 0.0,
        "event_rate": event_rate,
        "lift_vs_panel": event_rate / base_rate if base_rate else 0.0,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = ["online_baseline", "online_fixed_decay", "online_DCSI", "online_relation_DCSI"]
    for (target, horizon, model), part in df.groupby(["target", "horizon", "model"], sort=True):
        if model not in models:
            continue
        event_col = TARGET_EVENT[target]
        part = part.copy()
        part[event_col] = pd.to_numeric(part[event_col], errors="coerce").fillna(0.0)
        part["target_arrivals"] = pd.to_numeric(part["target_arrivals"], errors="coerce").fillna(0.0)
        part["pred_prob"] = pd.to_numeric(part["pred_prob"], errors="coerce")
        part = part.dropna(subset=["pred_prob"])
        if part.empty:
            continue

        ranks_desc = part["pred_prob"].rank(method="first", ascending=False)
        ranks_asc = part["pred_prob"].rank(method="first", ascending=True)
        n = len(part)
        high_mask = ranks_desc <= max(1, int(round(0.10 * n)))
        low_mask = ranks_asc <= max(1, int(round(0.50 * n)))

        high = selected_metrics(part, high_mask, event_col)
        low = selected_metrics(part, low_mask, event_col)
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "model_label": MODEL_LABEL.get(model, model),
                "band": "top_10_review",
                **high,
            }
        )
        rows.append(
            {
                "target": target,
                "horizon": int(horizon),
                "model": model,
                "model_label": MODEL_LABEL.get(model, model),
                "band": "bottom_50_release",
                **low,
            }
        )
    return pd.DataFrame(rows)


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    ref = summary[summary["model"].eq("online_baseline")].copy()
    cand = summary[summary["model"].eq("online_relation_DCSI")].copy()
    keys = ["target", "horizon", "band"]
    merged = cand.merge(ref, on=keys, suffixes=("_relation", "_context"))
    rows = []
    for r in merged.itertuples(index=False):
        event_diff = r.event_capture_or_leakage_relation - r.event_capture_or_leakage_context
        lift_diff = r.lift_vs_panel_relation - r.lift_vs_panel_context
        rows.append(
            {
                "target": r.target,
                "horizon": int(r.horizon),
                "band": r.band,
                "relation_event_share": r.event_capture_or_leakage_relation,
                "context_event_share": r.event_capture_or_leakage_context,
                "event_share_diff": event_diff,
                "relation_lift": r.lift_vs_panel_relation,
                "context_lift": r.lift_vs_panel_context,
                "lift_diff": lift_diff,
                "selected_hours": r.selected_hours_relation,
            }
        )
    return pd.DataFrame(rows)


def write_assessment(comp: pd.DataFrame, out_dir: Path, mode: str) -> None:
    lines = ["# Two-sided triage assessment", ""]
    gate = True
    for r in comp[(comp["horizon"].eq(1)) & (comp["band"].eq("top_10_review"))].itertuples(index=False):
        gate = gate and r.event_share_diff > 0
        label = "long-delay arrivals" if r.target == "long_arrival_delay" else "cancelled arrivals"
        lines.append(
            f"- 1 h {label}: top 10% Relation-DCSI captures {r.relation_event_share:.3f}, "
            f"context captures {r.context_event_share:.3f}, difference {r.event_share_diff:+.3f}."
        )
    for r in comp[(comp["horizon"].eq(1)) & (comp["band"].eq("bottom_50_release"))].itertuples(index=False):
        gate = gate and r.event_share_diff < 0
        label = "long-delay arrivals" if r.target == "long_arrival_delay" else "cancelled arrivals"
        lines.append(
            f"- 1 h {label}: bottom 50% Relation-DCSI leakage {r.relation_event_share:.3f}, "
            f"context leakage {r.context_event_share:.3f}, difference {r.event_share_diff:+.3f}."
        )
    lines.extend(["", f"Smoke gate: {'pass' if gate else 'fail'} in {mode} mode."])
    (out_dir / "two_sided_triage_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--months", default="1,7,12")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS.exists():
        raise SystemExit(
            "Online lead-time prediction table is missing. Set DCSI_PROJECT_ROOT "
            "to the reconstructed project root or rerun dcsi_online_lead_validation.py first."
        )
    usecols = [
        "row_id",
        "airport",
        "utc_hour",
        "month",
        "target_arrivals",
        "target_arr_delay60_count",
        "target_cancel_count",
        "pred_prob",
        "target",
        "model",
        "horizon",
    ]
    df = pd.read_csv(PREDICTIONS, usecols=usecols)
    if args.mode == "smoke":
        months = [int(x) for x in args.months.split(",") if x.strip()]
        df = df[df["month"].isin(months)].copy()
    summary = summarize(df)
    comp = compare(summary)
    summary.to_csv(out_dir / "two_sided_triage_summary.csv", index=False)
    comp.to_csv(out_dir / "two_sided_triage_comparison.csv", index=False)
    write_assessment(comp, out_dir, args.mode)


if __name__ == "__main__":
    main()
