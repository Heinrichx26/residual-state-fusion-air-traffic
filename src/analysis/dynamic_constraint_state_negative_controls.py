from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dynamic_constraint_state_inversion import (
    DYNAMIC_NUMERIC,
    MAIN_10,
    MAIN_2025_PANEL,
    MAIN_CATS,
    TARGETS,
    attach_dynamic,
    fit_predict,
    load_panel,
    model_rows,
    parse_airports,
    parse_months,
)
from fusion_prediction_increment import evaluate
from fusion_strengthening_common import ROOT_OUT


def selected_rho_table(path: Path) -> pd.DataFrame:
    rho = pd.read_csv(path)
    return rho.sort_values(["outer_month", "inner_mean_auc"], ascending=[True, False]).drop_duplicates("outer_month")


def airport_rotated_panel(panel: pd.DataFrame, rho: float) -> pd.DataFrame:
    airports = sorted(panel["airport"].unique())
    mapping = {airport: airports[(i + 1) % len(airports)] for i, airport in enumerate(airports)}
    donor = panel[["airport", "utc_hour", "active_minutes"]].copy()
    donor["airport"] = donor["airport"].map(mapping)
    donor = donor.rename(columns={"active_minutes": "rotated_active_minutes"})
    tmp = panel.merge(donor, on=["airport", "utc_hour"], how="left")
    tmp["active_minutes"] = tmp["rotated_active_minutes"].fillna(0.0)
    return attach_dynamic(tmp.drop(columns=["rotated_active_minutes"]), rho)


def airport_reversed_panel(panel: pd.DataFrame, rho: float) -> pd.DataFrame:
    parts = []
    for _, g in panel.sort_values(["airport", "utc_hour"]).groupby("airport", sort=False):
        use = g.copy()
        use["active_minutes"] = g["active_minutes"].to_numpy()[::-1]
        parts.append(use)
    tmp = pd.concat(parts, ignore_index=True).sort_values("row_id")
    return attach_dynamic(tmp, rho)


def evaluate_control(panel: pd.DataFrame, rho_selected: pd.DataFrame, control: str) -> pd.DataFrame:
    cache: dict[float, pd.DataFrame] = {}
    predictions = []
    for outer_month, rho in rho_selected[["outer_month", "rho"]].itertuples(index=False):
        rho = float(rho)
        if rho not in cache:
            if control == "real_action":
                cache[rho] = model_rows(attach_dynamic(panel, rho))
            elif control == "airport_rotated_action":
                cache[rho] = model_rows(airport_rotated_panel(panel, rho))
            elif control == "airport_reversed_action":
                cache[rho] = model_rows(airport_reversed_panel(panel, rho))
            else:
                raise ValueError(control)
        scored = cache[rho]
        for target, success_col in TARGETS.items():
            train = scored[scored["month"] != int(outer_month)].copy()
            test = scored[scored["month"] == int(outer_month)].copy()
            pred, _ = fit_predict(train, test, success_col, DYNAMIC_NUMERIC, MAIN_CATS)
            pred["control"] = control
            pred["target"] = target
            predictions.append(pred)
    pred_all = pd.concat(predictions, ignore_index=True)
    rows = []
    for (control_name, target), g in pred_all.groupby(["control", "target"]):
        success_col = TARGETS[target]
        rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"control": control_name, "target": target}
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    source = ROOT_OUT / args.source_output
    out = ROOT_OUT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    airports = parse_airports(args.airports) or MAIN_10
    panel = load_panel(MAIN_2025_PANEL, 2025, months, airports)
    rho_selected = selected_rho_table(source / "dcsi_rho_selection.csv")
    controls = [
        evaluate_control(panel, rho_selected, "real_action"),
        evaluate_control(panel, rho_selected, "airport_rotated_action"),
        evaluate_control(panel, rho_selected, "airport_reversed_action"),
    ]
    metrics = pd.concat(controls, ignore_index=True)
    fixed = pd.read_csv(source / "dcsi_cv_gains.csv")
    fixed = fixed[fixed["model"].eq("fixed_window")][["target", "auc", "brier"]].rename(
        columns={"auc": "fixed_auc", "brier": "fixed_brier"}
    )
    real = metrics[metrics["control"].eq("real_action")][["target", "auc", "brier"]].rename(
        columns={"auc": "real_auc", "brier": "real_brier"}
    )
    summary = metrics.merge(fixed, on="target", how="left").merge(real, on="target", how="left")
    summary["auc_gap_vs_fixed"] = summary["auc"] - summary["fixed_auc"]
    summary["auc_gap_vs_real"] = summary["auc"] - summary["real_auc"]
    summary["brier_gain_vs_fixed"] = summary["fixed_brier"] - summary["brier"]
    summary.to_csv(out / "dcsi_negative_control_metrics.csv", index=False)
    lines = ["# Dynamic constraint-state negative controls", ""]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.control}, {row.target}: AUC {float(row.auc):.3f}; gap vs real {float(row.auc_gap_vs_real):+.3f}; gap vs fixed {float(row.auc_gap_vs_fixed):+.3f}."
        )
    (out / "dcsi_negative_control_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", default="dynamic_constraint_state_inversion_full_2025")
    parser.add_argument("--output-name", default="dynamic_constraint_state_negative_controls")
    parser.add_argument("--months", default="1-12")
    parser.add_argument("--airports", default="ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
