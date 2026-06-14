from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CQS_REQUIRED = [
    "README.md",
    "requirements.txt",
    "docs/data_sources.md",
    "docs/reproduction.md",
    "docs/cqs_rank_release_manifest.md",
    "data/raw/ourairports/airports.csv",
    "src/analysis/asoc_cqs_rank.py",
    "src/analysis/asoc_cqs_rank_smoke.py",
    "src/analysis/asoc_cqs_rank_full.py",
    "src/analysis/asoc_cqs_rank_bootstrap.py",
    "src/analysis/asoc_cqs_carrier_audit.py",
    "src/analysis/asoc_cqs_carrier_audit_tests.py",
    "src/analysis/asoc_cqs_decision_audit.py",
    "src/analysis/asoc_cqs_decision_audit_tests.py",
    "src/analysis/asoc_cqs_robust_certificate.py",
    "src/analysis/asoc_cqs_robust_certificate_tests.py",
    "src/analysis/asoc_cqs_decision_strengthening.py",
    "src/analysis/asoc_raeg_rank.py",
    "src/analysis/asoc_raeg_rank_smoke.py",
    "src/analysis/asoc_raeg_rank_full.py",
    "src/analysis/asoc_raeg_rank_postprocess.py",
    "src/analysis/asoc_raeg_rank_tests.py",
    "src/analysis/asoc_soft_computing_smoke.py",
    "src/analysis/asoc_fuzzy_residual_evidence_smoke.py",
    "src/analysis/asoc_temporal_frequency_smoke.py",
    "src/analysis/asoc_graph_temporal_baselines_smoke.py",
    "src/analysis/asoc_graph_temporal_feature_tests.py",
    "src/analysis/asoc_model_comparison_bootstrap.py",
    "src/analysis/asoc_reporting_tables.py",
    "src/analysis/fusion_prediction_increment.py",
    "src/analysis/fusion_strengthening_common.py",
    "src/analysis/fusion_strengthening_demand_residual.py",
    "src/analysis/fusion_strengthening_prediction_diagnostics.py",
    "src/plotting/build_gt_afre_article_figures.py",
    "src/plotting/build_gt_afre_article_figures_tests.py",
    "src/figures/make_cqs_graphical_abstract.py",
    "results/cqs_rank/primary_10airport_rolling/cqs_rank_assessment.md",
    "results/cqs_rank/primary_10airport_rolling/cqs_rank_metrics.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_rank_gains.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_budget_gain_ci_maintext.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_budget_gain_ci_summary.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_rank_bootstrap_summary.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_queue_carrier_audit.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_decision_pair_audit.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_decision_fold_audit.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_decision_traffic_strata.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_calibration_robust_certificate.csv",
    "results/cqs_rank/primary_10airport_rolling/cqs_calibration_gap_bins.csv",
    "results/cqs_rank/decision_strengthening/cqs_decision_strengthening_assessment.md",
    "results/cqs_rank/decision_strengthening/cqs_strengthening_matrix.csv",
    "results/cqs_rank/decision_strengthening/all_scenario_cqs_gains.csv",
    "results/cqs_rank/decision_strengthening/all_scenario_cqs_metrics.csv",
    "results/cqs_rank/decision_strengthening/all_scenario_calibration_stress.csv",
    "results/cqs_rank/decision_strengthening/all_scenario_decision_value.csv",
    "results/cqs_rank/decision_strengthening/all_scenario_placebo_checks.csv",
    "results/cqs_rank/decision_strengthening/expanded_comparator_protocol.csv",
    "results/cqs_rank/smoke/cqs_rank_assessment.md",
    "results/cqs_rank/smoke/cqs_rank_metrics.csv",
    "results/cqs_rank/smoke/cqs_rank_gains.csv",
]

PROVENANCE_REQUIRED = [
    "data_schema/digital_thread_entities.csv",
    "data_schema/kbs_knowledge_entities.csv",
    "data_schema/kbs_knowledge_relations.csv",
    "data_schema/kbs_knowledge_rules.csv",
    "data_schema/dcsi_airport_hour_schema.json",
    "data_schema/reasoning_operators.csv",
    "data_schema/reasoning_queries.csv",
    "data_schema/operators.csv",
    "data_schema/feedback_policy.csv",
    "data_schema/queries.csv",
    "data_schema/rule_execution_example.csv",
    "data_schema/source_fields.csv",
    "data_schema/split_ids.csv",
    "src/analysis/dcsi_hmm_state_reference.py",
    "src/analysis/dcsi_online_lead_validation.py",
    "src/analysis/dcsi_two_sided_triage.py",
    "src/analysis/build_knowledge_reasoning_feedback.py",
    "results/benchmark/benchmark_field_dictionary.csv",
    "results/benchmark/benchmark_task_definitions.csv",
    "results/benchmark/benchmark_split_definitions.csv",
    "results/benchmark/benchmark_baseline_scores.csv",
    "results/scorecards/paper_scorecard.csv",
]

EXPECTED_TABLE_MIN_ROWS = {
    "results/cqs_rank/primary_10airport_rolling/cqs_rank_metrics.csv": 9,
    "results/cqs_rank/primary_10airport_rolling/cqs_budget_gain_ci_maintext.csv": 3,
    "results/cqs_rank/primary_10airport_rolling/cqs_calibration_robust_certificate.csv": 3,
    "results/cqs_rank/decision_strengthening/cqs_strengthening_matrix.csv": 9,
    "results/cqs_rank/decision_strengthening/all_scenario_calibration_stress.csv": 81,
    "results/cqs_rank/decision_strengthening/all_scenario_placebo_checks.csv": 9,
    "results/cqs_rank/smoke/cqs_rank_metrics.csv": 1,
}


def row(path: str, group: str) -> dict[str, object]:
    full = ROOT / path
    return {
        "group": group,
        "path": path,
        "exists": full.exists(),
        "bytes": full.stat().st_size if full.exists() else 0,
    }


def table_summary(path: str) -> dict[str, object]:
    full = ROOT / path
    if not full.exists() or full.suffix.lower() != ".csv":
        return {"path": path, "rows": "", "columns": "", "min_rows": "", "row_check": ""}
    data = pd.read_csv(full)
    min_rows = EXPECTED_TABLE_MIN_ROWS.get(path, 0)
    return {
        "path": path,
        "rows": len(data),
        "columns": len(data.columns),
        "min_rows": min_rows if min_rows else "",
        "row_check": "ok" if len(data) >= min_rows else "too_few_rows",
    }


def main() -> None:
    required_rows = [row(path, "cqs_rank") for path in CQS_REQUIRED]
    required_rows.extend(row(path, "provenance") for path in PROVENANCE_REQUIRED)
    audit = pd.DataFrame(required_rows)
    missing = audit[~audit["exists"]]

    out = ROOT / "results" / "release_package_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)

    summaries = pd.DataFrame([table_summary(path) for path in CQS_REQUIRED + PROVENANCE_REQUIRED])
    summaries.to_csv(ROOT / "results" / "release_table_summary.csv", index=False)
    short_tables = summaries[summaries["row_check"].eq("too_few_rows")]

    if not missing.empty:
        raise SystemExit("Missing required files:\n" + "\n".join(missing["path"].tolist()))
    if not short_tables.empty:
        lines = [f"{r.path}: rows={r.rows}, min_rows={r.min_rows}" for r in short_tables.itertuples()]
        raise SystemExit("Tables below minimum row count:\n" + "\n".join(lines))

    print("Release package audit passed.")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
