from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    "README.md",
    "requirements.txt",
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
    "docs/data_sources.md",
    "docs/reproduction.md",
    "src/analysis/dcsi_hmm_state_reference.py",
    "src/analysis/dcsi_online_lead_validation.py",
    "src/analysis/dcsi_two_sided_triage.py",
    "src/analysis/build_knowledge_reasoning_feedback.py",
    "results/benchmark/benchmark_field_dictionary.csv",
    "results/benchmark/benchmark_task_definitions.csv",
    "results/benchmark/benchmark_split_definitions.csv",
    "results/benchmark/benchmark_baseline_scores.csv",
    "results/scorecards/paper_scorecard.csv",
    "results/scorecards/knowledge_reasoning_feedback_audit.csv",
    "results/scorecards/knowledge_feedback_actions.csv",
    "results/scorecards/reasoning_audit.csv",
    "results/scorecards/feedback_actions.csv",
    "results/scorecards/query_priority_updates.csv",
    "results/scorecards/rule_gate_decisions.csv",
    "results/scorecards/dcsi_cv_gains.csv",
    "results/scorecards/dcsi_cv_metrics.csv",
    "results/scorecards/kbs_baseline_gains.csv",
    "results/scorecards/kbs_pair_bootstrap_auc_ci.csv",
    "results/scorecards/online_lead_gains.csv",
    "results/scorecards/online_lead_metrics.csv",
    "results/scorecards/online_lead_rho_selection.csv",
    "results/scorecards/online_lead_state_diffs.csv",
    "results/scorecards/online_lead_assessment.md",
    "results/scorecards/two_sided_triage_summary.csv",
    "results/scorecards/two_sided_triage_comparison.csv",
    "results/scorecards/two_sided_triage_assessment.md",
    "results/scorecards/dcsi_rho_selection.csv",
    "results/scorecards/dcsi_deciles.csv",
    "results/scorecards/dcsi_event_peak_curve.csv",
    "results/scorecards/dcsi_event_peak_summary.csv",
    "results/scorecards/dcsi_reason_memory.csv",
    "results/scorecards/dcsi_placebo_shift.csv",
    "results/scorecards/dcsi_negative_control_metrics.csv",
    "results/scorecards/dcsi_negative_control_assessment.md",
    "results/scorecards/dcsi_transfer_gains.csv",
    "results/scorecards/dcsi_transfer_metrics.csv",
    "results/scorecards/dcsi_transfer_rho_selection.csv",
]


def row(path: str) -> dict[str, object]:
    full = ROOT / path
    return {
        "path": path,
        "exists": full.exists(),
        "bytes": full.stat().st_size if full.exists() else 0,
    }


def table_summary(path: str) -> dict[str, object]:
    full = ROOT / path
    if not full.exists() or full.suffix.lower() != ".csv":
        return {"path": path, "rows": "", "columns": ""}
    data = pd.read_csv(full)
    return {"path": path, "rows": len(data), "columns": len(data.columns)}


def main() -> None:
    audit = pd.DataFrame([row(path) for path in REQUIRED])
    missing = audit[~audit["exists"]]
    out = ROOT / "results" / "release_package_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)

    summaries = pd.DataFrame([table_summary(path) for path in REQUIRED])
    summaries.to_csv(ROOT / "results" / "release_table_summary.csv", index=False)

    if not missing.empty:
        raise SystemExit("Missing required files:\n" + "\n".join(missing["path"].tolist()))
    print("Release package audit passed.")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
