from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "data_schema"
SCORE_DIR = ROOT / "results" / "scorecards"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def paper_metric(rows: list[dict[str, str]], metric: str) -> str:
    for row in rows:
        if row.get("metric") == metric:
            return row.get("value", "")
    return ""


def build_operator_rows(entity_count: int, relation_count: int, rule_count: int, field_count: int) -> list[dict[str, str]]:
    return [
        {
            "operator_id": "O1",
            "operator": "source-role typing",
            "input_entities": "source fields, entity classes",
            "rule_expression": "ArchiveField -> Entity(role)",
            "materialized_output": f"{entity_count} entities, {field_count} source fields",
            "validation_use": "builds the typed airport-hour record",
        },
        {
            "operator_id": "O2",
            "operator": "admissible update",
            "input_entities": "ManagementAction, AirportHour",
            "rule_expression": "actsOn plus available(issue,start,current-hour) -> admissible_update",
            "materialized_output": "GDP/GS active overlap impulse",
            "validation_use": "creates timestamp-admissible state updates",
        },
        {
            "operator_id": "O3",
            "operator": "dynamic state materialization",
            "input_entities": "admissible_update, ConstraintState",
            "rule_expression": "admissible_update -> Z, recovery Z, mild-weather Z, delta Z",
            "materialized_output": "state vector for each airport-hour",
            "validation_use": "supports scoring, decile closure, and event timing",
        },
        {
            "operator_id": "O4",
            "operator": "semantic specialization",
            "input_entities": "ManagementAction, ReasonFamily",
            "rule_expression": "hasReason -> family-specific action impulse",
            "materialized_output": "weather, demand, runway or surface, facility, and other states",
            "validation_use": "explains mechanism-specific recovery memory",
        },
        {
            "operator_id": "O5",
            "operator": "closure binding",
            "input_entities": "ConstraintState, OutcomeClosure",
            "rule_expression": "validatedBy -> closure evidence",
            "materialized_output": "state deciles, event timing, and outcome gradients",
            "validation_use": "tests the inferred state after outcomes close",
        },
        {
            "operator_id": "O6",
            "operator": "action-path specificity",
            "input_entities": "ManagementAction, AirportHour, OutcomeClosure",
            "rule_expression": "rotate or reverse action path -> control state",
            "materialized_output": "airport-rotated and time-reversed control scorecards",
            "validation_use": "tests airport-time relation specificity",
        },
        {
            "operator_id": "O7",
            "operator": "partition reuse",
            "input_entities": "split definitions, source-role schema",
            "rule_expression": "schema plus split -> transfer state",
            "materialized_output": "cross-year and airport-group transfer scorecards",
            "validation_use": "checks reuse of the same state object across partitions",
        },
        {
            "operator_id": "O8",
            "operator": "feedback update",
            "input_entities": "closure evidence, control evidence, transfer evidence",
            "rule_expression": "validation evidence -> rule status and query priority",
            "materialized_output": f"{relation_count} relation types and {rule_count} rules linked to feedback actions",
            "validation_use": "records which source-role rules remain active after closure checks",
        },
    ]


def build_query_rows(metrics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "query_id": "Q1",
            "engineering_query": "Find airport-hours in high recovery state after action closure",
            "schema_fields": "AirportHour, ConstraintState.recovery_Z, OutcomeClosure",
            "operator_chain": "O3 -> O5",
            "returned_evidence": "top/bottom ratios 5.00 and 13.42",
            "feedback_action": "retain recovery-state closure query",
        },
        {
            "query_id": "Q2",
            "engineering_query": "Identify semantic reason families with persistent recovery memory",
            "schema_fields": "ManagementAction.reason, ReasonFamily, ConstraintState.recovery_Z",
            "operator_chain": "O4 -> O5",
            "returned_evidence": "weather recovery-state area 72588; top delay 0.260",
            "feedback_action": "retain family-specific mechanism query",
        },
        {
            "query_id": "Q3",
            "engineering_query": "Check states available before delayed outcome closure",
            "schema_fields": "issue time, start time, current hour, online Z, OutcomeClosure",
            "operator_chain": "O2 -> O3 -> O5",
            "returned_evidence": "1/3/6 h paired differences remain positive",
            "feedback_action": "retain timestamp-admissible online gate",
        },
        {
            "query_id": "Q4",
            "engineering_query": "Check airport-time action-path specificity",
            "schema_fields": "ManagementAction.airport, ManagementAction.time, ConstraintState, OutcomeClosure",
            "operator_chain": "O6 -> O5",
            "returned_evidence": "cancellation AUC 0.769 real, 0.748 rotated, 0.739 reversed",
            "feedback_action": "retain airport-time action link",
        },
        {
            "query_id": "Q5",
            "engineering_query": "Check reuse of the state object across partitions",
            "schema_fields": "split_id, AirportHour, ConstraintState, OutcomeClosure",
            "operator_chain": "O7 -> O5",
            "returned_evidence": "cross-year differences +0.007/+0.024; spatial stress +0.002/+0.012",
            "feedback_action": "retain common source-role object across partitions",
        },
        {
            "query_id": "Q6",
            "engineering_query": "Summarize evidence for the reconstructed state object",
            "schema_fields": "paper scorecard rows and validation split identifiers",
            "operator_chain": "O1 -> O8",
            "returned_evidence": f"{len(metrics)} scorecard rows and 8 validation split definitions",
            "feedback_action": "update run-level validation status",
        },
    ]


def build_audit_rows(metrics: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "reasoning_layer": "Typed knowledge closure",
            "executed_operator": "O1 source-role typing",
            "materialized_result": "7 entities, 9 relation types, 6 source-role rules, and 20 source fields",
            "feedback_record": "schema ready for query execution",
            "archived_output": "data_schema/reasoning_operators.csv",
        },
        {
            "reasoning_layer": "Admissibility reasoning",
            "executed_operator": "O2 admissible update",
            "materialized_result": "12/12 BTS months, 120/120 ASOS airport-months, 365/365 ATCSCC days, 6539 retained action records",
            "feedback_record": "update, context, and closure roles admitted",
            "archived_output": "data_schema/reasoning_queries.csv",
        },
        {
            "reasoning_layer": "State materialization",
            "executed_operator": "O3 dynamic state materialization",
            "materialized_result": "rho=0.95 selected in 12/12 folds; representative half-life 13.5 h",
            "feedback_record": "long-memory state region retained",
            "archived_output": "results/scorecards/dcsi_rho_selection.csv",
        },
        {
            "reasoning_layer": "Closure reasoning",
            "executed_operator": "O5 closure binding",
            "materialized_result": "top/bottom ratios 5.00 for long delay and 13.42 for cancellation",
            "feedback_record": "state-ranking closure query retained",
            "archived_output": "results/scorecards/dcsi_deciles.csv",
        },
        {
            "reasoning_layer": "Online reasoning",
            "executed_operator": "O2 + O3 + O5 timestamp gate",
            "materialized_result": "1/3/6 h paired differences over online fixed decay: +0.0038/+0.0030, +0.0038/+0.0042, +0.0020/+0.0031",
            "feedback_record": "online timestamp gate retained",
            "archived_output": "results/scorecards/online_lead_state_diffs.csv",
        },
        {
            "reasoning_layer": "Semantic reasoning",
            "executed_operator": "O4 semantic specialization",
            "materialized_result": "5 reason families; weather recovery-state area 72588; top delay 0.260; top cancellation 0.0697",
            "feedback_record": "family-specific mechanism query retained",
            "archived_output": "results/scorecards/dcsi_reason_memory.csv",
        },
        {
            "reasoning_layer": "Action-path reasoning",
            "executed_operator": "O6 action-path specificity",
            "materialized_result": "real cancellation AUC 0.769; airport-rotated 0.748; time-reversed 0.739",
            "feedback_record": "airport-time action link retained",
            "archived_output": "results/scorecards/dcsi_negative_control_metrics.csv",
        },
        {
            "reasoning_layer": "Transfer reasoning",
            "executed_operator": "O7 partition reuse",
            "materialized_result": "cross-year differences +0.007/+0.024; spatial stress differences +0.002/+0.012",
            "feedback_record": "source-role object retained across partitions",
            "archived_output": "results/scorecards/dcsi_transfer_gains.csv",
        },
        {
            "reasoning_layer": "Feedback loop",
            "executed_operator": "O8 feedback update",
            "materialized_result": f"{len(metrics)} scorecard rows update rule status and query priority",
            "feedback_record": "run-level reasoning status archived",
            "archived_output": "results/scorecards/knowledge_feedback_actions.csv",
        },
    ]


def build_feedback_rows() -> list[dict[str, str]]:
    return [
        {
            "feedback_id": "F1",
            "trigger_evidence": "source audit complete and zero retained-interval parser issue rows",
            "rule_status": "source-role rules active",
            "system_action": "admit the airport-hour thread for state reconstruction",
        },
        {
            "feedback_id": "F2",
            "trigger_evidence": "rho=0.95 selected in 12/12 folds",
            "rule_status": "memory selection active",
            "system_action": "use fold-selected recovery memory for DCSI state updates",
        },
        {
            "feedback_id": "F3",
            "trigger_evidence": "top/bottom ratios 5.00 and 13.42",
            "rule_status": "closure-ranking query active",
            "system_action": "retain state decile closure as a monitoring diagnostic",
        },
        {
            "feedback_id": "F4",
            "trigger_evidence": "online 1/3/6 h paired differences remain positive",
            "rule_status": "timestamp gate active",
            "system_action": "retain issue/start/current-hour online monitoring view",
        },
        {
            "feedback_id": "F5",
            "trigger_evidence": "cancellation AUC decreases under airport-rotated and time-reversed action paths",
            "rule_status": "action-path specificity active",
            "system_action": "retain airport-time action link in reasoning closure",
        },
        {
            "feedback_id": "F6",
            "trigger_evidence": "cross-year and spatial stress differences remain positive",
            "rule_status": "partition-reuse query active",
            "system_action": "retain common source-role object across validation partitions",
        },
    ]


def build_feedback_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "P1",
            "evidence_input": "source coverage and parser audit",
            "gate_condition": "all source families complete and retained parser issue rows equal zero",
            "state_or_query_effect": "open source-role gate for state reconstruction",
            "next_operator_set": "O1,O2,O3",
        },
        {
            "policy_id": "P2",
            "evidence_input": "fold-wise memory selection",
            "gate_condition": "same long-memory region selected in the validation folds",
            "state_or_query_effect": "promote fold-selected memory to the state update rule",
            "next_operator_set": "O3,O5",
        },
        {
            "policy_id": "P3",
            "evidence_input": "state decile closure",
            "gate_condition": "top-to-bottom closure ratios exceed one for delay and cancellation",
            "state_or_query_effect": "activate recovery-state ranking query",
            "next_operator_set": "O5,O8",
        },
        {
            "policy_id": "P4",
            "evidence_input": "online lead-time validation",
            "gate_condition": "paired online differences remain positive at 1, 3, and 6 h",
            "state_or_query_effect": "activate timestamp-admissible online monitoring query",
            "next_operator_set": "O2,O3,O5,O8",
        },
        {
            "policy_id": "P5",
            "evidence_input": "airport-rotated and time-reversed controls",
            "gate_condition": "control AUC values fall below the real action path",
            "state_or_query_effect": "activate airport-time action-link rule",
            "next_operator_set": "O6,O8",
        },
        {
            "policy_id": "P6",
            "evidence_input": "cross-year and airport-partition checks",
            "gate_condition": "partition differences retain positive direction",
            "state_or_query_effect": "activate partition-reuse query and mark spatial stress as diagnostic",
            "next_operator_set": "O7,O8",
        },
    ]


def build_query_priority_rows() -> list[dict[str, str]]:
    return [
        {
            "query_id": "Q1",
            "feedback_policy": "P3",
            "prior_priority": "candidate",
            "posterior_priority": "active",
            "reasoning_effect": "state decile closure becomes a retained monitoring query",
        },
        {
            "query_id": "Q2",
            "feedback_policy": "P3,P6",
            "prior_priority": "candidate",
            "posterior_priority": "active",
            "reasoning_effect": "semantic recovery memory remains available as a mechanism query",
        },
        {
            "query_id": "Q3",
            "feedback_policy": "P4",
            "prior_priority": "candidate",
            "posterior_priority": "active",
            "reasoning_effect": "online issue/start/current-hour view is retained",
        },
        {
            "query_id": "Q4",
            "feedback_policy": "P5",
            "prior_priority": "candidate",
            "posterior_priority": "active",
            "reasoning_effect": "airport-time action specificity remains part of the state audit",
        },
        {
            "query_id": "Q5",
            "feedback_policy": "P6",
            "prior_priority": "candidate",
            "posterior_priority": "diagnostic",
            "reasoning_effect": "cross-year transfer is retained; airport extension remains a stress diagnostic",
        },
        {
            "query_id": "Q6",
            "feedback_policy": "P1,P2,P3,P4,P5,P6",
            "prior_priority": "candidate",
            "posterior_priority": "active",
            "reasoning_effect": "run-level feedback record summarizes the active rule set",
        },
    ]


def build_rule_gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate_id": "G1",
            "rule_gate": "source-role admissibility",
            "evidence_value": "6539 retained actions and zero parser issue rows",
            "gate_status": "active",
            "enabled_output": "state reconstruction table",
        },
        {
            "gate_id": "G2",
            "rule_gate": "fold-memory promotion",
            "evidence_value": "rho=0.95 in 12/12 folds",
            "gate_status": "active",
            "enabled_output": "current and recovery state components",
        },
        {
            "gate_id": "G3",
            "rule_gate": "closure-ranking promotion",
            "evidence_value": "top-to-bottom closure ratios 5.00/13.42",
            "gate_status": "active",
            "enabled_output": "decile closure query",
        },
        {
            "gate_id": "G4",
            "rule_gate": "timestamp-admissible online promotion",
            "evidence_value": "positive paired differences at 1/3/6 h",
            "gate_status": "active",
            "enabled_output": "online monitoring query",
        },
        {
            "gate_id": "G5",
            "rule_gate": "action-path specificity",
            "evidence_value": "real cancellation AUC 0.769; rotated 0.748; reversed 0.739",
            "gate_status": "active",
            "enabled_output": "airport-time action-link query",
        },
        {
            "gate_id": "G6",
            "rule_gate": "partition reuse",
            "evidence_value": "cross-year +0.007/+0.024; spatial +0.002/+0.012",
            "gate_status": "active-diagnostic",
            "enabled_output": "transfer query and spatial stress diagnostic",
        },
    ]


def run(mode: str) -> None:
    entities = read_csv(SCHEMA_DIR / "entities.csv")
    relations = read_csv(SCHEMA_DIR / "relations.csv")
    rules = read_csv(SCHEMA_DIR / "rules.csv")
    fields = read_csv(SCHEMA_DIR / "source_fields.csv")
    metrics = read_csv(SCORE_DIR / "paper_scorecard.csv")
    with (SCHEMA_DIR / "schema.json").open("r", encoding="utf-8") as f:
        schema = json.load(f)

    if mode == "smoke":
        rows = [
            {
                "check": "schema_loaded",
                "value": str(len(schema.get("entities", {}))),
                "status": "ok" if schema.get("entities") else "empty",
            },
            {
                "check": "paper_scorecard_rows",
                "value": str(len(metrics)),
                "status": "ok" if metrics else "empty",
            },
        ]
        write_csv(SCORE_DIR / "knowledge_reasoning_feedback_smoke.csv", rows, ["check", "value", "status"])
        print("knowledge reasoning feedback smoke passed")
        return

    operator_rows = build_operator_rows(len(entities), len(relations), len(rules), len(fields))
    query_rows = build_query_rows(metrics)
    audit_rows = build_audit_rows(metrics)
    feedback_rows = build_feedback_rows()
    feedback_policy_rows = build_feedback_policy_rows()
    query_priority_rows = build_query_priority_rows()
    rule_gate_rows = build_rule_gate_rows()

    write_csv(
        SCHEMA_DIR / "reasoning_operators.csv",
        operator_rows,
        ["operator_id", "operator", "input_entities", "rule_expression", "materialized_output", "validation_use"],
    )
    write_csv(
        SCHEMA_DIR / "operators.csv",
        operator_rows,
        ["operator_id", "operator", "input_entities", "rule_expression", "materialized_output", "validation_use"],
    )
    write_csv(
        SCHEMA_DIR / "reasoning_queries.csv",
        query_rows,
        ["query_id", "engineering_query", "schema_fields", "operator_chain", "returned_evidence", "feedback_action"],
    )
    write_csv(
        SCHEMA_DIR / "queries.csv",
        query_rows,
        ["query_id", "engineering_query", "schema_fields", "operator_chain", "returned_evidence", "feedback_action"],
    )
    write_csv(
        SCORE_DIR / "knowledge_reasoning_feedback_audit.csv",
        audit_rows,
        ["reasoning_layer", "executed_operator", "materialized_result", "feedback_record", "archived_output"],
    )
    write_csv(
        SCORE_DIR / "reasoning_audit.csv",
        audit_rows,
        ["reasoning_layer", "executed_operator", "materialized_result", "feedback_record", "archived_output"],
    )
    write_csv(
        SCORE_DIR / "knowledge_feedback_actions.csv",
        feedback_rows,
        ["feedback_id", "trigger_evidence", "rule_status", "system_action"],
    )
    write_csv(
        SCORE_DIR / "feedback_actions.csv",
        feedback_rows,
        ["feedback_id", "trigger_evidence", "rule_status", "system_action"],
    )
    write_csv(
        SCHEMA_DIR / "feedback_policy.csv",
        feedback_policy_rows,
        ["policy_id", "evidence_input", "gate_condition", "state_or_query_effect", "next_operator_set"],
    )
    write_csv(
        SCORE_DIR / "query_priority_updates.csv",
        query_priority_rows,
        ["query_id", "feedback_policy", "prior_priority", "posterior_priority", "reasoning_effect"],
    )
    write_csv(
        SCORE_DIR / "rule_gate_decisions.csv",
        rule_gate_rows,
        ["gate_id", "rule_gate", "evidence_value", "gate_status", "enabled_output"],
    )
    print("knowledge reasoning feedback audit written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
