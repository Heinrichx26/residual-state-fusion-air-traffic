# CQS-Rank Release Manifest

This manifest maps the manuscript-facing claim to repository files.

## Manuscript identity

Title: Calibrated Queue-Set Evidential Graph Ranking for Airport Disruption Monitoring from Public Operational Records

Primary method roles:

- GT-AFRE: calibrated graph-temporal fuzzy residual evidence probability carrier.
- RAEG-Rank: reliability-aware probability comparator and mechanism audit.
- CQS-Rank: calibrated fixed-capacity queue-set selector using expected event mass.

Static metadata:

- `data/raw/ourairports/airports.csv`: 30-airport latitude, longitude, and elevation subset for airport-metadata graph checks.

## Main result files

Primary 2025 10-airport rolling-quarter CQS results:

- `results/cqs_rank/primary_10airport_rolling/cqs_rank_metrics.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_rank_gains.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_budget_gain_ci_maintext.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_rank_bootstrap_summary.csv`

Carrier and exchanged-record audit:

- `results/cqs_rank/primary_10airport_rolling/cqs_queue_carrier_audit.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_queue_carrier_audit.md`
- `results/cqs_rank/primary_10airport_rolling/cqs_decision_pair_audit.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_decision_fold_audit.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_decision_traffic_strata.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_decision_audit_assessment.md`

Calibration-robust lower event-mass certificate:

- `results/cqs_rank/primary_10airport_rolling/cqs_calibration_robust_certificate.csv`
- `results/cqs_rank/primary_10airport_rolling/cqs_calibration_gap_bins.csv`

Cross-scope, external-year, calibration-stress, and falsification checks:

- `results/cqs_rank/decision_strengthening/cqs_decision_strengthening_assessment.md`
- `results/cqs_rank/decision_strengthening/cqs_strengthening_matrix.csv`
- `results/cqs_rank/decision_strengthening/all_scenario_cqs_gains.csv`
- `results/cqs_rank/decision_strengthening/all_scenario_cqs_metrics.csv`
- `results/cqs_rank/decision_strengthening/all_scenario_calibration_stress.csv`
- `results/cqs_rank/decision_strengthening/all_scenario_decision_value.csv`
- `results/cqs_rank/decision_strengthening/all_scenario_placebo_checks.csv`
- `results/cqs_rank/decision_strengthening/expanded_comparator_protocol.csv`

Smoke outputs:

- `results/cqs_rank/smoke/cqs_rank_assessment.md`
- `results/cqs_rank/smoke/cqs_rank_metrics.csv`
- `results/cqs_rank/smoke/cqs_rank_gains.csv`

## Script entry points

CQS-Rank scoring and summaries:

- `src/analysis/asoc_cqs_rank.py`
- `src/analysis/asoc_cqs_rank_smoke.py`
- `src/analysis/asoc_cqs_rank_full.py`
- `src/analysis/asoc_cqs_rank_bootstrap.py`

Decision-layer audits:

- `src/analysis/asoc_cqs_carrier_audit.py`
- `src/analysis/asoc_cqs_decision_audit.py`
- `src/analysis/asoc_cqs_robust_certificate.py`
- `src/analysis/asoc_cqs_decision_strengthening.py`

Probability comparator support:

- `src/analysis/asoc_raeg_rank.py`
- `src/analysis/asoc_raeg_rank_smoke.py`
- `src/analysis/asoc_raeg_rank_full.py`
- `src/analysis/asoc_raeg_rank_postprocess.py`

Figure builders:

- `src/plotting/build_gt_afre_article_figures.py`
- `src/figures/make_cqs_graphical_abstract.py`

## Release audit

Run:

```bash
python src/analysis/verify_release_package.py
```

The release audit checks that the files above are present and records row/column counts for CSV tables.
