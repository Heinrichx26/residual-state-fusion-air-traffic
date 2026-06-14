# CQS-Rank Airport Disruption Monitoring

This repository contains the release package for:

**Calibrated Queue-Set Evidential Graph Ranking for Airport Disruption Monitoring from Public Operational Records**

The package supports the manuscript's public-record airport-hour monitoring pipeline. It includes source-data acquisition notes, field schemas, analysis scripts, archived result tables, figure builders, smoke-test commands, and release checks for Calibrated Queue-Set Evidential Graph Ranking (CQS-Rank).

CQS-Rank separates two audited objects from the same calibrated score:

- scheduled-arrival event probability for Brier score, expected calibration error, and probability audit;
- expected event mass, computed as scheduled arrivals multiplied by calibrated probability, for fixed-capacity airport-hour review.

Graph-Temporal Adaptive Fuzzy Residual Evidence (GT-AFRE) supplies the calibrated probability carrier. Reliability-Aware Action-Evidential Graph Ranking (RAEG-Rank) supplies the reliability-aware probability comparator and mechanism audit. CQS-Rank supplies the calibrated fixed-capacity queue-set decision layer.

## Data sources

The study uses public records from official providers:

- Bureau of Transportation Statistics (BTS) Airline On-Time Performance records.
- Iowa Environmental Mesonet (IEM) Automated Surface Observing Systems (ASOS) reports.
- Federal Aviation Administration (FAA) Air Traffic Control System Command Center (ATCSCC) advisory archive.
- National Centers for Environmental Information (NCEI) Storm Events records for supplementary weather-context checks.

Large raw source files are not redistributed. The `manifests/` directory records source-file manifests, and the scripts in `src/data/` document the acquisition workflow. A 30-airport OurAirports metadata subset is included to reproduce static airport-neighborhood graph checks.

## Repository layout

- `data_schema/`: airport-hour schemas, source-field definitions, split identifiers, evidence-role definitions, and rule traces retained from the public-record monitoring artifact.
- `data/raw/ourairports/airports.csv`: 30-airport public metadata subset used by static airport-neighborhood graph checks.
- `docs/`: data-source notes, reproduction commands, and the CQS-Rank release manifest.
- `src/data/`: public source acquisition and parsing scripts.
- `src/analysis/`: CQS-Rank, RAEG-Rank, GT-AFRE, comparator, calibration, robust-certificate, and release-verification scripts.
- `src/plotting/`: figure builders that read archived result tables.
- `src/figures/`: graphical abstract generation script.
- `results/cqs_rank/`: archived CQS-Rank tables used by the manuscript.
- `results/benchmark/` and `results/scorecards/`: earlier public-record monitoring benchmark tables retained for provenance and compatibility.
- `manifests/`: source-data download manifests.

## Quick verification

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the release audit:

```bash
python src/analysis/verify_release_package.py
```

The verifier checks the CQS-Rank manuscript-facing files, archived result tables, CQS analysis scripts, earlier monitoring-artifact files, and table dimensions. It writes:

- `results/release_package_audit.csv`
- `results/release_table_summary.csv`

## Smoke path

Run the CQS smoke path before any full reconstruction:

```bash
python src/analysis/asoc_cqs_rank_smoke.py
python src/analysis/asoc_cqs_carrier_audit_tests.py
python src/analysis/asoc_cqs_decision_audit_tests.py
python src/analysis/asoc_cqs_robust_certificate_tests.py
```

The smoke run uses a small airport-month panel and writes outputs under `results/experiments/applied_soft_computing_smoke/` when reconstructed source panels are available. The unit checks exercise the carrier audit, exchanged-record audit, and calibration-robust certificate logic without requiring a full-year run.

## Full analysis path

After reconstructing public source panels from the manifests, run the main CQS chain:

```bash
python src/analysis/asoc_raeg_rank_full.py
python src/analysis/asoc_cqs_rank_full.py
python src/analysis/asoc_cqs_rank_bootstrap.py
python src/analysis/asoc_cqs_carrier_audit.py
python src/analysis/asoc_cqs_decision_audit.py
python src/analysis/asoc_cqs_robust_certificate.py
python src/analysis/asoc_cqs_decision_strengthening.py
```

The plotting scripts read archived tables and do not require rerunning the full experiments:

```bash
python src/plotting/build_gt_afre_article_figures.py
python src/figures/make_cqs_graphical_abstract.py
```

## Archived manuscript tables

The current manuscript is supported by:

- `results/cqs_rank/primary_10airport_rolling/`: 2025 10-airport rolling-quarter CQS metrics, gains, bootstrap intervals, carrier audit, exchanged-record audit, traffic-stratum audit, and calibration-robust certificate.
- `results/cqs_rank/decision_strengthening/`: primary, 30-airport, and 2024-to-2025 cross-year CQS stress audits, calibration-radius stress checks, falsification checks, and comparator-protocol audit.
- `results/cqs_rank/smoke/`: small-panel CQS smoke outputs.

Full reconstructed source panels and raw public records are excluded from this repository because they are large and remain available from the public providers.

## Citation

Please cite the associated article after publication if you use this code or the archived result tables.
