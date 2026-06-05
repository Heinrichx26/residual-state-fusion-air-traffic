# A Decision-Support Artifact for Air Traffic Disruption Monitoring

This repository contains source-data instructions, field schemas, split definitions, analysis code, archived result tables, and figure scripts for:

**A Decision-Support Artifact for Air Traffic Disruption Monitoring: Dynamic Constraint-State Inversion from Public Operational Records**

The package organizes public air traffic records into a timestamp-admissible decision-support artifact. The artifact links physical observations, traffic-management action records, planned workload, semantic advisory reasons, inferred dynamic constraint states, alert rankings, review-workload outputs, routine-monitoring triage, and flight-outcome closure for airport-hour and advisory-event monitoring.

## Data sources

The study uses public records from official providers:

- Bureau of Transportation Statistics (BTS) Airline On-Time Performance data.
- Iowa Environmental Mesonet (IEM) Automated Surface Observing Systems (ASOS) hourly weather reports.
- Federal Aviation Administration (FAA) Air Traffic Control System Command Center (ATCSCC) advisory archive.
- National Centers for Environmental Information (NCEI) Storm Events Database.

Large raw source files are not redistributed. The `manifests/` directory records the files and download status used in the study. The scripts in `src/data/` document the acquisition workflow.

## Repository layout

- `data_schema/`: monitoring-artifact definitions, source fields, source-role rules, diagnostic queries, rule-execution example, and validation split identifiers.
- `docs/`: data-source and reproduction notes.
- `src/data/`: source acquisition and parsing scripts.
- `src/analysis/`: airport-hour panel construction, dynamic constraint-state inversion (DCSI), online lead-time validation, two-sided triage, optional state-space diagnostics, negative controls, transfer checks, and package verification.
- `src/plotting/`: figure builders that read archived result tables.
- `results/benchmark/`: task definitions and baseline scores for the DCSI monitoring benchmark.
- `results/scorecards/`: archived DCSI result tables, advisory-feature references, review-cost outputs, two-sided triage outputs, paired bootstrap intervals, online lead-time outputs, and paper scorecards.
- `manifests/`: source-data manifests.

## Quick reproducibility path

Install the Python environment:

```bash
python -m pip install -r requirements.txt
```

Verify the package:

```bash
python src/analysis/verify_release_package.py
```

The verifier checks the README, environment file, source-role schemas, diagnostic queries, task definitions, split identifiers, DCSI scorecards, online lead-time tables, two-sided triage tables, negative-control tables, and transfer tables.

If the reconstructed source panels are stored outside this reproduction package, point the scripts at that project root before running analysis commands:

```powershell
$env:DCSI_PROJECT_ROOT="<path-to-reconstructed-project>"
```

For bash:

```bash
export DCSI_PROJECT_ROOT="/path/to/reconstructed/project"
```

Run the DCSI smoke path before full reconstruction:

```bash
python src/analysis/dynamic_constraint_state_inversion.py --months 1,7,12 --airports ATL,ORD --output-name dcsi_smoke_2025
python src/analysis/dynamic_constraint_state_negative_controls.py --source-output dcsi_smoke_2025 --months 1,7,12 --airports ATL,ORD --output-name dcsi_negative_controls_smoke
```

Rebuild the full DCSI result tables after reconstructing the public source panels:

```bash
python src/analysis/dynamic_constraint_state_inversion.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dynamic_constraint_state_inversion_full_2025 --rho-grid 0,0.25,0.50,0.70,0.85,0.90,0.93,0.95,0.97,0.985
python src/analysis/dynamic_constraint_state_negative_controls.py --source-output dynamic_constraint_state_inversion_full_2025 --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dynamic_constraint_state_negative_controls
```

Run the relation-augmented DCSI scorecard after the full DCSI table exists:

```bash
python src/analysis/dcsi_kbs_strengthening.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dcsi_kbs_strengthening_full --bootstrap 300
```

Run the online lead-time validation after the advisory issue-time table has been reconstructed:

```bash
python src/analysis/dcsi_online_lead_validation.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --horizons 1,3,6 --rho-grid 0.90,0.95,0.97 --output-name online_lead_full_2025
```

Build the two-sided review and routine-monitoring triage scorecards from archived online predictions:

```bash
python src/analysis/dcsi_two_sided_triage.py --mode smoke --months 1,7,12
python src/analysis/dcsi_two_sided_triage.py --mode full
```

Run the optional state-space diagnostic on a small panel when comparing DCSI with a hidden-state reference:

```bash
python src/analysis/dcsi_hmm_state_reference.py --months 1,7,12 --airports ATL,ORD --output-name dcsi_hmm_state_reference_smoke
```

Rebuild figures from archived result tables:

```bash
python src/plotting/build_dynamic_constraint_figures.py
```

## Evaluation tasks

The package defines seven evaluation tasks:

1. Dynamic constraint-state monitoring.
2. Long-delay risk scoring.
3. Cancellation risk scoring.
4. Action-trajectory negative controls.
5. Online lead-time monitoring.
6. Two-sided review and routine-monitoring triage.
7. Cross-year and airport-group transfer.

The package includes fixed split definitions for leave-one-month, 2024-to-2025, main-to-extension airport transfer, online lead-time horizons, shifted timing, airport-rotated actions, time-reversed actions, and event timing. Selected scorecards include DCSI model differences, online lead-time differences, advisory-feature references, review-cost sensitivity, two-sided triage, paired bootstrap intervals, fold metrics, memory-parameter selection, state decile closure, semantic constraint memory, shifted timing diagnostics, negative controls, transfer checks, and event peak timing.

## Reproducibility note

Article figures are generated from archived result tables independently of the full experimental runs. Full data reconstruction requires downloading source files listed in the manifests.

## Citation

Please cite the associated publication after publication if you use this code or the derived tables.
