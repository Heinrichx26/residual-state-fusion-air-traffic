# Air Traffic Knowledge-Guided Dynamic Constraint-State Digital Thread

This repository contains source-data instructions, field schemas, split definitions, analysis code, archived result tables, and figure scripts for:

**Knowledge-Guided Dynamic Constraint-State Inversion for Digital-Thread Monitoring of Air Traffic Disruptions under Traffic-Management Advisories**

The package organizes public air traffic records into an operational digital thread. The thread links physical observations, traffic-management action records, planned workload, semantic advisory reasons, inferred dynamic constraint states, and flight-outcome closure for airport-hour and advisory-event monitoring.

## Data sources

The study uses public records from official providers:

- Bureau of Transportation Statistics (BTS) Airline On-Time Performance data.
- Iowa Environmental Mesonet (IEM) Automated Surface Observing Systems (ASOS) hourly weather reports.
- Federal Aviation Administration (FAA) Air Traffic Control System Command Center (ATCSCC) advisory archive.
- National Centers for Environmental Information (NCEI) Storm Events Database.

Large raw source files are not redistributed. The `manifests/` directory records the files and download status used in the study. The scripts in `src/data/` document the acquisition workflow.

## Repository layout

- `data_schema/`: digital-thread entity definitions, source fields, knowledge relations, source-role rules, and validation split identifiers.
- `docs/`: data-source and reproduction notes.
- `src/data/`: source acquisition and parsing scripts.
- `src/analysis/`: airport-hour panel construction, dynamic constraint-state inversion (DCSI), knowledge-guided strengthening baselines, optional state-space diagnostics, negative controls, transfer checks, and package verification.
- `src/plotting/`: figure builders that read archived result tables.
- `results/benchmark/`: task definitions and baseline scores for the DCSI monitoring benchmark.
- `results/scorecards/`: archived DCSI result tables, strong advisory-feature references, paired bootstrap intervals, and paper scorecards.
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

The verifier checks the README, environment file, digital-thread schemas, task definitions, split identifiers, DCSI scorecards, negative-control tables, and transfer tables.

Run the DCSI smoke path before full reconstruction:

```bash
python src/analysis/dynamic_constraint_state_inversion.py --months 1,7,12 --airports ATL,ORD --output-name dcsi_smoke_2025
python src/analysis/dynamic_constraint_state_negative_controls.py --input-name dcsi_smoke_2025 --output-name dcsi_negative_controls_smoke
```

Rebuild the full DCSI result tables after reconstructing the public source panels:

```bash
python src/analysis/dynamic_constraint_state_inversion.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dynamic_constraint_state_inversion_full_2025 --rho-grid 0,0.25,0.50,0.70,0.85,0.90,0.93,0.95,0.97,0.985
python src/analysis/dynamic_constraint_state_negative_controls.py --input-name dynamic_constraint_state_inversion_full_2025 --output-name dynamic_constraint_state_negative_controls
```

Run the knowledge-guided strengthening table after the full DCSI table exists:

```bash
python src/analysis/dcsi_kbs_strengthening.py --input-name dynamic_constraint_state_inversion_full_2025 --output-name dcsi_kbs_strengthening_full --bootstrap 300
```

Run the optional state-space diagnostic on a small panel when comparing DCSI with a hidden-state reference:

```bash
python src/analysis/dcsi_hmm_state_reference.py --input-name dynamic_constraint_state_inversion_full_2025 --months 1,7,12 --airports ATL,ORD --output-name dcsi_hmm_state_reference_smoke
```

Rebuild figures from archived result tables:

```bash
python src/plotting/build_dynamic_constraint_figures.py
```

## Evaluation tasks

The package defines five evaluation tasks:

1. Dynamic constraint-state monitoring.
2. Long-delay risk scoring.
3. Cancellation risk scoring.
4. Action-trajectory negative controls.
5. Cross-year and airport-group transfer.

The package includes fixed split definitions for leave-one-month, 2024-to-2025, main-to-extension airport transfer, shifted timing, airport-rotated actions, time-reversed actions, and event timing. Selected scorecards include DCSI model gains, strong advisory-feature references, paired bootstrap intervals, fold metrics, memory-parameter selection, state decile closure, semantic constraint memory, shifted timing diagnostics, negative controls, transfer checks, and event peak timing.

## Reproducibility note

Article figures are generated from archived result tables independently of the full experimental runs. Full data reconstruction requires downloading source files listed in the manifests.

## Citation

Please cite the associated publication after publication if you use this code or the derived tables.
