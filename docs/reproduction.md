# Reproduction

This document gives the reproduction path for the CQS-Rank manuscript.

Use the smoke path first. Full reconstruction downloads public data from the original providers and can take much longer than the release-table checks.

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

The full RAEG-Rank residual path uses PyTorch and LightGBM. The archived table audit and CQS unit checks use standard tabular dependencies.

## 2. Verify the release package

```bash
python src/analysis/verify_release_package.py
```

Expected result:

```text
Release package audit passed.
```

The verifier checks manuscript-facing CQS-Rank files, CQS summary tables, decision stress tables, source schemas, earlier monitoring-artifact files, and key table dimensions. It writes `results/release_package_audit.csv` and `results/release_table_summary.csv`.

## 3. Run CQS smoke checks

```bash
python src/analysis/asoc_cqs_carrier_audit_tests.py
python src/analysis/asoc_cqs_decision_audit_tests.py
python src/analysis/asoc_cqs_robust_certificate_tests.py
```

These tests check the carrier audit, exchanged-record audit, traffic-stratum audit, and calibration-robust lower event-mass certificate. They do not require full raw public data reconstruction.

If reconstructed panels are available, run the small CQS smoke experiment:

```bash
python src/analysis/asoc_cqs_rank_smoke.py
```

The smoke experiment writes CQS smoke tables under `results/experiments/applied_soft_computing_smoke/`.

## 4. Reconstruct public source panels

Large raw source files are not redistributed. Use the manifests and acquisition scripts:

```bash
python src/data/download_initial_source_data.py
python src/data/download_supplemental_source_data.py
python src/data/reparse_cached_atcscc_2025.py
```

If reconstructed source panels are stored outside this repository, set the project root before running full experiments:

```powershell
$env:DCSI_PROJECT_ROOT="<path-to-reconstructed-project>"
```

For bash:

```bash
export DCSI_PROJECT_ROOT="/path/to/reconstructed-project"
```

The environment variable name is retained for compatibility with earlier public-record monitoring scripts.

## 5. Run the full CQS-Rank chain

Run the RAEG-Rank probability comparator, then apply CQS-Rank:

```bash
python src/analysis/asoc_raeg_rank_full.py
python src/analysis/asoc_cqs_rank_full.py
```

Run paired uncertainty and decision audits:

```bash
python src/analysis/asoc_cqs_rank_bootstrap.py
python src/analysis/asoc_cqs_carrier_audit.py
python src/analysis/asoc_cqs_decision_audit.py
python src/analysis/asoc_cqs_robust_certificate.py
```

Run wider-scope, external-year, calibration-stress, and falsification checks:

```bash
python src/analysis/asoc_cqs_decision_strengthening.py
```

The manuscript uses the archived outputs in:

- `results/cqs_rank/primary_10airport_rolling/`
- `results/cqs_rank/decision_strengthening/`
- `results/cqs_rank/smoke/`

## 6. Rebuild manuscript figures

Figures are generated from archived result tables and are independent of full experimental reruns:

```bash
python src/plotting/build_gt_afre_article_figures.py
python src/figures/make_cqs_graphical_abstract.py
```

## 7. Read the release manifest

The table-to-manuscript mapping is in:

```text
docs/cqs_rank_release_manifest.md
```
