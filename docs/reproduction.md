# Reproduction

Use the smoke path first to verify source parsing and the DCSI state update on a small subset. Then run the full table path after the raw public data have been reconstructed from the manifests.

Figures are generated from archived result tables. Figure rendering is independent of the full experimental runs.

When reconstructed panels are stored outside this release package, set `DCSI_PROJECT_ROOT` to the reconstructed project root before running analysis commands.

The full DCSI run is reproduced with:

```bash
python src/analysis/dynamic_constraint_state_inversion.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dynamic_constraint_state_inversion_full_2025 --rho-grid 0,0.25,0.50,0.70,0.85,0.90,0.93,0.95,0.97,0.985
python src/analysis/dynamic_constraint_state_negative_controls.py --source-output dynamic_constraint_state_inversion_full_2025 --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dynamic_constraint_state_negative_controls
```

The knowledge-guided strengthening table is reproduced with:

```bash
python src/analysis/dcsi_kbs_strengthening.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --output-name dcsi_kbs_strengthening_full --bootstrap 300
```

The online lead-time check is reproduced with:

```bash
python src/analysis/dcsi_online_lead_validation.py --months 1-12 --airports ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO --horizons 1,3,6 --rho-grid 0.90,0.95,0.97 --output-name online_lead_full_2025
```

The archived scorecards in `results/scorecards/` report held-out AUC gains, online lead-time gains, strong advisory-feature references, paired bootstrap intervals, memory selection, decile closure, event timing, semantic constraint memory, shifted timing diagnostics, negative controls, and transfer checks.
