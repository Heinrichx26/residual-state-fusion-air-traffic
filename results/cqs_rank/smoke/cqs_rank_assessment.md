# CQS-Rank assessment

Scope: months=[1, 2]; airports=['ATL', 'DFW', 'EWR', 'ORD']; validation=month.

Gate: usable if CQS-Rank improves top-10 capture over GT-AFRE/RAEG-Rank for a delay target and cancellation while keeping Brier and ECE no worse than the calibrated probability anchor.

## cancellation
- CQS-Rank: AUC 0.8702; PR-AUC 0.1897; Brier 0.03261; ECE 0.02295; top-10 0.6931.
- vs Graph-temporal evidence: usable signal; PR-AUC +0.0317; top-10 +0.1464; Brier +0.00082; ECE +0.00150.
- vs RAEG-Rank: usable signal; PR-AUC +0.0262; top-10 +0.1113; Brier +0.00074; ECE +0.00344.

## long_arrival_delay
- CQS-Rank: AUC 0.7050; PR-AUC 0.2279; Brier 0.06389; ECE 0.00548; top-10 0.4123.
- vs Graph-temporal evidence: usable signal; PR-AUC +0.0184; top-10 +0.0665; Brier +0.00112; ECE +0.00374.
- vs RAEG-Rank: usable signal; PR-AUC +0.0180; top-10 +0.0646; Brier +0.00117; ECE +0.00712.

Overall smoke gate: pass.