# CQS-Rank assessment

Scope: months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; airports=['ATL', 'CLT', 'DEN', 'DFW', 'EWR', 'JFK', 'LAX', 'LGA', 'ORD', 'SFO']; validation=rolling_quarter.

Gate: usable if CQS-Rank improves top-10 capture over GT-AFRE/RAEG-Rank for a delay target and cancellation while keeping Brier and ECE no worse than the calibrated probability anchor.

## cancellation
- CQS-Rank: AUC 0.8658; PR-AUC 0.2195; Brier 0.01383; ECE 0.00123; top-10 0.7149.
- vs Graph-temporal evidence: usable signal; PR-AUC -0.0370; top-10 +0.0397; Brier -0.00005; ECE -0.00051.
- vs RAEG-Rank: usable signal; PR-AUC -0.0361; top-10 +0.0406; Brier -0.00005; ECE +0.00127.

## long_arrival_delay
- CQS-Rank: AUC 0.7219; PR-AUC 0.2814; Brier 0.07229; ECE 0.00221; top-10 0.4613.
- vs Graph-temporal evidence: usable signal; PR-AUC -0.0590; top-10 +0.0612; Brier +0.00000; ECE +0.00003.
- vs RAEG-Rank: usable signal; PR-AUC -0.0578; top-10 +0.0603; Brier -0.00003; ECE +0.00102.

## severe_arrival_delay
- CQS-Rank: AUC 0.7187; PR-AUC 0.1565; Brier 0.03655; ECE 0.00280; top-10 0.4901.
- vs Graph-temporal evidence: usable signal; PR-AUC -0.0324; top-10 +0.0486; Brier -0.00000; ECE -0.00009.
- vs RAEG-Rank: usable signal; PR-AUC -0.0321; top-10 +0.0474; Brier -0.00000; ECE +0.00161.

Overall smoke gate: pass.