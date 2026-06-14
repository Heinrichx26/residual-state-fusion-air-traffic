# CQS-Rank decision audit

The audit compares the CQS-Rank fixed-budget queue with the RAEG-Rank probability queue on identical airport-hours.
CQS-only rows enter the CQS queue and do not enter the RAEG queue; reference-only rows have the opposite status.

## long_arrival_delay
- Top-10 queue overlap 0.657; churn 0.343; event gain +9115; capture gain +0.0602.
- CQS-only hours have mean arrivals 67.87, mean calibrated risk 0.1274, and mean event mass 8.393; reference-only hours have mean arrivals 14.27, mean calibrated risk 0.2727, and mean event mass 3.706.
- Fold-level event-gain win rate: 1.000 over 3 rolling-quarter folds.

## severe_arrival_delay
- Top-10 queue overlap 0.672; churn 0.328; event gain +3217; capture gain +0.0475.
- CQS-only hours have mean arrivals 70.37, mean calibrated risk 0.0475, and mean event mass 3.262; reference-only hours have mean arrivals 15.56, mean calibrated risk 0.1014, and mean event mass 1.458.
- Fold-level event-gain win rate: 1.000 over 3 rolling-quarter folds.

## cancellation
- Top-10 queue overlap 0.739; churn 0.261; event gain +1086; capture gain +0.0406.
- CQS-only hours have mean arrivals 64.24, mean calibrated risk 0.0218, and mean event mass 1.339; reference-only hours have mean arrivals 12.21, mean calibrated risk 0.0516, and mean event mass 0.538.
- Fold-level event-gain win rate: 1.000 over 3 rolling-quarter folds.

## Traffic-stratum evidence
- long_arrival_delay: high-traffic stratum event gain +15882 with CQS-selected 3381 and reference-selected 1566 airport-hours.
- severe_arrival_delay: high-traffic stratum event gain +5822 with CQS-selected 3468 and reference-selected 1735 airport-hours.
- cancellation: high-traffic stratum event gain +1838 with CQS-selected 2798 and reference-selected 1460 airport-hours.