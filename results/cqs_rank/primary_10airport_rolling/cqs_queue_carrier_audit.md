# CQS-Rank carrier audit

The audit compares the closed-form CQS queue against two probability queues on the same airport-hours.
Positive top-10 gains against both probability carriers indicate that the queue-set decision layer is not dependent on the RAEG residual layer.

- cancellation: CQS top-10 0.7149; gain vs GT-AFRE probability +0.0397; gain vs RAEG probability +0.0406.
- long_arrival_delay: CQS top-10 0.4613; gain vs GT-AFRE probability +0.0612; gain vs RAEG probability +0.0603.
- severe_arrival_delay: CQS top-10 0.4901; gain vs GT-AFRE probability +0.0486; gain vs RAEG probability +0.0474.