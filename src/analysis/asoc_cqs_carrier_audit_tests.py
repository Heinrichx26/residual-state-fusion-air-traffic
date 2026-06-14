from __future__ import annotations

import pandas as pd

from asoc_cqs_carrier_audit import build_carrier_audit


def test_build_carrier_audit_reports_gain_against_both_probability_carriers() -> None:
    metrics = pd.DataFrame(
        [
            {"target": "delay", "model": "CQS-Rank", "top10_capture": 0.50},
            {"target": "delay", "model": "Graph-temporal evidence", "top10_capture": 0.42},
            {"target": "delay", "model": "RAEG-Rank", "top10_capture": 0.43},
        ]
    )
    gains = pd.DataFrame(
        [
            {
                "target": "delay",
                "reference": "Graph-temporal evidence",
                "top10_capture_gain": 0.08,
                "brier_gain": 0.0001,
                "ece_10_gain": 0.0002,
            },
            {
                "target": "delay",
                "reference": "RAEG-Rank",
                "top10_capture_gain": 0.07,
                "brier_gain": -0.0001,
                "ece_10_gain": 0.0003,
            },
        ]
    )

    audit = build_carrier_audit(metrics, gains)

    assert len(audit) == 1
    row = audit.iloc[0]
    assert row["target"] == "delay"
    assert row["cqs_top10_capture"] == 0.50
    assert row["gain_vs_gt_probability"] == 0.08
    assert row["gain_vs_raeg_probability"] == 0.07
    assert bool(row["positive_against_both"]) is True


if __name__ == "__main__":
    test_build_carrier_audit_reports_gain_against_both_probability_carriers()
