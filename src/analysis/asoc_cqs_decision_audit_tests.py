from __future__ import annotations

import pandas as pd

from asoc_cqs_decision_audit import audit_pair, traffic_stratum_audit


def toy_predictions() -> pd.DataFrame:
    rows = []
    for i, (arrivals, events, cqs_score, raeg_prob) in enumerate(
        [
            (100, 8, 0.95, 0.30),
            (80, 5, 0.90, 0.20),
            (20, 3, 0.40, 0.99),
            (12, 2, 0.30, 0.95),
        ]
    ):
        base = {
            "airport": f"A{i}",
            "utc_hour": f"2025-01-01 {i:02d}:00:00",
            "month": 1,
            "arrivals": arrivals,
            "arr_delay60_count": events,
            "arr_delay120_count": 0,
            "cancel_count": 0,
            "excess_delay60_minutes": events * 60,
            "target": "long_arrival_delay",
        }
        rows.append({**base, "model": "CQS-Rank", "pred_prob": 0.05, "cqs_rank_score": cqs_score})
        rows.append({**base, "model": "RAEG-Rank", "pred_prob": raeg_prob})
    return pd.DataFrame(rows)


def test_audit_pair_reports_queue_swap_event_gain() -> None:
    audit = audit_pair(toy_predictions(), "long_arrival_delay", "arr_delay60_count", 0.50)

    assert audit["queue_size"] == 2
    assert audit["overlap_share"] == 0.0
    assert audit["cqs_only_events"] == 13.0
    assert audit["reference_only_events"] == 5.0
    assert audit["event_gain"] == 8.0
    assert audit["cqs_only_arrivals_mean"] == 90.0
    assert audit["reference_only_arrivals_mean"] == 16.0


def test_traffic_stratum_audit_allocates_gain_to_high_volume() -> None:
    strata = traffic_stratum_audit(toy_predictions(), "long_arrival_delay", "arr_delay60_count", 0.50)
    high = strata[strata["traffic_stratum"].eq("high")].iloc[0]
    mid = strata[strata["traffic_stratum"].eq("mid")].iloc[0]
    low = strata[strata["traffic_stratum"].eq("low")].iloc[0]

    assert high["event_gain"] == 8.0
    assert mid["event_gain"] == 5.0
    assert low["event_gain"] == -5.0
    assert high["cqs_selected"] == 1
    assert mid["cqs_selected"] == 1
    assert low["reference_selected"] == 2


if __name__ == "__main__":
    test_audit_pair_reports_queue_swap_event_gain()
    test_traffic_stratum_audit_allocates_gain_to_high_volume()
