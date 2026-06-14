from __future__ import annotations

import pandas as pd

from asoc_cqs_robust_certificate import add_calibration_lower_score, calibration_gap_table


def toy_calibration_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "airport": ["A", "B", "C", "D"],
            "utc_hour": pd.date_range("2025-01-01", periods=4, freq="h"),
            "month": [1, 1, 1, 1],
            "arrivals": [100, 50, 20, 10],
            "pred_prob": [0.10, 0.10, 0.80, 0.80],
            "arr_delay60_count": [5, 10, 10, 4],
        }
    )


def test_calibration_gap_is_arrival_weighted_by_risk_bin() -> None:
    gaps = calibration_gap_table(toy_calibration_frame(), "arr_delay60_count", n_bins=2)

    low = gaps.sort_values("pred_rate").iloc[0]
    high = gaps.sort_values("pred_rate").iloc[1]

    assert round(float(low["pred_rate"]), 4) == 0.1000
    assert round(float(low["obs_rate"]), 4) == 0.1000
    assert round(float(low["gap"]), 4) == 0.0000
    assert round(float(high["pred_rate"]), 4) == 0.8000
    assert round(float(high["obs_rate"]), 4) == 0.4667
    assert round(float(high["gap"]), 4) == 0.3333


def test_lower_score_subtracts_bin_calibration_gap() -> None:
    scored = add_calibration_lower_score(toy_calibration_frame(), "arr_delay60_count", n_bins=2)

    low = scored.sort_values("pred_prob").iloc[0]
    high = scored.sort_values("pred_prob").iloc[-1]

    assert round(float(low["calibration_lower_prob"]), 4) == 0.1000
    assert round(float(low["calibration_lower_score"]), 4) == 10.0000
    assert round(float(high["calibration_lower_prob"]), 4) == 0.4667
    assert round(float(high["calibration_lower_score"]), 4) == 4.6667


if __name__ == "__main__":
    test_calibration_gap_is_arrival_weighted_by_risk_bin()
    test_lower_score_subtracts_bin_calibration_gap()
