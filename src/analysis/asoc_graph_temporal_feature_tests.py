from __future__ import annotations

import numpy as np
import pandas as pd

from asoc_graph_temporal_baselines_smoke import (
    GRAPH_SIGNALS,
    add_graph_features,
    add_graph_features_fast,
    airport_metadata_graph_weights,
)


def build_small_panel() -> pd.DataFrame:
    rows = []
    hours = pd.date_range("2025-01-01", periods=3, freq="h")
    airports = ["AAA", "BBB", "CCC"]
    for hour_idx, hour in enumerate(hours):
        for airport_idx, airport in enumerate(airports):
            row = {
                "airport": airport,
                "utc_hour": hour,
            }
            for signal_idx, signal in enumerate(GRAPH_SIGNALS):
                row[signal] = 1.0 + hour_idx * 10.0 + airport_idx * 2.0 + signal_idx * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_fast_graph_features_match_slow_version() -> None:
    panel = build_small_panel()
    weights = pd.DataFrame(
        {
            "AAA": {"AAA": 0.0, "BBB": 0.25, "CCC": 0.75},
            "BBB": {"AAA": 0.40, "BBB": 0.0, "CCC": 0.60},
            "CCC": {"AAA": 0.55, "BBB": 0.45, "CCC": 0.0},
        }
    )
    weights = weights.div(weights.sum(axis=1), axis=0)
    medians = {signal: float(panel[signal].median()) for signal in GRAPH_SIGNALS}
    slow = add_graph_features(panel, weights, medians).sort_values(["utc_hour", "airport"]).reset_index(drop=True)
    fast = add_graph_features_fast(panel, weights, medians).sort_values(["utc_hour", "airport"]).reset_index(drop=True)
    for signal in GRAPH_SIGNALS:
        col = f"graph_{signal}"
        np.testing.assert_allclose(fast[col].to_numpy(float), slow[col].to_numpy(float), rtol=1e-10, atol=1e-10)


def test_airport_metadata_graph_weights_are_normalized() -> None:
    weights = airport_metadata_graph_weights(["ATL", "CLT", "DEN", "DFW"])
    assert list(weights.index) == ["ATL", "CLT", "DEN", "DFW"]
    assert list(weights.columns) == ["ATL", "CLT", "DEN", "DFW"]
    np.testing.assert_allclose(np.diag(weights.to_numpy(float)), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(float), np.ones(4), atol=1e-12)
    assert (weights.to_numpy(float) >= 0.0).all()


if __name__ == "__main__":
    test_fast_graph_features_match_slow_version()
    test_airport_metadata_graph_weights_are_normalized()
    print("graph temporal feature tests passed")
