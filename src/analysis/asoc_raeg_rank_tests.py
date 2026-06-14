from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from asoc_raeg_rank import (
    RAEGConfig,
    airport_metadata_graph_weights,
    choose_device,
    fit_action_propensity,
    multi_view_airport_graph_weights,
    select_raeg_feature_columns,
    soft_topk_capture_loss,
)


def test_soft_topk_prefers_event_ranking() -> None:
    events = torch.tensor([4.0, 0.0, 0.0, 0.0])
    good = torch.tensor([3.0, 0.0, -1.0, -2.0])
    bad = torch.tensor([-2.0, 3.0, 2.0, 1.0])
    good_loss = soft_topk_capture_loss(good, events, budget_fraction=0.25)
    bad_loss = soft_topk_capture_loss(bad, events, budget_fraction=0.25)
    assert torch.isfinite(good_loss)
    assert float(good_loss) < float(bad_loss)


def test_metadata_graph_rows_are_normalized() -> None:
    airports = ["ATL", "DFW", "ORD", "SFO"]
    weights = airport_metadata_graph_weights(airports)
    assert list(weights.index) == sorted(airports)
    rows = weights.sum(axis=1).to_numpy(float)
    assert np.allclose(rows, 1.0)
    assert np.allclose(np.diag(weights.to_numpy(float)), 0.0)


def test_feature_selector_excludes_outcomes() -> None:
    frame = pd.DataFrame(
        {
            "weather_score": [1.0],
            "relation_score": [0.2],
            "graph_weather_score": [0.3],
            "mv_graph_weather_score": [0.25],
            "temp_lag1_weather_score": [0.4],
            "raeg_action_active_state": [0.5],
            "raeg_action_cf_gap": [0.1],
            "arrivals": [10.0],
            "arr_delay60_count": [2.0],
            "arr_delay120_count": [1.0],
            "excess_delay60_minutes": [35.0],
            "cancel_count": [0.0],
            "arr_delay60_rate": [0.2],
            "arr_delay120_rate": [0.1],
        }
    )
    cols = select_raeg_feature_columns(frame)
    assert "weather_score" in cols
    assert "relation_score" in cols
    assert "graph_weather_score" in cols
    assert "mv_graph_weather_score" in cols
    assert "temp_lag1_weather_score" in cols
    assert "raeg_action_active_state" in cols
    assert "raeg_action_cf_gap" in cols
    assert "arrivals" not in cols
    assert "arr_delay60_count" not in cols
    assert "arr_delay120_count" not in cols
    assert "excess_delay60_minutes" not in cols
    assert "cancel_count" not in cols
    assert "arr_delay60_rate" not in cols
    assert "arr_delay120_rate" not in cols


def test_multiview_graph_rows_are_normalized() -> None:
    airports = ["ATL", "DFW", "ORD", "SFO"]
    rows = []
    for i, airport in enumerate(airports):
        for hour in range(3):
            rows.append(
                {
                    "airport": airport,
                    "scheduled_arrivals": 20 + i * 3 + hour,
                    "scheduled_departures": 18 + i * 2,
                    "weather_score": 0.4 + 0.1 * i,
                    "active_minutes": 10.0 * (i % 2),
                    "post_3h_known_minutes": 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    weights = multi_view_airport_graph_weights(frame, frame, airports)
    assert list(weights.index) == sorted(airports)
    assert np.allclose(weights.sum(axis=1).to_numpy(float), 1.0)
    assert np.allclose(np.diag(weights.to_numpy(float)), 0.0)


def test_action_propensity_is_bounded() -> None:
    rows = []
    for i in range(80):
        rows.append(
            {
                "airport": "ATL" if i < 40 else "DFW",
                "arrivals": 20 + i % 5,
                "weather_score": float(i % 7),
                "wind_speed_mps": 2.0 + i % 6,
                "visibility_km": 12.0 - (i % 4),
                "scheduled_arrivals": 15 + i % 8,
                "scheduled_departures": 12 + i % 6,
                "arrival_bank_intensity": 0.8 + (i % 4) * 0.1,
                "departure_bank_intensity": 0.7 + (i % 3) * 0.1,
                "month_sin": 0.0,
                "month_cos": 1.0,
                "active_minutes": 60.0 if i % 5 == 0 else 0.0,
                "active_within_minutes": 0.0,
                "active_before_minutes": 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    train_prop, test_prop = fit_action_propensity(frame.iloc[:60].copy(), frame.iloc[60:].copy())
    assert np.all(np.isfinite(train_prop))
    assert np.all(np.isfinite(test_prop))
    assert np.all((train_prop > 0.0) & (train_prop < 1.0))
    assert np.all((test_prop > 0.0) & (test_prop < 1.0))


def test_device_selection() -> None:
    device = choose_device(require_cuda=False)
    assert str(device) in {"cuda", "cpu"}
    if torch.cuda.is_available():
        cuda_device = choose_device(require_cuda=True)
        assert str(cuda_device) == "cuda"


def main() -> None:
    test_soft_topk_prefers_event_ranking()
    test_metadata_graph_rows_are_normalized()
    test_feature_selector_excludes_outcomes()
    test_multiview_graph_rows_are_normalized()
    test_action_propensity_is_bounded()
    test_device_selection()
    cfg = RAEGConfig(epochs=1, batch_size=16, require_cuda=False)
    assert cfg.epochs == 1
    print("asoc_raeg_rank_tests passed")


if __name__ == "__main__":
    main()
