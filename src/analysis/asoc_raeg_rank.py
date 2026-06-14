from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asoc_fuzzy_residual_evidence_smoke import (
    ADVISORY_NUM,
    BASE_NUM,
    FUZZY_MEMBERSHIP_NUM,
    PRIOR_NUM,
    TARGETS as BASE_TARGETS,
    load_panel as default_load_panel,
    relation_scores as default_relation_scores,
    validation_folds,
)
from asoc_graph_temporal_baselines_smoke import (
    GRAPH_SIGNALS,
    airport_metadata_graph_weights,
    add_graph_features_fast,
    design as graph_design,
    feature_sets,
    fold_prepare,
    train_medians,
)
from asoc_soft_computing_smoke import (
    OUT_ROOT,
    align_design,
    capture_at_fraction,
    ece,
    expanded_binomial_frame,
    isotonic_calibrate,
    metric_row,
    parse_airports,
    parse_int_list,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


MODEL_NAME = "RAEG-Rank"
GT_AFRE_MODEL = "Graph-temporal evidence"
RELATION_MODEL = "Relation-DCSI h1"
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30]
BOOTSTRAP_METRICS = ["pr_auc", "top5_capture", "top10_capture", "top20_capture", "brier", "ece_10"]
TARGETS = dict(BASE_TARGETS)
TARGETS.setdefault("severe_arrival_delay", "arr_delay120_count")
SEVERE_DELAY_COL = "arr_delay120_count"
SEVERITY_COL = "excess_delay60_minutes"

MAIN_AIRPORTS = "ATL,CLT,DEN,DFW,EWR,JFK,LAX,LGA,ORD,SFO"
SMOKE_AIRPORTS = "ATL,DFW,EWR,ORD"

EVIDENCE_MEMBERSHIPS = [
    "weather_high_membership",
    "arrival_pressure_membership",
    "departure_pressure_membership",
    "arrival_bank_membership",
    "departure_bank_membership",
    "target_prior_membership",
    "active_memory_membership",
    "post_memory_membership",
    "soft_pressure_index",
    "relation_score",
]

ACTION_SOURCE_COLS = [
    "active_before_minutes",
    "active_within_minutes",
    "active_minutes",
    "post_1h_known_minutes",
    "post_3h_known_minutes",
    "post_3h_minutes",
]

ACTION_PROPENSITY_FEATURES = [
    "weather_score",
    "wind_speed_mps",
    "visibility_km",
    "ceiling_m",
    "temperature_c",
    "scheduled_arrivals",
    "scheduled_departures",
    "arrival_bank_intensity",
    "departure_bank_intensity",
    "arrival_carrier_hhi",
    "departure_carrier_hhi",
    "month_sin",
    "month_cos",
    "prior_hour_arrivals",
]

MULTIVIEW_PROFILE_COLS = [
    "scheduled_arrivals",
    "scheduled_departures",
    "arrival_bank_intensity",
    "departure_bank_intensity",
    "arrival_carrier_hhi",
    "departure_carrier_hhi",
    "weather_score",
    "wind_speed_mps",
    "visibility_km",
    "ceiling_m",
    "temperature_c",
    "active_before_minutes",
    "active_within_minutes",
    "active_minutes",
    "post_1h_known_minutes",
    "post_3h_known_minutes",
]

VARIANTS = {
    "RAEG-Rank": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without reliability gate": {
        "use_reliability": False,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without action-coupled transition": {
        "use_reliability": True,
        "use_action": False,
        "use_counterfactual_action": False,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without action-counterfactual state": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": False,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without dual graph": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": False,
        "use_multiview_graph": False,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without multi-view graph": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": True,
        "use_multiview_graph": False,
        "use_queue_objective": True,
        "use_severity_queue": True,
        "require_neural_residual": True,
    },
    "RAEG without queue-calibrated objective": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": False,
        "use_severity_queue": False,
        "require_neural_residual": True,
    },
    "RAEG without severity-aware queue": {
        "use_reliability": True,
        "use_action": True,
        "use_counterfactual_action": True,
        "use_dual_graph": True,
        "use_multiview_graph": True,
        "use_queue_objective": True,
        "use_severity_queue": False,
        "require_neural_residual": True,
    },
}


@dataclass(frozen=True)
class RAEGConfig:
    hidden_dim: int = 64
    epochs: int = 12
    batch_size: int = 768
    learning_rate: float = 0.0018
    weight_decay: float = 0.001
    queue_weight: float = 0.25
    calibration_weight: float = 0.05
    budget_fraction: float = 0.10
    queue_temperature: float = 0.12
    severity_queue_weight: float = 0.35
    seed: int = 42
    require_cuda: bool = True


@dataclass
class RAEGFitResult:
    test_prob: np.ndarray
    raw_test_prob: np.ndarray
    train_prob: np.ndarray
    feature_columns: list[str]
    device: str
    final_loss: float
    blend: str = "neural"


class RAEGNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ramp(values: pd.Series, low: float, high: float) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if high <= low:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((x - low) / (high - low)).clip(0.0, 1.0).fillna(0.0)


def action_ramp(values: pd.Series, low_minutes: float, high_minutes: float) -> pd.Series:
    if not isinstance(values, pd.Series):
        values = pd.Series([values])
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    if not finite.empty and finite.quantile(0.99) <= 8.5:
        x = x * 60.0
    return ramp(x, low_minutes, high_minutes)


def row_normalize_graph(weights: pd.DataFrame) -> pd.DataFrame:
    out = weights.copy().astype(float)
    if out.empty:
        return out
    arr = np.array(out.to_numpy(float), dtype=float, copy=True)
    np.fill_diagonal(arr, 0.0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    for i in range(arr.shape[0]):
        row_sum = float(arr[i].sum())
        if row_sum <= 1e-12:
            if arr.shape[0] > 1:
                arr[i, :] = 1.0 / (arr.shape[0] - 1)
                arr[i, i] = 0.0
            else:
                arr[i, i] = 0.0
        else:
            arr[i, :] = arr[i, :] / row_sum
    return pd.DataFrame(arr, index=out.index, columns=out.columns)


def profile_similarity_graph_weights(frame: pd.DataFrame, airports: list[str], columns: list[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in frame.columns]
    if len(airports) <= 1 or not cols:
        return row_normalize_graph(pd.DataFrame(np.zeros((len(airports), len(airports))), index=airports, columns=airports))
    profile_rows: list[pd.DataFrame] = []
    for stat_name, stat_fn in [("mean", "mean"), ("std", "std")]:
        profile = frame.groupby("airport")[cols].agg(stat_fn).reindex(airports)
        profile.columns = [f"{col}_{stat_name}" for col in cols]
        profile_rows.append(profile)
    profile = pd.concat(profile_rows, axis=1)
    for col in profile.columns:
        values = pd.to_numeric(profile[col], errors="coerce")
        fill = float(values.median()) if values.notna().any() else 0.0
        profile[col] = values.fillna(fill)
    arr = profile.to_numpy(float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    z = (arr - mean) / std
    dist = np.sqrt(np.maximum(((z[:, None, :] - z[None, :, :]) ** 2).mean(axis=2), 0.0))
    sigma = float(np.nanmedian(dist[dist > 0])) if np.any(dist > 0) else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    sim = np.exp(-dist / sigma)
    np.fill_diagonal(sim, 0.0)
    return row_normalize_graph(pd.DataFrame(sim, index=airports, columns=airports))


def multi_view_airport_graph_weights(train: pd.DataFrame, test: pd.DataFrame, airports: list[str]) -> pd.DataFrame:
    context = pd.concat([train, test], ignore_index=True, sort=False)
    metadata = airport_metadata_graph_weights(airports)
    demand_weather_action = profile_similarity_graph_weights(context, airports, MULTIVIEW_PROFILE_COLS)
    combined = 0.45 * metadata + 0.55 * demand_weather_action
    return row_normalize_graph(combined)


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 3:
        return 0.0
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    w = w[mask].astype(float)
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    vx = np.average((x - wx) ** 2, weights=w)
    vy = np.average((y - wy) ** 2, weights=w)
    denom = np.sqrt(max(vx * vy, 0.0))
    if denom <= 1e-12:
        return 0.0
    return float(cov / denom)


def evidence_reliability(train: pd.DataFrame, cols: Iterable[str], success_col: str) -> dict[str, float]:
    totals = pd.to_numeric(train["arrivals"], errors="coerce").fillna(0.0).to_numpy(float)
    successes = pd.to_numeric(train[success_col], errors="coerce").fillna(0.0).to_numpy(float)
    rate = np.divide(successes, np.maximum(totals, 1.0))
    out: dict[str, float] = {}
    for col in cols:
        values = pd.to_numeric(train.get(col, 0.0), errors="coerce")
        missing = float(values.isna().mean())
        x = values.fillna(values.median() if values.notna().any() else 0.0).to_numpy(float)
        corr = abs(weighted_corr(x, rate, totals))
        evidence_strength = min(corr / 0.20, 1.0)
        out[col] = float(np.clip(0.25 + 0.75 * evidence_strength * np.sqrt(1.0 - missing), 0.20, 1.0))
    return out


def add_metadata_graph_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    airports = sorted(set(train["airport"].astype(str)).union(set(test["airport"].astype(str))))
    weights = airport_metadata_graph_weights(airports)
    multiview_weights = multi_view_airport_graph_weights(train, test, airports)
    medians = train_medians(train, sorted(set(GRAPH_SIGNALS)))
    out_frames = []
    for frame in [train, test]:
        base = frame.drop(columns=[col for col in frame.columns if col.startswith("graph_")], errors="ignore")
        meta = add_graph_features_fast(base, weights, medians)
        multiview = add_graph_features_fast(base, multiview_weights, medians)
        out = frame.copy()
        for signal in GRAPH_SIGNALS:
            src = f"graph_{signal}"
            dst = f"meta_graph_{signal}"
            out[dst] = pd.to_numeric(meta[src], errors="coerce").fillna(0.0)
            mv_dst = f"mv_graph_{signal}"
            out[mv_dst] = pd.to_numeric(multiview[src], errors="coerce").fillna(0.0)
            graph_col = f"graph_{signal}"
            if graph_col in out.columns:
                out[f"dual_graph_gap_{signal}"] = (
                    pd.to_numeric(out[graph_col], errors="coerce").fillna(0.0) - out[dst]
                )
                out[f"target_mv_graph_gap_{signal}"] = (
                    pd.to_numeric(out[graph_col], errors="coerce").fillna(0.0) - out[mv_dst]
                )
            out[f"meta_mv_graph_gap_{signal}"] = out[dst] - out[mv_dst]
        out_frames.append(out)
    return out_frames[0], out_frames[1]


def available_evidence_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in EVIDENCE_MEMBERSHIPS:
        if col in frame.columns:
            cols.append(col)
        for prefix in ["graph_", "meta_graph_", "mv_graph_", "dual_graph_gap_", "target_mv_graph_gap_", "meta_mv_graph_gap_"]:
            candidate = f"{prefix}{col}"
            if candidate in frame.columns:
                cols.append(candidate)
    for col in [
        "temp_lag1_soft_pressure_index",
        "temp_mean3_soft_pressure_index",
        "temp_max6_soft_pressure_index",
        "temp_mean12_target_prior_membership",
        "temp_max12_active_memory_membership",
        "temp_mean24_post_memory_membership",
    ]:
        if col in frame.columns:
            cols.append(col)
    return list(dict.fromkeys(cols))


def fit_action_propensity(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    features = [col for col in ACTION_PROPENSITY_FEATURES if col in train.columns]
    active_train = np.maximum(
        action_ramp(train.get("active_minutes", 0.0), 10.0, 75.0),
        action_ramp(train.get("active_within_minutes", 0.0), 10.0, 75.0),
    )
    active_train = np.maximum(active_train, action_ramp(train.get("active_before_minutes", 0.0), 10.0, 90.0))
    y = (active_train >= 0.20).astype(int)
    mean_prop = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    if len(features) < 2 or pd.Series(y).nunique() < 2:
        return np.full(len(train), mean_prop, dtype=float), np.full(len(test), mean_prop, dtype=float)

    x_train = train[features].copy()
    x_test = test[features].copy()
    for col in features:
        train_values = pd.to_numeric(x_train[col], errors="coerce")
        fill = float(train_values.median()) if train_values.notna().any() else 0.0
        x_train[col] = train_values.fillna(fill)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(fill)
    weights = pd.to_numeric(train.get("arrivals", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    try:
        model = LGBMClassifier(
            n_estimators=90,
            num_leaves=10,
            learning_rate=0.05,
            min_child_samples=50,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=3.0,
            verbosity=-1,
            random_state=seed,
        )
        model.fit(x_train, y, sample_weight=weights)
        train_prop = model.predict_proba(x_train)[:, 1]
        test_prop = model.predict_proba(x_test)[:, 1]
    except Exception:
        train_prop = np.full(len(train), mean_prop, dtype=float)
        test_prop = np.full(len(test), mean_prop, dtype=float)
    return np.clip(train_prop, 1e-4, 1 - 1e-4), np.clip(test_prop, 1e-4, 1 - 1e-4)


def add_raeg_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    success_col: str,
    use_reliability: bool,
    use_action: bool,
    use_dual_graph: bool,
    use_counterfactual_action: bool,
    use_multiview_graph: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train = train.copy()
    test = test.copy()
    evidence_cols = available_evidence_cols(train)
    if not use_dual_graph:
        graph_prefixes = (
            "graph_",
            "meta_graph_",
            "mv_graph_",
            "dual_graph_gap_",
            "target_mv_graph_gap_",
            "meta_mv_graph_gap_",
        )
        evidence_cols = [col for col in evidence_cols if not col.startswith(graph_prefixes)]
        for frame in [train, test]:
            for col in frame.columns:
                if col.startswith(graph_prefixes):
                    frame[col] = 0.0
    elif not use_multiview_graph:
        mv_prefixes = ("mv_graph_", "target_mv_graph_gap_", "meta_mv_graph_gap_")
        evidence_cols = [col for col in evidence_cols if not col.startswith(mv_prefixes)]
        for frame in [train, test]:
            for col in frame.columns:
                if col.startswith(mv_prefixes):
                    frame[col] = 0.0
    reliabilities = evidence_reliability(train, evidence_cols, success_col)
    if not use_reliability:
        reliabilities = {col: 1.0 for col in evidence_cols}

    if use_action and use_counterfactual_action:
        train_action_prop, test_action_prop = fit_action_propensity(train, test)
    else:
        train_action_prop = np.zeros(len(train), dtype=float)
        test_action_prop = np.zeros(len(test), dtype=float)

    for frame, action_prop in [(train, train_action_prop), (test, test_action_prop)]:
        for col in evidence_cols:
            values = pd.to_numeric(frame.get(col, 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            r = reliabilities[col]
            safe = col.replace(" ", "_").replace("-", "_")
            frame[f"raeg_mass_pressure_{safe}"] = r * values
            frame[f"raeg_mass_normal_{safe}"] = r * (1.0 - values)
            frame[f"raeg_mass_uncertainty_{safe}"] = 1.0 - r
            frame[f"raeg_mass_balance_{safe}"] = frame[f"raeg_mass_pressure_{safe}"] - frame[f"raeg_mass_normal_{safe}"]

        active = action_ramp(frame.get("active_minutes", 0.0), 10.0, 75.0)
        active_within = action_ramp(frame.get("active_within_minutes", 0.0), 10.0, 75.0)
        pre_action = action_ramp(frame.get("active_before_minutes", 0.0), 10.0, 90.0)
        post_1h = action_ramp(frame.get("post_1h_known_minutes", 0.0), 10.0, 90.0)
        post_3h = action_ramp(frame.get("post_3h_known_minutes", 0.0), 15.0, 180.0)
        weather = pd.to_numeric(frame.get("weather_high_membership", 0.0), errors="coerce").fillna(0.0)
        demand = np.maximum(
            pd.to_numeric(frame.get("arrival_pressure_membership", 0.0), errors="coerce").fillna(0.0),
            pd.to_numeric(frame.get("departure_pressure_membership", 0.0), errors="coerce").fillna(0.0),
        )
        prior = pd.to_numeric(frame.get("target_prior_membership", 0.0), errors="coerce").fillna(0.0)
        context_pressure = np.maximum.reduce([weather.to_numpy(float), demand.to_numpy(float), prior.to_numpy(float)])
        if use_action:
            frame["raeg_action_pre_state"] = pre_action
            frame["raeg_action_active_state"] = np.maximum(active, active_within)
            frame["raeg_action_recovery_state"] = np.maximum(post_1h, post_3h)
            frame["raeg_action_hysteresis"] = frame["raeg_action_recovery_state"] - frame["raeg_action_active_state"]
            frame["raeg_action_weather_coupling"] = np.maximum(active, post_3h) * weather
            frame["raeg_action_demand_coupling"] = np.maximum(active, post_3h) * demand
            frame["raeg_action_prior_coupling"] = np.maximum(active, post_3h) * prior
            if use_counterfactual_action:
                observed_action = frame["raeg_action_active_state"].to_numpy(float)
                recovery_state = frame["raeg_action_recovery_state"].to_numpy(float)
                prop = np.asarray(action_prop, dtype=float)
                frame["raeg_action_propensity"] = prop
                frame["raeg_action_cf_gap"] = observed_action - prop
                frame["raeg_action_unexpected_active"] = observed_action * (1.0 - prop)
                frame["raeg_action_expected_but_inactive"] = (1.0 - observed_action) * prop
                frame["raeg_action_observed_pressure"] = observed_action * context_pressure
                frame["raeg_action_counterfactual_pressure"] = (1.0 - observed_action) * prop * context_pressure
                frame["raeg_action_recovery_cf_gap"] = recovery_state - prop
            else:
                for col in [
                    "raeg_action_propensity",
                    "raeg_action_cf_gap",
                    "raeg_action_unexpected_active",
                    "raeg_action_expected_but_inactive",
                    "raeg_action_observed_pressure",
                    "raeg_action_counterfactual_pressure",
                    "raeg_action_recovery_cf_gap",
                ]:
                    frame[col] = 0.0
        else:
            for col in [
                "raeg_action_pre_state",
                "raeg_action_active_state",
                "raeg_action_recovery_state",
                "raeg_action_hysteresis",
                "raeg_action_weather_coupling",
                "raeg_action_demand_coupling",
                "raeg_action_prior_coupling",
                "raeg_action_propensity",
                "raeg_action_cf_gap",
                "raeg_action_unexpected_active",
                "raeg_action_expected_but_inactive",
                "raeg_action_observed_pressure",
                "raeg_action_counterfactual_pressure",
                "raeg_action_recovery_cf_gap",
            ]:
                frame[col] = 0.0

    return train, test, reliabilities


def prepare_raeg_fold(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    target: str,
    success_col: str,
    use_reliability: bool,
    use_action: bool,
    use_dual_graph: bool,
    use_counterfactual_action: bool,
    use_multiview_graph: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train, test = fold_prepare(raw_train, raw_test, target, success_col, graph_source="target_correlation")
    train, test = add_metadata_graph_features(train, test)
    return add_raeg_features(
        train,
        test,
        success_col,
        use_reliability,
        use_action,
        use_dual_graph,
        use_counterfactual_action,
        use_multiview_graph,
    )


def select_raeg_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("raeg_", "graph_", "meta_graph_", "mv_graph_", "dual_graph_gap_", "target_mv_graph_gap_", "meta_mv_graph_gap_", "temp_")
    base = BASE_NUM + ADVISORY_NUM + PRIOR_NUM + FUZZY_MEMBERSHIP_NUM + ["relation_score"]
    cols = [col for col in base if col in frame.columns]
    cols.extend([col for col in frame.columns if col.startswith(prefixes)])
    forbidden = {
        "arrivals",
        "arr_delay60_count",
        "arr_delay120_count",
        "cancel_count",
        "divert_count",
        "arr_delay60_rate",
        "arr_delay120_rate",
        "excess_delay60_minutes",
        "excess_delay60_per_arrival",
        "cancel_rate",
        "divert_rate",
        "mean_arr_delay",
        "p90_arr_delay",
    }
    return [col for col in list(dict.fromkeys(cols)) if col not in forbidden]


def design_raeg(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = train[columns].copy()
    x_test = test[columns].copy()
    for col in columns:
        median = pd.to_numeric(x_train[col], errors="coerce").median()
        fill = float(median) if np.isfinite(median) else 0.0
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce").fillna(fill)
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(fill)
    cats = ["airport", "local_hour", "day_of_week"]
    d_train = pd.get_dummies(train[cats].astype(str), prefix=cats, dtype=float)
    d_test = pd.get_dummies(test[cats].astype(str), prefix=cats, dtype=float)
    x_train = pd.concat([x_train, d_train], axis=1)
    x_test = pd.concat([x_test, d_test], axis=1)
    return align_design(x_train, x_test)


def standardize(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_arr = train_x.to_numpy(dtype=np.float32)
    test_arr = test_x.to_numpy(dtype=np.float32)
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_arr - mean) / std).astype(np.float32), ((test_arr - mean) / std).astype(np.float32)


def fit_tree_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x_raw: pd.DataFrame,
    test_x_raw: pd.DataFrame,
    success_col: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_expanded, y_expanded, w_expanded = expanded_binomial_frame(train, train_x_raw, success_col)
    model = LGBMClassifier(
        n_estimators=320,
        num_leaves=18,
        learning_rate=0.025,
        min_child_samples=100,
        subsample=0.82,
        colsample_bytree=0.78,
        reg_lambda=4.0,
        reg_alpha=0.1,
        verbosity=-1,
        random_state=seed,
    )
    model.fit(x_expanded, y_expanded, sample_weight=w_expanded)
    train_prob = np.clip(model.predict_proba(train_x_raw)[:, 1], 1e-5, 1 - 1e-5)
    test_prob = np.clip(model.predict_proba(test_x_raw)[:, 1], 1e-5, 1 - 1e-5)
    return train_prob, test_prob


def fit_gt_afre_probability_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    success_col: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_train_raw = graph_design(train, features)
    x_test_raw = graph_design(test, features)
    x_train_raw, x_test_raw = align_design(x_train_raw, x_test_raw)
    x_expanded, y_expanded, w_expanded = expanded_binomial_frame(train, x_train_raw, success_col)
    model = LGBMClassifier(
        n_estimators=220,
        num_leaves=20,
        learning_rate=0.04,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        verbosity=-1,
        random_state=42,
    )
    model.fit(x_expanded, y_expanded, sample_weight=w_expanded)
    train_prob = np.clip(model.predict_proba(x_train_raw)[:, 1], 1e-5, 1 - 1e-5)
    test_prob = np.clip(model.predict_proba(x_test_raw)[:, 1], 1e-5, 1 - 1e-5)
    return train_prob, test_prob


def blend_objective(successes: np.ndarray, totals: np.ndarray, prob: np.ndarray) -> float:
    frame = pd.DataFrame({"arrivals": totals, "events": successes, "pred_prob": prob})
    row = metric_row("blend", "blend", frame, "events")
    cal = ece(successes, totals, prob)
    return float(row["pr_auc"] + 0.25 * row["top10_capture"] - 0.10 * cal)


def choose_blend(
    successes: np.ndarray,
    totals: np.ndarray,
    candidate_train: dict[str, np.ndarray],
    candidate_test: dict[str, np.ndarray],
    require_neural_residual: bool = False,
) -> tuple[str, np.ndarray, np.ndarray]:
    recipes: dict[str, dict[str, float]] = {}
    names = list(candidate_train)
    for name in names:
        if name != "gt_afre":
            recipes[name] = {name: 1.0}
    if "gt_afre" in names and "raeg_tree" in names:
        recipes["0.80_gt_afre+0.20_raeg_tree"] = {"gt_afre": 0.80, "raeg_tree": 0.20}
        recipes["0.65_gt_afre+0.35_raeg_tree"] = {"gt_afre": 0.65, "raeg_tree": 0.35}
    if "gt_afre" in names and "raeg_tree" in names and "raeg_neural" in names:
        recipes["0.79_gt_afre+0.20_raeg_tree+0.01_raeg_neural"] = {
            "gt_afre": 0.79,
            "raeg_tree": 0.20,
            "raeg_neural": 0.01,
        }
        recipes["0.64_gt_afre+0.35_raeg_tree+0.01_raeg_neural"] = {
            "gt_afre": 0.64,
            "raeg_tree": 0.35,
            "raeg_neural": 0.01,
        }
        recipes["0.75_gt_afre+0.20_raeg_tree+0.05_raeg_neural"] = {
            "gt_afre": 0.75,
            "raeg_tree": 0.20,
            "raeg_neural": 0.05,
        }
        recipes["0.60_gt_afre+0.30_raeg_tree+0.10_raeg_neural"] = {
            "gt_afre": 0.60,
            "raeg_tree": 0.30,
            "raeg_neural": 0.10,
        }
    if "raeg_tree" in names and "raeg_neural" in names:
        recipes["0.85_raeg_tree+0.15_raeg_neural"] = {"raeg_tree": 0.85, "raeg_neural": 0.15}

    best_name = ""
    best_score = -np.inf
    best_train = np.zeros_like(successes, dtype=float)
    best_test = np.zeros(len(next(iter(candidate_test.values()))), dtype=float)
    for recipe_name, weights in recipes.items():
        if require_neural_residual and "raeg_neural" not in weights:
            continue
        train_prob = np.zeros_like(successes, dtype=float)
        test_prob = np.zeros(len(next(iter(candidate_test.values()))), dtype=float)
        for name, weight in weights.items():
            train_prob += weight * candidate_train[name]
            test_prob += weight * candidate_test[name]
        train_prob = np.clip(train_prob, 1e-5, 1 - 1e-5)
        score = blend_objective(successes, totals, train_prob)
        if score > best_score:
            best_name = recipe_name
            best_score = score
            best_train = train_prob
            best_test = np.clip(test_prob, 1e-5, 1 - 1e-5)
    if not best_name and require_neural_residual:
        return choose_blend(successes, totals, candidate_train, candidate_test, require_neural_residual=False)
    return best_name, best_train, best_test


def soft_topk_capture_loss(
    logits: torch.Tensor,
    event_counts: torch.Tensor,
    budget_fraction: float = 0.10,
    temperature: float = 0.12,
) -> torch.Tensor:
    if logits.numel() < 2 or torch.sum(event_counts) <= 0:
        return logits.sum() * 0.0
    q = float(np.clip(1.0 - budget_fraction, 0.0, 1.0))
    threshold = torch.quantile(logits.detach(), q)
    gate = torch.sigmoid((logits - threshold) / max(temperature, 1e-4))
    capture = torch.sum(gate * event_counts) / (torch.sum(event_counts) + 1e-6)
    budget_penalty = (torch.mean(gate) - budget_fraction) ** 2
    return -capture + budget_penalty


def choose_device(require_cuda: bool) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA is required for RAEG-Rank but torch.cuda.is_available() is False.")
    return torch.device("cpu")


def fit_predict_raeg(
    train: pd.DataFrame,
    test: pd.DataFrame,
    success_col: str,
    variant_name: str,
    config: RAEGConfig,
    anchor_train_prob: np.ndarray | None = None,
    anchor_test_prob: np.ndarray | None = None,
) -> RAEGFitResult:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown RAEG variant: {variant_name}")
    variant = VARIANTS[variant_name]
    device = choose_device(config.require_cuda)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_p, test_p, reliabilities = prepare_raeg_fold(
        train,
        test,
        target=str(train["target_name"].iloc[0]) if "target_name" in train.columns else "target",
        success_col=success_col,
        use_reliability=variant["use_reliability"],
        use_action=variant["use_action"],
        use_dual_graph=variant["use_dual_graph"],
        use_counterfactual_action=variant["use_counterfactual_action"],
        use_multiview_graph=variant["use_multiview_graph"],
    )
    columns = select_raeg_feature_columns(train_p)
    x_train_raw, x_test_raw = design_raeg(train_p, test_p, columns)
    tree_train_prob, tree_test_prob = fit_tree_probability(
        train_p,
        test_p,
        x_train_raw,
        x_test_raw,
        success_col,
        seed=config.seed,
    )
    x_train, x_test = standardize(x_train_raw, x_test_raw)
    totals = pd.to_numeric(train_p["arrivals"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    successes = pd.to_numeric(train_p[success_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
    queue_events = successes.copy()
    if variant.get("use_severity_queue", False) and SEVERITY_COL in train_p.columns:
        severity = pd.to_numeric(train_p[SEVERITY_COL], errors="coerce").fillna(0.0).to_numpy(np.float32)
        if float(np.sum(severity)) > 0.0 and float(np.sum(successes)) > 0.0:
            severity_scale = float(np.sum(successes) / max(np.sum(severity), 1.0))
            queue_events = successes + config.severity_queue_weight * severity * severity_scale
    y = np.divide(successes, np.maximum(totals, 1.0)).astype(np.float32)
    weights = totals / max(float(np.mean(totals[totals > 0])) if np.any(totals > 0) else 1.0, 1.0)

    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(queue_events, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    model = RAEGNet(x_train.shape[1], config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    final_loss = 0.0
    queue_weight = config.queue_weight if variant["use_queue_objective"] else 0.0

    model.train()
    for _ in range(config.epochs):
        losses = []
        for bx, by, bw, bevents in loader:
            bx = bx.to(device)
            by = by.to(device)
            bw = bw.to(device)
            bevents = bevents.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            prob = torch.sigmoid(logits)
            weighted_bce = torch.mean(bce(logits, by) * bw)
            queue_loss = soft_topk_capture_loss(
                logits,
                bevents,
                budget_fraction=config.budget_fraction,
                temperature=config.queue_temperature,
            )
            calibration_loss = (torch.mean(prob * bw) - torch.mean(by * bw)) ** 2
            loss = weighted_bce + queue_weight * queue_loss + config.calibration_weight * calibration_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else float("nan")

    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(torch.tensor(x_train, dtype=torch.float32, device=device))).detach().cpu().numpy()
        raw_test_prob = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32, device=device))).detach().cpu().numpy()
    try:
        neural_test_prob = isotonic_calibrate(train_prob, successes, totals, raw_test_prob)
    except Exception:
        neural_test_prob = np.clip(raw_test_prob, 1e-5, 1 - 1e-5)

    candidate_train = {
        "raeg_neural": np.clip(train_prob, 1e-5, 1 - 1e-5),
        "raeg_tree": np.clip(tree_train_prob, 1e-5, 1 - 1e-5),
    }
    candidate_test = {
        "raeg_neural": np.clip(neural_test_prob, 1e-5, 1 - 1e-5),
        "raeg_tree": np.clip(tree_test_prob, 1e-5, 1 - 1e-5),
    }
    if anchor_train_prob is not None and anchor_test_prob is not None:
        candidate_train["gt_afre"] = np.clip(anchor_train_prob, 1e-5, 1 - 1e-5)
        candidate_test["gt_afre"] = np.clip(anchor_test_prob, 1e-5, 1 - 1e-5)
    blend_name, blend_train, blend_test = choose_blend(
        successes,
        totals,
        candidate_train,
        candidate_test,
        require_neural_residual=bool(variant.get("require_neural_residual", False)),
    )
    try:
        test_prob = isotonic_calibrate(blend_train, successes, totals, blend_test)
    except Exception:
        test_prob = np.clip(blend_test, 1e-5, 1 - 1e-5)

    result = RAEGFitResult(
        test_prob=np.clip(test_prob, 1e-5, 1 - 1e-5),
        raw_test_prob=np.clip(blend_test, 1e-5, 1 - 1e-5),
        train_prob=np.clip(blend_train, 1e-5, 1 - 1e-5),
        feature_columns=list(x_train_raw.columns),
        device=str(device),
        final_loss=final_loss,
        blend=blend_name,
    )
    result.reliabilities = reliabilities  # type: ignore[attr-defined]
    return result


def parse_models(text: str) -> list[str]:
    if text.strip().upper() in {"ALL", "*"}:
        return list(VARIANTS)
    selected = [item.strip() for item in text.split(";") if item.strip()]
    unknown = [item for item in selected if item not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown RAEG models: {unknown}")
    return selected


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("At least one budget fraction is required.")
    return values


def load_experiment_panel(months: list[int], airports: list[str] | None, panel_path: str | None = None) -> pd.DataFrame:
    if not panel_path:
        return default_load_panel(months, airports)
    path = Path(panel_path)
    panel = pd.read_csv(path, parse_dates=["utc_hour"])
    panel = panel[panel["month"].isin(months)].copy()
    if airports:
        panel = panel[panel["airport"].isin(airports)].copy()
    panel = panel[(pd.to_numeric(panel["arrivals"], errors="coerce") > 0) & panel["weather_score"].notna()].copy()
    numeric_cols = sorted(set(BASE_NUM + ADVISORY_NUM + PRIOR_NUM + ACTION_SOURCE_COLS + ["active_strong", "post_3h_strong"]))
    for col in numeric_cols:
        if col not in panel.columns:
            panel[col] = 0.0
        values = pd.to_numeric(panel[col], errors="coerce")
        fill = float(values.median()) if values.notna().any() else 0.0
        panel[col] = values.fillna(fill)
    for col in ["airport", "local_hour", "day_of_week"]:
        panel[col] = panel[col].astype(str)
    return panel.reset_index(drop=True)


def load_relation_scores(
    months: list[int],
    target: str,
    horizon: int,
    relation_path: str | None = None,
) -> pd.DataFrame:
    relation_target = "long_arrival_delay" if target == "severe_arrival_delay" else target
    if not relation_path:
        return default_relation_scores(months, relation_target, horizon)
    rel = pd.read_csv(relation_path, parse_dates=["utc_hour"])
    rel = rel[
        rel["month"].isin(months)
        & rel["target"].eq(relation_target)
        & rel["horizon"].eq(horizon)
        & rel["model"].eq("online_relation_DCSI")
    ].copy()
    if rel.empty:
        raise ValueError(f"No relation scores for target={relation_target}, horizon={horizon}, months={months}, path={relation_path}")
    return rel[["airport", "utc_hour", "month", "pred_prob"]].rename(columns={"pred_prob": "relation_score"})


def raeg_validation_folds(
    panel: pd.DataFrame,
    validation: str,
    first_test_month: int,
    min_train_months: int,
) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    if validation != "rolling_quarter":
        yield from validation_folds(panel, validation, first_test_month, min_train_months)
        return
    months = sorted(int(m) for m in panel["month"].dropna().unique())
    for quarter_start in [1, 4, 7, 10]:
        test_months = [m for m in months if quarter_start <= m <= quarter_start + 2]
        if not test_months or min(test_months) < first_test_month:
            continue
        train_months = [m for m in months if m < min(test_months)]
        if len(train_months) < min_train_months:
            continue
        train = panel[panel["month"].isin(train_months)].copy()
        test = panel[panel["month"].isin(test_months)].copy()
        if not train.empty and not test.empty:
            yield f"rolling_q{quarter_start:02d}_{quarter_start + 2:02d}", train, test


def capture_column(fraction: float) -> str:
    return f"top{int(round(100 * fraction))}_capture"


def capture_at_score(successes: np.ndarray, totals: np.ndarray, score: np.ndarray, fraction: float) -> tuple[float, float, float, int]:
    n = max(1, int(np.ceil(len(score) * fraction)))
    idx = np.argsort(score)[::-1][:n]
    total_events = float(np.sum(successes))
    selected_events = float(np.sum(successes[idx]))
    selected_arrivals = float(np.sum(totals[idx]))
    if total_events <= 0:
        capture = np.nan
    else:
        capture = selected_events / total_events
    if selected_arrivals <= 0:
        selected_rate = np.nan
    else:
        selected_rate = selected_events / selected_arrivals
    full_rate = total_events / max(float(np.sum(totals)), 1.0)
    lift = selected_rate / full_rate if np.isfinite(selected_rate) and full_rate > 0 else np.nan
    return float(capture), float(selected_events), float(lift), int(n)


def extended_metric_row(
    target: str,
    model: str,
    pred: pd.DataFrame,
    success_col: str,
    budgets: list[float] | None = None,
) -> dict[str, object]:
    budgets = budgets or BUDGET_FRACTIONS
    row = metric_row(target, model, pred, success_col)
    successes = pred[success_col].to_numpy(float)
    totals = pred["arrivals"].to_numpy(float)
    prob = pred["pred_prob"].to_numpy(float)
    for fraction in budgets:
        row[capture_column(fraction)] = capture_at_fraction(successes, totals, prob, fraction)
    return row


def queue_budget_curve(
    predictions: pd.DataFrame,
    budgets: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=True):
        success_col = TARGETS[str(target)]
        successes = group[success_col].to_numpy(float)
        totals = group["arrivals"].to_numpy(float)
        prob = group["pred_prob"].to_numpy(float)
        severity = (
            pd.to_numeric(group[SEVERITY_COL], errors="coerce").fillna(0.0).to_numpy(float)
            if SEVERITY_COL in group.columns
            else np.zeros(len(group), dtype=float)
        )
        queue_scores = {
            "probability": prob,
            "expected_event": totals * prob,
        }
        full_events = float(np.sum(successes))
        full_severity = float(np.sum(severity))
        full_arrivals = float(np.sum(totals))
        full_rate = full_events / max(full_arrivals, 1.0)
        for queue_name, score in queue_scores.items():
            for fraction in budgets:
                capture, selected_events, lift, selected_count = capture_at_score(successes, totals, score, fraction)
                idx = np.argsort(score)[::-1][:selected_count]
                selected_severity = float(np.sum(severity[idx])) if len(idx) else 0.0
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "queue": queue_name,
                        "budget_fraction": fraction,
                        "selected_airport_hours": selected_count,
                        "captured_events": selected_events,
                        "total_events": full_events,
                        "capture": capture,
                        "lift": lift,
                        "full_event_rate": full_rate,
                        "captured_excess_delay60_minutes": selected_severity,
                        "total_excess_delay60_minutes": full_severity,
                        "severity_capture": selected_severity / full_severity if full_severity > 0 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def metric_gain(candidate: dict[str, object], reference: dict[str, object], metric: str) -> float:
    if metric in {"brier", "ece_10", "log_loss"}:
        return float(reference[metric]) - float(candidate[metric])
    return float(candidate[metric]) - float(reference[metric])


def paired_bootstrap(
    predictions: pd.DataFrame,
    budgets: list[float],
    reps: int,
    seed: int,
    reference_model: str = GT_AFRE_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reps <= 0:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    key_cols = ["airport", "utc_hour", "month", "arrivals"]

    for target in sorted(predictions["target"].unique()):
        success_col = TARGETS[str(target)]
        target_pred = predictions[predictions["target"].eq(target)].copy()
        target_pred["utc_hour"] = pd.to_datetime(target_pred["utc_hour"])
        target_pred["airport_day"] = target_pred["airport"].astype(str) + "_" + target_pred["utc_hour"].dt.strftime("%Y-%m-%d")
        ref = target_pred[target_pred["model"].eq(reference_model)][key_cols + [success_col, "airport_day", "pred_prob"]].rename(
            columns={"pred_prob": "ref_prob"}
        )
        if ref.empty:
            continue
        for candidate_model in sorted(set(target_pred["model"]) - {reference_model}):
            cand = target_pred[target_pred["model"].eq(candidate_model)][key_cols + [success_col, "airport_day", "pred_prob"]].rename(
                columns={"pred_prob": "cand_prob"}
            )
            aligned = cand.merge(
                ref,
                on=key_cols + [success_col, "airport_day"],
                how="inner",
                validate="one_to_one",
            )
            if aligned.empty:
                continue
            clusters = aligned["airport_day"].drop_duplicates().to_numpy()
            cluster_index = {cluster: np.flatnonzero(aligned["airport_day"].to_numpy() == cluster) for cluster in clusters}
            full_candidate = extended_metric_row(
                target,
                candidate_model,
                aligned.assign(pred_prob=aligned["cand_prob"]),
                success_col,
                budgets,
            )
            full_reference = extended_metric_row(
                target,
                reference_model,
                aligned.assign(pred_prob=aligned["ref_prob"]),
                success_col,
                budgets,
            )
            observed = {metric: metric_gain(full_candidate, full_reference, metric) for metric in BOOTSTRAP_METRICS}
            draws = {metric: [] for metric in BOOTSTRAP_METRICS}
            for rep in range(reps):
                sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
                sampled_idx = np.concatenate([cluster_index[cluster] for cluster in sampled_clusters])
                sample = aligned.iloc[sampled_idx]
                sample_candidate = extended_metric_row(
                    target,
                    candidate_model,
                    sample.assign(pred_prob=sample["cand_prob"]),
                    success_col,
                    budgets,
                )
                sample_reference = extended_metric_row(
                    target,
                    reference_model,
                    sample.assign(pred_prob=sample["ref_prob"]),
                    success_col,
                    budgets,
                )
                for metric in BOOTSTRAP_METRICS:
                    value = metric_gain(sample_candidate, sample_reference, metric)
                    draws[metric].append(value)
                    draw_rows.append(
                        {
                            "target": target,
                            "candidate_model": candidate_model,
                            "reference_model": reference_model,
                            "metric": metric,
                            "rep": rep,
                            "gain": value,
                        }
                    )
            for metric, values in draws.items():
                arr = np.asarray(values, dtype=float)
                summary_rows.append(
                    {
                        "target": target,
                        "candidate_model": candidate_model,
                        "reference_model": reference_model,
                        "metric": metric,
                        "observed_gain": observed[metric],
                        "mean_gain": float(np.nanmean(arr)),
                        "ci_low": float(np.nanpercentile(arr, 2.5)),
                        "ci_high": float(np.nanpercentile(arr, 97.5)),
                        "bootstrap_reps": reps,
                        "clusters": int(len(clusters)),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def add_gain_table(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for ref_name, prefix in [(RELATION_MODEL, "relation"), (GT_AFRE_MODEL, "gt_afre")]:
        metric_cols = [
            col
            for col in [
                "auc",
                "pr_auc",
                "brier",
                "ece_10",
                "top1_capture",
                "top5_capture",
                "top10_capture",
                "top20_capture",
                "top30_capture",
            ]
            if col in out.columns
        ]
        ref = out[out["model"].eq(ref_name)][["target"] + metric_cols].rename(
            columns={col: f"{prefix}_{col}" for col in metric_cols}
        )
        out = out.merge(ref, on="target", how="left")
        for col in metric_cols:
            gain_col = f"{col}_gain_vs_{prefix}"
            if col in {"brier", "ece_10", "log_loss"}:
                out[gain_col] = out[f"{prefix}_{col}"] - out[col]
            else:
                out[gain_col] = out[col] - out[f"{prefix}_{col}"]
        if f"ece_10_gain_vs_{prefix}" in out.columns:
            out[f"ece10_gain_vs_{prefix}"] = out[f"ece_10_gain_vs_{prefix}"]
    return out


def prediction_frame(
    test: pd.DataFrame,
    target: str,
    model_name: str,
    fold_id: str,
    success_col: str,
    prob: np.ndarray,
) -> pd.DataFrame:
    out = test[["airport", "utc_hour", "month", "arrivals", success_col]].copy()
    for extra_col in [
        "arr_delay60_count",
        SEVERE_DELAY_COL,
        "cancel_count",
        "divert_count",
        SEVERITY_COL,
        "mean_arr_delay",
        "p90_arr_delay",
    ]:
        if extra_col in test.columns and extra_col not in out.columns:
            out[extra_col] = pd.to_numeric(test[extra_col], errors="coerce").fillna(0.0).to_numpy(float)
    out["target"] = target
    out["model"] = model_name
    out["fold_id"] = fold_id
    out["pred_prob"] = np.clip(prob, 1e-5, 1 - 1e-5)
    return out


def run_raeg_experiment(
    months: list[int],
    airports: list[str] | None,
    output_name: str,
    models: list[str],
    validation: str,
    first_test_month: int,
    min_train_months: int,
    config: RAEGConfig,
    include_gt_afre: bool = True,
    budgets: list[float] | None = None,
    bootstrap_reps: int = 0,
    bootstrap_seed: int = 20260611,
    panel_path: str | None = None,
    relation_path: str | None = None,
) -> Path:
    out = OUT_ROOT / output_name
    out.mkdir(parents=True, exist_ok=True)
    budgets = budgets or BUDGET_FRACTIONS
    base_panel = load_experiment_panel(months, airports, panel_path)
    metrics_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    relation_prediction_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    target_availability_rows: list[dict[str, object]] = []

    for target, success_col in TARGETS.items():
        if success_col not in base_panel.columns:
            target_availability_rows.append(
                {"target": target, "success_col": success_col, "available": False, "reason": "missing_success_column"}
            )
            continue
        target_availability_rows.append({"target": target, "success_col": success_col, "available": True, "reason": ""})
        panel = base_panel.merge(load_relation_scores(months, target, 1, relation_path), on=["airport", "utc_hour", "month"], how="inner")
        panel["target_name"] = target
        fold_payloads = [
            (fold_id, raw_train.assign(target_name=target), raw_test.assign(target_name=target))
            for fold_id, raw_train, raw_test in raeg_validation_folds(panel, validation, first_test_month, min_train_months)
            if not raw_train.empty and not raw_test.empty
        ]
        if not fold_payloads:
            raise ValueError(f"No folds for {target}; months={months}, validation={validation}.")

        relation_parts = []
        for fold_id, _, raw_test in fold_payloads:
            fold = prediction_frame(raw_test, target, RELATION_MODEL, fold_id, success_col, raw_test["relation_score"].to_numpy(float))
            relation_parts.append(fold)
            fold_rows.append(extended_metric_row(target, RELATION_MODEL, fold, success_col, budgets) | {"fold_id": fold_id})
        relation_pred = pd.concat(relation_parts, ignore_index=True)
        metrics_rows.append(extended_metric_row(target, RELATION_MODEL, relation_pred, success_col, budgets))
        relation_prediction_parts.append(relation_pred)

        gt_anchor: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        gt_parts = []
        for fold_id, raw_train, raw_test in fold_payloads:
            train_gt, test_gt = fold_prepare(raw_train, raw_test, target, success_col, graph_source="target_correlation")
            features = feature_sets(train_gt)[GT_AFRE_MODEL]
            gt_train_prob, gt_test_prob = fit_gt_afre_probability_train_test(train_gt, test_gt, success_col, features)
            gt_anchor[fold_id] = (gt_train_prob, gt_test_prob)
            if include_gt_afre:
                fold = prediction_frame(test_gt, target, GT_AFRE_MODEL, fold_id, success_col, gt_test_prob)
                gt_parts.append(fold)
                fold_rows.append(extended_metric_row(target, GT_AFRE_MODEL, fold, success_col, budgets) | {"fold_id": fold_id})
                print(f"finished {target} {fold_id} {GT_AFRE_MODEL}", flush=True)
        if include_gt_afre:
            gt_pred = pd.concat(gt_parts, ignore_index=True)
            metrics_rows.append(extended_metric_row(target, GT_AFRE_MODEL, gt_pred, success_col, budgets))
            prediction_parts.append(gt_pred)

        for model_name in models:
            model_parts = []
            for fold_id, raw_train, raw_test in fold_payloads:
                anchor_train, anchor_test = gt_anchor[fold_id]
                result = fit_predict_raeg(
                    raw_train,
                    raw_test,
                    success_col,
                    model_name,
                    config,
                    anchor_train_prob=anchor_train,
                    anchor_test_prob=anchor_test,
                )
                fold = prediction_frame(raw_test, target, model_name, fold_id, success_col, result.test_prob)
                model_parts.append(fold)
                fold_rows.append(extended_metric_row(target, model_name, fold, success_col, budgets) | {"fold_id": fold_id})
                audit_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "fold_id": fold_id,
                        "device": result.device,
                        "feature_count": len(result.feature_columns),
                        "final_loss": result.final_loss,
                        "blend": result.blend,
                        "mean_raw_prob": float(np.mean(result.raw_test_prob)),
                        "mean_calibrated_prob": float(np.mean(result.test_prob)),
                    }
                )
                print(f"finished {target} {fold_id} {model_name}", flush=True)
            pred = pd.concat(model_parts, ignore_index=True)
            metrics_rows.append(extended_metric_row(target, model_name, pred, success_col, budgets))
            prediction_parts.append(pred)

    if not prediction_parts and not relation_prediction_parts:
        raise ValueError("No target could be evaluated because required success columns were unavailable.")

    metrics = pd.DataFrame(metrics_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    gains = add_gain_table(metrics)
    predictions = pd.concat(relation_prediction_parts + prediction_parts, ignore_index=True)
    budget_curve = queue_budget_curve(predictions, budgets)
    bootstrap_summary, bootstrap_draws = paired_bootstrap(
        predictions,
        budgets=budgets,
        reps=bootstrap_reps,
        seed=bootstrap_seed,
    )
    metrics.to_csv(out / "raeg_rank_metrics.csv", index=False)
    fold_metrics.to_csv(out / "raeg_rank_fold_metrics.csv", index=False)
    gains.to_csv(out / "raeg_rank_gains.csv", index=False)
    budget_curve.to_csv(out / "raeg_rank_budget_curve.csv", index=False)
    predictions.to_csv(out / "raeg_rank_predictions.csv", index=False)
    if not bootstrap_summary.empty:
        bootstrap_summary.to_csv(out / "raeg_rank_bootstrap_summary.csv", index=False)
        bootstrap_draws.to_csv(out / "raeg_rank_bootstrap_draws.csv", index=False)
    pd.DataFrame(target_availability_rows).to_csv(out / "raeg_rank_target_availability.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out / "raeg_rank_training_audit.csv", index=False)
    write_assessment(out, gains, bootstrap_summary, months, airports, validation, config)
    write_manifest(out, months, airports, models, validation, config, budgets, bootstrap_reps, bootstrap_seed, panel_path, relation_path)
    return out


def write_manifest(
    out: Path,
    months: list[int],
    airports: list[str] | None,
    models: list[str],
    validation: str,
    config: RAEGConfig,
    budgets: list[float],
    bootstrap_reps: int,
    bootstrap_seed: int,
    panel_path: str | None,
    relation_path: str | None,
) -> None:
    manifest = {
        "months": months,
        "airports": airports if airports is not None else "ALL",
        "models": models,
        "validation": validation,
        "budget_fractions": budgets,
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "panel_path": panel_path or "default temporal availability panel",
        "relation_path": relation_path or "default online lead DCSI predictions",
        "config": config.__dict__,
    }
    (out / "raeg_rank_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_assessment(
    out: Path,
    gains: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    months: list[int],
    airports: list[str] | None,
    validation: str,
    config: RAEGConfig,
) -> None:
    lines = [
        "# RAEG-Rank smoke assessment",
        "",
        f"Scope: months {','.join(str(m) for m in months)}; airports {','.join(airports) if airports else 'ALL'}; validation {validation}.",
        f"Training: epochs {config.epochs}; batch size {config.batch_size}; CUDA required {config.require_cuda}.",
        "",
        "Gate: RAEG-Rank should improve PR-AUC or top-10 capture versus GT-AFRE for at least one outcome without a material ECE loss.",
        "",
    ]
    usable_any = False
    for target in sorted(gains["target"].unique()):
        group = gains[gains["target"].eq(target)].copy()
        if MODEL_NAME not in set(group["model"]):
            continue
        row = group[group["model"].eq(MODEL_NAME)].iloc[0]
        ece_loss = -row["ece10_gain_vs_gt_afre"]
        usable = (
            row["pr_auc_gain_vs_gt_afre"] > 0
            or row["top10_capture_gain_vs_gt_afre"] > 0
        ) and ece_loss <= 0.01
        usable_any = usable_any or usable
        verdict = "usable for expansion" if usable else "diagnostic only"
        lines.append(f"## {target}")
        lines.append(
            f"- Verdict: {verdict}. RAEG-Rank vs GT-AFRE: AUC {row['auc_gain_vs_gt_afre']:+.4f}, "
            f"PR-AUC {row['pr_auc_gain_vs_gt_afre']:+.4f}, Brier {row['brier_gain_vs_gt_afre']:+.5f}, "
            f"ECE {row['ece10_gain_vs_gt_afre']:+.5f}, top-10 capture {row['top10_capture_gain_vs_gt_afre']:+.4f}."
        )
        if "top5_capture_gain_vs_gt_afre" in row and "top20_capture_gain_vs_gt_afre" in row:
            lines.append(
                f"- Queue width check: top-5 gain {row['top5_capture_gain_vs_gt_afre']:+.4f}; "
                f"top-20 gain {row['top20_capture_gain_vs_gt_afre']:+.4f}."
            )
        if not bootstrap_summary.empty:
            use_boot = bootstrap_summary[
                bootstrap_summary["target"].eq(target)
                & bootstrap_summary["candidate_model"].eq(MODEL_NAME)
                & bootstrap_summary["metric"].isin(["pr_auc", "top10_capture"])
            ]
            for boot in use_boot.itertuples(index=False):
                lines.append(
                    f"- Bootstrap {boot.metric}: observed gain {boot.observed_gain:+.4f}, "
                    f"95% CI [{boot.ci_low:+.4f}, {boot.ci_high:+.4f}] from {boot.bootstrap_reps} airport-day resamples."
                )
        lines.append("")
    lines.append(f"Overall expansion gate: {'pass' if usable_any else 'hold for method tuning'}.")
    lines.append("")
    gains.to_markdown(out / "raeg_rank_gains.md", index=False)
    (out / "raeg_rank_assessment.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser(default_output: str, default_months: str, default_airports: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default=default_months)
    parser.add_argument("--airports", default=default_airports)
    parser.add_argument("--output-name", default=default_output)
    parser.add_argument("--models", default=MODEL_NAME)
    parser.add_argument("--validation", choices=["month", "rolling", "airport_group", "rolling_quarter"], default="month")
    parser.add_argument("--first-test-month", type=int, default=4)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--severity-queue-weight", type=float, default=0.35)
    parser.add_argument("--budget-fractions", default="0.01,0.05,0.10,0.20,0.30")
    parser.add_argument("--bootstrap-reps", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260611)
    parser.add_argument("--panel-path", default="")
    parser.add_argument("--relation-path", default="")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--skip-gt-afre", action="store_true")
    return parser


def run_from_args(args: argparse.Namespace) -> Path:
    config = RAEGConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        severity_queue_weight=args.severity_queue_weight,
        require_cuda=not args.allow_cpu,
    )
    return run_raeg_experiment(
        months=parse_int_list(args.months),
        airports=parse_airports(args.airports),
        output_name=args.output_name,
        models=parse_models(args.models),
        validation=args.validation,
        first_test_month=args.first_test_month,
        min_train_months=args.min_train_months,
        config=config,
        include_gt_afre=not args.skip_gt_afre,
        budgets=parse_float_list(args.budget_fractions),
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
        panel_path=args.panel_path or None,
        relation_path=args.relation_path or None,
    )
