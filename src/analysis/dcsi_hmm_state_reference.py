from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from dynamic_constraint_state_inversion import (
    BASE_NUMERIC,
    FIXED_NUMERIC,
    MAIN_10,
    MAIN_2025_PANEL,
    MAIN_CATS,
    TARGETS,
    fit_predict,
    load_panel,
    model_rows,
    parse_months,
)
from fusion_prediction_increment import evaluate
from fusion_strengthening_common import ROOT_OUT


OUT_ROOT = ROOT_OUT / "dcsi_hmm_state_reference"
EPS = 1e-8


@dataclass
class GaussianHMM:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def parse_airports(text: str) -> list[str] | None:
    if text.strip().upper() in {"ALL", "*"}:
        return None
    return [x.strip().upper() for x in text.split(",") if x.strip()]


def add_relation_fields(panel: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    frames = []
    for _, g in panel.sort_values(["airport", "utc_hour"]).groupby("airport", sort=False):
        g = g.copy()
        active = (g["active_minutes"].to_numpy(float) > 0).astype(int)
        since = np.full(len(g), horizon + 1, dtype=float)
        run = horizon + 1
        for i, val in enumerate(active):
            if val == 1:
                run = 0
            else:
                run = min(horizon + 1, run + 1)
            since[i] = run
        g["active_impulse"] = (g["active_minutes"] / 60.0).clip(0, 1)
        g["hours_since_active"] = since
        g["recent_action_clock"] = np.where(since <= horizon, 1.0 / (1.0 + since), 0.0)
        g["post_relation_6h"] = ((since > 0) & (since <= 6)).astype(float)
        g["post_relation_12h"] = ((since > 0) & (since <= 12)).astype(float)
        g["post_relation_24h"] = ((since > 0) & (since <= 24)).astype(float)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def logit_rate(success: pd.Series, total: pd.Series) -> np.ndarray:
    rate = (pd.to_numeric(success, errors="coerce").fillna(0.0) + 0.5) / (
        pd.to_numeric(total, errors="coerce").fillna(0.0) + 1.0
    )
    rate = np.clip(rate.to_numpy(float), 1e-5, 1 - 1e-5)
    return np.log(rate / (1 - rate))


def feature_matrix(panel: pd.DataFrame, mode: str) -> tuple[np.ndarray, list[str]]:
    if mode == "action":
        cols = ["active_impulse", "hours_since_active", "recent_action_clock", "post_relation_6h", "post_relation_12h", "post_relation_24h"]
        x = panel[cols].copy()
        x["hours_since_active"] = np.log1p(x["hours_since_active"].clip(0, 25))
        return x.to_numpy(float), cols
    if mode == "closure":
        x = pd.DataFrame(
            {
                "active_impulse": panel["active_impulse"].to_numpy(float),
                "delay_logit": logit_rate(panel["arr_delay60_count"], panel["arrivals"]),
                "cancel_logit": logit_rate(panel["cancel_count"], panel["arrivals"]),
            },
            index=panel.index,
        )
        return x.to_numpy(float), x.columns.tolist()
    raise ValueError(f"unknown HMM mode: {mode}")


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def split_sequences(x: np.ndarray, groups: pd.Series) -> list[np.ndarray]:
    seqs = []
    for _, idx in groups.groupby(groups, sort=False).groups.items():
        seqs.append(x[np.asarray(list(idx), dtype=int)])
    return seqs


def log_gaussian_diag(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    var = np.maximum(variances, 1e-4)
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (np.sum(np.log(2 * np.pi * var), axis=1)[None, :] + np.sum(diff * diff / var[None, :, :], axis=2))


def forward_backward(log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n, k = log_b.shape
    log_start = np.log(np.maximum(startprob, EPS))
    log_trans = np.log(np.maximum(transmat, EPS))
    alpha = np.empty((n, k), dtype=float)
    beta = np.empty((n, k), dtype=float)
    alpha[0] = log_start + log_b[0]
    for t in range(1, n):
        alpha[t] = log_b[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
    beta[-1] = 0.0
    for t in range(n - 2, -1, -1):
        beta[t] = logsumexp(log_trans + log_b[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    loglik = float(logsumexp(alpha[-1]))
    gamma = np.exp(alpha + beta - loglik)
    xi_sum = np.zeros((k, k), dtype=float)
    for t in range(n - 1):
        log_xi = alpha[t][:, None] + log_trans + log_b[t + 1][None, :] + beta[t + 1][None, :] - loglik
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, loglik


def fit_hmm(train_x: np.ndarray, train_groups: pd.Series, n_states: int, max_iter: int, seed: int) -> GaussianHMM:
    km = KMeans(n_clusters=n_states, n_init=10, random_state=seed)
    labels = km.fit_predict(train_x)
    means = km.cluster_centers_.astype(float)
    variances = np.vstack(
        [
            np.var(train_x[labels == s], axis=0) if np.any(labels == s) else np.var(train_x, axis=0)
            for s in range(n_states)
        ]
    )
    variances = np.maximum(variances, 0.05)
    startprob = np.full(n_states, 1.0 / n_states)
    transmat = np.full((n_states, n_states), 0.05 / max(1, n_states - 1))
    np.fill_diagonal(transmat, 0.95)

    seqs = split_sequences(train_x, train_groups.reset_index(drop=True))
    for _ in range(max_iter):
        gamma_sum = np.zeros(n_states)
        x_sum = np.zeros_like(means)
        x2_sum = np.zeros_like(means)
        start_sum = np.zeros(n_states)
        trans_sum = np.zeros_like(transmat)
        for seq in seqs:
            log_b = log_gaussian_diag(seq, means, variances)
            gamma, xi_sum, _ = forward_backward(log_b, startprob, transmat)
            start_sum += gamma[0]
            trans_sum += xi_sum
            gamma_sum += gamma.sum(axis=0)
            x_sum += gamma.T @ seq
            x2_sum += gamma.T @ (seq * seq)
        startprob = (start_sum + EPS) / (start_sum.sum() + EPS * n_states)
        transmat = trans_sum + EPS
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        means = x_sum / np.maximum(gamma_sum[:, None], EPS)
        variances = x2_sum / np.maximum(gamma_sum[:, None], EPS) - means * means
        variances = np.maximum(variances, 0.03)
    return GaussianHMM(startprob=startprob, transmat=transmat, means=means, variances=variances)


def posterior(hmm: GaussianHMM, x: np.ndarray, groups: pd.Series) -> np.ndarray:
    out = np.zeros((len(x), hmm.means.shape[0]), dtype=float)
    groups = groups.reset_index(drop=True)
    for _, idx in groups.groupby(groups, sort=False).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        log_b = log_gaussian_diag(x[idx], hmm.means, hmm.variances)
        gamma, _, _ = forward_backward(log_b, hmm.startprob, hmm.transmat)
        out[idx] = gamma
    return out


def order_states_by_training_closure(train: pd.DataFrame, gamma: np.ndarray, n_states: int) -> np.ndarray:
    arrivals = train["arrivals"].to_numpy(float)
    delay = train["arr_delay60_count"].to_numpy(float)
    cancel = train["cancel_count"].to_numpy(float)
    risk = []
    for state in range(n_states):
        weight = gamma[:, state] * arrivals
        denom = max(float(weight.sum()), EPS)
        delay_rate = float((gamma[:, state] * delay).sum() / denom)
        cancel_rate = float((gamma[:, state] * cancel).sum() / denom)
        risk.append(delay_rate + cancel_rate)
    return np.argsort(np.asarray(risk, dtype=float))


def attach_hmm_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mode: str,
    n_states: int,
    max_iter: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_x_raw, _ = feature_matrix(train, mode)
    test_x_raw, _ = feature_matrix(test, mode)
    train_x, mean, std = standardize(train_x_raw, train_x_raw)
    test_x = (test_x_raw - mean) / std
    hmm = fit_hmm(train_x, train["airport"], n_states=n_states, max_iter=max_iter, seed=seed)
    train_post = posterior(hmm, train_x, train["airport"])
    order = order_states_by_training_closure(train, train_post, n_states)
    high_state = int(order[-1])
    test_post = posterior(hmm, test_x, test["airport"])
    state_scores = np.arange(n_states, dtype=float)
    state_rank = np.empty(n_states, dtype=float)
    state_rank[order] = state_scores
    train = train.copy()
    test = test.copy()
    train["hmm_high_prob"] = train_post[:, high_state]
    test["hmm_high_prob"] = test_post[:, high_state]
    train["hmm_expected_state"] = train_post @ state_rank
    test["hmm_expected_state"] = test_post @ state_rank
    info = {
        "high_state": high_state,
        "self_transition_high": float(hmm.transmat[high_state, high_state]),
        "states": float(n_states),
    }
    return train, test, info


def evaluate_state_auc(df: pd.DataFrame, score_col: str) -> list[dict[str, object]]:
    rows = []
    for target, success_col in TARGETS.items():
        metrics = evaluate(df[success_col].to_numpy(float), df["arrivals"].to_numpy(float), df[score_col].to_numpy(float))
        rows.append({"target": target, "score": score_col, **metrics})
    return rows


def run(args: argparse.Namespace) -> None:
    months = parse_months(args.months)
    airports = parse_airports(args.airports)
    out = OUT_ROOT / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    panel = model_rows(add_relation_fields(load_panel(MAIN_2025_PANEL, 2025, months, airports or MAIN_10)))
    metric_rows = []
    state_rows = []
    fold_infos = []
    predictions = []
    for outer_month in months:
        train = panel[panel["month"] != outer_month].copy()
        test = panel[panel["month"] == outer_month].copy()
        if train.empty or test.empty:
            continue
        action_train, action_test, action_info = attach_hmm_features(
            train, test, "action", args.states, args.max_iter, args.seed + outer_month
        )
        closure_train, closure_test, closure_info = attach_hmm_features(
            train, test, "closure", args.states, args.max_iter, args.seed + 100 + outer_month
        )
        action_info.update({"fold_month": outer_month, "mode": "action_only"})
        closure_info.update({"fold_month": outer_month, "mode": "closure_informed"})
        fold_infos.extend([action_info, closure_info])
        for target, success_col in TARGETS.items():
            pred, metrics = fit_predict(
                action_train,
                action_test,
                success_col,
                FIXED_NUMERIC + ["hmm_high_prob", "hmm_expected_state"],
                MAIN_CATS,
            )
            metric_rows.append(metrics | {"target": target, "model": "hmm_action_state", "fold_month": outer_month})
            pred["target"] = target
            pred["model"] = "hmm_action_state"
            predictions.append(pred)
        action_eval = action_test[["row_id", "airport", "utc_hour", "month", "arrivals", "arr_delay60_count", "cancel_count", "hmm_high_prob", "hmm_expected_state"]].copy()
        closure_eval = closure_test[["row_id", "airport", "utc_hour", "month", "arrivals", "arr_delay60_count", "cancel_count", "hmm_high_prob", "hmm_expected_state"]].copy()
        action_eval["mode"] = "action_only"
        closure_eval["mode"] = "closure_informed"
        state_rows.append(action_eval)
        state_rows.append(closure_eval)
    pred_all = pd.concat(predictions, ignore_index=True)
    for target, g in pred_all.groupby("target"):
        success_col = TARGETS[target]
        metric_rows.append(
            evaluate(g[success_col].to_numpy(float), g["arrivals"].to_numpy(float), g["pred_prob"].to_numpy(float))
            | {"target": target, "model": "hmm_action_state", "fold_month": "all"}
        )
    states = pd.concat(state_rows, ignore_index=True)
    state_auc = []
    for mode, g in states.groupby("mode"):
        for row in evaluate_state_auc(g, "hmm_high_prob"):
            row["mode"] = mode
            state_auc.append(row)
    pd.DataFrame(metric_rows).to_csv(out / "hmm_action_state_metrics.csv", index=False)
    pd.DataFrame(fold_infos).to_csv(out / "hmm_state_fold_info.csv", index=False)
    pd.DataFrame(state_auc).to_csv(out / "hmm_state_auc.csv", index=False)
    states.to_csv(out / "hmm_state_scores.csv", index=False)
    pred_all.to_csv(out / "hmm_action_state_predictions.csv", index=False)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1,7,12")
    parser.add_argument("--airports", default="ATL,ORD")
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="smoke")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
