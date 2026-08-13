"""Bradley-Terry global rank solver."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from closer.config import RankingConfig
from closer.errors import RankingSolverError
from closer.ranking.graph import PreferenceGraph
from closer.schemas.ranking import SolverResult


def clamp_confidence(confidence: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, confidence))


def observation_weight(confidence: float, config: RankingConfig) -> float:
    c = clamp_confidence(confidence, config.confidence_clamp_min, config.confidence_clamp_max)
    if config.weight_mode == "confidence":
        return c
    return -math.log(1.0 - c)


def fit_bradley_terry(
    variant_ids: list[str],
    graph: PreferenceGraph,
    config: RankingConfig,
) -> SolverResult:
    if not variant_ids:
        raise RankingSolverError("no variants to rank")
    ids = list(variant_ids)
    index = {vid: i for i, vid in enumerate(ids)}
    n = len(ids)

    winners: list[int] = []
    losers: list[int] = []
    weights: list[float] = []
    n_obs = 0
    for obs in graph.observations:
        n_obs += 1
        if obs.winner == "uncertain":
            continue
        if obs.winner == "tie":
            if config.tie_mode == "ignore":
                continue
            # Weak 0.5 constraint implemented as two tiny opposite edges.
            i = index[obs.left_id]
            j = index[obs.right_id]
            w = 0.05 * observation_weight(max(obs.confidence, 0.5), config)
            winners.extend([i, j])
            losers.extend([j, i])
            weights.extend([w, w])
            continue
        winner = obs.left_id if obs.winner == "left" else obs.right_id
        loser = obs.right_id if obs.winner == "left" else obs.left_id
        winners.append(index[winner])
        losers.append(index[loser])
        weights.append(observation_weight(obs.confidence, config))

    if not winners:
        scores = {vid: 0.0 for vid in ids}
        ranking = sorted(ids)
        return SolverResult(
            ranking=ranking,
            scores=scores,
            method="bradley_terry",
            converged=True,
            n_observations=n_obs,
            n_informative_edges=0,
        )

    w_idx = np.asarray(winners, dtype=int)
    l_idx = np.asarray(losers, dtype=int)
    ww = np.asarray(weights, dtype=float)
    l2 = config.l2_regularization

    def unpack(x: np.ndarray) -> np.ndarray:
        scores = np.zeros(n, dtype=float)
        scores[:-1] = x
        scores[-1] = -float(np.sum(x))
        return scores

    def objective(x: np.ndarray) -> float:
        s = unpack(x)
        diff = s[w_idx] - s[l_idx]
        nll = np.sum(ww * np.logaddexp(0.0, -diff))
        nll += l2 * float(np.dot(s, s))
        return float(nll)

    x0 = np.zeros(n - 1, dtype=float)
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": config.max_optimize_iter},
    )
    scores_vec = unpack(result.x)
    scores = {vid: float(scores_vec[i]) for i, vid in enumerate(ids)}
    ranking = sorted(ids, key=lambda vid: (-scores[vid], vid))
    return SolverResult(
        ranking=ranking,
        scores=scores,
        method="bradley_terry",
        converged=bool(result.success),
        n_observations=n_obs,
        n_informative_edges=len(winners),
    )


def borda_from_setwise(variant_ids: list[str], rankings: list[list[str]]) -> SolverResult:
    """Defined ablation aggregator when the BT solver is disabled."""
    scores = {vid: 0.0 for vid in variant_ids}
    for ranking in rankings:
        n = len(ranking)
        for pos, vid in enumerate(ranking):
            scores[vid] += float(n - pos)
    order = sorted(variant_ids, key=lambda vid: (-scores[vid], vid))
    return SolverResult(
        ranking=order,
        scores=scores,
        method="borda",
        converged=True,
        n_observations=len(rankings),
        n_informative_edges=sum(max(0, len(r) - 1) for r in rankings),
    )
