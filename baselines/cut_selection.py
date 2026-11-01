"""
Cut selection: score candidate Benders cuts and keep top-SELECTION_TOP_K per iteration.

Linear score w^T f; scorer_params may later supply NN weights.
Also supports random_k: uniform sample of SELECTION_TOP_K cuts (eval baseline).
"""

# Current cut selection policy:
# - score-based ranking using geometric features
# - select top SELECTION_TOP_K cuts each iteration (fewer if not enough candidates)
# - random_k: uniform sample of SELECTION_TOP_K cuts (no scoring)
# - scorer parameters are replaceable (NN-ready)
# Future extensions:
# - learned scoring weights
# - diversity-aware features
# - adaptive top-k or learned k

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

_EPS = 1e-9

CUT_FEATURE_DIM = 4


def compute_cut_features(
    cut: Dict[str, Any],
    z_current: np.ndarray,
    theta_k_current: Optional[float] = None,
) -> np.ndarray:
    """
    Fixed-size feature vector for one cut (numpy only).

    cut: rhs_const, coef_z, optional alpha, beta.
    z_current: first-stage z aligned with coef_z.
    theta_k_current: master theta for scenario k; if set, violation uses true slack.
    """
    rhs = float(cut.get("rhs_const", 0.0))
    coef_z = np.asarray(cut.get("coef_z", []), dtype=float)
    alpha = np.asarray(cut.get("alpha", []), dtype=float)
    beta = np.asarray(cut.get("beta", []), dtype=float)

    if z_current.size != coef_z.size:
        z_current = np.broadcast_to(z_current, coef_z.shape)[: coef_z.size]
    z_current = np.asarray(z_current, dtype=float).flat[: coef_z.size]
    coef_z = coef_z.flat[: z_current.size]

    # violation for cut theta_k >= rhs + coef_z^T z
    rhs_at_z = rhs + float(np.dot(coef_z, z_current))
    if theta_k_current is not None:
        violation = max(0.0, rhs_at_z - theta_k_current)
    else:
        violation = max(0.0, rhs_at_z)
    violation = float(violation)

    ab = np.concatenate([alpha.ravel(), beta.ravel()])
    dual_norm = float(np.sqrt(np.sum(ab * ab)) + _EPS)

    rhs_magnitude = abs(rhs) + _EPS

    coef_norm = float(np.sqrt(np.sum(coef_z * coef_z)) + _EPS)

    return np.array([violation, dual_norm, rhs_magnitude, coef_norm], dtype=np.float64)


def compute_cut_features_batch(
    cuts: List[Dict[str, Any]],
    z_current: np.ndarray,
    theta_val: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Stack features for n cuts -> (n, CUT_FEATURE_DIM)."""
    n = len(cuts)
    out = np.zeros((n, CUT_FEATURE_DIM), dtype=np.float64)
    for i, c in enumerate(cuts):
        theta_k = float(theta_val[c["k"]]) if theta_val is not None and "k" in c else None
        out[i, :] = compute_cut_features(c, z_current, theta_k)
    return out


def compute_scores(
    cut_features: np.ndarray,
    state: Any,
    params: Optional[Union[Dict[str, Any], np.ndarray]] = None,
) -> np.ndarray:
    """
    Per-cut score = w^T features (higher = add first).
    cut_features: (n_t, d_f). state reserved. params: weights vector or dict with "weights".
    """
    d_f = cut_features.shape[1]
    if params is None:
        w = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64).ravel()
    elif isinstance(params, np.ndarray):
        w = np.asarray(params, dtype=np.float64).ravel()
    else:
        w = np.asarray(params.get("weights", DEFAULT_WEIGHTS), dtype=np.float64).ravel()
    if w.size < d_f:
        w = np.concatenate([w, np.zeros(d_f - w.size, dtype=np.float64)])
    w = w[:d_f]
    return np.asarray(cut_features @ w, dtype=np.float64)


DEFAULT_WEIGHTS = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

SELECTION_TOP_K = 10


def _selection_mode(scorer_params: Optional[Union[Dict[str, Any], np.ndarray]]) -> str:
    """Return 'random_k' or 'score' from scorer_params."""
    if isinstance(scorer_params, dict):
        mode = scorer_params.get("mode") or scorer_params.get("selection_mode")
        if mode == "random_k":
            return "random_k"
    return "score"


def select_random_cuts(
    cuts_t: List[Dict[str, Any]],
    k: int = SELECTION_TOP_K,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """Uniform sample of up to k cuts (no scoring)."""
    n_t = len(cuts_t)
    if n_t == 0:
        return []
    K_t = min(n_t, k)
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(n_t, size=K_t, replace=False)
    return [cuts_t[int(i)] for i in indices]


def select_cuts(
    cuts_t: List[Dict[str, Any]],
    z_current: np.ndarray,
    state_t: Optional[Any] = None,
    scorer_params: Optional[Union[Dict[str, Any], np.ndarray]] = None,
    theta_val: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Any]]:
    """
    Return up to SELECTION_TOP_K cuts.
    Default: highest scores. If scorer_params mode is random_k: uniform sample.
    """
    n_t = len(cuts_t)
    if n_t == 0:
        return []

    if _selection_mode(scorer_params) == "random_k":
        return select_random_cuts(cuts_t, k=SELECTION_TOP_K, rng=rng)

    cut_features = compute_cut_features_batch(cuts_t, z_current, theta_val)
    scores = compute_scores(cut_features, state_t, scorer_params)
    K_t = min(n_t, SELECTION_TOP_K)
    top_indices = np.argsort(scores)[::-1][:K_t]
    return [cuts_t[i] for i in top_indices]


class CutScorer:
    """Linear scorer; params may later come from NN."""

    def __init__(self, default_weights: Optional[np.ndarray] = None):
        self.default_weights = np.asarray(
            default_weights if default_weights is not None else DEFAULT_WEIGHTS,
            dtype=np.float64,
        )

    def compute_scores(
        self,
        cut_features: np.ndarray,
        state: Optional[Any] = None,
        params: Optional[Union[Dict, np.ndarray]] = None,
    ) -> np.ndarray:
        w = params if params is not None else self.default_weights
        return compute_scores(cut_features, state, w)


class CutSelector:
    """Thin wrapper; k capped by SELECTION_TOP_K in select_cuts / select_random_cuts."""

    def __init__(
        self,
        scorer: Optional[CutScorer] = None,
        seed: Optional[int] = None,
    ):
        self.scorer = scorer or CutScorer()
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        cuts_t: List[Dict[str, Any]],
        z_current: np.ndarray,
        state_t: Optional[Any] = None,
        scorer_params: Optional[Union[Dict, np.ndarray]] = None,
        theta_val: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        return select_cuts(
            cuts_t,
            z_current,
            state_t=state_t,
            scorer_params=scorer_params,
            theta_val=theta_val,
            rng=self._rng,
        )


__all__ = [
    "CUT_FEATURE_DIM",
    "DEFAULT_WEIGHTS",
    "SELECTION_TOP_K",
    "compute_cut_features",
    "compute_cut_features_batch",
    "compute_scores",
    "select_random_cuts",
    "select_cuts",
    "CutScorer",
    "CutSelector",
]
