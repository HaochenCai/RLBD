"""
Cut selection: feature computation + probabilistic sampling.

- compute_cut_features_batch: features for NN inputs (commander).
- sample_indices_from_probs: sample k indices without replacement from probs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_EPS = 1e-9
CUT_FEATURE_DIM = 5


def sample_indices_from_probs(
    probs: np.ndarray,
    k: int = 5,
    seed: Optional[int] = None,
) -> List[int]:
    """
    Sample k distinct indices without replacement from probs.
    Empty list if n=0; all indices if n<=k.
    If NaN/inf or non-positive sum, fall back to uniform; negatives clipped then normalized.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    n = len(probs)
    if n == 0:
        return []
    k = min(k, n)
    if k == 0:
        return []

    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)) or np.all(probs <= 0):
        probs = np.ones(n, dtype=np.float64) / n
    else:
        probs = np.maximum(probs, 0.0)
        s = probs.sum()
        if s <= 0:
            probs = np.ones(n, dtype=np.float64) / n
        else:
            probs = probs / s

    rng = np.random.default_rng(seed)
    selected: List[int] = []
    probs_copy = probs.copy()

    for _ in range(k):
        idx = int(rng.choice(n, p=probs_copy))
        selected.append(idx)
        probs_copy[idx] = 0.0
        s = probs_copy.sum()
        if s <= 0:
            break
        probs_copy = probs_copy / s

    return selected


def sample_indices_from_logits_sequential(
    logits: torch.Tensor,
    k: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[int], torch.Tensor]:
    """
    Sequential masked sampling without replacement from logits.
    Re-softmax each step; total_log_prob sums log-probs from the same distributions.
    """
    logits = logits.reshape(-1)
    n = int(logits.numel())
    if n == 0 or k <= 0:
        return [], torch.tensor(0.0, device=logits.device)

    max_select = min(k, n)
    mask = torch.zeros_like(logits)
    total_log_prob = torch.tensor(0.0, device=logits.device)
    selected: List[int] = []

    rng_state = None
    if seed is not None:
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(int(seed))

    try:
        for _ in range(max_select):
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=0)
            dist = torch.distributions.Categorical(probs=probs)
            sampled_idx = dist.sample()
            total_log_prob = total_log_prob + dist.log_prob(sampled_idx)
            idx = int(sampled_idx.item())
            selected.append(idx)
            mask[idx] = -1e9
    finally:
        if rng_state is not None:
            torch.random.set_rng_state(rng_state)

    return selected, total_log_prob


def compute_cut_features(
    cut: Dict[str, Any],
    z_current: np.ndarray,
    theta_k_current: Optional[float] = None,
    nc_value: float = 0.0,
) -> np.ndarray:
    """Feature vector for one cut."""
    rhs = float(cut.get("rhs_const", 0.0))
    coef_z = np.asarray(cut.get("coef_z", []), dtype=float)
    alpha = np.asarray(cut.get("alpha", []), dtype=float)
    beta = np.asarray(cut.get("beta", []), dtype=float)

    if z_current.size != coef_z.size:
        z_current = np.broadcast_to(z_current, coef_z.shape)[: coef_z.size]
    z_current = np.asarray(z_current, dtype=float).flat[: coef_z.size]
    coef_z = coef_z.flat[: z_current.size]

    rhs_at_z = rhs + float(np.dot(coef_z, z_current))
    if theta_k_current is not None:
        violation = max(0.0, rhs_at_z - theta_k_current)
    else:
        violation = max(0.0, rhs_at_z)

    ab = np.concatenate([alpha.ravel(), beta.ravel()])
    dual_norm = float(np.sqrt(np.sum(ab * ab)) + _EPS)
    rhs_magnitude = abs(rhs) + _EPS
    coef_norm = float(np.sqrt(np.sum(coef_z * coef_z)) + _EPS)

    try:
        nc_val = float(nc_value)
    except (TypeError, ValueError):
        nc_val = 0.0
    if not np.isfinite(nc_val):
        nc_val = 0.0
    nc_val = max(0.0, nc_val)

    return np.array([violation, dual_norm, rhs_magnitude, coef_norm, nc_val], dtype=np.float64)


def compute_cut_features_batch(
    cuts: List[Dict[str, Any]],
    z_current: np.ndarray,
    theta_val: Optional[np.ndarray] = None,
    nc_per_cut: Optional[np.ndarray] = None,
    debug: bool = False,
) -> np.ndarray:
    """Batch cut features (n, CUT_FEATURE_DIM); log1p then per-column normalization."""
    n = len(cuts)
    out = np.zeros((n, CUT_FEATURE_DIM), dtype=np.float64)
    for i, c in enumerate(cuts):
        theta_k = float(theta_val[c["k"]]) if theta_val is not None and "k" in c else None
        if nc_per_cut is not None and i < len(nc_per_cut):
            nc_i = float(nc_per_cut[i])
        else:
            nc_i = 0.0
        out[i, :] = compute_cut_features(c, z_current, theta_k, nc_value=nc_i)

    # 1) log1p scale
    out = np.log1p(np.maximum(out, 0.0))

    # 2) per-column normalize
    mean = out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True)
    out = (out - mean) / (std + 1e-6)

    if debug:
        print("cut_features (normalized) stats: min={} max={} mean={}".format(
            out.min(), out.max(), out.mean()
        ))

    return out


__all__ = [
    "CUT_FEATURE_DIM",
    "compute_cut_features",
    "compute_cut_features_batch",
    "sample_indices_from_probs",
    "sample_indices_from_logits_sequential",
]
