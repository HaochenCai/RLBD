"""
Reward: log-gap improvement with RMP-time penalty.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

DEFAULT_EPS = 1e-8
DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 0.01
DEFAULT_TREF = 0.01
DEFAULT_LAMBDA_PENALTY = 0.01

_RMP_TIME_KEYS = [
    "solve_time_master",
    "rmp_solve_time",
    "master_solve_time",
]


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _search_keys(mapping: Any, keys: list[str]) -> Optional[float]:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            value = _to_float(mapping.get(key))
            if value is not None:
                return value
    return None


def log_gap_rmp_time_penalty_reward(gap_old: float, gap_new: float, info: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
    eps = DEFAULT_EPS
    alpha = DEFAULT_ALPHA
    beta = DEFAULT_BETA
    t_ref = DEFAULT_TREF
    lambda_penalty = DEFAULT_LAMBDA_PENALTY

    if isinstance(info, dict):
        v = _to_float(info.get("reward_eps"))
        if v is not None and v > 0.0:
            eps = v
        v = _to_float(info.get("reward_alpha"))
        if v is not None:
            alpha = v
        v = _to_float(info.get("reward_beta"))
        if v is not None:
            beta = v
        v = _to_float(info.get("reward_tref"))
        if v is not None and v > 0.0:
            t_ref = v
        v = _to_float(info.get("reward_lambda_penalty"))
        if v is not None:
            lambda_penalty = max(0.0, v)

    gap_old_val = _to_float(gap_old)
    gap_new_val = _to_float(gap_new)

    rmp_time_val = _search_keys(info, _RMP_TIME_KEYS)
    if rmp_time_val is None:
        rmp_time_val = 0.0
    elif rmp_time_val < 0.0:
        rmp_time_val = 0.0

    if gap_old_val is None or gap_new_val is None or gap_old_val < 0.0 or gap_new_val < 0.0:
        reward = -lambda_penalty
        scaling_dict = {
            "scaling_reward_name": "log_gap_rmp_time_penalty",
            "scaling_global_raw": 0.0,
            "scaling_global_final": 0.0,
            "scaling_time_raw": 0.0,
            "scaling_time_final": 0.0,
            "scaling_penalty": -lambda_penalty,
            "scaling_gap_old": gap_old_val,
            "scaling_gap_new": gap_new_val,
            "scaling_rmp_time": rmp_time_val,
            "scaling_alpha": alpha,
            "scaling_beta": beta,
            "scaling_tref": t_ref,
            "scaling_lambda": lambda_penalty,
        }
        return float(reward), scaling_dict

    gap_old_safe = max(gap_old_val, eps)
    gap_new_safe = max(gap_new_val, eps)
    global_raw = math.log(gap_old_safe / gap_new_safe)
    global_final = alpha * global_raw

    time_raw = rmp_time_val / t_ref
    time_final = beta * time_raw

    penalty = -lambda_penalty
    reward = global_final - time_final + penalty

    scaling_dict = {
        "scaling_reward_name": "log_gap_rmp_time_penalty",
        "scaling_global_raw": global_raw,
        "scaling_global_final": global_final,
        "scaling_time_raw": time_raw,
        "scaling_time_final": time_final,
        "scaling_penalty": penalty,
        "scaling_gap_old": gap_old_val,
        "scaling_gap_new": gap_new_val,
        "scaling_rmp_time": rmp_time_val,
        "scaling_alpha": alpha,
        "scaling_beta": beta,
        "scaling_tref": t_ref,
        "scaling_lambda": lambda_penalty,
    }
    return float(reward), scaling_dict


__all__ = ["log_gap_rmp_time_penalty_reward"]
