"""
Reward plugins for Benders MDP: inject reward_fn into BendersMDPEnv.

Env calls reward_fn(gap_old, gap_new, info) only; formulas live here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

RewardFn = Callable[[float, float, Dict[str, Any]], float]


def gap_reduction_reward(gap_old: float, gap_new: float, info: Dict[str, Any]) -> float:
    """Return gap_old - gap_new (larger is more gap reduction)."""
    return gap_old - gap_new


def zero_reward(gap_old: float, gap_new: float, info: Dict[str, Any]) -> float:
    """Always zero; for logging-only runs."""
    return 0.0


__all__ = ["RewardFn", "gap_reduction_reward", "zero_reward"]
