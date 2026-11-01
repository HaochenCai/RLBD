from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from reward_log_gap_rmp_time_penalty import log_gap_rmp_time_penalty_reward

RewardFn = Callable[[float, float, Dict[str, Any]], Any]

REWARD_REGISTRY: Dict[str, RewardFn] = {
    "log_gap_rmp_time_penalty": log_gap_rmp_time_penalty_reward,
}


def get_reward_fn(name: str) -> RewardFn:
    if name not in REWARD_REGISTRY:
        available = ", ".join(sorted(REWARD_REGISTRY.keys()))
        raise ValueError(f"unknown reward_name={name!r}. available: {available}")
    return REWARD_REGISTRY[name]


def with_reward_overrides(reward_fn: RewardFn, overrides: Mapping[str, float] | None) -> RewardFn:
    if not overrides:
        return reward_fn
    override_dict = {str(k): float(v) for k, v in dict(overrides).items()}

    def _wrapped(gap_old: float, gap_new: float, info: Dict[str, Any]) -> Any:
        merged_info = dict(info) if isinstance(info, dict) else {}
        for k, v in override_dict.items():
            if k not in merged_info:
                merged_info[k] = v
        return reward_fn(gap_old, gap_new, merged_info)

    return _wrapped


__all__ = ["RewardFn", "REWARD_REGISTRY", "get_reward_fn", "with_reward_overrides"]
