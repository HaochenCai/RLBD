"""
Benders MDP state: fixed-size normalized vectors from trajectory snapshots.

For cut-selection / RL inputs. Uses logger snapshots only (no Gurobi handles).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


_EPS = 1e-9
_MAX_ITER_DEFAULT = 2000
_SCALE_LB_UB = 1e6
_SCALE_COST = 1e6
_SCALE_TIME = 300.0
_SCALE_CUTS = 500
_SCALE_CAPACITY = 1e4


@dataclass
class BendersState:
    """
    Scalar summary for one Benders iteration (no full y/z vectors).
    """

    iteration_index: int = 0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    optimality_gap: float = 0.0
    delta_lb: float = 0.0
    delta_ub: float = 0.0

    total_cuts_in_master: int = 0
    cuts_added_last_iter: int = 0
    cumulative_cuts_added: int = 0
    master_solve_time_last_iter: float = 0.0

    average_recourse_cost: float = 0.0
    max_recourse_cost: float = 0.0
    min_recourse_cost: float = 0.0
    recourse_std: float = 0.0

    num_open_stations: float = 0.0
    avg_capacity: float = 0.0
    max_capacity: float = 0.0
    capacity_utilization_proxy: float = 0.0

    gap_reduction_rate: float = 0.0
    lb_improvement_rate: float = 0.0
    ub_improvement_rate: float = 0.0

    mean_violation: float = 0.0
    max_violation: float = 0.0
    last_gap_reduction: float = 0.0

    extra_features: Dict[str, float] = field(default_factory=dict, repr=False)

    def to_vector(
        self,
        max_iter: int = _MAX_ITER_DEFAULT,
        scale_lb_ub: float = _SCALE_LB_UB,
        scale_cost: float = _SCALE_COST,
        scale_time: float = _SCALE_TIME,
        scale_cuts: int = _SCALE_CUTS,
        scale_capacity: float = _SCALE_CAPACITY,
    ) -> np.ndarray:
        """Fixed-length normalized feature vector for NN input."""
        def safe_div(a: float, b: float, default: float = 0.0) -> float:
            return a / (b + _EPS) if abs(b) > _EPS else default

        def clip01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        def clip11(x: float) -> float:
            return float(np.clip(x, -1.0, 1.0))

        v_iter = clip01(self.iteration_index / max(1, max_iter))
        v_lb = clip11(np.tanh(self.lower_bound / (scale_lb_ub + _EPS)))
        v_ub = clip11(np.tanh(self.upper_bound / (scale_lb_ub + _EPS)))
        v_gap = clip01(self.optimality_gap / (self.optimality_gap + 1.0 + _EPS))
        v_delta_lb = clip11(np.tanh(self.delta_lb / (scale_cost + _EPS)))
        v_delta_ub = clip11(np.tanh(self.delta_ub / (scale_cost + _EPS)))

        v_total_cuts = clip01(self.total_cuts_in_master / (scale_cuts + self.total_cuts_in_master))
        v_cuts_added = clip01(self.cuts_added_last_iter / (scale_cuts + self.cuts_added_last_iter))
        v_cumulative_cuts = clip01(self.cumulative_cuts_added / (scale_cuts + self.cumulative_cuts_added))
        v_master_time = clip01(self.master_solve_time_last_iter / (scale_time + _EPS))

        v_recourse_avg = clip11(np.tanh(self.average_recourse_cost / (scale_cost + _EPS)))
        v_recourse_max = clip11(np.tanh(self.max_recourse_cost / (scale_cost + _EPS)))
        v_recourse_min = clip11(np.tanh(self.min_recourse_cost / (scale_cost + _EPS)))
        v_recourse_std = clip01(self.recourse_std / (self.recourse_std + 1.0 + _EPS))

        v_num_open = clip01(self.num_open_stations / (20.0 + self.num_open_stations))
        v_avg_cap = clip01(np.tanh(self.avg_capacity / (scale_capacity + _EPS)) * 0.5 + 0.5)
        v_max_cap = clip01(np.tanh(self.max_capacity / (scale_capacity + _EPS)) * 0.5 + 0.5)
        v_util = clip01(self.capacity_utilization_proxy / (self.capacity_utilization_proxy + 1.0 + _EPS))

        v_gap_rate = clip11(self.gap_reduction_rate)
        v_lb_rate = clip11(self.lb_improvement_rate)
        v_ub_rate = clip11(self.ub_improvement_rate)

        v_mean_viol = clip01(self.mean_violation / (self.mean_violation + 1.0 + _EPS))
        v_max_viol = clip01(self.max_violation / (self.max_violation + 1.0 + _EPS))
        v_last_gap_red = clip11(self.last_gap_reduction)

        return np.array([
            v_iter,
            v_lb,
            v_ub,
            v_gap,
            v_delta_lb,
            v_delta_ub,
            v_total_cuts,
            v_cuts_added,
            v_cumulative_cuts,
            v_master_time,
            v_recourse_avg,
            v_recourse_max,
            v_recourse_min,
            v_recourse_std,
            v_num_open,
            v_avg_cap,
            v_max_cap,
            v_util,
            v_gap_rate,
            v_lb_rate,
            v_ub_rate,
            v_mean_viol,
            v_max_viol,
            v_last_gap_red,
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return 24


def _get_snapshots(logger: Union[Any, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize logger object or list[dict] into list of snapshot dicts."""
    if hasattr(logger, "get_trajectory"):
        trajectory = logger.get_trajectory()
        return [
            {
                "iteration": s.iteration,
                "lb": s.lb,
                "ub": s.ub,
                "gap": s.gap,
                "y_val": np.asarray(s.y_val),
                "z_val": np.asarray(s.z_val),
                "theta_val": np.asarray(s.theta_val),
                "first_stage_cost": s.first_stage_cost,
                "recourse_avg": s.recourse_avg,
                "Q_vals": np.asarray(s.Q_vals),
                "num_cuts_total": s.num_cuts_total,
                "master_num_constraints": getattr(s, "master_num_constraints", 0),
                "master_num_variables": getattr(s, "master_num_variables", 0),
                "master_solve_time": getattr(s, "master_solve_time", 0.0),
                "cuts_this_iter": getattr(s, "cuts_this_iter", []),
            }
            for s in trajectory
        ]
    if isinstance(logger, list) and logger and isinstance(logger[0], dict):
        return logger
    return []


def build_state(
    logger: Union[Any, List[Dict[str, Any]]],
    snapshot_index: Optional[int] = None,
    prev_state: Optional[BendersState] = None,
) -> BendersState:
    """
    Build BendersState from trajectory logger or list of snapshot dicts.

    snapshot_index: which iteration; None = last.
    prev_state: optional previous state for consistent deltas (unused if snapshots have prev).
    """
    snapshots = _get_snapshots(logger)
    if not snapshots:
        return BendersState()

    idx = snapshot_index if snapshot_index is not None else (len(snapshots) - 1)
    idx = max(0, min(idx, len(snapshots) - 1))
    cur = snapshots[idx]
    prev = snapshots[idx - 1] if idx > 0 else None

    y_val = np.asarray(cur["y_val"], dtype=float)
    z_val = np.asarray(cur["z_val"], dtype=float)
    Q_vals = np.asarray(cur["Q_vals"], dtype=float)

    lb_cur = float(cur["lb"])
    ub_cur = float(cur["ub"])
    gap_cur = max(float(cur["gap"]), 0.0)
    delta_lb = 0.0
    delta_ub = 0.0
    gap_reduction_rate = 0.0
    lb_improvement_rate = 0.0
    ub_improvement_rate = 0.0
    gap_prev = 0.0

    if prev is not None:
        lb_prev = float(prev["lb"])
        ub_prev = float(prev["ub"])
        gap_prev = max(float(prev["gap"]), 0.0)
        delta_lb = lb_cur - lb_prev
        delta_ub = ub_cur - ub_prev
        gap_reduction_rate = (gap_prev - gap_cur) / (gap_prev + _EPS)
        lb_improvement_rate = (lb_cur - lb_prev) / (abs(lb_prev) + _EPS)
        ub_improvement_rate = (ub_prev - ub_cur) / (abs(ub_prev) + _EPS)

    if Q_vals.size > 0:
        avg_recourse = float(np.mean(Q_vals))
        max_recourse = float(np.max(Q_vals))
        min_recourse = float(np.min(Q_vals))
        recourse_std_val = float(np.std(Q_vals))
    else:
        avg_recourse = float(cur.get("recourse_avg", 0.0))
        max_recourse = min_recourse = avg_recourse
        recourse_std_val = 0.0

    num_open = float(np.sum(y_val))
    total_cap = float(np.sum(z_val)) + _EPS
    avg_cap = float(np.mean(z_val)) if z_val.size > 0 else 0.0
    max_cap = float(np.max(z_val)) if z_val.size > 0 else 0.0
    capacity_utilization_proxy = avg_recourse / total_cap

    num_cuts_total = int(cur.get("num_cuts_total", 0))
    cuts_this_iter = cur.get("cuts_this_iter", [])
    cuts_added = len(cuts_this_iter) if isinstance(cuts_this_iter, list) else 0
    cumulative_cuts = sum(
        len(s.get("cuts_this_iter") or []) for s in snapshots[: idx + 1]
    )
    master_solve_time = float(cur.get("master_solve_time", 0.0))

    return BendersState(
        iteration_index=int(cur["iteration"]),
        lower_bound=lb_cur,
        upper_bound=ub_cur,
        optimality_gap=gap_cur,
        delta_lb=delta_lb,
        delta_ub=delta_ub,
        total_cuts_in_master=num_cuts_total,
        cuts_added_last_iter=cuts_added,
        cumulative_cuts_added=cumulative_cuts,
        master_solve_time_last_iter=master_solve_time,
        average_recourse_cost=avg_recourse,
        max_recourse_cost=max_recourse,
        min_recourse_cost=min_recourse,
        recourse_std=recourse_std_val,
        num_open_stations=num_open,
        avg_capacity=avg_cap,
        max_capacity=max_cap,
        capacity_utilization_proxy=capacity_utilization_proxy,
        gap_reduction_rate=gap_reduction_rate,
        lb_improvement_rate=lb_improvement_rate,
        ub_improvement_rate=ub_improvement_rate,
        mean_violation=0.0,
        max_violation=0.0,
        last_gap_reduction=(gap_prev - gap_cur) if prev is not None else 0.0,
    )


def build_state_sequence(
    logger: Union[Any, List[Dict[str, Any]]],
    snapshot_indices: Optional[List[int]] = None,
) -> List[BendersState]:
    """Build one BendersState per snapshot (all indices if snapshot_indices is None)."""
    snapshots = _get_snapshots(logger)
    if not snapshots:
        return []
    indices = snapshot_indices if snapshot_indices is not None else list(range(len(snapshots)))
    return [build_state(logger, snapshot_index=i) for i in indices]


__all__ = ["BendersState", "build_state", "build_state_sequence", "_get_snapshots"]
