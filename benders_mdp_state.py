"""
Benders-MDP state representation.

Builds fixed-size, normalizable state vectors from trajectory logger data
for cut-selection policies (MDP + RL / NN).

Design:
- Logger-only; no Gurobi handles
- Stable numerics, fixed dim, low cost
- Room for extensions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Constants: scaling / numerical stability
# ---------------------------------------------------------------------------
_EPS = 1e-9
_MAX_ITER_DEFAULT = 2000
_SCALE_LB_UB = 1e6
_SCALE_COST = 1e6
_SCALE_TIME = 300.0  # seconds
_SCALE_CUTS = 500
_SCALE_CAPACITY = 1e4


# ---------------------------------------------------------------------------
# BendersState container
# ---------------------------------------------------------------------------
@dataclass
class BendersState:
    """
    MDP state for one Benders iteration; fields derivable from logger.
    No full y/z vectors; summary stats only.
    """

    # ---- 1. Convergence ----
    iteration_index: int = 0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    optimality_gap: float = 0.0
    delta_lb: float = 0.0
    delta_ub: float = 0.0

    # ---- 2. Master ----
    total_cuts_in_master: int = 0  # cuts in RMP
    cuts_added_last_iter: int = 0  # cuts added this step
    cumulative_cuts_added: int = 0  # same as total_cuts_in_master (total cuts)
    master_solve_time_last_iter: float = 0.0

    # ---- 3. Subproblem / recourse ----
    average_recourse_cost: float = 0.0
    max_recourse_cost: float = 0.0
    min_recourse_cost: float = 0.0
    recourse_std: float = 0.0

    # ---- 4. First-stage summary (no full vectors) ----
    num_open_stations: float = 0.0
    avg_capacity: float = 0.0
    max_capacity: float = 0.0
    capacity_utilization_proxy: float = 0.0

    # ---- 5. Convergence trends ----
    gap_reduction_rate: float = 0.0
    lb_improvement_rate: float = 0.0
    ub_improvement_rate: float = 0.0

    # ---- 6. Env / violation (matches env contract) ----
    mean_violation: float = 0.0
    max_violation: float = 0.0
    last_gap_reduction: float = 0.0

    # Optional extras (e.g. cut-violation), serializable
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
        """
        Fixed-size numpy vector for NN input.
        Smooth bounded transforms (tanh, rational saturators); no hard clip.
        """
        t = float(self.iteration_index)
        M = float(max(1, max_iter))
        # Iter progress: t / max(1, M, t) in [0,1]; saturates at 1 when t>M (vs clip(t/M))
        v_iter = float(t / max(1.0, M, t))

        # Convergence
        v_lb = float(np.tanh(self.lower_bound / (scale_lb_ub + _EPS)))
        v_ub = float(np.tanh(self.upper_bound / (scale_lb_ub + _EPS)))
        v_gap = float(self.optimality_gap / (self.optimality_gap + 1.0 + _EPS))
        v_delta_lb = float(np.tanh(self.delta_lb / (scale_cost + _EPS)))
        v_delta_ub = float(np.tanh(self.delta_ub / (scale_cost + _EPS)))

        # Master
        k_tot = float(max(0, self.total_cuts_in_master))
        k_add = float(max(0, self.cuts_added_last_iter))
        k_cum = float(max(0, self.cumulative_cuts_added))
        s_k = float(scale_cuts)
        v_total_cuts = float(k_tot / (s_k + k_tot + _EPS))
        v_cuts_added = float(k_add / (s_k + k_add + _EPS))
        v_cumulative_cuts = float(k_cum / (s_k + k_cum + _EPS))
        v_master_time = float(self.master_solve_time_last_iter / (scale_time + _EPS))

        # Recourse
        v_recourse_avg = float(np.tanh(self.average_recourse_cost / (scale_cost + _EPS)))
        v_recourse_max = float(np.tanh(self.max_recourse_cost / (scale_cost + _EPS)))
        v_recourse_min = float(np.tanh(self.min_recourse_cost / (scale_cost + _EPS)))
        v_recourse_std = float(self.recourse_std / (self.recourse_std + 1.0 + _EPS))

        # First-stage summary
        n_o = float(max(0.0, self.num_open_stations))
        v_num_open = float(n_o / (20.0 + n_o + _EPS))
        v_avg_cap = float(np.tanh(self.avg_capacity / (scale_capacity + _EPS)) * 0.5 + 0.5)
        v_max_cap = float(np.tanh(self.max_capacity / (scale_capacity + _EPS)) * 0.5 + 0.5)
        u = float(self.capacity_utilization_proxy)
        v_util = float(u / (u + 1.0 + _EPS))

        # Trends (real-valued, no truncation)
        v_gap_rate = float(self.gap_reduction_rate)
        v_lb_rate = float(self.lb_improvement_rate)
        v_ub_rate = float(self.ub_improvement_rate)

        # violation / last_gap_reduction (env-aligned)
        v_m = float(max(0.0, self.mean_violation))
        v_mx = float(max(0.0, self.max_violation))
        v_mean_viol = float(v_m / (v_m + 1.0 + _EPS))
        v_max_viol = float(v_mx / (v_mx + 1.0 + _EPS))
        v_last_gap_red = float(self.last_gap_reduction)

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
        """State vector dim (excluding extra_features)."""
        return 24


# ---------------------------------------------------------------------------
# Build state from logger
# ---------------------------------------------------------------------------
def _get_snapshots(logger: Union[Any, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize to list of snapshot dicts (logger object or list of dict)."""
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
    Build BendersState from logger records.

    Parameters
    ----------
    logger : BendersTrajectoryLogger or List[Dict]
        Trajectory source: logger or exported list of snapshot dicts (e.g. from JSON).
    snapshot_index : int, optional
        Which iteration to build. None = last round.
    prev_state : BendersState, optional
        Previous state for delta/rate consistency; if omitted, taken from prior snapshot in logger.

    Returns
    -------
    BendersState
    """
    snapshots = _get_snapshots(logger)
    if not snapshots:
        return BendersState()

    idx = snapshot_index if snapshot_index is not None else (len(snapshots) - 1)
    idx = max(0, min(idx, len(snapshots) - 1))
    cur = snapshots[idx]
    prev = snapshots[idx - 1] if idx > 0 else None

    # Arrays from cur (may already be list)
    y_val = np.asarray(cur["y_val"], dtype=float)
    z_val = np.asarray(cur["z_val"], dtype=float)
    Q_vals = np.asarray(cur["Q_vals"], dtype=float)

    # Convergence (non-negative gap clamp when LB > UB hurts features/reward)
    lb_cur = float(cur["lb"])
    ub_cur = float(cur["ub"])
    gap_cur = max(float(cur["gap"]), 0.0)
    delta_lb = 0.0
    delta_ub = 0.0
    gap_reduction_rate = 0.0
    lb_improvement_rate = 0.0
    ub_improvement_rate = 0.0

    if prev is not None:
        lb_prev = float(prev["lb"])
        ub_prev = float(prev["ub"])
        gap_prev = max(float(prev["gap"]), 0.0)  # non-negative gap for rates / stability
        delta_lb = lb_cur - lb_prev
        delta_ub = ub_cur - ub_prev
        gap_reduction_rate = (gap_prev - gap_cur) / (gap_prev + _EPS)
        lb_improvement_rate = (lb_cur - lb_prev) / (abs(lb_prev) + _EPS)
        ub_improvement_rate = (ub_prev - ub_cur) / (abs(ub_prev) + _EPS)

    # Recourse stats
    if Q_vals.size > 0:
        avg_recourse = float(np.mean(Q_vals))
        max_recourse = float(np.max(Q_vals))
        min_recourse = float(np.min(Q_vals))
        recourse_std_val = float(np.std(Q_vals))
    else:
        avg_recourse = float(cur.get("recourse_avg", 0.0))
        max_recourse = min_recourse = avg_recourse
        recourse_std_val = 0.0

    # First-stage summary
    num_open = float(np.sum(y_val))
    total_cap = float(np.sum(z_val)) + _EPS
    avg_cap = float(np.mean(z_val)) if z_val.size > 0 else 0.0
    max_cap = float(np.max(z_val)) if z_val.size > 0 else 0.0
    # capacity_utilization_proxy ~ recourse_avg / total_capacity
    capacity_utilization_proxy = avg_recourse / total_cap

    # Master: total = cuts in master now; cumulative = sum added over rounds 1..idx
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
    """
    Build a sequence of states for multi-round trajectories.
    If snapshot_indices is None, use all snapshots.
    """
    snapshots = _get_snapshots(logger)
    if not snapshots:
        return []
    indices = snapshot_indices if snapshot_indices is not None else list(range(len(snapshots)))
    return [build_state(logger, snapshot_index=i) for i in indices]


def build_state_from_env_snapshot(snapshot: Dict[str, Any]) -> BendersState:
    """
    Build BendersState from env snapshot (single source of truth).
    Snapshot is built by env at reset/step end with all stats the solver exposes.
    Missing fields filled with proxies to avoid many zeros in to_vector.
    """
    def _f(key: str, default: float = 0.0) -> float:
        v = snapshot.get(key)
        return float(v) if v is not None else default

    def _i(key: str, default: int = 0) -> int:
        v = snapshot.get(key)
        return int(v) if v is not None else default

    iteration = _i("iteration", 0)
    lb = _f("lb", 0.0)
    ub = _f("ub", 0.0)
    gap = max(_f("gap", 0.0), 0.0)
    delta_lb = _f("delta_lb", 0.0)
    delta_ub = _f("delta_ub", 0.0)
    last_gap_reduction = _f("last_gap_reduction", 0.0)
    master_solve_time = _f("master_solve_time", 0.0)
    num_cuts_total = _i("num_cuts_total", 0)
    cuts_added = _i("cuts_added", 0)
    violation_mean = _f("violation_mean", 0.0)
    violation_max = _f("violation_max", 0.0)

    # Recourse: prefer Q_vals; else violation as proxy
    Q_vals = snapshot.get("Q_vals")
    if Q_vals is not None:
        Q_arr = np.asarray(Q_vals, dtype=float)
        if Q_arr.size > 0:
            avg_recourse = float(np.mean(Q_arr))
            max_recourse = float(np.max(Q_arr))
            min_recourse = float(np.min(Q_arr))
            recourse_std_val = float(np.std(Q_arr)) if Q_arr.size > 1 else 0.0
        else:
            proxy_r = max(violation_max * 1e3, _EPS)
            avg_recourse = proxy_r
            max_recourse = min_recourse = proxy_r
            recourse_std_val = 0.0
    else:
        # proxy violation scale for recourse when missing
        proxy = max(violation_max * 1e3, _EPS) if violation_max > 0 else _EPS
        avg_recourse = _f("recourse_avg", proxy)
        max_recourse = _f("recourse_max", proxy * 1.2)
        min_recourse = _f("recourse_min", proxy * 0.8)
        recourse_std_val = _f("recourse_std", proxy * 0.1)

    # First-stage: prefer z_val / y_val
    z_val = snapshot.get("z_val")
    y_val = snapshot.get("y_val")
    if z_val is not None and y_val is not None:
        z_arr = np.asarray(z_val, dtype=float)
        y_arr = np.asarray(y_val, dtype=float)
        num_open = float(np.sum(y_arr))
        total_cap = float(np.sum(z_arr)) + _EPS
        avg_cap = float(np.mean(z_arr)) if z_arr.size > 0 else 0.0
        max_cap = float(np.max(z_arr)) if z_arr.size > 0 else 0.0
        capacity_utilization_proxy = avg_recourse / total_cap
    else:
        # proxy from cuts/gap; 0.5 sentinel when missing
        num_open = float(max(num_cuts_total, 1)) * 0.1
        avg_cap = max_recourse * 0.5 if max_recourse > 0 else _EPS
        max_cap = avg_cap * 1.5
        capacity_utilization_proxy = min(gap, 1.0) if gap > 0 else 0.5

    # Trend rates
    prev_gap = snapshot.get("prev_gap")
    prev_lb = snapshot.get("prev_lb")
    prev_ub = snapshot.get("prev_ub")
    gap_reduction_rate = 0.0
    lb_improvement_rate = 0.0
    ub_improvement_rate = 0.0
    if prev_gap is not None and prev_gap > _EPS:
        gap_reduction_rate = (float(prev_gap) - gap) / (float(prev_gap) + _EPS)
    if prev_lb is not None and abs(float(prev_lb)) > _EPS:
        lb_improvement_rate = (lb - float(prev_lb)) / (abs(float(prev_lb)) + _EPS)
    if prev_ub is not None and abs(float(prev_ub)) > _EPS:
        ub_improvement_rate = (float(prev_ub) - ub) / (abs(float(prev_ub)) + _EPS)

    return BendersState(
        iteration_index=iteration,
        lower_bound=lb,
        upper_bound=ub,
        optimality_gap=gap,
        delta_lb=delta_lb,
        delta_ub=delta_ub,
        total_cuts_in_master=num_cuts_total,
        cuts_added_last_iter=cuts_added,
        cumulative_cuts_added=num_cuts_total,
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
        mean_violation=violation_mean,
        max_violation=violation_max,
        last_gap_reduction=last_gap_reduction,
    )


__all__ = [
    "BendersState",
    "build_state",
    "build_state_from_env_snapshot",
    "build_state_sequence",
    "_get_snapshots",
]


# ---------------------------------------------------------------------------
# Minimal usage sketch (debug / MDP hookup)
# ---------------------------------------------------------------------------
# From in-memory logger, last round:
#   from benders_mdp_state import build_state, BendersState
#   state = build_state(trajectory_logger)
#   state_k = build_state(trajectory_logger, snapshot_index=k)
#
# From JSON trajectory (offline):
#   import json
#   with open("import_json/benders_trajectory.json") as f:
#       trajectory = json.load(f)
#   state = build_state(trajectory, snapshot_index=10)
#
# Full trajectory sequence:
#   states = build_state_sequence(trajectory)
#   for s in states:
#       vec = s.to_vector()
#
# Optional extras slot:
#   state.extra_features["my_feature"] = 1.0
