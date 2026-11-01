"""
Benders trajectory logger: records per-iteration state and cut info.

For MDP / RL / cut-selection policy / convergence analysis.
Does not change optimization logic; caller passes data after each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Per-iteration snapshot (MDP state / RL features / convergence)
# ---------------------------------------------------------------------------
@dataclass
class IterationSnapshot:
    """Full snapshot of one Benders iteration."""

    iteration: int
    lb: float
    ub: float
    gap: float
    y_val: np.ndarray          # first-stage y
    z_val: np.ndarray          # first-stage z
    theta_val: np.ndarray      # master theta_k
    first_stage_cost: float
    recourse_avg: float
    Q_vals: np.ndarray         # per-scenario subproblem optimal Q_k
    num_cuts_total: int        # cuts in master after this iter (incl. new)
    master_num_constraints: int = 0   # master constraint count
    master_num_variables: int = 0    # master variable count
    master_solve_time: float = 0.0   # master solve time this iter (sec), optional
    wall_clock_sec: float = 0.0      # cumulative wall since Benders loop start (sec)
    cuts_this_iter: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"k": int, "rhs_const": float, "coef_z": np.ndarray, "alpha": np.ndarray, "beta": np.ndarray}
    # alpha/beta optional, for cut features / MDP

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict (numpy -> list) for JSON."""
        return {
            "iteration": self.iteration,
            "lb": float(self.lb),
            "ub": float(self.ub),
            "gap": float(self.gap),
            "y_val": self.y_val.tolist(),
            "z_val": self.z_val.tolist(),
            "theta_val": self.theta_val.tolist(),
            "first_stage_cost": float(self.first_stage_cost),
            "recourse_avg": float(self.recourse_avg),
            "Q_vals": self.Q_vals.tolist(),
            "num_cuts_total": self.num_cuts_total,
            "master_num_constraints": self.master_num_constraints,
            "master_num_variables": self.master_num_variables,
            "master_solve_time": float(self.master_solve_time),
            "wall_clock_sec": float(self.wall_clock_sec),
            "cuts_this_iter": [
                {
                    "k": c["k"],
                    "rhs_const": float(c["rhs_const"]),
                    "coef_z": np.asarray(c["coef_z"]).tolist(),
                    "alpha": np.asarray(c.get("alpha", [])).tolist(),
                    "beta": np.asarray(c.get("beta", [])).tolist(),
                }
                for c in self.cuts_this_iter
            ],
        }


# ---------------------------------------------------------------------------
# Trajectory logger: append snapshots, export / analyze later
# ---------------------------------------------------------------------------
class BendersTrajectoryLogger:
    """
    Append-only trajectory log; does not participate in solving.
    Caller invokes log_iteration at end of each iteration with that round's data.
    """

    def __init__(self, log_dual_vectors: bool = True):
        """
        log_dual_vectors: if True, store alpha, beta in cuts_this_iter (cut features / MDP).
        """
        self.log_dual_vectors = log_dual_vectors
        self.snapshots: List[IterationSnapshot] = []
        self._num_cuts_before_iter: int = 0  # cuts in master at iter start

    def start_iteration(self, num_cuts_in_master: int) -> None:
        """Call at iteration start; records cut count before adding this round's cuts."""
        self._num_cuts_before_iter = num_cuts_in_master

    def log_iteration(
        self,
        iteration: int,
        lb: float,
        ub: float,
        gap: float,
        y_val: np.ndarray,
        z_val: np.ndarray,
        theta_val: np.ndarray,
        first_stage_cost: float,
        recourse_avg: float,
        Q_vals: np.ndarray,
        cuts_added: int,
        cut_rhs_const: Optional[List[float]] = None,
        cut_coef_z: Optional[List[np.ndarray]] = None,
        cut_alpha: Optional[List[np.ndarray]] = None,
        cut_beta: Optional[List[np.ndarray]] = None,
        master_num_constraints: int = 0,
        master_num_variables: int = 0,
        master_solve_time: float = 0.0,
        wall_clock_sec: float = 0.0,
    ) -> None:
        """
        Log one iteration; vanilla_benders calls this at end of each round.

        cut_rhs_const[k], cut_coef_z[k]: cut constant and z coeffs for scenario k.
        cut_alpha[k], cut_beta[k]: optional subproblem duals (MDP / features).
        wall_clock_sec: cumulative wall seconds since the Benders loop started.
        """
        num_cuts_total = self._num_cuts_before_iter + cuts_added

        cuts_this_iter: List[Dict[str, Any]] = []
        if cut_rhs_const is not None and cut_coef_z is not None:
            for k in range(len(cut_rhs_const)):
                entry: Dict[str, Any] = {
                    "k": k,
                    "rhs_const": float(cut_rhs_const[k]),
                    "coef_z": np.asarray(cut_coef_z[k], dtype=float),
                }
                if self.log_dual_vectors and cut_alpha is not None and cut_beta is not None:
                    if k < len(cut_alpha) and k < len(cut_beta):
                        entry["alpha"] = np.asarray(cut_alpha[k], dtype=float)
                        entry["beta"] = np.asarray(cut_beta[k], dtype=float)
                cuts_this_iter.append(entry)

        snap = IterationSnapshot(
            iteration=iteration,
            lb=lb,
            ub=ub,
            gap=gap,
            y_val=np.asarray(y_val, dtype=float),
            z_val=np.asarray(z_val, dtype=float),
            theta_val=np.asarray(theta_val, dtype=float),
            first_stage_cost=first_stage_cost,
            recourse_avg=recourse_avg,
            Q_vals=np.asarray(Q_vals, dtype=float),
            num_cuts_total=num_cuts_total,
            master_num_constraints=master_num_constraints,
            master_num_variables=master_num_variables,
            master_solve_time=master_solve_time,
            wall_clock_sec=float(wall_clock_sec),
            cuts_this_iter=cuts_this_iter,
        )
        self.snapshots.append(snap)
        self._num_cuts_before_iter = num_cuts_total

    def get_trajectory(self) -> List[IterationSnapshot]:
        """Return list of recorded iteration snapshots."""
        return self.snapshots

    def get_trajectory_dict(self) -> List[Dict[str, Any]]:
        """Serializable trajectory (list of dict)."""
        return [s.to_dict() for s in self.snapshots]

    def summary(self) -> Dict[str, Any]:
        """Short stats for convergence checks."""
        if not self.snapshots:
            return {"num_iterations": 0}
        last = self.snapshots[-1]
        return {
            "num_iterations": len(self.snapshots),
            "final_lb": last.lb,
            "final_ub": last.ub,
            "final_gap": last.gap,
            "final_num_cuts": last.num_cuts_total,
            "final_master_num_constraints": last.master_num_constraints,
            "final_master_num_variables": last.master_num_variables,
        }


__all__ = ["BendersTrajectoryLogger", "IterationSnapshot"]
