"""
Benders MDP environment: one step = one Benders iteration.

- action must include "selected_indices" (list of int); env does not sample,
  it uses candidate_cuts[i] for i in selected_indices.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from benders_mdp_state import BendersState, build_state_from_env_snapshot


class BendersSolverInterface(Protocol):
    def reset(self) -> None: ...
    def add_cuts(self, cuts: List[Dict[str, Any]]) -> None: ...
    def solve_master(self) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]: ...
    def solve_subproblems(self, z: np.ndarray) -> Tuple[
        List[Dict[str, Any]], float, Dict[str, float], Dict[str, Any],
    ]: ...
    def num_cuts(self) -> int: ...


class MockSolver:
    def __init__(self, n_z: int = 4, n_scenarios: int = 3, lb_init: float = 100.0, ub_init: float = 200.0, gap_decay_per_iter: float = 0.05):
        self.n_z = n_z
        self.n_scenarios = n_scenarios
        self._lb = lb_init
        self._ub = ub_init
        self._gap_decay = gap_decay_per_iter
        self._cuts: List[Dict[str, Any]] = []
        self._iter = 0
        self.last_subproblem_stats: Dict[str, Any] = {}

    def reset(self) -> None:
        self._cuts = []
        self._iter = 0
        self._lb = 100.0
        self._ub = 200.0
        self.last_subproblem_stats = {}

    def add_cuts(self, cuts: List[Dict[str, Any]]) -> None:
        self._cuts.extend(cuts)

    def solve_master(self) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        self._iter += 1
        z = np.ones(self.n_z, dtype=float) * (1.0 + 0.1 * self._iter)
        theta = np.ones(self.n_scenarios, dtype=float) * 10.0
        self._lb = self._lb + 2.0 + len(self._cuts) * 0.1
        return z, theta, self._lb, {"solve_time_master": 0.01}

    def solve_subproblems(self, z: np.ndarray) -> Tuple[List[Dict[str, Any]], float, Dict[str, float], Dict[str, Any]]:
        self._ub = self._ub - self._gap_decay * (self._ub - self._lb)
        candidate_cuts = []
        Q_vals = []
        for k in range(self.n_scenarios):
            candidate_cuts.append({"k": k, "rhs_const": 5.0 + k, "coef_z": np.random.randn(self.n_z).tolist(), "alpha": [0.1], "beta": np.random.rand(self.n_z).tolist()})
            Q_vals.append(10.0 + k * 2.0 + np.random.rand() * 5.0)
        viol = max(0.0, (self._ub - self._lb) / max(1.0, abs(self._ub)))
        violation_stats = {"mean": viol * 0.5, "max": viol}
        self.last_subproblem_stats = {
            "Q_vals": np.array(Q_vals, dtype=float),
            "recourse_avg": float(np.mean(Q_vals)),
            "recourse_max": float(np.max(Q_vals)),
            "recourse_min": float(np.min(Q_vals)),
            "recourse_std": float(np.std(Q_vals)) if len(Q_vals) > 1 else 0.0,
            "y_val": np.ones(self.n_z) * 0.5,
            "z_val": np.asarray(z, dtype=float),
        }
        return candidate_cuts, self._ub, violation_stats, {"solve_time_sub": 0.02}

    def num_cuts(self) -> int:
        return len(self._cuts)


class BendersMDPEnv:
    def __init__(
        self,
        solver: BendersSolverInterface,
        reward_fn: Callable[[float, float, Dict[str, Any]], float],
        max_iters: int = 500,
        tol_gap: float = 1e-4,
    ):
        self.solver = solver
        self.reward_fn = reward_fn
        self.max_iters = max_iters
        self.tol_gap = tol_gap

        self._iteration: int = 0
        self._candidate_cuts: List[Dict[str, Any]] = []
        self._z_current: Optional[np.ndarray] = None
        self._theta_current: Optional[np.ndarray] = None
        self._LB: float = 0.0
        self._UB: float = 0.0
        self._gap: float = 1.0
        self._violation_stats: Dict[str, float] = {"mean": 0.0, "max": 0.0}

        self._prev_lower_bound: Optional[float] = None
        self._prev_upper_bound: Optional[float] = None
        self._prev_gap: Optional[float] = None
        self._prev_total_cuts: Optional[int] = None
        self._lb_baseline: Optional[float] = None

    def _build_snapshot_and_state(
        self,
        iteration: int,
        LB: float,
        UB: float,
        num_cuts: int,
        violation_stats: Dict[str, float],
        gap_old: float,
        master_solve_time: Optional[float] = None,
        cuts_added: int = 0,
        update_history: bool = True,
    ) -> BendersState:
        """Build snapshot and call build_state_from_env_snapshot (single source of truth)."""
        gap = (UB - LB) / max(1.0, abs(UB))
        gap = max(0.0, float(gap))

        delta_lb = 0.0 if self._prev_lower_bound is None else LB - self._prev_lower_bound
        delta_ub = 0.0 if self._prev_upper_bound is None else UB - self._prev_upper_bound
        delta_gap = 0.0 if self._prev_gap is None else self._prev_gap - gap

        # Save prev before update for trend rates in build_state
        prev_gap = float(self._prev_gap) if self._prev_gap is not None else None
        prev_lb = float(self._prev_lower_bound) if self._prev_lower_bound is not None else None
        prev_ub = float(self._prev_upper_bound) if self._prev_upper_bound is not None else None

        if update_history:
            self._prev_lower_bound = LB
            self._prev_upper_bound = UB
            self._prev_gap = gap
            self._prev_total_cuts = num_cuts

        # Subproblem stats from solver if available
        sub_stats = getattr(self.solver, "last_subproblem_stats", {}) or {}

        snapshot = {
            "iteration": iteration,
            "max_iters": self.max_iters,
            "lb": LB,
            "ub": UB,
            "gap": gap,
            "delta_lb": delta_lb,
            "delta_ub": delta_ub,
            "last_gap_reduction": delta_gap,
            "master_solve_time": float(master_solve_time) if master_solve_time is not None else 0.0,
            "num_cuts_total": num_cuts,
            "cuts_added": cuts_added,
            "violation_mean": float(violation_stats.get("mean", 0.0)),
            "violation_max": float(violation_stats.get("max", 0.0)),
            "prev_gap": prev_gap,
            "prev_lb": prev_lb,
            "prev_ub": prev_ub,
        }
        snapshot["Q_vals"] = sub_stats.get("Q_vals")
        snapshot["recourse_avg"] = sub_stats.get("recourse_avg")
        snapshot["recourse_max"] = sub_stats.get("recourse_max")
        snapshot["recourse_min"] = sub_stats.get("recourse_min")
        snapshot["recourse_std"] = sub_stats.get("recourse_std")
        snapshot["z_val"] = sub_stats.get("z_val") if sub_stats.get("z_val") is not None else self._z_current
        snapshot["y_val"] = sub_stats.get("y_val") if sub_stats.get("y_val") is not None else None

        return build_state_from_env_snapshot(snapshot)

    def get_candidate_cuts(self) -> List[Dict[str, Any]]:
        return list(self._candidate_cuts)

    def get_z_current(self) -> Optional[np.ndarray]:
        if self._z_current is None:
            return None
        return np.asarray(self._z_current, dtype=np.float64)

    def get_theta_current(self) -> Optional[np.ndarray]:
        if self._theta_current is None:
            return None
        return np.asarray(self._theta_current, dtype=np.float64)

    def reset(self) -> BendersState:
        self.solver.reset()
        self._iteration = 0
        self._candidate_cuts = []
        self._z_current = None
        self._theta_current = None
        self._LB = 0.0
        self._UB = 0.0
        self._gap = 1.0
        self._violation_stats = {"mean": 0.0, "max": 0.0}
        self._prev_lower_bound = None
        self._prev_upper_bound = None
        self._prev_gap = None
        self._prev_total_cuts = None
        self._lb_baseline = None

        t0 = time.time()
        z, theta, LB, _ = self.solver.solve_master()
        master_solve_time = time.time() - t0
        candidate_cuts, UB, violation_stats, _ = self.solver.solve_subproblems(z)

        self._z_current = z
        self._theta_current = theta
        self._LB = LB
        self._UB = UB
        self._violation_stats = violation_stats
        self._gap = (UB - LB) / max(1.0, abs(UB))
        self._gap = max(0.0, self._gap)
        self._candidate_cuts = candidate_cuts
        self._lb_baseline = float(LB)

        return self._build_snapshot_and_state(
            iteration=0,
            LB=LB,
            UB=UB,
            num_cuts=self.solver.num_cuts(),
            violation_stats=violation_stats,
            gap_old=1.0,
            master_solve_time=master_solve_time,
            cuts_added=0,
            update_history=True,
        )

    def step(self, action: Dict[str, Any]) -> Tuple[BendersState, float, bool, Dict[str, Any]]:
        if "selected_indices" not in action:
            raise ValueError("action must contain 'selected_indices'; env does not sample")

        selected_indices = action["selected_indices"]
        if not isinstance(selected_indices, list):
            raise ValueError("selected_indices must be a list")
        selected_indices = [int(i) for i in selected_indices]
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError("selected_indices must be unique")
        n_candidates = len(self._candidate_cuts)
        for idx in selected_indices:
            if idx < 0 or idx >= n_candidates:
                raise ValueError(f"selected_indices index {idx} out of range [0, {n_candidates-1}]")

        gap_old = self._gap
        LB_old, UB_old = self._LB, self._UB
        violation_old_mean = float(self._violation_stats.get("mean", 0.0))
        violation_old_max = float(self._violation_stats.get("max", 0.0))

        selected_cuts = [self._candidate_cuts[i] for i in selected_indices]
        self.solver.add_cuts(selected_cuts)

        t0 = time.time()
        z, theta, LB, master_info = self.solver.solve_master()
        master_solve_time = time.time() - t0
        self._z_current = z
        self._theta_current = theta
        self._LB = LB

        candidate_cuts_next, UB, violation_stats, sub_info = self.solver.solve_subproblems(z)
        self._UB = UB
        self._violation_stats = violation_stats
        self._candidate_cuts = candidate_cuts_next

        gap_new = (UB - LB) / max(1.0, abs(UB))
        gap_new = max(0.0, gap_new)
        self._gap = gap_new
        self._iteration += 1

        next_state = self._build_snapshot_and_state(
            iteration=self._iteration,
            LB=LB,
            UB=UB,
            num_cuts=self.solver.num_cuts(),
            violation_stats=violation_stats,
            gap_old=gap_old,
            master_solve_time=master_solve_time,
            cuts_added=len(selected_cuts),
            update_history=True,
        )

        done = gap_new <= self.tol_gap or self._iteration >= self.max_iters

        selected_scenarios = [c.get("k", None) for c in selected_cuts]
        violation_new_mean = float(violation_stats.get("mean", 0.0))
        violation_new_max = float(violation_stats.get("max", 0.0))

        info = {
            "selected_cut_count": len(selected_cuts),
            "candidate_cut_count": len(self._candidate_cuts),
            "gap_old": gap_old,
            "gap_new": gap_new,
            "violation_old_mean": violation_old_mean,
            "violation_new_mean": violation_new_mean,
            "violation_old_max": violation_old_max,
            "violation_new_max": violation_new_max,
            "vl_old": violation_old_mean,
            "vl_new": violation_new_mean,
            "LB_old": LB_old,
            "LB_new": LB,
            "UB_old": UB_old,
            "UB_new": UB,
            "LB_baseline": self._lb_baseline,
            "solve_time_master": master_info.get("solve_time_master"),
            "solve_time_sub": sub_info.get("solve_time_sub"),
            "sub_iter_count_sum": sub_info.get("sub_iter_count_sum"),
            "sub_iter_count_mean": sub_info.get("sub_iter_count_mean"),
            "sub_iter_count_max": sub_info.get("sub_iter_count_max"),
            "sub_gurobi_runtime_sum": sub_info.get("sub_gurobi_runtime_sum"),
            "sub_gurobi_runtime_mean": sub_info.get("sub_gurobi_runtime_mean"),
            "z_l1": sub_info.get("z_l1"),
            "n_zero_cap": sub_info.get("n_zero_cap"),
            "n_positive_beta": sub_info.get("n_positive_beta"),
            "n_scenarios_solved": sub_info.get("n_scenarios_solved"),
            "selected_indices": selected_indices,
            "selected_scenarios": selected_scenarios,
            "total_cuts_in_rmp": self.solver.num_cuts(),
        }

        reward_out = self.reward_fn(gap_old, gap_new, info)
        reward_debug: Dict[str, Any] = {}
        if isinstance(reward_out, tuple) and len(reward_out) == 2:
            reward, dbg = reward_out
            if isinstance(dbg, dict):
                reward_debug = dbg
        else:
            reward = reward_out
        reward = float(reward)
        info["reward_debug"] = reward_debug
        assert info["selected_indices"] == action["selected_indices"], "selected_indices must match action"

        return next_state, reward, done, info


if __name__ == "__main__":
    from reward_registry import get_reward_fn

    solver = MockSolver(n_z=4, n_scenarios=3, gap_decay_per_iter=0.05)
    env = BendersMDPEnv(solver, reward_fn=get_reward_fn("log_gap_rmp_time_penalty"), max_iters=10, tol_gap=0.01)
    state = env.reset()
    for t in range(3):
        cuts = env.get_candidate_cuts()
        indices = [0] if cuts else []
        next_state, reward, done, info = env.step({"selected_indices": indices})
        print(f"step {t+1}: gap={next_state.optimality_gap:.6f} reward={reward:.6f} done={done}")
        if done:
            break
