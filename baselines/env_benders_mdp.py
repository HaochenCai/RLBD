"""
Benders MDP environment: one step = one Benders iteration.

Solver handles optimization; env wires action -> cut selection -> add cuts -> master -> sub -> state/reward.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from benders_mdp_state import BendersState

try:
    from cut_selection import CutSelector, select_cuts
except ImportError:
    CutSelector = None  # type: ignore
    select_cuts = None  # type: ignore


class BendersSolverInterface(Protocol):
    """Abstract Benders solver; concrete Gurobi code lives elsewhere."""

    def reset(self) -> None:
        """Reset RMP so solve_master can run."""
        ...

    def add_cuts(self, cuts: List[Dict[str, Any]]) -> None:
        """Append cuts (dicts with rhs_const, coef_z, alpha, beta, k, ...)."""
        ...

    def solve_master(self) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """
        Solve current master.
        Returns:
            z, theta, LB, master_info (e.g. solve_time_master).
        """
        ...

    def solve_subproblems(self, z: np.ndarray) -> Tuple[
        List[Dict[str, Any]],
        float,
        Dict[str, float],
        Dict[str, Any],
    ]:
        """
        Solve subproblems at z; return candidate cuts, UB, violation_stats, sub_info.
        """
        ...

    def num_cuts(self) -> int:
        """Number of cuts in master."""
        ...


class MockSolver:
    """Toy solver for tests; no Gurobi."""

    def __init__(
        self,
        n_z: int = 4,
        n_scenarios: int = 3,
        lb_init: float = 100.0,
        ub_init: float = 200.0,
        gap_decay_per_iter: float = 0.05,
    ):
        self.n_z = n_z
        self.n_scenarios = n_scenarios
        self._lb = lb_init
        self._ub = ub_init
        self._gap_decay = gap_decay_per_iter
        self._cuts: List[Dict[str, Any]] = []
        self._iter = 0

    def reset(self) -> None:
        self._cuts = []
        self._iter = 0
        self._lb = 100.0
        self._ub = 200.0

    def add_cuts(self, cuts: List[Dict[str, Any]]) -> None:
        self._cuts.extend(cuts)

    def solve_master(self) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        self._iter += 1
        z = np.ones(self.n_z, dtype=float) * (1.0 + 0.1 * self._iter)
        theta = np.ones(self.n_scenarios, dtype=float) * 10.0
        self._lb = self._lb + 2.0 + len(self._cuts) * 0.1
        return z, theta, self._lb, {"solve_time_master": 0.01}

    def solve_subproblems(self, z: np.ndarray) -> Tuple[
        List[Dict[str, Any]],
        float,
        Dict[str, float],
        Dict[str, Any],
    ]:
        self._ub = self._ub - self._gap_decay * (self._ub - self._lb)
        candidate_cuts = []
        for k in range(self.n_scenarios):
            candidate_cuts.append({
                "k": k,
                "rhs_const": 5.0 + k,
                "coef_z": np.random.randn(self.n_z).tolist(),
                "alpha": [0.1],
                "beta": np.random.rand(self.n_z).tolist(),
            })
        viol = max(0.0, (self._ub - self._lb) / max(1.0, abs(self._ub)))
        violation_stats = {"mean": viol * 0.5, "max": viol}
        return candidate_cuts, self._ub, violation_stats, {"solve_time_sub": 0.02}

    def num_cuts(self) -> int:
        return len(self._cuts)


class BendersMDPEnv:
    """
    MDP view of Benders: step = select cuts, add cuts, solve master, solve subs, update state.
    action carries scorer_params (etc.); reward_fn(gap_old, gap_new, info) is injected.
    Observations are BendersState; use state.to_vector() for NN input.
    """

    def __init__(
        self,
        solver: BendersSolverInterface,
        reward_fn: Callable[[float, float, Dict[str, Any]], float],
        selector: Optional[Any] = None,
        max_iters: int = 500,
        tol_gap: float = 1e-4,
    ):
        self.solver = solver
        self.reward_fn = reward_fn
        self.selector = selector if selector is not None else _default_selector()
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

    def _make_benders_state(
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
        """Build BendersState from current env scalars; optionally refresh _prev_* caches."""
        _eps = 1e-9
        gap = (UB - LB) / max(1.0, abs(UB))
        gap = max(0.0, float(gap))

        delta_lb = 0.0 if self._prev_lower_bound is None else LB - self._prev_lower_bound
        delta_ub = 0.0 if self._prev_upper_bound is None else UB - self._prev_upper_bound
        delta_gap = 0.0 if self._prev_gap is None else self._prev_gap - gap

        cuts_total = num_cuts
        cuts_added_last_iter = cuts_added
        delta_cuts_total = (
            0 if self._prev_total_cuts is None else cuts_total - self._prev_total_cuts
        )

        gap_reduction_rate = (delta_gap / (self._prev_gap + _eps)) if self._prev_gap is not None else 0.0
        lb_improvement_rate = (delta_lb / (abs(self._prev_lower_bound) + _eps)) if self._prev_lower_bound is not None else 0.0
        ub_improvement_rate = (-delta_ub / (abs(self._prev_upper_bound) + _eps)) if self._prev_upper_bound is not None else 0.0

        master_time = float(master_solve_time) if master_solve_time is not None else 0.0

        if update_history:
            self._prev_lower_bound = LB
            self._prev_upper_bound = UB
            self._prev_gap = gap
            self._prev_total_cuts = cuts_total

        return BendersState(
            iteration_index=iteration,
            lower_bound=LB,
            upper_bound=UB,
            optimality_gap=gap,
            delta_lb=delta_lb,
            delta_ub=delta_ub,
            total_cuts_in_master=cuts_total,
            cuts_added_last_iter=cuts_added_last_iter,
            cumulative_cuts_added=cuts_total,
            master_solve_time_last_iter=master_time,
            mean_violation=float(violation_stats.get("mean", 0.0)),
            max_violation=float(violation_stats.get("max", 0.0)),
            last_gap_reduction=delta_gap,
            gap_reduction_rate=gap_reduction_rate,
            lb_improvement_rate=lb_improvement_rate,
            ub_improvement_rate=ub_improvement_rate,
        )

    def reset(self) -> BendersState:
        """Reset solver; one master + subs to seed candidate cuts and state."""
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

        state = self._make_benders_state(
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
        return state

    def step(
        self,
        action: Dict[str, Any],
    ) -> Tuple[BendersState, float, bool, Dict[str, Any]]:
        """One Benders iteration; returns next_state, reward, done, info."""
        gap_old = self._gap
        LB_old, UB_old = self._LB, self._UB

        current_state = self._make_benders_state(
            iteration=self._iteration,
            LB=self._LB,
            UB=self._UB,
            num_cuts=self.solver.num_cuts(),
            violation_stats=self._violation_stats,
            gap_old=gap_old,
            master_solve_time=None,
            cuts_added=0,
            update_history=False,
        )

        scorer_params = action.get("scorer_params") if isinstance(action, dict) else None
        if self._z_current is None or self._theta_current is None:
            self._z_current = np.zeros(1)
            self._theta_current = np.zeros(1)
        selected_cuts = _select_cuts(
            self.selector,
            self._candidate_cuts,
            self._z_current,
            self._theta_current,
            scorer_params,
            state_t=current_state,
        )

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

        next_state = self._make_benders_state(
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

        info = {
            "selected_cut_count": len(selected_cuts),
            "candidate_cut_count": len(self._candidate_cuts),
            "effective_top_k": len(selected_cuts),
            "gap_old": gap_old,
            "gap_new": gap_new,
            "LB_old": LB_old,
            "LB_new": LB,
            "UB_old": UB_old,
            "UB_new": UB,
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
        }

        reward = self.reward_fn(gap_old, gap_new, info)

        return next_state, reward, done, info


def _default_selector() -> Any:
    if CutSelector is not None:
        return CutSelector()
    return None


def _select_cuts(
    selector: Any,
    candidate_cuts: List[Dict[str, Any]],
    z_current: np.ndarray,
    theta_val: Optional[np.ndarray],
    scorer_params: Optional[Any],
    state_t: Optional[BendersState] = None,
) -> List[Dict[str, Any]]:
    """Dispatch to CutSelector.select or select_cuts; if no selector, return all candidates."""
    if selector is None or not candidate_cuts:
        return list(candidate_cuts)
    if hasattr(selector, "select"):
        return selector.select(
            candidate_cuts,
            z_current,
            state_t=state_t,
            scorer_params=scorer_params,
            theta_val=theta_val,
        )
    if select_cuts is not None:
        return select_cuts(
            candidate_cuts,
            z_current,
            state_t=state_t,
            scorer_params=scorer_params,
            theta_val=theta_val,
        )
    return list(candidate_cuts)


if __name__ == "__main__":
    from reward_plugins import gap_reduction_reward

    solver = MockSolver(n_z=4, n_scenarios=3, gap_decay_per_iter=0.05)
    env = BendersMDPEnv(
        solver,
        reward_fn=gap_reduction_reward,
        max_iters=10,
        tol_gap=0.01,
    )

    state = env.reset()
    print("reset state (BendersState):", state.iteration_index, state.optimality_gap, state.to_vector().shape)

    for t in range(5):
        action = {"scorer_params": {"weights": np.array([1.0, 0.0, 0.0, 0.0])}}
        next_state, reward, done, info = env.step(action)
        print(f"step {t+1}: gap={next_state.optimality_gap:.6f} reward={reward:.6f} done={done} "
              f"| selected={info['selected_cut_count']} candidate={info['candidate_cut_count']}")
        if done:
            break
