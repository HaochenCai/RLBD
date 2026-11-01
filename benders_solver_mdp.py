"""
MDP-compatible incremental Benders solver.

- Builds master, solves master/subproblems, emits candidate cuts, adds selected cuts.
- Cut selection is env/action, not here.
- Implements BendersSolverInterface for env_benders_mdp.
- Reuses vanilla_benders read_inputs, solve_subproblem_duals (does not edit vanilla_benders.py).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from vanilla_benders import read_inputs, solve_subproblem_duals


class BendersMDPSolver:
    """
    Incremental Benders: one step per call, persistent master model.
    Driven by env_benders_mdp: after action selects cuts, env calls add_cuts(selected).
    base_dir at construct; reset() has no args for env.reset().
    """

    def __init__(
        self,
        base_dir: str,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._base_dir = base_dir
        self._seed = seed
        self._master: Optional[gp.Model] = None
        self._y: Any = None
        self._z: Any = None
        self._theta: Any = None
        self._rng: Optional[np.random.Generator] = None
        self._cut_constrs: List[Any] = []

        # Dimensions and data (cached from read_inputs on reset)
        self.I: int = 0
        self.J: int = 0
        self.K: int = 0  # N scenarios
        self._f: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._M: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        self._p: Optional[np.ndarray] = None
        self._r: Optional[np.ndarray] = None
        self._c: Optional[np.ndarray] = None
        self._d: Optional[np.ndarray] = None
        self._const_k_list: Optional[List[float]] = None

        # Internal iteration / bound state
        self._it: int = 0
        self._LB: float = -np.inf
        self._UB_best: float = np.inf
        self._cut_count: int = 0
        self._y_val: Optional[np.ndarray] = None
        self._z_val: Optional[np.ndarray] = None
        self._theta_val: Optional[np.ndarray] = None

        # Filled in solve_subproblems; read-only for env snapshot
        self.last_subproblem_stats: Dict[str, Any] = {}

    def reset(self) -> None:
        """
        Env-compatible, no args. If master missing: read data and build model.
        Else: remove Benders cuts and reset iter state only (no re-read).
        """
        if self._master is None:
            (
                _station_ids,
                _customer_ids,
                self._f,
                self._b,
                self._M,
                self._C,
                self._p,
                self._r,
                self._c,
                self._d,
                self.I,
                self.J,
                self.K,
            ) = read_inputs(self._base_dir)
            self._const_k_list = [
                -float(np.dot(self._r, self._d[k, :])) for k in range(self.K)
            ]
            self._rng = np.random.default_rng(self._seed)
            self._build_master()
        else:
            self._remove_cut_constraints()

        self._it = 0
        self._LB = -np.inf
        self._UB_best = np.inf
        self._cut_count = 0
        self._y_val = None
        self._z_val = None
        self._theta_val = None
        self.last_subproblem_stats = {}

    def _build_master(self) -> None:
        """Build Gurobi master (y, z, theta, linking, theta_lb, objective); matches vanilla."""
        self._master = gp.Model("master_benders_mdp")
        self._master.Params.OutputFlag = 0
        if self._seed is not None:
            self._master.Params.Seed = int(self._seed)
        self._master.Params.Threads = 1

        I, K = self.I, self.K
        Iset = range(I)
        Kset = range(K)

        self._y = self._master.addVars(Iset, vtype=GRB.BINARY, name="y")
        self._z = self._master.addVars(Iset, vtype=GRB.INTEGER, lb=0, name="z")
        self._theta = self._master.addVars(
            Kset, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta"
        )

        for k in Kset:
            self._master.addConstr(
                self._theta[k] >= self._const_k_list[k], name=f"theta_lb_{k}"
            )
        for i in Iset:
            self._master.addConstr(
                self._z[i] <= float(self._M[i]) * self._y[i], name=f"link_{i}"
            )

        N = float(K)
        self._master.setObjective(
            gp.quicksum(self._f[i] * self._y[i] + self._b[i] * self._z[i] for i in Iset)
            + (1.0 / N) * gp.quicksum(self._theta[k] for k in Kset),
            GRB.MINIMIZE,
        )
        self._cut_constrs = []

    def _remove_cut_constraints(self) -> None:
        """Remove all Benders cut constraints and reset Gurobi model state."""
        for c in self._cut_constrs:
            self._master.remove(c)
        self._cut_constrs.clear()
        self._cut_count = 0
        self._master.update()
        self._master.reset()

    def solve_master(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """
        Solve current RMP; cache solution.
        Returns:
            z_current, theta_current, LB, master_info (incl. solve_time_master)
        """
        if self._master is None:
            raise RuntimeError("solver not initialized: call reset() first")
        t0 = time.time()
        self._master.optimize()
        solve_time = time.time() - t0
        if self._master.Status != GRB.OPTIMAL:
            raise RuntimeError(
                f"Master not optimal. Status={self._master.Status}"
            )

        self._it += 1
        self._LB = float(self._master.ObjVal)
        Iset = range(self.I)
        Kset = range(self.K)
        self._y_val = np.array([self._y[i].X for i in Iset], dtype=float)
        self._z_val = np.array([self._z[i].X for i in Iset], dtype=float)
        self._theta_val = np.array([self._theta[k].X for k in Kset], dtype=float)

        return (
            self._z_val,
            self._theta_val,
            self._LB,
            {"solve_time_master": solve_time},
        )

    def solve_subproblems(
        self,
        z: Optional[np.ndarray] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        float,
        Dict[str, float],
        Dict[str, Any],
    ]:
        """
        Solve all scenario subproblems at current z; candidate cuts and UB update.
        solve_master() must be called first or RuntimeError.
        Returns:
            candidate_cuts, UB, violation_stats, sub_info (incl. solve_time_sub)
        """
        if self._z_val is None:
            raise RuntimeError(
                "solve_master() must be called before solve_subproblems()"
            )
        z_use = z if z is not None else self._z_val
        z_use = np.asarray(z_use, dtype=float).ravel()[: self.I]
        if z_use.size < self.I:
            raise ValueError(f"z length {z_use.size} < I={self.I}")

        I, J, K = self.I, self.J, self.K
        Iset = range(I)
        Kset = range(K)
        f, b, C, c, d, p, r = (
            self._f,
            self._b,
            self._C,
            self._c,
            self._d,
            self._p,
            self._r,
        )

        t0 = time.time()
        candidate_cuts: List[Dict[str, Any]] = []
        Q_vals = np.zeros(K, dtype=float)
        iter_counts: List[int] = []
        gurobi_runtimes: List[float] = []
        n_positive_beta = 0
        z_l1 = float(np.sum(z_use))
        n_zero_cap = int(np.sum(np.asarray(C, dtype=float).ravel()[:I] * z_use <= 1e-12))

        for k in Kset:
            sub_seed = None
            if self._seed is not None:
                sub_seed = int(self._seed) + int(self._it) * 1_000_003 + int(k) * 997
            Qk, alpha_k, beta_k, _, sub_stats = solve_subproblem_duals(
                c=c,
                d_k=d[k, :],
                p=p,
                r=r,
                C=C,
                z_val=z_use,
                OutputFlag=0,
                seed=sub_seed,
            )
            Q_vals[k] = Qk
            iter_counts.append(int(sub_stats.get("iter_count", 0)))
            gurobi_runtimes.append(float(sub_stats.get("runtime_sec", 0.0)))
            n_positive_beta += int(np.sum(np.asarray(beta_k, dtype=float) > 1e-8))
            # Same as vanilla: theta_k >= Q_k(z^t) + sum_i (beta_i^k * C_i)(z_i - z_i^t)
            coef_z = np.array([beta_k[i] * float(C[i]) for i in Iset], dtype=float)
            rhs_const = Qk - float(np.dot(coef_z, z_use))
            assert np.isfinite(rhs_const), "rhs_const must be finite"
            assert coef_z.shape == (I,), f"coef_z.shape {coef_z.shape} != (I={I},)"
            candidate_cuts.append({
                "k": k,
                "rhs_const": rhs_const,
                "coef_z": coef_z,
                "Qk": Qk,
                "alpha": alpha_k,
                "beta": beta_k,
            })

        first_stage_cost = float(np.dot(f, self._y_val) + np.dot(b, z_use))
        recourse_avg = float(np.mean(Q_vals))
        UB_candidate = first_stage_cost + recourse_avg
        self._UB_best = min(self._UB_best, UB_candidate)

        # violation: max(0, rhs + coef_z@z - theta_k); z_use matches passed z
        violations = []
        for cut in candidate_cuts:
            k = cut["k"]
            rhs = cut["rhs_const"]
            coef = np.asarray(cut["coef_z"])
            viol = max(
                0.0,
                rhs + float(np.dot(coef, z_use)) - self._theta_val[k],
            )
            violations.append(viol)
        violations_arr = np.array(violations)
        violation_stats = {
            "mean": float(np.mean(violations_arr)),
            "max": float(np.max(violations_arr)),
        }

        # Read-only stats for env state
        recourse_avg = float(np.mean(Q_vals))
        self.last_subproblem_stats = {
            "Q_vals": np.asarray(Q_vals, dtype=float),
            "recourse_avg": recourse_avg,
            "recourse_max": float(np.max(Q_vals)),
            "recourse_min": float(np.min(Q_vals)),
            "recourse_std": float(np.std(Q_vals)) if K > 1 else 0.0,
            "y_val": np.asarray(self._y_val, dtype=float),
            "z_val": np.asarray(z_use, dtype=float),
        }

        solve_time_sub = time.time() - t0
        iter_arr = np.asarray(iter_counts, dtype=float)
        rt_arr = np.asarray(gurobi_runtimes, dtype=float)
        sub_info = {
            "solve_time_sub": solve_time_sub,
            "sub_iter_count_sum": float(np.sum(iter_arr)) if iter_arr.size else 0.0,
            "sub_iter_count_mean": float(np.mean(iter_arr)) if iter_arr.size else 0.0,
            "sub_iter_count_max": float(np.max(iter_arr)) if iter_arr.size else 0.0,
            "sub_gurobi_runtime_sum": float(np.sum(rt_arr)) if rt_arr.size else 0.0,
            "sub_gurobi_runtime_mean": float(np.mean(rt_arr)) if rt_arr.size else 0.0,
            "z_l1": z_l1,
            "n_zero_cap": int(n_zero_cap),
            "n_positive_beta": int(n_positive_beta),
            "n_scenarios_solved": int(K),
        }
        return candidate_cuts, self._UB_best, violation_stats, sub_info

    def add_cuts(self, selected_cuts: List[Dict[str, Any]]) -> None:
        """Add selected cuts to RMP; no internal selection."""
        if self._master is None:
            raise RuntimeError("solver not initialized: call reset() first")
        I = self.I
        Iset = range(I)

        for cut in selected_cuts:
            k = cut["k"]
            assert isinstance(k, int), "cut k must be int"
            assert 0 <= k < self.K, f"cut k={k} must satisfy 0 <= k < K={self.K}"
            coef_z = np.asarray(cut["coef_z"], dtype=float)
            assert coef_z.shape == (self.I,), f"coef_z.shape {coef_z.shape} != (I={self.I},)"
            rhs_const = float(cut["rhs_const"])
            assert np.isfinite(rhs_const), "rhs_const must be finite"

            rhs_z = gp.quicksum(coef_z[i] * self._z[i] for i in Iset)
            constr = self._master.addConstr(
                self._theta[k] >= rhs_const + rhs_z,
                name=f"benders_it{self._it}_k{k}_{len(self._cut_constrs)}",
            )
            self._cut_constrs.append(constr)

        self._cut_count += len(selected_cuts)

    def num_cuts(self) -> int:
        """Number of Benders cuts in RMP (env interface)."""
        return self._cut_count

    def get_master_cut_count(self) -> int:
        """Same as num_cuts()."""
        return self._cut_count

    def get_current_z(self) -> np.ndarray:
        """Current master z (shape (I,))."""
        if self._z_val is None:
            raise RuntimeError("no solution yet: solve_master() first")
        return self._z_val

    def get_current_theta(self) -> np.ndarray:
        """Current master theta (shape (K,))."""
        if self._theta_val is None:
            raise RuntimeError("no solution yet: solve_master() first")
        return self._theta_val


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # Same default data dir as vanilla (project root or cwd)
    base_dir = "dataset test Feb"
    if not os.path.isdir(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), "dataset test Feb")
    if not os.path.isdir(base_dir):
        print("Skip smoke test: dataset test Feb not found")
    else:
        try:
            solver = BendersMDPSolver(base_dir=base_dir)
            solver.reset()

            for t in range(3):
                z, theta, lb, _ = solver.solve_master()
                cand, ub, viol, _ = solver.solve_subproblems(z)
                # Simple rule: first 20% or at least 1, at most 5
                n = len(cand)
                k_sel = max(1, min(5, max(1, int(0.2 * n))))
                selected = cand[:k_sel]
                solver.add_cuts(selected)

                gap = abs(ub - lb) / (abs(ub) + 1e-9)
                print(
                    f"iter {t+1} | LB={lb:.4f} UB={ub:.4f} gap={gap:.6f} "
                    f"cut_count={solver.num_cuts()} selected={len(selected)}"
                )

            print("Smoke test OK.")
        except Exception as e:
            print(f"Smoke test skipped or failed: {e}")
            print("(Ensure dataset test Feb exists and openpyxl is installed if using Excel.)")
