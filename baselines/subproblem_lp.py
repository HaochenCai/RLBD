"""
Per-scenario second-stage LP (Gurobi) for incremental Benders / MDP.

Split from data-only module so read path stays free of master-loop Gurobi code.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def solve_subproblem_duals(
    c,
    d_k,
    p,
    r,
    C,
    z_val,
    OutputFlag=0,
    *,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """
    Primal subproblem (always feasible because s exists):
      min sum_{i,j} c_ij x_ij + sum_j p_j s_j - sum_j r_j d_kj
      s.t. sum_i x_ij + s_j = d_kj         (alpha_j free)
           sum_j x_ij <= C_i z_i           (beta_i >= 0)
           x,s >= 0
    Returns:
      Qk, alpha, beta, const_k,
      solver_stats (iter_count, runtime_sec, z_l1, n_zero_cap)
    """
    I, J = c.shape
    z_arr = np.asarray(z_val, dtype=float).ravel()[:I]
    C_arr = np.asarray(C, dtype=float).ravel()[:I]
    sub = gp.Model("sub_k")
    sub.Params.OutputFlag = OutputFlag
    sub.Params.Threads = 1
    sub.Params.Method = 1
    if seed is not None:
        sub.Params.Seed = int(seed)

    x = sub.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    s = sub.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    demand_con = {}
    for j in range(J):
        demand_con[j] = sub.addConstr(
            gp.quicksum(x[i, j] for i in range(I)) + s[j] == float(d_k[j]),
            name=f"demand_{j}",
        )

    cap_con = {}
    for i in range(I):
        cap_con[i] = sub.addConstr(
            gp.quicksum(x[i, j] for j in range(J)) <= float(C_arr[i]) * float(z_arr[i]),
            name=f"cap_{i}",
        )

    const_k = -float(np.dot(r, d_k))
    sub.setObjective(
        gp.quicksum(c[i, j] * x[i, j] for i in range(I) for j in range(J))
        + gp.quicksum(p[j] * s[j] for j in range(J))
        + const_k,
        GRB.MINIMIZE,
    )

    sub.optimize()
    if sub.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Subproblem not optimal. Status={sub.Status}")

    alpha = np.array([demand_con[j].Pi for j in range(J)], dtype=float)
    beta = np.array([cap_con[i].Pi for i in range(I)], dtype=float)
    cap_rhs = C_arr * z_arr
    solver_stats = {
        "iter_count": int(getattr(sub, "IterCount", 0) or 0),
        "runtime_sec": float(getattr(sub, "Runtime", 0.0) or 0.0),
        "z_l1": float(np.sum(z_arr)),
        "n_zero_cap": int(np.sum(cap_rhs <= 1e-12)),
    }

    return float(sub.ObjVal), alpha, beta, const_k, solver_stats


__all__ = ["solve_subproblem_duals"]
