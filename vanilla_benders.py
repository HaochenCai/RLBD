import os

import pandas as pd
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, List, Optional

from benders_trajectory_logger import BendersTrajectoryLogger


# -----------------------------
# Helpers (same spirit as your direct_solve_SAA.py)
# -----------------------------
def read_first_sheet(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


def _col(df: pd.DataFrame, candidates):
    """Find first existing column name from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def build_cost_matrix(dist_raw: pd.DataFrame, station_ids, customer_ids) -> np.ndarray:
    """
    Build c[i,j]. Supports:
      - pure matrix with shape I x J (no id columns)
      - first row/col as headers with ids
    """
    I = len(station_ids)
    J = len(customer_ids)

    df = dist_raw.copy()

    # If looks like it already is numeric matrix I x J
    if df.shape == (I, J) and np.issubdtype(df.dtypes.iloc[0], np.number):
        return df.to_numpy(dtype=float)

    # Try header row customer ids, first col station ids
    # If first column is station id
    first_col = df.columns[0]
    if not np.issubdtype(df[first_col].dtype, np.number):
        # Treat first col as station ids
        df = df.set_index(first_col)

    # Now columns should be customer ids (maybe numeric or str)
    # Align index/columns to station_ids/customer_ids
    # Convert both sides to string for robust match
    df.index = df.index.map(str)
    df.columns = df.columns.map(str)

    sid_str = list(map(str, station_ids))
    cid_str = list(map(str, customer_ids))

    missing_i = [s for s in sid_str if s not in df.index]
    missing_j = [c for c in cid_str if c not in df.columns]
    if missing_i or missing_j:
        raise ValueError(
            f"Distance alignment failed. Missing stations: {missing_i[:5]} ... "
            f"Missing customers: {missing_j[:5]} ..."
        )

    df = df.loc[sid_str, cid_str]
    return df.to_numpy(dtype=float)


def build_demand_matrix(dem_raw: pd.DataFrame, customer_ids) -> np.ndarray:
    """
    Build d[k,j]. Supports:
      - numeric matrix N x J (no id column)
      - first row has customer ids, first col scenario id
    """
    df = dem_raw.copy()
    J = len(customer_ids)

    # If already numeric N x J
    if df.shape[1] == J and all(np.issubdtype(t, np.number) for t in df.dtypes):
        return df.to_numpy(dtype=float)

    # If first col is scenario id -> set as index
    first_col = df.columns[0]
    if not np.issubdtype(df[first_col].dtype, np.number):
        df = df.set_index(first_col)

    df.columns = df.columns.map(str)
    cid_str = list(map(str, customer_ids))

    missing = [c for c in cid_str if c not in df.columns]
    if missing:
        # Try cust1, cust2, ... style (demand sheet columns)
        cust_cols = [f"cust{c}" for c in cid_str]
        if all(col in df.columns for col in cust_cols):
            df = df.loc[:, cust_cols]
            return df.to_numpy(dtype=float)
        raise ValueError(f"Demand alignment failed. Missing customers in scenario sheet: {missing[:8]} ...")

    df = df.loc[:, cid_str]
    return df.to_numpy(dtype=float)


def read_inputs(base_dir: str):
    """
    Expect the same four files (same names as your direct solver uses).
    """
    def _resolve_input_path(
        *,
        prefix: str,
        legacy_name: str,
    ) -> str:
        legacy_path = os.path.join(base_dir, legacy_name)
        if os.path.isfile(legacy_path):
            return legacy_path

        candidates = [
            fn
            for fn in os.listdir(base_dir)
            if fn.lower().startswith(f"{prefix.lower()}_") and fn.lower().endswith(".xlsx")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Cannot find input file for prefix '{prefix}' under {base_dir}. "
                f"Tried legacy name '{legacy_name}' and pattern '{prefix}_*.xlsx'."
            )
        if len(candidates) == 1:
            return os.path.join(base_dir, candidates[0])

        candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(base_dir, fn)), reverse=True)
        return os.path.join(base_dir, candidates[0])

    station_path = _resolve_input_path(prefix="station_side", legacy_name="station_side_episode_test.xlsx")
    customer_path = _resolve_input_path(prefix="customer_side", legacy_name="customer_side_episode_test.xlsx")
    dist_path = _resolve_input_path(prefix="distance", legacy_name="distance_episode_test.xlsx")
    demand_path = _resolve_input_path(prefix="demand_scenario", legacy_name="demand_scenario_episode_test.xlsx")

    st = read_first_sheet(station_path)
    cu = read_first_sheet(customer_path)
    dist = read_first_sheet(dist_path)
    dem = read_first_sheet(demand_path)

    # --- station side ---
    sid_col = _col(st, ["station_id", "station", "id"])
    if sid_col is None:
        raise ValueError("station_side.xlsx must contain a station id column like station_id.")

    station_ids = st[sid_col].tolist()
    I = len(station_ids)

    f_col = _col(st, ["fixed_cost_f", "f", "fixed_cost", "open_cost"])
    b_col = _col(st, ["charge_cost_b", "charger_cost_b", "b", "pole_cost", "charger_cost"])
    M_col = _col(st, ["max_chargers_M", "m", "M", "max_poles", "max_chargers"])
    C_col = _col(st, ["capacity_per_charger_C", "cap_per_charger_C", "c", "C", "cap", "capacity", "capacity_per_pole"])

    if f_col is None or b_col is None or M_col is None:
        raise ValueError("station_side.xlsx must contain columns for f, b, M (fixed cost, pole cost, max poles).")

    f = st[f_col].to_numpy(dtype=float)
    b = st[b_col].to_numpy(dtype=float)
    M = st[M_col].to_numpy(dtype=float)
    if C_col is None:
        C = np.full(I, 100.0, dtype=float)
    else:
        C = st[C_col].to_numpy(dtype=float)

    # --- customer side ---
    cid_col = _col(cu, ["customer_id", "customer", "id"])
    if cid_col is None:
        raise ValueError("customer_side.xlsx must contain a customer id column like customer_id.")

    customer_ids = cu[cid_col].tolist()
    J = len(customer_ids)

    p_col = _col(cu, ["shortage_penalty_p", "penalty_p", "p", "penalty"])
    r_col = _col(cu, ["revenue_r", "revenue", "r", "reward"])

    if p_col is None or r_col is None:
        raise ValueError("customer_side.xlsx must contain columns for p and r (penalty, revenue).")

    p = cu[p_col].to_numpy(dtype=float)
    r = cu[r_col].to_numpy(dtype=float)

    # --- c and d ---
    c = build_cost_matrix(dist, station_ids, customer_ids)        # (I,J)
    d = build_demand_matrix(dem, customer_ids)                    # (N,J)
    N = d.shape[0]

    return station_ids, customer_ids, f, b, M, C, p, r, c, d, I, J, N


# -----------------------------
# Subproblem: scenario-k LP given z
# -----------------------------
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
):
    """
    Primal subproblem (always feasible because s exists):
      min sum_{i,j} c_ij x_ij + sum_j p_j s_j - sum_j r_j d_kj
      s.t. sum_i x_ij + s_j = d_kj         (alpha_j free)
           sum_j x_ij <= C_i z_i           (beta_i >= 0)
           x,s >= 0
    Returns:
      Qk (optimal obj),
      alpha (J,),
      beta  (I,),
      const_k,
      solver_stats (dict: iter_count, runtime_sec, z_l1, n_zero_cap)
    """
    I, J = c.shape
    z_arr = np.asarray(z_val, dtype=float).ravel()[:I]
    C_arr = np.asarray(C, dtype=float).ravel()[:I]
    sub = gp.Model("sub_k")
    sub.Params.OutputFlag = OutputFlag
    # Use simplex to ensure Pi is available and stable
    sub.Params.Method = 1
    sub.Params.Threads = 1
    if seed is not None:
        sub.Params.Seed = int(seed)

    x = sub.addVars(I, J, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    s = sub.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    demand_con = {}
    for j in range(J):
        demand_con[j] = sub.addConstr(gp.quicksum(x[i, j] for i in range(I)) + s[j] == float(d_k[j]),
                                      name=f"demand_{j}")

    cap_con = {}
    for i in range(I):
        cap_con[i] = sub.addConstr(gp.quicksum(x[i, j] for j in range(J)) <= float(C_arr[i]) * float(z_arr[i]),
                                   name=f"cap_{i}")

    const_k = -float(np.dot(r, d_k))
    sub.setObjective(
        gp.quicksum(c[i, j] * x[i, j] for i in range(I) for j in range(J))
        + gp.quicksum(p[j] * s[j] for j in range(J))
        + const_k,
        GRB.MINIMIZE
    )

    sub.optimize()
    if sub.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Subproblem not optimal. Status={sub.Status}")

    alpha = np.array([demand_con[j].Pi for j in range(J)], dtype=float)  # free duals
    beta = np.array([cap_con[i].Pi for i in range(I)], dtype=float)      # >=0 duals for <= constraints
    cap_rhs = C_arr * z_arr
    solver_stats = {
        "iter_count": int(getattr(sub, "IterCount", 0) or 0),
        "runtime_sec": float(getattr(sub, "Runtime", 0.0) or 0.0),
        "z_l1": float(np.sum(z_arr)),
        "n_zero_cap": int(np.sum(cap_rhs <= 1e-12)),
    }

    return float(sub.ObjVal), alpha, beta, const_k, solver_stats


# -----------------------------
# Vanilla Benders (multi-cut)
# -----------------------------
def solve_benders_multicut(
    base_dir: str,
    max_iters: int = 200,
    tol_rel_gap: float = 1e-2,
    verbose: bool = True,
    trajectory_logger: Optional[BendersTrajectoryLogger] = None,
    return_timing: bool = False,
    rng_seed: Optional[int] = None,
    max_elapsed_sec: Optional[float] = None,
):
    """
    trajectory_logger: if a BendersTrajectoryLogger, logs each iter and cuts (MDP/RL/analysis).
    rng_seed: if set, Gurobi Seed + single-thread subs for reproducible runs.
    max_elapsed_sec: if >0, stop when cumulative master+sub solve time reaches cap
        (same as master_sub_total_sec / benchmark vanilla_solve_time_sec); 0 or None disables.
    """
    station_ids, customer_ids, f, b, M, C, p, r, c, d, I, J, N = read_inputs(base_dir)

    Iset = range(I)
    Kset = range(N)

    # Master model
    master = gp.Model("master_benders_multicut")
    master.Params.OutputFlag = 0  # we'll do our own printing
    master.Params.Threads = 1
    if rng_seed is not None:
        master.Params.Seed = int(rng_seed)

    y = master.addVars(Iset, vtype=GRB.BINARY, name="y")
    z = master.addVars(Iset, vtype=GRB.INTEGER, lb=0, name="z")

    # theta_k: one per scenario (multi-cut)
    # Lower bound: theta_k >= const_k because c*x + p*s >= 0 always
    const_k_list = [-float(np.dot(r, d[k, :])) for k in Kset]
    theta = master.addVars(Kset, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    for k in Kset:
        master.addConstr(theta[k] >= const_k_list[k], name=f"theta_lb_{k}")

    # linking constraints
    for i in Iset:
        master.addConstr(z[i] <= float(M[i]) * y[i], name=f"link_{i}")

    master.setObjective(
        gp.quicksum(f[i] * y[i] + b[i] * z[i] for i in Iset)
        + (1.0 / N) * gp.quicksum(theta[k] for k in Kset),
        GRB.MINIMIZE
    )

    # Benders loop
    UB = float("inf")
    LB = -float("inf")
    best_y = None
    best_z = None
    master_time_cum = 0.0
    sub_time_cum = 0.0
    iteration_curve: List[Dict[str, float]] = []

    if verbose:
        print(f"=== Vanilla Benders (multi-cut) ===")
        print(f"I={I}, J={J}, N(scenarios)={N}")
        print("Iter |        LB |              UB |   gap(%) | cuts | y, z | fs_cost | rec_avg")

    phase_t0 = time.perf_counter()
    cap = float(max_elapsed_sec) if max_elapsed_sec is not None and float(max_elapsed_sec) > 0.0 else None
    last_it = 0
    stop_reason = "max_iters"
    for it in range(1, max_iters + 1):
        last_it = it
        t_master0 = time.perf_counter()
        master.optimize()
        master_time_iter = time.perf_counter() - t_master0
        master_time_cum += master_time_iter
        if master.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Master not optimal. Status={master.Status}")

        LB = float(master.ObjVal)

        y_val = np.array([y[i].X for i in Iset], dtype=float)
        z_val = np.array([z[i].X for i in Iset], dtype=float)
        theta_val = np.array([theta[k].X for k in Kset], dtype=float)

        # Solve all scenario subproblems (multi-cut)
        Q_vals = np.zeros(N, dtype=float)
        cuts_added = 0
        sub_time_before = sub_time_cum
        if trajectory_logger is not None:
            cut_rhs_list, cut_coef_list, alpha_list, beta_list = [], [], [], []
            trajectory_logger.start_iteration((it - 1) * N)

        for k in Kset:
            t_sub0 = time.perf_counter()
            sub_seed = None
            if rng_seed is not None:
                sub_seed = int(rng_seed) + int(it) * 1_000_003 + int(k) * 997
            Qk, alpha_k, beta_k, const_k, _sub_stats = solve_subproblem_duals(
                c=c,
                d_k=d[k, :],
                p=p,
                r=r,
                C=C,
                z_val=z_val,
                OutputFlag=0,
                seed=sub_seed,
            )
            sub_time_cum += time.perf_counter() - t_sub0
            Q_vals[k] = Qk

            # Benders optimality cut (correct form):
            #   theta_k >= Q_k(z^t) + sum_i (beta_i^k * C_i) * (z_i - z_i^t)
            coef_z = np.array([beta_k[i] * float(C[i]) for i in Iset], dtype=float)
            rhs_const = Qk - float(np.dot(coef_z, z_val))
            rhs_z = gp.quicksum(coef_z[i] * z[i] for i in Iset)
            master.addConstr(theta[k] >= rhs_const + rhs_z, name=f"benders_it{it}_k{k}")
            cuts_added += 1
            if trajectory_logger is not None:
                cut_rhs_list.append(rhs_const)
                cut_coef_list.append(coef_z)
                alpha_list.append(alpha_k)
                beta_list.append(beta_k)

        # Compute UB with current first-stage solution + avg subproblem optimal values
        first_stage_cost = float(np.dot(f, y_val) + np.dot(b, z_val))
        recourse_avg = float(np.mean(Q_vals))
        UB_candidate = first_stage_cost + recourse_avg
        if UB_candidate < UB:
            UB = UB_candidate
            best_y = y_val.copy()
            best_z = z_val.copy()

        # Gap
        gap = (UB - LB) / (abs(UB) + 1e-9)
        wall_clock_sec = float(time.perf_counter() - phase_t0)
        sub_time_iter = float(sub_time_cum - sub_time_before)
        iteration_curve.append(
            {
                "iteration": float(it),
                "lb": float(LB),
                "ub": float(UB),
                "rel_gap": float(gap),
                "cuts_added": float(cuts_added),
                "wall_clock_sec": wall_clock_sec,
                "solve_time_master": float(master_time_iter),
                "solve_time_sub": sub_time_iter,
                "master_time_cum_sec": float(master_time_cum),
                "sub_time_cum_sec": float(sub_time_cum),
            }
        )
        if trajectory_logger is not None:
            trajectory_logger.log_iteration(
                iteration=it,
                lb=LB,
                ub=UB,
                gap=gap,
                y_val=y_val,
                z_val=z_val,
                theta_val=theta_val,
                first_stage_cost=first_stage_cost,
                recourse_avg=recourse_avg,
                Q_vals=Q_vals,
                cuts_added=cuts_added,
                cut_rhs_const=cut_rhs_list,
                cut_coef_z=cut_coef_list,
                cut_alpha=alpha_list,
                cut_beta=beta_list,
                master_num_constraints=len(master.getConstrs()),
                master_num_variables=len(master.getVars()),
                master_solve_time=master_time_iter,
                wall_clock_sec=wall_clock_sec,
            )
        if verbose:
            print(f"{it:>4d} | {LB:>16.6f} | {UB:>16.6f} | {100*gap:>7.3f} | {cuts_added:>10d} | "
                  f"y={y_val}, z={z_val} | fs={first_stage_cost:.2f} rec={recourse_avg:.2f}")

        if gap <= tol_rel_gap:
            stop_reason = "gap"
            break
        solve_cum = master_time_cum + sub_time_cum
        if cap is not None and solve_cum >= cap:
            stop_reason = "time"
            break

    # Report best incumbent
    print("\n=== Best incumbent (by station_id) ===")
    for i in Iset:
        print(f"station {station_ids[i]}: y={best_y[i]:.0f}, z={best_z[i]:.0f}")

    print("\n=== Final bounds ===")
    print(f"LB (last master): {LB:.6f}")
    print(f"UB (best incumbent): {UB:.6f}")
    print(f"Final rel gap: {100*((UB-LB)/(abs(UB)+1e-9)):.6f}%")

    if return_timing:
        return best_y, best_z, LB, UB, {
            "master_time_sec": float(master_time_cum),
            "sub_time_sec": float(sub_time_cum),
            "master_sub_total_sec": float(master_time_cum + sub_time_cum),
            "iters_used": int(last_it),
            "benders_wall_elapsed_sec": float(time.perf_counter() - phase_t0),
            "stop_reason": stop_reason,
            "iteration_curve": iteration_curve,
        }
    return best_y, best_z, LB, UB
