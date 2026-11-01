from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np

from benders_trajectory_logger import BendersTrajectoryLogger
from vanilla_benders import solve_benders_multicut


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_vanilla(
    *,
    base_dir: str,
    out_dir: str,
    run_name: str | None,
    seed: int,
    max_iters: int,
    tol_rel_gap: float,
    verbose: bool,
    save_trajectory: bool,
    max_elapsed_sec: float | None = None,
) -> Dict[str, Any]:
    _set_seed(seed)

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = BendersTrajectoryLogger(log_dual_vectors=True) if save_trajectory else None
    t0 = time.time()
    y, z, lb, ub, timing = solve_benders_multicut(
        base_dir=base_dir,
        max_iters=max_iters,
        tol_rel_gap=tol_rel_gap,
        verbose=verbose,
        trajectory_logger=logger,
        return_timing=True,
        rng_seed=int(seed),
        max_elapsed_sec=max_elapsed_sec,
    )
    elapsed = time.time() - t0

    if not np.isfinite(lb) or not np.isfinite(ub):
        raise RuntimeError(f"invalid vanilla outputs: LB={lb}, UB={ub}")
    denom = max(1e-12, abs(float(ub)))
    rel_gap = float((float(ub) - float(lb)) / denom)
    converged = bool(rel_gap <= float(tol_rel_gap))
    iters_used = int(timing.get("iters_used", 0))
    benders_wall = float(timing.get("benders_wall_elapsed_sec", 0.0))
    stop_reason = str(timing.get("stop_reason", "max_iters"))

    summary: Dict[str, Any] = {
        "base_dir": base_dir,
        "seed": seed,
        "max_iters": max_iters,
        "max_elapsed_sec": float(max_elapsed_sec) if max_elapsed_sec and max_elapsed_sec > 0 else None,
        "tol_rel_gap": tol_rel_gap,
        "LB": float(lb),
        "UB": float(ub),
        "rel_gap": rel_gap,
        "converged": converged,
        "iters_used": iters_used,
        "stop_reason": stop_reason,
        "benders_wall_elapsed_sec": benders_wall,
        "elapsed_sec": elapsed,
        "master_time_sec": float(timing.get("master_time_sec", 0.0)),
        "sub_time_sec": float(timing.get("sub_time_sec", 0.0)),
        "master_sub_total_sec": float(timing.get("master_sub_total_sec", 0.0)),
        "num_stations": int(len(y)),
        "num_solution_z": int(len(z)),
        "run_name": run_name,
    }

    curve = list(timing.get("iteration_curve") or [])
    curve_path = os.path.join(run_dir, "iterations.csv")
    curve_fields = [
        "iteration",
        "lb",
        "ub",
        "rel_gap",
        "cuts_added",
        "wall_clock_sec",
        "solve_time_master",
        "solve_time_sub",
        "master_time_cum_sec",
        "sub_time_cum_sec",
    ]
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=curve_fields)
        w.writeheader()
        for row in curve:
            w.writerow({k: row.get(k) for k in curve_fields})
    summary["iterations_path"] = curve_path

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    solution_path = os.path.join(run_dir, "solution.json")
    with open(solution_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "y": [float(v) for v in y],
                "z": [float(v) for v in z],
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    if logger is not None:
        traj_path = os.path.join(run_dir, "trajectory.json")
        with open(traj_path, "w", encoding="utf-8") as fh:
            json.dump(logger.get_trajectory_dict(), fh, ensure_ascii=False, indent=2)
        summary["trajectory_path"] = traj_path
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vanilla multi-cut Benders baseline")
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./vanilla_runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_iters", type=int, default=200)
    p.add_argument(
        "--max_elapsed_sec",
        type=float,
        default=0.0,
        help=(
            "Cap on cumulative master+sub solve seconds inside the Benders loop "
            "(same quantity as master_sub_total_sec / benchmark vanilla_solve_time_sec); "
            "0 disables. gap<=tol still stops earlier. Last full iteration may slightly exceed the cap."
        ),
    )
    p.add_argument("--tol_rel_gap", type=float, default=1e-2)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--save_trajectory", action="store_true")
    p.add_argument("--fail_on_not_converged", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_vanilla(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        seed=args.seed,
        max_iters=args.max_iters,
        tol_rel_gap=args.tol_rel_gap,
        verbose=not args.quiet,
        save_trajectory=args.save_trajectory,
        max_elapsed_sec=float(args.max_elapsed_sec) if args.max_elapsed_sec and args.max_elapsed_sec > 0 else None,
    )
    print("Vanilla run finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_not_converged and not bool(summary.get("converged", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
