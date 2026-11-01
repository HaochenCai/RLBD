"""
MDP Commander: orchestrate Benders MDP experiments.

Batch layout (optional group layer):
  dataset_root/seed_*/          (no --group)
  dataset_root/<group>/seed_*/  (--group <folder name>)

Default: one episode per immediate subdirectory of the scan root (top-k incremental Benders).
Each episode writes steps.jsonl and summary.csv under its run folder.
Per-iter steps.jsonl includes wall_clock_sec (cumulative since episode start),
plus solve_time_master / solve_time_sub when available.
Batch mode: one CLI invocation uses a single session folder under out_dir; each dataset subdir is a subfolder inside it.
Optional per-episode wall-clock cap via --time_limit (seconds, from reset); when reached, that episode stops and batch continues; batch_summaries.json aggregates episode summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from benders_solver_mdp import BendersMDPSolver
from env_benders_mdp import BendersMDPEnv
from reward_plugins import gap_reduction_reward
from cut_selection import CutSelector, DEFAULT_WEIGHTS, CUT_FEATURE_DIM, SELECTION_TOP_K


def discover_dataset_dirs(parent_dir: str) -> List[str]:
    """
    List one-level subdirs of parent_dir (skip names starting with . or ~).
    seed_<digits> first, sorted numerically; other dirs after, sorted by name.
    """
    if not os.path.isdir(parent_dir):
        return []
    out: List[str] = []
    for name in os.listdir(parent_dir):
        if name.startswith(".") or name.startswith("~"):
            continue
        path = os.path.join(parent_dir, name)
        if os.path.isdir(path):
            out.append(path)

    def sort_key(p: str) -> tuple:
        bn = os.path.basename(p)
        m = re.match(r"^seed_(\d+)$", bn, flags=re.IGNORECASE)
        if m:
            return (0, int(m.group(1)), bn)
        return (1, bn.lower(), bn)

    out.sort(key=sort_key)
    return out


def _dataset_seed_from_base_dir(base_dir: str) -> Optional[int]:
    m = re.search(r"seed_(\d+)$", os.path.basename(os.path.normpath(base_dir)), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def _make_policy(policy_name: str, seed: int):
    """Return a small policy object with .act(state) -> action dict."""

    if policy_name == "random_k":
        class _RandomKPolicy:
            def act(self, state: Any) -> Dict[str, Any]:
                return {"scorer_params": {"mode": "random_k"}}

        return _RandomKPolicy()

    if policy_name == "default":
        weights = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)
    elif policy_name == "topk":
        weights = np.zeros(CUT_FEATURE_DIM, dtype=np.float64)
        weights[0] = 1.0
    elif policy_name == "random":
        # Random linear weights (score-based); not the same as random_k sampling.
        rng = np.random.default_rng(seed)
        weights = rng.random(CUT_FEATURE_DIM, dtype=np.float64)
    else:
        weights = np.asarray(DEFAULT_WEIGHTS, dtype=np.float64)

    class _Policy:
        def act(self, state: Any) -> Dict[str, Any]:
            return {"scorer_params": {"weights": weights}}

    return _Policy()


def run_episode(
    base_dir: str,
    seed: int = 0,
    max_iters: int = 999999999999,
    policy_name: str = "default",
    out_dir: str = "./runs",
    run_name: Optional[str] = None,
    max_episode_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    One MDP episode: reset + step loop; write steps.jsonl and summary.csv.
    total_master_solve_time_sec sums reset master + each step's master wall time.
    If max_episode_time_sec is set (> 0), wall time since episode start (including reset)
    is capped; further steps are skipped and batch caller may continue with the next seed.
    """
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    solver = BendersMDPSolver(base_dir=base_dir, seed=seed)
    selector = CutSelector(seed=seed)
    env = BendersMDPEnv(
        solver=solver,
        reward_fn=gap_reduction_reward,
        selector=selector,
        max_iters=max_iters,
        tol_gap=1e-2,
    )
    policy = _make_policy(policy_name, seed)

    t0 = time.time()
    state = env.reset()
    total_reward = 0.0
    total_selected_cuts = 0
    total_master_solve_time_sec = float(
        getattr(state, "master_solve_time_last_iter", 0.0) or 0.0
    )
    total_sub_iter_count = 0.0
    total_sub_gurobi_runtime = 0.0
    total_sub_time_sec = 0.0
    last_n_scenarios = 0
    z_l1_vals: List[float] = []
    n_zero_cap_vals: List[float] = []
    n_positive_beta_vals: List[float] = []
    steps_path = os.path.join(run_dir, "steps.jsonl")
    iters_used = 0
    limit_active = max_episode_time_sec is not None and max_episode_time_sec > 0.0
    stopped_for_episode_time_limit = False

    with open(steps_path, "w", encoding="utf-8") as fh:
        if limit_active and (time.time() - t0) >= max_episode_time_sec:
            stopped_for_episode_time_limit = True
        else:
            for step in range(1, max_iters + 1):
                if limit_active and (time.time() - t0) >= max_episode_time_sec:
                    stopped_for_episode_time_limit = True
                    break
                action = policy.act(state)
                next_state, reward, done, info = env.step(action)

                assert np.isfinite(reward), "reward must be finite"
                assert np.isfinite(next_state.optimality_gap), "gap must be finite"
                assert isinstance(info, dict), "info must be dict"

                total_reward += reward
                total_selected_cuts += info.get("selected_cut_count", 0)
                mt = info.get("solve_time_master")
                if mt is not None:
                    total_master_solve_time_sec += float(mt)
                total_sub_iter_count += float(info.get("sub_iter_count_sum") or 0.0)
                total_sub_gurobi_runtime += float(info.get("sub_gurobi_runtime_sum") or 0.0)
                total_sub_time_sec += float(info.get("solve_time_sub") or 0.0)
                if info.get("n_scenarios_solved") is not None:
                    last_n_scenarios = int(info.get("n_scenarios_solved") or 0)
                z_l1_vals.append(float(info.get("z_l1") or 0.0))
                n_zero_cap_vals.append(float(info.get("n_zero_cap") or 0.0))
                n_positive_beta_vals.append(float(info.get("n_positive_beta") or 0.0))
                iters_used = step

                # Cumulative wall since episode start (includes reset + prior steps).
                wall_clock_sec = float(time.time() - t0)
                row = {
                    "iter": step,
                    "gap_old": info.get("gap_old"),
                    "gap_new": info.get("gap_new"),
                    "reward": reward,
                    "done": done,
                    "selected": info.get("selected_cut_count"),
                    "candidate": info.get("candidate_cut_count"),
                    "effective_top_k": info.get("effective_top_k"),
                    "LB_new": info.get("LB_new"),
                    "UB_new": info.get("UB_new"),
                    "wall_clock_sec": wall_clock_sec,
                    "solve_time_master": info.get("solve_time_master"),
                    "solve_time_sub": info.get("solve_time_sub"),
                    "sub_iter_count_sum": info.get("sub_iter_count_sum"),
                    "sub_iter_count_mean": info.get("sub_iter_count_mean"),
                    "sub_iter_count_max": info.get("sub_iter_count_max"),
                    "sub_gurobi_runtime_sum": info.get("sub_gurobi_runtime_sum"),
                    "sub_gurobi_runtime_mean": info.get("sub_gurobi_runtime_mean"),
                    "z_l1": info.get("z_l1"),
                    "n_zero_cap": info.get("n_zero_cap"),
                    "n_positive_beta": info.get("n_positive_beta"),
                    "n_scenarios_solved": info.get("n_scenarios_solved"),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

                state = next_state
                if done:
                    break

    total_time_sec = time.time() - t0
    final_gap = float(state.optimality_gap)

    ds_seed = _dataset_seed_from_base_dir(base_dir)
    summary = {
        "base_dir": base_dir,
        "dataset_seed": ds_seed if ds_seed is not None else "",
        "seed": seed,
        "policy": policy_name,
        "top_k": SELECTION_TOP_K,
        "max_episode_time_sec": max_episode_time_sec if max_episode_time_sec is not None else "",
        "max_iters": max_iters,
        "iters_used": iters_used,
        "final_gap": final_gap,
        "total_reward": total_reward,
        "total_time_sec": total_time_sec,
        "total_master_solve_time_sec": total_master_solve_time_sec,
        "total_sub_time_sec": total_sub_time_sec,
        "total_sub_iter_count": total_sub_iter_count,
        "mean_sub_iter_count_per_iter": (
            float(total_sub_iter_count / iters_used) if iters_used > 0 else 0.0
        ),
        "mean_sub_iter_count_per_scenario": (
            float(total_sub_iter_count / (iters_used * max(last_n_scenarios, 1)))
            if iters_used > 0
            else 0.0
        ),
        "n_scenarios_solved": last_n_scenarios if last_n_scenarios > 0 else "",
        "total_sub_gurobi_runtime": total_sub_gurobi_runtime,
        "mean_sub_gurobi_runtime_per_iter": (
            float(total_sub_gurobi_runtime / iters_used) if iters_used > 0 else 0.0
        ),
        "mean_z_l1": float(np.mean(z_l1_vals)) if z_l1_vals else "",
        "mean_n_zero_cap": float(np.mean(n_zero_cap_vals)) if n_zero_cap_vals else "",
        "mean_n_positive_beta": (
            float(np.mean(n_positive_beta_vals)) if n_positive_beta_vals else ""
        ),
        "total_selected_cuts": total_selected_cuts,
        "run_name": run_name,
        "stopped_for_episode_time_limit": stopped_for_episode_time_limit,
    }

    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)

    return summary


def _normalized_group(group: Optional[str]) -> Optional[str]:
    if group is None:
        return None
    g = group.strip()
    return g if g else None


def _write_batch_summaries_json(
    batch_root: str,
    summaries: List[Dict[str, Any]],
    max_episode_time_sec: Optional[float] = None,
) -> None:
    """One JSON file with meta plus every run_episode return dict (all summary fields)."""
    path = os.path.join(batch_root, "batch_summaries.json")
    cumulative = sum(float(s.get("total_time_sec") or 0.0) for s in summaries)
    any_time_limit = any(bool(s.get("stopped_for_episode_time_limit")) for s in summaries)
    payload: Dict[str, Any] = {
        "meta": {
            "max_episode_time_sec": max_episode_time_sec,
            "cumulative_total_time_sec": cumulative,
            "episode_count": len(summaries),
            "any_episode_stopped_for_time_limit": any_time_limit,
        },
        "episodes": summaries,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run top-k Benders MDP on each seed subdir under dataset_root or dataset_root/--group, "
            "or --run_one for a single data folder."
        )
    )
    _root = os.path.dirname(os.path.abspath(__file__))
    _default_dataset_root = os.path.join(_root, "dataset")
    p.add_argument(
        "--dataset_root",
        type=str,
        default=_default_dataset_root,
        help="Batch: root folder that contains seed_* dirs, or group folders when using --group (default: <script_dir>/dataset).",
    )
    p.add_argument(
        "--group",
        type=str,
        default=None,
        help=(
            "Batch only: scan dataset_root/<group>/ for seed subdirs (e.g. --group 'group A'). "
            "Omit to scan immediate children of dataset_root (flat layout)."
        ),
    )
    p.add_argument(
        "--run_one",
        type=str,
        default=None,
        help="If set, run a single episode on this directory (absolute or cwd-relative; e.g. .../dataset/group A/seed_110).",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for solver/policy.")
    p.add_argument("--max_iters", type=int, default=999999999999, help="Max MDP steps per episode.")
    p.add_argument(
        "--time_limit",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Per-episode wall-clock cap in seconds (from reset). "
            "Omit for no limit. When reached, the episode stops and batch continues to the next seed."
        ),
    )
    p.add_argument(
        "--policy",
        type=str,
        default="default",
        choices=("default", "topk", "random", "random_k"),
        help=(
            "Cut-selection policy: topk ranks by violation and keeps K cuts; "
            "default uses DEFAULT_WEIGHTS; random draws score weights; "
            "random_k draws a uniform subset of K cuts."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./runs",
        help="Output root. Batch: creates out_dir/<session>/ per invocation; each episode is out_dir/<session>/<seed_tag>/.",
    )
    p.add_argument("--run_name", type=str, default=None, help="Used with --run_one; batch mode auto-names runs.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Leading/trailing spaces make " /abs/path" relative on POSIX → wrong abspath join with cwd.
    args.dataset_root = args.dataset_root.strip()
    args.out_dir = args.out_dir.strip()
    if args.run_one is not None:
        args.run_one = args.run_one.strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries: List[Dict[str, Any]] = []

    if args.run_one:
        one_path = args.run_one
        if not os.path.isabs(one_path):
            one_path = os.path.normpath(os.path.join(os.getcwd(), one_path))
        if not os.path.isdir(one_path):
            raise SystemExit(f"--run_one is not a directory: {one_path}")
        s = run_episode(
            base_dir=one_path,
            seed=args.seed,
            max_iters=args.max_iters,
            policy_name=args.policy,
            out_dir=args.out_dir,
            run_name=args.run_name,
            max_episode_time_sec=args.time_limit,
        )
        summaries.append(s)
        print("Run finished:", s)
        print("Outputs:", os.path.join(args.out_dir, s["run_name"]))
    else:
        dataset_root = os.path.normpath(os.path.abspath(args.dataset_root))
        if not os.path.isdir(dataset_root):
            raise SystemExit(f"dataset_root does not exist: {dataset_root}")
        ng = _normalized_group(args.group)
        if ng:
            scan_root = os.path.join(dataset_root, ng)
            if not os.path.isdir(scan_root):
                raise SystemExit(
                    f"Group directory does not exist: {scan_root}\n"
                    f"(dataset_root={dataset_root}, group={ng!r})"
                )
        else:
            scan_root = dataset_root
        dataset_dirs = discover_dataset_dirs(scan_root)
        if not dataset_dirs:
            hint = f" under group {ng!r}" if ng else ""
            raise SystemExit(f"No subdirectories under scan root: {scan_root}{hint}")
        print(f"Batch: {len(dataset_dirs)} dataset dir(s) under {scan_root!r}")
        if ng:
            session_name = f"{ts}_{'_'.join(ng.split())}"
        else:
            session_name = ts
        batch_root = os.path.normpath(os.path.join(os.path.abspath(args.out_dir), session_name))
        os.makedirs(batch_root, exist_ok=True)
        for base_dir in dataset_dirs:
            tag = os.path.basename(os.path.normpath(base_dir))
            s = run_episode(
                base_dir=base_dir,
                seed=args.seed,
                max_iters=args.max_iters,
                policy_name=args.policy,
                out_dir=batch_root,
                run_name=tag,
                max_episode_time_sec=args.time_limit,
            )
            summaries.append(s)
            print(f"Finished {tag!r}:", s.get("final_gap"), "iters", s.get("iters_used"))
            print("  ->", os.path.join(batch_root, s["run_name"]))
            if s.get("stopped_for_episode_time_limit"):
                print(
                    f"  Episode time limit ({args.time_limit}s) reached for {tag!r}; continuing to next seed."
                )
        _write_batch_summaries_json(
            batch_root,
            summaries,
            max_episode_time_sec=args.time_limit,
        )
        print(f"\nCompleted {len(summaries)} episode(s); session folder: {batch_root!r}")
        print("Aggregate summaries:", os.path.join(batch_root, "batch_summaries.json"))
