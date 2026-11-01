from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from dataset_factory import (
    DatasetConfig,
    build_demand_dataframe,
    build_static_dataframes,
    sample_demand_mu,
    write_dataset_files,
)
from seed_control import derive_seeds as _derive_seeds
from seed_control import resolve_eval_seed_start


def _run_command(
    stage_name: str,
    cmd: List[str],
    cwd: str,
    log_path: str,
    env: Optional[Dict[str, str]] = None,
) -> None:
    print(f"\n[stage:{stage_name}] start")
    print(f"[stage:{stage_name}] command: {' '.join(cmd)}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("COMMAND:\n")
        fh.write(" ".join(cmd))
        fh.write("\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        elapsed = time.time() - t0
        fh.write(f"\n[stage:{stage_name}] elapsed_sec={elapsed:.3f}\n")
    print(f"[stage:{stage_name}] finished in {elapsed:.3f}s (code={proc.returncode})")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _latest_subdir(path: str) -> str | None:
    if not os.path.isdir(path):
        return None
    subdirs = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isdir(os.path.join(path, name))
    ]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _compute_stats(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    n = len(finite)
    mean = sum(finite) / float(n)
    var = sum((x - mean) ** 2 for x in finite) / float(n)
    return {
        "count": float(n),
        "mean": float(mean),
        "std": float(var ** 0.5),
        "min": float(min(finite)),
        "max": float(max(finite)),
    }


def _single_thread_env() -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = "1"
    return env


def _run_cut_baseline(
    *,
    stage_prefix: str,
    policy: str,
    eval_dir: str,
    scenario_seed: int,
    solver_seed: int,
    run_dir: str,
    folder_name: str,
    logs_dir: str,
    baseline_dir: str,
) -> Dict[str, Any]:
    """One limited-cut episode. K and the gap tolerance stay at the baseline defaults."""
    out_root = os.path.abspath(os.path.join(run_dir, folder_name))
    os.makedirs(out_root, exist_ok=True)
    episode_name = f"seed_{scenario_seed}"
    cmd = [
        sys.executable,
        "-u",
        "mdp_commander.py",
        "--run_one",
        os.path.abspath(eval_dir),
        "--policy",
        policy,
        "--seed",
        str(solver_seed),
        "--out_dir",
        out_root,
        "--run_name",
        episode_name,
    ]
    _run_command(
        f"{stage_prefix}_seed_{scenario_seed}",
        cmd,
        cwd=baseline_dir,
        log_path=os.path.join(logs_dir, f"{stage_prefix}_seed_{scenario_seed}.log"),
        env=_single_thread_env(),
    )
    row: Dict[str, Any] = {}
    summary_csv = os.path.join(out_root, episode_name, "summary.csv")
    if os.path.isfile(summary_csv):
        with open(summary_csv, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows:
            row = rows[0]
    return {
        "total_time_sec": _parse_optional_float(row.get("total_time_sec")),
        "master_time_sec": _parse_optional_float(row.get("total_master_solve_time_sec")),
        "iters": _parse_optional_float(row.get("iters_used")),
        "final_gap": _parse_optional_float(row.get("final_gap")),
        "cuts": _parse_optional_float(row.get("total_selected_cuts")),
        "solver_seed": row.get("seed"),
    }


def _parse_optional_float(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return float(x)


def _parse_optional_int(raw: Any) -> Optional[int]:
    x = _parse_optional_float(raw)
    if x is None:
        return None
    return int(round(x))


def _parse_optional_bool(raw: Any) -> Optional[bool]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _greedy_checkpoint_record_from_mapping(
    *,
    eval_scenario_seed: int,
    eval_index: int,
    dataset_dir: str,
    policy_name: Optional[str],
    policy_path: Optional[str],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "eval_scenario_seed": int(eval_scenario_seed),
        "eval_index": int(eval_index),
        "dataset_dir": dataset_dir,
        "policy_name": policy_name,
        "policy_path": policy_path,
        "elapsed_sec": _parse_optional_float(row.get("elapsed_sec")),
        "total_master_time": _parse_optional_float(row.get("total_master_time")),
        "total_sub_time": _parse_optional_float(row.get("total_sub_time")),
        "num_steps": _parse_optional_int(row.get("num_steps")),
        "converged": _parse_optional_bool(row.get("converged")),
        "total_selected_cuts": _parse_optional_int(row.get("total_selected_cuts")),
        "total_reward": _parse_optional_float(row.get("total_reward")),
        "reward_name": row.get("reward_name"),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EV Benders experiment pipeline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="./experiment_runs")
    p.add_argument("--run_name", type=str, default=None)

    # dataset
    p.add_argument("--num_stations", type=int, default=10)
    p.add_argument("--num_customers", type=int, default=20)
    p.add_argument("--num_scenarios", type=int, default=30)
    p.add_argument("--fixed_cost_min", type=int, default=80)
    p.add_argument("--fixed_cost_max", type=int, default=300)
    p.add_argument("--charge_cost_min", type=int, default=5)
    p.add_argument("--charge_cost_max", type=int, default=40)
    p.add_argument("--max_chargers_min", type=int, default=10)
    p.add_argument("--max_chargers_max", type=int, default=80)
    p.add_argument("--capacity_min", type=int, default=40)
    p.add_argument("--capacity_max", type=int, default=120)
    p.add_argument("--penalty_min", type=int, default=30)
    p.add_argument("--penalty_max", type=int, default=120)
    p.add_argument("--revenue_min", type=int, default=5)
    p.add_argument("--revenue_max", type=int, default=60)
    p.add_argument("--distance_min", type=int, default=1)
    p.add_argument("--distance_max", type=int, default=80)
    p.add_argument("--demand_min", type=int, default=5)
    p.add_argument("--demand_max", type=int, default=120)
    p.add_argument("--demand_mu_min", type=int, default=20)
    p.add_argument("--demand_mu_max", type=int, default=100)
    p.add_argument("--demand_sigma_ratio", type=float, default=0.1)
    p.add_argument("--truncate_negative_demand", type=int, default=1)
    p.add_argument(
        "--train_demand_distribution",
        type=str,
        default="normal",
        choices=["normal", "left_skewed", "right_skewed"],
    )
    p.add_argument(
        "--eval_demand_distribution",
        type=str,
        default="normal",
        choices=["normal", "left_skewed", "right_skewed"],
    )
    p.add_argument("--demand_skew_alpha", type=float, default=5.0)
    p.add_argument("--num_eval_datasets", type=int, default=5)
    p.add_argument("--eval_scenario_seed_start", type=int, default=None)

    # vanilla
    p.add_argument("--vanilla_max_iters", type=int, default=200)
    p.add_argument(
        "--vanilla_max_elapsed_sec",
        type=float,
        default=0.0,
        help=(
            "Passed to run_vanilla as --max_elapsed_sec: cap on cumulative master+sub solve seconds "
            "(same as benchmark vanilla_solve_time_sec). 0 disables."
        ),
    )
    p.add_argument("--vanilla_tol_rel_gap", type=float, default=1e-2)
    p.add_argument("--vanilla_quiet", action="store_true")
    p.add_argument("--vanilla_save_trajectory", action="store_true")
    p.add_argument("--require_vanilla_converged", action="store_true")

    # training
    p.add_argument("--train_max_iters", type=int, default=2000)
    p.add_argument(
        "--train_max_elapsed_sec",
        type=float,
        default=0.0,
        help="Wall-clock cap (perf_counter) on RL training; 0 disables (use --train_max_iters only).",
    )
    p.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help=(
            "Passed to mdp_commander.py as --seed (RL + BendersMDPSolver / sampling). "
            "If omitted, uses derived value from --seed (same as seeds['train_seed'] = seed+37)."
        ),
    )
    p.add_argument("--episodic", action="store_true")
    p.add_argument("--num_episodes", type=int, default=10)
    p.add_argument("--k_select", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--reward_ma_window", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--reward_name", type=str, default="log_gap_rmp_time_penalty")
    p.add_argument("--tol_gap", type=float, default=1e-2)
    p.add_argument("--reward_eps", type=float, default=None)
    p.add_argument("--reward_alpha", type=float, default=None)
    p.add_argument("--reward_beta", type=float, default=None)
    p.add_argument("--reward_tref", type=float, default=None)
    p.add_argument("--reward_lambda_penalty", type=float, default=None)
    p.add_argument("--reward_vl_ref", type=float, default=None)
    p.add_argument("--reward_c", type=float, default=None)
    p.add_argument("--reward_gamma", type=float, default=None)
    p.add_argument("--reward_eta", type=float, default=None)
    p.add_argument("--reward_param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip RL training (mdp_commander). Use with --skip_eval to run only vanilla + SVM.",
    )
    p.add_argument(
        "--vanilla_svm_only",
        action="store_true",
        help="Shortcut: sets --skip_train and --skip_eval (vanilla Benders + SVM on eval sets only).",
    )

    # evaluation
    p.add_argument("--eval_max_iters", type=int, default=2000)
    p.add_argument(
        "--eval_max_elapsed_sec",
        type=float,
        default=0.0,
        help=(
            "Passed to evaluate_greedy_policy: cap on cumulative master+sub solve seconds "
            "(same as greedy summary solve_time_sec / benchmark). 0 disables."
        ),
    )
    p.add_argument("--eval_k_select", type=int, default=5)
    p.add_argument("--eval_tol_gap", type=float, default=1e-2)
    p.add_argument("--eval_all_checkpoints", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    # svm evaluation
    p.add_argument("--svm_stage1_paths", type=int, default=10)
    p.add_argument("--svm_stage1_iters", type=int, default=30)
    p.add_argument("--svm_key", type=float, default=1.5)
    p.add_argument("--svm_key_low", type=float, default=1.0)
    p.add_argument("--svm_key_change_time", type=float, default=5.0)
    p.add_argument("--svm_itelimit", type=int, default=2000)
    p.add_argument(
        "--svm_timelimit",
        type=float,
        default=3600.0,
        help=(
            "Passed to run_svm_eval as --timelimit: Phase-3 stop when cumulative master+sub solve seconds "
            "reach this cap (same units as svm_stage3_solve_time_sec / phase3_solve_time_sec in svm summary). "
            "Example: --svm_timelimit 61.56336483699999 to match a reference benchmark budget."
        ),
    )
    p.add_argument("--svm_pretrain", type=int, default=0)
    p.add_argument("--skip_svm_eval", action="store_true")
    p.add_argument(
        "--skip_vanilla",
        action="store_true",
        help="Skip Vanilla Benders on eval sets.",
    )
    p.add_argument(
        "--random_k_only",
        action="store_true",
        help=(
            "Generate datasets, then run only uniform Random-K "
            "(baselines/mdp_commander.py --policy random_k). "
            "Skips training, Vanilla, policy evaluation, and LearnBD."
        ),
    )
    p.add_argument(
        "--top_k_only",
        action="store_true",
        help=(
            "Generate datasets, then run only violation Top-K "
            "(baselines/mdp_commander.py --policy topk). "
            "Skips training, Vanilla, policy evaluation, and LearnBD."
        ),
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Generate datasets, then run only RLBD policy evaluation. "
            "Skips training, Vanilla, and LearnBD. "
            "Place the policy at <run_dir>/train/policy_final.pt before launch, "
            "or pass --eval_all_checkpoints with checkpoints already in <run_dir>/train/."
        ),
    )
    p.add_argument(
        "--eval_then_topk",
        action="store_true",
        help=(
            "Generate datasets, run RLBD policy evaluation, then violation Top-K. "
            "Skips training, Vanilla, and LearnBD. "
            "Place the policy at <run_dir>/train/policy_final.pt before launch."
        ),
    )

    args = p.parse_args()
    if bool(getattr(args, "vanilla_svm_only", False)):
        args.skip_train = True
        args.skip_eval = True
    if bool(getattr(args, "random_k_only", False)) and bool(getattr(args, "top_k_only", False)):
        p.error("choose one of --random_k_only or --top_k_only")
    if bool(getattr(args, "random_k_only", False)) or bool(getattr(args, "top_k_only", False)):
        args.skip_train = True
        args.skip_vanilla = True
        args.skip_eval = True
        args.skip_svm_eval = True
    exclusive = [
        name
        for name, enabled in (
            ("--eval_only", bool(getattr(args, "eval_only", False))),
            ("--eval_then_topk", bool(getattr(args, "eval_then_topk", False))),
            ("--random_k_only", bool(getattr(args, "random_k_only", False))),
            ("--top_k_only", bool(getattr(args, "top_k_only", False))),
            ("--vanilla_svm_only", bool(getattr(args, "vanilla_svm_only", False))),
        )
        if enabled
    ]
    if len(exclusive) > 1:
        p.error("choose only one of " + ", ".join(exclusive))
    if bool(getattr(args, "eval_only", False)) or bool(getattr(args, "eval_then_topk", False)):
        if bool(getattr(args, "skip_eval", False)):
            p.error("policy evaluation stays on; do not pass --skip_eval")
        args.skip_train = True
        args.skip_vanilla = True
        args.skip_svm_eval = True
    return args


def main() -> None:
    args = _parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    seeds = _derive_seeds(args.seed)
    eval_seed_start = resolve_eval_seed_start(
        int(args.seed),
        eval_scenario_seed_start=(
            int(args.eval_scenario_seed_start)
            if args.eval_scenario_seed_start is not None
            else None
        ),
        seeds=seeds,
    )
    eval_scenario_seeds = [eval_seed_start + i for i in range(max(1, int(args.num_eval_datasets)))]

    train_seed_effective = int(args.train_seed) if args.train_seed is not None else int(seeds["train_seed"])

    dataset_train_dir = os.path.join(run_dir, "dataset_train")
    dataset_eval_root_dir = os.path.join(run_dir, "dataset_eval")
    vanilla_root_dir = os.path.join(run_dir, "vanilla")
    eval_root_dir = os.path.join(run_dir, "eval")
    svm_eval_root_dir = os.path.join(run_dir, "svm_eval")

    config = {
        "run_name": run_name,
        "root_dir": root,
        "run_dir": run_dir,
        "seeds": seeds,
        "train_seed_effective": int(train_seed_effective),
        "train_seed_derived_default": int(seeds["train_seed"]),
        "args": vars(args),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)

    # 1) dataset generation (fixed structure + fixed mu, scenario-only variation)
    base_cfg = DatasetConfig(
        num_stations=args.num_stations,
        num_customers=args.num_customers,
        num_scenarios=args.num_scenarios,
        seed=0,
        fixed_cost_min=args.fixed_cost_min,
        fixed_cost_max=args.fixed_cost_max,
        charge_cost_min=args.charge_cost_min,
        charge_cost_max=args.charge_cost_max,
        max_chargers_min=args.max_chargers_min,
        max_chargers_max=args.max_chargers_max,
        capacity_min=args.capacity_min,
        capacity_max=args.capacity_max,
        penalty_min=args.penalty_min,
        penalty_max=args.penalty_max,
        revenue_min=args.revenue_min,
        revenue_max=args.revenue_max,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        demand_min=args.demand_min,
        demand_max=args.demand_max,
        demand_mu_min=args.demand_mu_min,
        demand_mu_max=args.demand_mu_max,
        demand_sigma_ratio=args.demand_sigma_ratio,
        truncate_negative_demand=bool(args.truncate_negative_demand),
        demand_distribution="normal",
        demand_skew_alpha=args.demand_skew_alpha,
    )

    structure_rng = np.random.default_rng(int(seeds["dataset_structure_seed"]))
    station_df, customer_df, distance_df, _station_ids, customer_ids = build_static_dataframes(base_cfg, structure_rng)

    mu_rng = np.random.default_rng(int(seeds["demand_mu_seed"]))
    demand_mu = sample_demand_mu(base_cfg, mu_rng, base_cfg.num_customers)

    train_scenario_seed = int(seeds["dataset_train_scenario_seed"])
    train_cfg = DatasetConfig(
        **{
            **base_cfg.__dict__,
            "seed": train_scenario_seed,
            "demand_distribution": args.train_demand_distribution,
            "demand_skew_alpha": args.demand_skew_alpha,
        }
    )
    train_rng = np.random.default_rng(train_scenario_seed)
    train_demand_df = build_demand_dataframe(train_cfg, train_rng, customer_ids, demand_mu)
    dataset_train_paths = write_dataset_files(
        base_dir=dataset_train_dir,
        cfg=train_cfg,
        station_df=station_df,
        customer_df=customer_df,
        distance_df=distance_df,
        demand_df=train_demand_df,
        extra_config={
            "dataset_role": "train",
            "dataset_structure_seed": int(seeds["dataset_structure_seed"]),
            "dataset_scenario_seed": train_scenario_seed,
            "demand_mu_seed": int(seeds["demand_mu_seed"]),
            "demand_mu": [int(x) for x in demand_mu.tolist()],
            "scenario_probability_distribution": "uniform",
            "demand_distribution": args.train_demand_distribution,
            "demand_skew_alpha": float(args.demand_skew_alpha),
        },
    )

    eval_dataset_infos: List[Dict[str, Any]] = []
    for idx, sce_seed in enumerate(eval_scenario_seeds):
        eval_dir = os.path.join(dataset_eval_root_dir, f"seed_{sce_seed}")
        eval_cfg = DatasetConfig(
            **{
                **base_cfg.__dict__,
                "seed": int(sce_seed),
                "demand_distribution": args.eval_demand_distribution,
                "demand_skew_alpha": args.demand_skew_alpha,
            }
        )
        eval_rng = np.random.default_rng(int(sce_seed))
        eval_demand_df = build_demand_dataframe(eval_cfg, eval_rng, customer_ids, demand_mu)
        eval_paths = write_dataset_files(
            base_dir=eval_dir,
            cfg=eval_cfg,
            station_df=station_df,
            customer_df=customer_df,
            distance_df=distance_df,
            demand_df=eval_demand_df,
            extra_config={
                "dataset_role": "eval",
                "dataset_structure_seed": int(seeds["dataset_structure_seed"]),
                "dataset_scenario_seed": int(sce_seed),
                "demand_mu_seed": int(seeds["demand_mu_seed"]),
                "demand_mu": [int(x) for x in demand_mu.tolist()],
                "scenario_probability_distribution": "uniform",
                "eval_dataset_index": int(idx),
                "demand_distribution": args.eval_demand_distribution,
                "demand_skew_alpha": float(args.demand_skew_alpha),
            },
        )
        eval_dataset_infos.append(
            {
                "eval_index": int(idx),
                "eval_scenario_seed": int(sce_seed),
                "dataset_dir": eval_dir,
                "paths": eval_paths,
            }
        )

    with open(os.path.join(run_dir, "dataset_paths.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_train_dir": dataset_train_dir,
                "dataset_eval_root_dir": dataset_eval_root_dir,
                "eval_scenario_seeds": eval_scenario_seeds,
                "dataset_train_paths": dataset_train_paths,
                "dataset_eval_sets": eval_dataset_infos,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    # 2) RL training (train dataset only)
    train_cmd = [
        sys.executable,
        os.path.join(root, "mdp_commander.py"),
        "--base_dir",
        dataset_train_dir,
        "--seed",
        str(train_seed_effective),
        "--max_iters",
        str(args.train_max_iters),
        "--out_dir",
        run_dir,
        "--run_name",
        "train",
        "--lr",
        str(args.lr),
        "--temperature",
        str(args.temperature),
        "--log_every",
        str(args.log_every),
        "--k_select",
        str(args.k_select),
        "--reward_ma_window",
        str(args.reward_ma_window),
        "--reward_name",
        str(args.reward_name),
        "--tol_gap",
        str(args.tol_gap),
        "--ckpt_every",
        str(args.ckpt_every),
    ]
    _reward_cli_keys = (
        "reward_eps",
        "reward_alpha",
        "reward_beta",
        "reward_tref",
        "reward_lambda_penalty",
        "reward_vl_ref",
        "reward_c",
        "reward_gamma",
        "reward_eta",
    )
    for _rk in _reward_cli_keys:
        _rv = getattr(args, _rk, None)
        if _rv is not None:
            train_cmd.extend([f"--{_rk}", str(_rv)])
    if args.episodic:
        train_cmd.extend(["--episodic", "--num_episodes", str(args.num_episodes)])
    for rp in args.reward_param:
        train_cmd.extend(["--reward_param", str(rp)])
    if args.train_max_elapsed_sec > 0.0:
        train_cmd.extend(["--train_max_elapsed_sec", str(args.train_max_elapsed_sec)])

    train_log_path = os.path.join(logs_dir, "train.log")
    if args.skip_train:
        with open(train_log_path, "w", encoding="utf-8") as fh:
            fh.write("[skip_train] RL training skipped (--skip_train or --vanilla_svm_only).\n")
        print("\n[stage:train] skipped (--skip_train)")
    else:
        _run_command("train", train_cmd, cwd=root, log_path=train_log_path)

    # 3) benchmarks on all eval datasets
    per_eval_seed_results: List[Dict[str, Any]] = []
    greedy_checkpoint_records: List[Dict[str, Any]] = []
    svm_training_records_dir = os.path.join(run_dir, "svm_training_records")
    svm_training_records_path = os.path.join(svm_training_records_dir, "svm_training_records.csv")
    os.makedirs(svm_training_records_dir, exist_ok=True)
    for eval_info in eval_dataset_infos:
        eval_seed = int(eval_info["eval_scenario_seed"])
        eval_dir = str(eval_info["dataset_dir"])
        result: Dict[str, Any] = {
            "eval_index": int(eval_info["eval_index"]),
            "eval_scenario_seed": eval_seed,
            "dataset_dir": eval_dir,
            "vanilla_solve_time_sec": None,
            "greedy_eval_checkpoint_solve_times_sec": [],
            "svm_stage3_solve_time_sec": None,
            "random_k_total_time_sec": None,
            "random_k_master_time_sec": None,
            "random_k_iters": None,
            "random_k_final_gap": None,
            "random_k_cuts": None,
            "random_k_solver_seed": None,
            "top_k_total_time_sec": None,
            "top_k_master_time_sec": None,
            "top_k_iters": None,
            "top_k_final_gap": None,
            "top_k_cuts": None,
            "top_k_solver_seed": None,
        }

        if not args.skip_vanilla:
            vanilla_out_dir = os.path.join(vanilla_root_dir, f"eval_seed_{eval_seed}")
            vanilla_cmd = [
                sys.executable,
                "-u",
                os.path.join(root, "run_vanilla.py"),
                "--base_dir",
                eval_dir,
                "--out_dir",
                vanilla_out_dir,
                "--run_name",
                "vanilla",
                "--seed",
                str(int(seeds["vanilla_seed"]) + int(eval_info["eval_index"])),
                "--max_iters",
                str(args.vanilla_max_iters),
                "--tol_rel_gap",
                str(args.vanilla_tol_rel_gap),
            ]
            if args.vanilla_quiet:
                vanilla_cmd.append("--quiet")
            if args.vanilla_save_trajectory:
                vanilla_cmd.append("--save_trajectory")
            if args.vanilla_max_elapsed_sec > 0.0:
                vanilla_cmd.extend(["--max_elapsed_sec", str(args.vanilla_max_elapsed_sec)])
            _run_command(
                f"vanilla_eval_seed_{eval_seed}",
                vanilla_cmd,
                cwd=root,
                log_path=os.path.join(logs_dir, f"vanilla_eval_seed_{eval_seed}.log"),
            )
            vanilla_summary_path = os.path.join(vanilla_out_dir, "vanilla", "summary.json")
            if os.path.isfile(vanilla_summary_path):
                with open(vanilla_summary_path, "r", encoding="utf-8") as fh:
                    vanilla_summary = json.load(fh)
                result["vanilla_solve_time_sec"] = vanilla_summary.get("master_sub_total_sec")
                if args.require_vanilla_converged and not bool(vanilla_summary.get("converged", False)):
                    raise RuntimeError(f"vanilla failed to converge on eval seed {eval_seed}")

        if not args.skip_eval:
            eval_out_dir = os.path.join(eval_root_dir, f"eval_seed_{eval_seed}")
            _greedy_reward_cli: List[str] = [
                "--reward_name",
                str(args.reward_name),
            ]
            _rk_eval = (
                "reward_eps",
                "reward_alpha",
                "reward_beta",
                "reward_tref",
                "reward_lambda_penalty",
                "reward_vl_ref",
                "reward_c",
                "reward_gamma",
                "reward_eta",
            )
            for _rk in _rk_eval:
                _rv = getattr(args, _rk, None)
                if _rv is not None:
                    _greedy_reward_cli.extend([f"--{_rk}", str(_rv)])
            for rp in args.reward_param:
                _greedy_reward_cli.extend(["--reward_param", str(rp)])

            if args.eval_all_checkpoints:
                eval_cmd = [
                    sys.executable,
                    os.path.join(root, "evaluate_greedy_policy.py"),
                    "--policy_dir",
                    os.path.join(run_dir, "train"),
                    "--base_dir",
                    eval_dir,
                    "--seed",
                    str(int(seeds["eval_seed"]) + int(eval_info["eval_index"])),
                    "--max_iters",
                    str(args.eval_max_iters),
                    "--k_select",
                    str(args.eval_k_select),
                    "--tol_gap",
                    str(args.eval_tol_gap),
                    "--out_dir",
                    eval_out_dir,
                ] + _greedy_reward_cli
            else:
                eval_cmd = [
                    sys.executable,
                    os.path.join(root, "evaluate_greedy_policy.py"),
                    "--policy_path",
                    os.path.join(run_dir, "train", "policy_final.pt"),
                    "--base_dir",
                    eval_dir,
                    "--seed",
                    str(int(seeds["eval_seed"]) + int(eval_info["eval_index"])),
                    "--max_iters",
                    str(args.eval_max_iters),
                    "--k_select",
                    str(args.eval_k_select),
                    "--tol_gap",
                    str(args.eval_tol_gap),
                    "--out_dir",
                    eval_out_dir,
                ] + _greedy_reward_cli
            if args.eval_max_elapsed_sec > 0.0:
                eval_cmd.extend(["--eval_max_elapsed_sec", str(args.eval_max_elapsed_sec)])
            _run_command(
                f"greedy_eval_seed_{eval_seed}",
                eval_cmd,
                cwd=root,
                log_path=os.path.join(logs_dir, f"greedy_eval_seed_{eval_seed}.log"),
            )
            eval_run_dir = _latest_subdir(eval_out_dir)
            if eval_run_dir is not None:
                batch_csv = os.path.join(eval_run_dir, "summary.csv")
                single_json = os.path.join(eval_run_dir, "summary.json")
                if os.path.isfile(batch_csv):
                    with open(batch_csv, "r", encoding="utf-8") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            master_val = row.get("total_master_time")
                            sub_val = row.get("total_sub_time")
                            solve_val = None
                            if master_val not in (None, "") and sub_val not in (None, ""):
                                solve_val = float(master_val) + float(sub_val)
                            result["greedy_eval_checkpoint_solve_times_sec"].append(
                                {
                                    "policy_name": row.get("policy_name"),
                                    "policy_path": row.get("policy_path"),
                                    "solve_time_sec": solve_val,
                                }
                            )
                            greedy_checkpoint_records.append(
                                _greedy_checkpoint_record_from_mapping(
                                    eval_scenario_seed=eval_seed,
                                    eval_index=int(eval_info["eval_index"]),
                                    dataset_dir=eval_dir,
                                    policy_name=row.get("policy_name"),
                                    policy_path=row.get("policy_path"),
                                    row=row,
                                )
                            )
                elif os.path.isfile(single_json):
                    with open(single_json, "r", encoding="utf-8") as fh:
                        one = json.load(fh)
                    solve_val = None
                    master_val = one.get("total_master_time")
                    sub_val = one.get("total_sub_time")
                    if master_val is not None and sub_val is not None:
                        solve_val = float(master_val) + float(sub_val)
                    result["greedy_eval_checkpoint_solve_times_sec"].append(
                        {
                            "policy_name": "policy_single",
                            "policy_path": one.get("policy_path"),
                            "solve_time_sec": solve_val,
                        }
                    )
                    greedy_checkpoint_records.append(
                        _greedy_checkpoint_record_from_mapping(
                            eval_scenario_seed=eval_seed,
                            eval_index=int(eval_info["eval_index"]),
                            dataset_dir=eval_dir,
                            policy_name="policy_single",
                            policy_path=one.get("policy_path"),
                            row=one,
                        )
                    )

        if not args.skip_svm_eval:
            svm_out_dir = os.path.join(svm_eval_root_dir, f"eval_seed_{eval_seed}")
            svm_cmd = [
                sys.executable,
                os.path.join(root, "run_svm_eval.py"),
                "--base_dir",
                eval_dir,
                "--train_base_dir",
                dataset_train_dir,
                "--training_records_path",
                svm_training_records_path,
                "--out_dir",
                svm_out_dir,
                "--run_name",
                "svm_eval",
                "--seed",
                str(int(seeds["svm_eval_seed"]) + int(eval_info["eval_index"])),
                "--stage1_paths",
                str(args.svm_stage1_paths),
                "--stage1_iters",
                str(args.svm_stage1_iters),
                "--tol_rel_gap",
                str(args.eval_tol_gap),
                "--key",
                str(args.svm_key),
                "--key_low",
                str(args.svm_key_low),
                "--key_change_time",
                str(args.svm_key_change_time),
                "--itelimit",
                str(args.svm_itelimit),
                "--timelimit",
                str(args.svm_timelimit),
                "--pretrain",
                str(args.svm_pretrain),
            ]
            _run_command(
                f"svm_eval_seed_{eval_seed}",
                svm_cmd,
                cwd=root,
                log_path=os.path.join(logs_dir, f"svm_eval_seed_{eval_seed}.log"),
            )
            svm_summary_path = os.path.join(svm_out_dir, "svm_eval", "summary.json")
            if os.path.isfile(svm_summary_path):
                with open(svm_summary_path, "r", encoding="utf-8") as fh:
                    svm_summary = json.load(fh)
                result["svm_stage3_solve_time_sec"] = svm_summary.get(
                    "svm_stage3_solve_time_sec", svm_summary.get("phase3_solve_time_sec")
                )

        if args.random_k_only or args.top_k_only or args.eval_then_topk:
            solver_seed = int(seeds["eval_seed"]) + int(eval_info["eval_index"])
            if args.random_k_only:
                cut_stage = ("random_k", "random_k", "randomk")
            else:
                cut_stage = ("top_k", "topk", "topk")
            stage_prefix, policy_name, folder_name = cut_stage
            cut_stats = _run_cut_baseline(
                stage_prefix=stage_prefix,
                policy=policy_name,
                eval_dir=eval_dir,
                scenario_seed=eval_seed,
                solver_seed=solver_seed,
                run_dir=run_dir,
                folder_name=folder_name,
                logs_dir=logs_dir,
                baseline_dir=os.path.join(root, "baselines"),
            )
            result[f"{stage_prefix}_total_time_sec"] = cut_stats["total_time_sec"]
            result[f"{stage_prefix}_master_time_sec"] = cut_stats["master_time_sec"]
            result[f"{stage_prefix}_iters"] = cut_stats["iters"]
            result[f"{stage_prefix}_final_gap"] = cut_stats["final_gap"]
            result[f"{stage_prefix}_cuts"] = cut_stats["cuts"]
            result[f"{stage_prefix}_solver_seed"] = cut_stats["solver_seed"]

        per_eval_seed_results.append(result)

    # 4) consolidated timing summary on eval datasets
    report_dir = os.path.join(run_dir, "benchmark_time_summary")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "summary.json")

    vanilla_values = [r.get("vanilla_solve_time_sec") for r in per_eval_seed_results]
    svm_values = [r.get("svm_stage3_solve_time_sec") for r in per_eval_seed_results]
    randomk_values = [r.get("random_k_total_time_sec") for r in per_eval_seed_results]
    topk_values = [r.get("top_k_total_time_sec") for r in per_eval_seed_results]
    greedy_all_values: List[Optional[float]] = []
    greedy_by_policy: Dict[str, List[Optional[float]]] = {}
    for r in per_eval_seed_results:
        for rec in r.get("greedy_eval_checkpoint_solve_times_sec", []):
            t_val = rec.get("solve_time_sec")
            greedy_all_values.append(t_val)
            policy_name = str(rec.get("policy_name"))
            greedy_by_policy.setdefault(policy_name, []).append(t_val)

    benchmark_summary = {
        "dataset_train_dir": dataset_train_dir,
        "dataset_eval_root_dir": dataset_eval_root_dir,
        "eval_scenario_seeds": eval_scenario_seeds,
        "per_eval_seed": per_eval_seed_results,
        "aggregate_stats": {
            "vanilla_benders_solve_time_sec": _compute_stats(vanilla_values),
            "random_k_total_time_sec": _compute_stats(randomk_values),
            "top_k_total_time_sec": _compute_stats(topk_values),
            "svm_stage3_solve_time_sec": _compute_stats(svm_values),
            "greedy_eval_overall_checkpoint_solve_time_sec": _compute_stats(greedy_all_values),
            "greedy_eval_by_policy": {
                policy: _compute_stats(values) for policy, values in greedy_by_policy.items()
            },
        },
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(benchmark_summary, fh, ensure_ascii=False, indent=2)

    greedy_metrics_path = os.path.join(report_dir, "greedy_checkpoint_metrics.json")
    greedy_metrics_payload: Dict[str, Any] = {
        "run_dir": run_dir,
        "eval_scenario_seeds": eval_scenario_seeds,
        "evaluation": "skipped" if args.skip_eval else "ok",
        "records": greedy_checkpoint_records,
    }
    with open(greedy_metrics_path, "w", encoding="utf-8") as fh:
        json.dump(greedy_metrics_payload, fh, ensure_ascii=False, indent=2)

    stage_status = {
        "dataset_train": "ok",
        "dataset_eval": "ok",
        "vanilla": "skipped" if args.skip_vanilla else "ok",
        "training": "skipped" if args.skip_train else "ok",
        "evaluation": "skipped" if args.skip_eval else "ok",
        "svm_evaluation": "skipped" if args.skip_svm_eval else "ok",
        "random_k": "ok" if args.random_k_only else "skipped",
        "top_k": "ok" if (args.top_k_only or args.eval_then_topk) else "skipped",
    }
    with open(os.path.join(run_dir, "stage_status.json"), "w", encoding="utf-8") as fh:
        json.dump(stage_status, fh, ensure_ascii=False, indent=2)

    print("Experiment completed.")
    print("Run dir:", run_dir)
    print("Train dataset dir:", dataset_train_dir)
    print("Eval dataset root dir:", dataset_eval_root_dir)
    print("Eval scenario seeds:", eval_scenario_seeds)
    td = os.path.join(run_dir, "train")
    print("Training dir:", td + (" (skipped)" if args.skip_train else ""))
    if not args.skip_eval:
        print("Eval root dir:", eval_root_dir)
    if not args.skip_svm_eval:
        print("SVM eval root dir:", svm_eval_root_dir)
    if args.random_k_only:
        print("Random-K root dir:", os.path.join(run_dir, "randomk"))
    if args.top_k_only or args.eval_then_topk:
        print("Top-K root dir:", os.path.join(run_dir, "topk"))
    print("Benchmark timing summary:", report_path)
    print("Greedy checkpoint metrics:", greedy_metrics_path)


if __name__ == "__main__":
    main()
