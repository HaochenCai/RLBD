from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

from vanilla_benders import read_inputs


def _load_svm_module(module_path: str):
    spec = importlib.util.spec_from_file_location("jia_shen_svm_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from path: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure decorators/types that inspect sys.modules (e.g., dataclass on py3.9)
    # can resolve this dynamically loaded module during execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_problem_payload(base_dir: str):
    _, _, f, b, M, C, p, r, c, d, I, J, N = read_inputs(base_dir)
    warehouse = list(range(I))
    factory = list(range(J))
    scenario = list(range(N))
    pair = [(i, j) for i in range(I) for j in range(J)]
    prob = np.full(N, 1.0 / float(N), dtype=float)
    return {
        "Warehouse": warehouse,
        "setup": f,
        "capacity": C,
        "Factory": factory,
        "lostsalecost": p,
        "Pair": pair,
        "cost": c,
        "Scenario": scenario,
        "probability": prob,
        "Demand": d,
        "charge_cost": b,
        "max_chargers": M,
        "revenue": r,
    }


def run_svm_eval(
    *,
    base_dir: str,
    out_dir: str,
    run_name: str | None,
    seed: int,
    stage1_paths: int,
    stage1_iters: int,
    tol_rel_gap: float,
    key: float,
    key_low: float,
    key_change_time: float,
    itelimit: int,
    pretrain: int,
    timelimit: float,
    train_base_dir: str | None = None,
    training_records_path: str | None = None,
) -> Dict[str, Any]:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Jia&Shen_SVM.py")
    svm_module = _load_svm_module(module_path)
    payload = _build_problem_payload(base_dir)
    train_payload = _build_problem_payload(train_base_dir) if train_base_dir else None

    t0 = time.time()
    outputs = svm_module.learn_bd(
        payload["Warehouse"],
        payload["setup"],
        payload["capacity"],
        payload["Factory"],
        payload["lostsalecost"],
        payload["Pair"],
        payload["cost"],
        payload["Scenario"],
        payload["probability"],
        payload["Demand"],
        [float(tol_rel_gap)],
        int(stage1_paths),
        int(stage1_iters),
        run_dir + os.sep,
        run_dir + os.sep,
        float(key),
        float(key_low),
        float(key_change_time),
        int(itelimit),
        int(pretrain),
        float(timelimit),
        charge_cost=payload["charge_cost"],
        max_chargers=payload["max_chargers"],
        revenue=payload["revenue"],
        rng_seed=int(seed),
        train_payload=train_payload,
        training_records_path=training_records_path,
    )
    elapsed = float(time.time() - t0)

    # Accept both legacy 9-tuple and current 10-tuple (with walltimev).
    if len(outputs) >= 10:
        (
            lowerboundv,
            realgapv,
            iterationv,
            cutnv,
            ratiobindv,
            mastertimev,
            subtimev,
            pos_bd_flag,
            trainingtime_v,
            walltimev,
        ) = outputs[:10]
    else:
        (
            lowerboundv,
            realgapv,
            iterationv,
            cutnv,
            ratiobindv,
            mastertimev,
            subtimev,
            pos_bd_flag,
            trainingtime_v,
        ) = outputs[:9]
        # Fallback: master+sub cum is not wall, but better than empty for old builds.
        walltimev = [
            float(mastertimev[i]) + float(subtimev[i]) for i in range(len(iterationv))
        ]

    rows = []
    for idx in range(len(iterationv)):
        rows.append(
            {
                "iteration": int(iterationv[idx]),
                "lowerbound": float(lowerboundv[idx]),
                "rel_gap": float(realgapv[idx]),
                "num_constraints": int(cutnv[idx]),
                "ratio_binding": float(ratiobindv[idx]),
                "master_time_cum_sec": float(mastertimev[idx]),
                "sub_time_cum_sec": float(subtimev[idx]),
                "wall_clock_sec": float(walltimev[idx]),
                "bd_mode_or_key": pos_bd_flag[idx],
                "retrain_time_sec": float(trainingtime_v[idx]),
            }
        )

    curve_path = os.path.join(run_dir, "svm_phase3_curve.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "iteration",
            "lowerbound",
            "rel_gap",
            "num_constraints",
            "ratio_binding",
            "master_time_cum_sec",
            "sub_time_cum_sec",
            "wall_clock_sec",
            "bd_mode_or_key",
            "retrain_time_sec",
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    phase3_master = float(mastertimev[-1]) if mastertimev else 0.0
    phase3_sub = float(subtimev[-1]) if subtimev else 0.0
    phase3_solve = phase3_master + phase3_sub
    final_gap = float(realgapv[-1]) if realgapv else float("inf")
    converged = bool(np.isfinite(final_gap) and final_gap <= float(tol_rel_gap))

    summary = {
        "base_dir": base_dir,
        "train_base_dir": train_base_dir,
        "training_records_path": training_records_path,
        "seed": int(seed),
        "run_name": run_name,
        "stage1_paths": int(stage1_paths),
        "stage1_iters": int(stage1_iters),
        "tol_rel_gap": float(tol_rel_gap),
        "key": float(key),
        "key_low": float(key_low),
        "key_change_time": float(key_change_time),
        "itelimit": int(itelimit),
        "timelimit": float(timelimit),
        "pretrain": int(pretrain),
        "num_phase3_iterations": int(len(iterationv)),
        "final_lowerbound": float(lowerboundv[-1]) if lowerboundv else None,
        "final_rel_gap": final_gap if np.isfinite(final_gap) else None,
        "converged": converged,
        "phase3_master_time_sec": phase3_master,
        "phase3_sub_time_sec": phase3_sub,
        "phase3_solve_time_sec": phase3_solve,
        # Same value as phase3_solve_time_sec; name matches benchmark_time_summary / run_experiment.
        "svm_stage3_solve_time_sec": phase3_solve,
        "phase3_wall_clock_sec": float(walltimev[-1]) if walltimev else None,
        "elapsed_total_sec": elapsed,
        "curve_path": curve_path,
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SVM-based Benders benchmark (Jia & Shen style)")
    p.add_argument("--base_dir", type=str, required=True)
    p.add_argument(
        "--train_base_dir",
        type=str,
        default=None,
        help=(
            "If set, Phase-1 SVM label collection uses this directory (e.g. RLBD dataset_train); "
            "Phase-3 Benders still uses --base_dir (eval). If omitted, all phases use --base_dir."
        ),
    )
    p.add_argument("--out_dir", type=str, default="./svm_eval_runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage1_paths", type=int, default=10)
    p.add_argument("--stage1_iters", type=int, default=30)
    p.add_argument("--tol_rel_gap", type=float, default=1e-2)
    p.add_argument("--key", type=float, default=1.5)
    p.add_argument("--key_low", type=float, default=1.0)
    p.add_argument("--key_change_time", type=float, default=5.0)
    p.add_argument("--itelimit", type=int, default=2000)
    p.add_argument(
        "--timelimit",
        type=float,
        default=3600.0,
        help=(
            "Phase-3 only: stop when cumulative master+sub solve time (seconds) reaches this value; "
            "same quantity as phase3_solve_time_sec / svm_stage3_solve_time_sec in summary. "
            "Use a very large number for no practical cap. Values <= 0 disable this cap."
        ),
    )
    p.add_argument("--pretrain", type=int, default=0)
    p.add_argument(
        "--training_records_path",
        type=str,
        default=None,
        help="CSV for Phase-1 SVM records: load if file exists, else generate (pretrain=0) and save.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_svm_eval(
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        seed=args.seed,
        stage1_paths=args.stage1_paths,
        stage1_iters=args.stage1_iters,
        tol_rel_gap=args.tol_rel_gap,
        key=args.key,
        key_low=args.key_low,
        key_change_time=args.key_change_time,
        itelimit=args.itelimit,
        timelimit=args.timelimit,
        pretrain=args.pretrain,
        train_base_dir=args.train_base_dir,
        training_records_path=args.training_records_path,
    )
    print("SVM eval finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
