"""
Standalone greedy evaluation entrypoint for RL-enhanced Benders cut selection.

- Loads a trained CutPolicyNet checkpoint.
- Runs BendersMDPEnv in inference mode (no learning, no sampling).
- Selects cuts greedily by policy logits (fixed top-k or adaptive score-based K_t).
- Logs per-step metrics and a run summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from benders_solver_mdp import BendersMDPSolver
from cut_selection import compute_cut_features_batch
from env_benders_mdp import BendersMDPEnv
from nn_policy import CutPolicyNet
from reward_registry import get_reward_fn, with_reward_overrides


def _float_or_none(raw: Any) -> Optional[float]:
    if raw is None or raw == "":
        return None
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return float(x)


def _int_or_none(raw: Any) -> Optional[int]:
    x = _float_or_none(raw)
    if x is None:
        return None
    return int(round(x))


def _bool_or_none(raw: Any) -> Optional[bool]:
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


def _greedy_checkpoint_record(
    *,
    eval_scenario_seed: int,
    eval_index: int,
    dataset_dir: str,
    policy_name: Optional[str],
    policy_path: Optional[str],
    summary_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Same keys as run_experiment._greedy_checkpoint_record_from_mapping for benchmark_time_summary."""
    return {
        "eval_scenario_seed": int(eval_scenario_seed),
        "eval_index": int(eval_index),
        "dataset_dir": dataset_dir,
        "policy_name": policy_name,
        "policy_path": policy_path,
        "elapsed_sec": _float_or_none(summary_row.get("elapsed_sec")),
        "total_master_time": _float_or_none(summary_row.get("total_master_time")),
        "total_sub_time": _float_or_none(summary_row.get("total_sub_time")),
        "num_steps": _int_or_none(summary_row.get("num_steps")),
        "converged": _bool_or_none(summary_row.get("converged")),
        "total_selected_cuts": _int_or_none(summary_row.get("total_selected_cuts")),
        "total_reward": _float_or_none(summary_row.get("total_reward")),
        "reward_name": summary_row.get("reward_name"),
    }


def _write_greedy_checkpoint_metrics_json(
    *,
    run_dir: str,
    eval_scenario_seed: int,
    records: List[Dict[str, Any]],
) -> str:
    report_dir = os.path.join(run_dir, "benchmark_time_summary")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "greedy_checkpoint_metrics.json")
    payload: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "eval_scenario_seeds": [int(eval_scenario_seed)],
        "evaluation": "ok",
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def select_topk_greedy(logits: torch.Tensor, k: int) -> List[int]:
    """
    Greedy top-k selection by descending logits.

    - No softmax
    - No random sampling
    - Returns unique indices
    """
    logits = logits.reshape(-1)
    n = int(logits.numel())
    if n == 0 or k <= 0:
        return []
    k_eff = min(int(k), n)
    _, idx = torch.topk(logits, k=k_eff, largest=True, sorted=True)
    return [int(i) for i in idx.detach().cpu().tolist()]


def _logits_to_numpy(logits: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Detach torch logits to a 1-D float64 numpy vector."""
    if isinstance(logits, torch.Tensor):
        return logits.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    return np.asarray(logits, dtype=np.float64).reshape(-1)


@dataclass
class SelectionConfig:
    """Cut-selection settings for greedy evaluation."""

    mode: str = "fixed_k"
    k_select: int = 5
    adaptive_k_min: int = 3
    adaptive_k_max: int = 20
    adaptive_tau_base: float = 1.0
    adaptive_tau_min: float = 0.25
    adaptive_tau_max: float = 2.5
    adaptive_gap_ref: float = 0.20
    adaptive_stagnation_tol: float = 0.005
    adaptive_time_ref: float = 0.05
    adaptive_a_gap: float = 0.75
    adaptive_b_stag: float = 0.50
    adaptive_c_time: float = 0.50


def select_adaptive_score_greedy(
    logits: Union[torch.Tensor, np.ndarray],
    *,
    gap: float,
    previous_gap: float | None,
    master_time_last_iter: float,
    k_min: int = 3,
    k_max: int = 20,
    tau_base: float = 1.0,
    tau_min: float = 0.25,
    tau_max: float = 2.5,
    gap_ref: float = 0.20,
    stagnation_tol: float = 0.005,
    time_ref: float = 0.05,
    a_gap: float = 0.75,
    b_stag: float = 0.50,
    c_time: float = 0.50,
    eps: float = 1e-9,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Adaptive admission: policy scores rank cuts; K_t varies by gap/stagnation/time pressure.

    Threshold tau_t controls how many cuts pass the normalized score-drop filter; the final
    selection is always the top K_t cuts by raw policy logit (descending).
    """
    scores = _logits_to_numpy(logits)
    n = int(scores.size)

    def _empty_info(**extra: Any) -> Dict[str, Any]:
        base = {
            "adaptive_k": 0,
            "adaptive_tau": None,
            "score_max": None,
            "score_std": None,
            "gap_pressure": None,
            "stagnation_pressure": None,
            "time_pressure": None,
            "relative_gap_reduction": None,
            "num_admitted_before_clip": 0,
            "adaptive_k_min": int(k_min),
            "adaptive_k_max": int(k_max),
        }
        base.update(extra)
        return base

    if n == 0:
        return [], _empty_info()

    # Fewer candidates than k_min: admit all, sorted by score.
    if n < k_min:
        order = np.argsort(-scores)
        selected = [int(i) for i in order.tolist()]
        s_max = float(scores.max())
        sigma = float(scores.std())
        return selected, _empty_info(
            adaptive_k=len(selected),
            adaptive_tau=None,
            score_max=s_max,
            score_std=sigma,
            gap_pressure=min(1.0, float(gap) / (gap_ref + eps)) if gap_ref > 0 else 0.0,
            stagnation_pressure=0.0,
            time_pressure=min(1.0, float(master_time_last_iter) / (time_ref + eps)) if time_ref > 0 else 0.0,
            relative_gap_reduction=0.0,
            num_admitted_before_clip=n,
            note="n_candidates_lt_k_min_select_all",
        )

    k_max_eff = min(int(k_max), n)
    k_min_eff = min(int(k_min), n)

    s_max = float(scores.max())
    sigma = float(scores.std())
    drops = (s_max - scores) / (sigma + eps)

    gap_val = float(gap)
    p_gap = min(1.0, gap_val / (gap_ref + eps)) if gap_ref > 0 else 0.0

    delta_rel = 0.0
    if previous_gap is not None:
        try:
            prev = float(previous_gap)
            if np.isfinite(prev) and prev > 0.0 and np.isfinite(gap_val):
                delta_rel = (prev - gap_val) / (prev + eps)
        except (TypeError, ValueError):
            delta_rel = 0.0

    p_stag = 1.0 if delta_rel < stagnation_tol else 0.0

    master_t = float(master_time_last_iter) if master_time_last_iter is not None else 0.0
    if not np.isfinite(master_t) or master_t < 0.0:
        master_t = 0.0
    p_time = min(1.0, master_t / (time_ref + eps)) if time_ref > 0 else 0.0

    tau_raw = tau_base * (1.0 + a_gap * p_gap + b_stag * p_stag - c_time * p_time)
    tau = float(np.clip(tau_raw, tau_min, tau_max))

    admitted_mask = drops <= tau
    num_admitted = int(np.count_nonzero(admitted_mask))

    k_t = int(np.clip(num_admitted, k_min_eff, k_max_eff))

    order = np.argsort(-scores)
    selected = [int(i) for i in order[:k_t].tolist()]

    return selected, {
        "adaptive_k": len(selected),
        "adaptive_tau": tau,
        "score_max": s_max,
        "score_std": sigma,
        "gap_pressure": p_gap,
        "stagnation_pressure": p_stag,
        "time_pressure": p_time,
        "relative_gap_reduction": float(delta_rel),
        "num_admitted_before_clip": num_admitted,
        "adaptive_k_min": int(k_min),
        "adaptive_k_max": int(k_max),
        "adaptive_k_max_effective": k_max_eff,
        "adaptive_tau_raw": float(tau_raw),
    }


def _fixed_k_selection_info(selected_indices: List[int], k_select: int) -> Dict[str, Any]:
    """Diagnostics dict for fixed top-k mode."""
    return {
        "selection_mode": "fixed_k",
        "adaptive_k": len(selected_indices),
        "adaptive_tau": None,
        "score_max": None,
        "score_std": None,
        "gap_pressure": None,
        "stagnation_pressure": None,
        "time_pressure": None,
        "relative_gap_reduction": None,
        "num_admitted_before_clip": len(selected_indices),
        "adaptive_k_min": None,
        "adaptive_k_max": int(k_select),
    }


def _empty_selection_info(selection_cfg: SelectionConfig) -> Dict[str, Any]:
    """Diagnostics when there are zero candidate cuts."""
    if selection_cfg.mode == "fixed_k":
        return _fixed_k_selection_info([], selection_cfg.k_select)
    return {
        "selection_mode": "adaptive_score",
        "adaptive_k": 0,
        "adaptive_tau": None,
        "score_max": None,
        "score_std": None,
        "gap_pressure": None,
        "stagnation_pressure": None,
        "time_pressure": None,
        "relative_gap_reduction": None,
        "num_admitted_before_clip": 0,
        "adaptive_k_min": selection_cfg.adaptive_k_min,
        "adaptive_k_max": selection_cfg.adaptive_k_max,
    }


def _selection_log_fields(selection_info: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten selection_info for steps.jsonl rows."""
    return {
        "selection_mode": selection_info.get("selection_mode"),
        "adaptive_k": selection_info.get("adaptive_k"),
        "adaptive_tau": selection_info.get("adaptive_tau"),
        "score_max": selection_info.get("score_max"),
        "score_std": selection_info.get("score_std"),
        "gap_pressure": selection_info.get("gap_pressure"),
        "stagnation_pressure": selection_info.get("stagnation_pressure"),
        "time_pressure": selection_info.get("time_pressure"),
        "relative_gap_reduction": selection_info.get("relative_gap_reduction"),
        "num_admitted_before_clip": selection_info.get("num_admitted_before_clip"),
        "adaptive_k_min": selection_info.get("adaptive_k_min"),
        "adaptive_k_max": selection_info.get("adaptive_k_max"),
    }


def _aggregate_selection_stats(
    step_rows: List[Dict[str, Any]],
    selection_cfg: SelectionConfig,
) -> Dict[str, Any]:
    """Per-episode aggregate cut-count statistics for summary.json / summary.csv."""
    per_step_k = [int(r.get("adaptive_k") or 0) for r in step_rows if r.get("num_candidates", 0) > 0]
    if not per_step_k:
        per_step_k = [int(r.get("adaptive_k") or 0) for r in step_rows]

    if per_step_k:
        avg_k = float(np.mean(per_step_k))
        min_k = int(min(per_step_k))
        max_k = int(max(per_step_k))
    else:
        avg_k = 0.0
        min_k = 0
        max_k = 0

    if selection_cfg.mode == "fixed_k":
        k_min_out: Any = int(selection_cfg.k_select)
        k_max_out: Any = int(selection_cfg.k_select)
    else:
        k_min_out = int(selection_cfg.adaptive_k_min)
        k_max_out = int(selection_cfg.adaptive_k_max)

    return {
        "selection_mode": selection_cfg.mode,
        "avg_selected_cuts_per_iter": avg_k,
        "min_selected_cuts_per_iter": min_k,
        "max_selected_cuts_per_iter": max_k,
        "adaptive_k_min": k_min_out,
        "adaptive_k_max": k_max_out,
    }


def load_policy(policy_path: str, device: torch.device) -> CutPolicyNet:
    """Instantiate current architecture and load checkpoint strictly."""
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"policy checkpoint not found: {policy_path}")
    policy = CutPolicyNet().to(device)
    state_dict = torch.load(policy_path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def _init_nc_counter(env: BendersMDPEnv) -> np.ndarray:
    """Initialize NC counters from current scenario count if available."""
    theta = env.get_theta_current()
    if theta is not None and len(theta) > 0:
        return np.zeros(int(len(theta)), dtype=np.int32)
    cuts = env.get_candidate_cuts()
    if cuts:
        max_k = max(int(c["k"]) for c in cuts)
        return np.zeros(max_k + 1, dtype=np.int32)
    return np.zeros(0, dtype=np.int32)


def _build_nc_per_cut(cuts: Sequence[Dict[str, Any]], nc_added: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-cut NC values using old counter values (before current step).
    Returns (nc_per_cut, possibly_resized_nc_added).
    """
    if len(cuts) == 0:
        return np.zeros(0, dtype=np.float64), nc_added

    k_list: List[int] = []
    for c in cuts:
        try:
            k = int(c["k"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid cut scenario index: {c}") from e
        if k < 0:
            raise ValueError(f"invalid negative scenario index k={k}")
        if k >= nc_added.size:
            old = nc_added
            nc_added = np.zeros(k + 1, dtype=np.int32)
            nc_added[: old.size] = old
        k_list.append(k)

    nc_per_cut = np.asarray([nc_added[k] for k in k_list], dtype=np.float64)
    return nc_per_cut, nc_added


def run_greedy_episode(
    policy: CutPolicyNet,
    env: BendersMDPEnv,
    device: torch.device,
    max_iters: int,
    k_select: int,
    tol_gap: float,
    max_elapsed_sec: float = 0.0,
    selection_cfg: Optional[SelectionConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one full greedy evaluation episode and collect logs.

    max_elapsed_sec: if >0, stop after a step when cumulative master+sub solve seconds
    (same as summary total_master_time+total_sub_time / benchmark solve_time_sec) reaches the cap.
    """
    if selection_cfg is None:
        selection_cfg = SelectionConfig(k_select=k_select)

    # Cumulative wall from episode start (includes reset), same convention as Top-K steps.jsonl.
    loop_t0 = time.perf_counter()
    state = env.reset()
    nc_added = _init_nc_counter(env)
    step_rows: List[Dict[str, Any]] = []
    selected_scenarios_seen: set[int] = set()
    total_master_time = 0.0
    total_sub_time = 0.0
    total_sub_iter_count = 0.0
    total_sub_gurobi_runtime = 0.0
    z_l1_vals: List[float] = []
    n_zero_cap_vals: List[float] = []
    n_positive_beta_vals: List[float] = []
    total_selected_cuts = 0
    total_reward = 0.0

    last_info: Dict[str, Any] = {}
    done = False
    num_steps = 0
    stop_reason = "max_iters"

    previous_gap: Optional[float] = None
    master_time_last_iter = 0.0

    with torch.no_grad():
        for step in range(1, max_iters + 1):
            cuts = env.get_candidate_cuts()
            z = env.get_z_current()
            theta = env.get_theta_current()
            if z is None:
                z = np.zeros(1, dtype=np.float64)
            if theta is None:
                theta = np.zeros(1, dtype=np.float64)

            selected_indices: List[int]
            selected_logits: List[float]
            selection_info: Dict[str, Any]
            nc_per_cut, nc_added = _build_nc_per_cut(cuts, nc_added)

            current_gap = float(state.optimality_gap)

            if len(cuts) == 0:
                selected_indices = []
                selected_logits = []
                selection_info = _empty_selection_info(selection_cfg)
            else:
                cut_features = compute_cut_features_batch(cuts, z, theta, nc_per_cut=nc_per_cut)
                state_vec = state.to_vector()
                cf_t = torch.from_numpy(cut_features).float().to(device)
                sv_t = torch.from_numpy(state_vec).float().to(device)
                if sv_t.dim() == 1:
                    sv_t = sv_t.unsqueeze(0)
                logits = policy(cf_t, sv_t)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits

                if selection_cfg.mode == "fixed_k":
                    selected_indices = select_topk_greedy(logits, selection_cfg.k_select)
                    selection_info = _fixed_k_selection_info(selected_indices, selection_cfg.k_select)
                elif selection_cfg.mode == "adaptive_score":
                    selected_indices, selection_info = select_adaptive_score_greedy(
                        logits,
                        gap=current_gap,
                        previous_gap=previous_gap,
                        master_time_last_iter=master_time_last_iter,
                        k_min=selection_cfg.adaptive_k_min,
                        k_max=selection_cfg.adaptive_k_max,
                        tau_base=selection_cfg.adaptive_tau_base,
                        tau_min=selection_cfg.adaptive_tau_min,
                        tau_max=selection_cfg.adaptive_tau_max,
                        gap_ref=selection_cfg.adaptive_gap_ref,
                        stagnation_tol=selection_cfg.adaptive_stagnation_tol,
                        time_ref=selection_cfg.adaptive_time_ref,
                        a_gap=selection_cfg.adaptive_a_gap,
                        b_stag=selection_cfg.adaptive_b_stag,
                        c_time=selection_cfg.adaptive_c_time,
                    )
                    selection_info["selection_mode"] = "adaptive_score"
                else:
                    raise ValueError(f"Unknown selection_mode: {selection_cfg.mode!r}")

                logits_np = logits.detach().cpu().numpy().reshape(-1)
                selected_logits = [float(logits_np[i]) for i in selected_indices]

            action = {"selected_indices": selected_indices}
            next_state, reward_env, done, info = env.step(action)

            nc_selected_values: List[float] = []
            for idx in selected_indices:
                if 0 <= idx < len(cuts):
                    try:
                        k_sel = int(cuts[idx]["k"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if k_sel < 0:
                        continue
                    if k_sel >= nc_added.size:
                        old = nc_added
                        nc_added = np.zeros(k_sel + 1, dtype=np.int32)
                        nc_added[: old.size] = old
                    nc_selected_values.append(float(nc_added[k_sel]))
                    nc_added[k_sel] += 1
                    selected_scenarios_seen.add(k_sel)

            total_master_time += float(info.get("solve_time_master") or 0.0)
            total_sub_time += float(info.get("solve_time_sub") or 0.0)
            total_sub_iter_count += float(info.get("sub_iter_count_sum") or 0.0)
            total_sub_gurobi_runtime += float(info.get("sub_gurobi_runtime_sum") or 0.0)
            z_l1_vals.append(float(info.get("z_l1") or 0.0))
            n_zero_cap_vals.append(float(info.get("n_zero_cap") or 0.0))
            n_positive_beta_vals.append(float(info.get("n_positive_beta") or 0.0))
            if "selected_cut_count" in info and info["selected_cut_count"] is not None:
                step_selected = int(info["selected_cut_count"])
            else:
                step_selected = len(selected_indices)
            total_selected_cuts += max(0, step_selected)
            num_steps = step
            last_info = info
            total_reward += float(reward_env)

            wall_clock_sec = float(time.perf_counter() - loop_t0)
            row = {
                "step": step,
                "num_candidates": len(cuts),
                "gap_before_select": current_gap,
                "selected_indices": selected_indices,
                "selected_scenarios": info.get("selected_scenarios", []),
                "selected_logits": selected_logits,
                "LB_new": info.get("LB_new"),
                "UB_new": info.get("UB_new"),
                "gap_new": info.get("gap_new"),
                "reward": reward_env,
                "reward_env": reward_env,
                "total_cuts_in_master": info.get("total_cuts_in_rmp"),
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
                "nc_selected_mean": float(np.mean(nc_selected_values)) if nc_selected_values else 0.0,
                "nc_selected_max": float(np.max(nc_selected_values)) if nc_selected_values else 0.0,
                **_selection_log_fields(selection_info),
            }
            step_rows.append(row)

            previous_gap = current_gap
            master_time_last_iter = float(info.get("solve_time_master") or 0.0)

            state = next_state
            if done:
                stop_reason = "gap"
                break
            solve_cum = total_master_time + total_sub_time
            if max_elapsed_sec > 0.0 and solve_cum >= max_elapsed_sec:
                stop_reason = "time"
                break

    wall_loop_elapsed_sec = float(time.perf_counter() - loop_t0)
    final_gap = float(state.optimality_gap)
    final_lb = last_info.get("LB_new")
    final_ub = last_info.get("UB_new")
    final_total_cuts = (
        int(last_info.get("total_cuts_in_rmp"))
        if "total_cuts_in_rmp" in last_info
        else int(getattr(env.solver, "num_cuts")())
    )

    K_known = int(nc_added.size) if nc_added.size > 0 else None
    unique_count = int(len(selected_scenarios_seen))
    unique_ratio = (float(unique_count) / float(K_known)) if (K_known is not None and K_known > 0) else None

    summary = {
        "max_iters": max_iters,
        "max_elapsed_sec": float(max_elapsed_sec) if max_elapsed_sec > 0.0 else None,
        "wall_loop_elapsed_sec": wall_loop_elapsed_sec,
        "stop_reason": stop_reason,
        "k_select": k_select,
        "final_gap": final_gap,
        "final_LB": final_lb,
        "final_UB": final_ub,
        "num_steps": num_steps,
        "converged": bool(final_gap <= tol_gap),
        "total_master_time": total_master_time,
        "total_sub_time": total_sub_time,
        "total_sub_iter_count": total_sub_iter_count,
        "mean_sub_iter_count_per_iter": (
            float(total_sub_iter_count / num_steps) if num_steps > 0 else 0.0
        ),
        "mean_sub_iter_count_per_scenario": (
            float(total_sub_iter_count / (num_steps * float(last_info.get("n_scenarios_solved") or 1)))
            if num_steps > 0
            else 0.0
        ),
        "total_sub_gurobi_runtime": total_sub_gurobi_runtime,
        "mean_sub_gurobi_runtime_per_iter": (
            float(total_sub_gurobi_runtime / num_steps) if num_steps > 0 else 0.0
        ),
        "mean_z_l1": float(np.mean(z_l1_vals)) if z_l1_vals else None,
        "mean_n_zero_cap": float(np.mean(n_zero_cap_vals)) if n_zero_cap_vals else None,
        "mean_n_positive_beta": (
            float(np.mean(n_positive_beta_vals)) if n_positive_beta_vals else None
        ),
        "total_selected_cuts": total_selected_cuts,
        "final_total_cuts_in_master": final_total_cuts,
        "unique_selected_scenarios_count": unique_count,
        "unique_selected_scenarios_ratio": unique_ratio,
        "K_known": K_known,
        "total_reward": float(total_reward),
        "solve_time_sec": float(total_master_time + total_sub_time),
        **_aggregate_selection_stats(step_rows, selection_cfg),
    }
    return step_rows, summary


def discover_policies_in_dir(policy_dir: str) -> List[Tuple[str, str]]:
    """Discover checkpoint policies sorted by episode, then policy_final last."""
    checkpoints_dir = os.path.join(policy_dir, "checkpoints")
    policies: List[Tuple[str, str]] = []

    ep_items: List[Tuple[int, str, str]] = []
    if os.path.isdir(checkpoints_dir):
        for fname in sorted(os.listdir(checkpoints_dir)):
            if not (fname.startswith("policy_ep") and fname.endswith(".pt")):
                continue
            stem = os.path.splitext(fname)[0]
            ep_str = stem[len("policy_ep") :]
            try:
                ep = int(ep_str)
            except ValueError:
                continue
            ep_items.append((ep, stem, os.path.join(checkpoints_dir, fname)))
    ep_items.sort(key=lambda x: x[0])
    for _, stem, path in ep_items:
        policies.append((stem, path))

    final_path = os.path.join(policy_dir, "policy_final.pt")
    if os.path.isfile(final_path):
        policies.append(("policy_final", final_path))
    return policies


def evaluate_one_policy(
    *,
    policy_path: str,
    base_dir: str,
    seed: int,
    max_iters: int,
    k_select: int,
    tol_gap: float,
    device: torch.device,
    out_dir: str,
    run_name: str,
    reward_name: str,
    reward_overrides: Optional[Mapping[str, float]] = None,
    max_elapsed_sec: float = 0.0,
    selection_cfg: Optional[SelectionConfig] = None,
) -> Dict[str, Any]:
    """Run one policy evaluation and write steps.jsonl + summary.json."""
    if selection_cfg is None:
        selection_cfg = SelectionConfig(k_select=k_select)

    t0 = time.time()
    policy = load_policy(policy_path, device=device)
    solver = BendersMDPSolver(base_dir=base_dir, seed=seed)
    reward_fn = with_reward_overrides(get_reward_fn(reward_name), reward_overrides)
    env_cap = max(max_iters, 10**9) if max_elapsed_sec > 0.0 else max_iters
    env = BendersMDPEnv(
        solver=solver,
        reward_fn=reward_fn,
        max_iters=env_cap,
        tol_gap=tol_gap,
    )

    step_rows, summary = run_greedy_episode(
        policy=policy,
        env=env,
        device=device,
        max_iters=env_cap,
        k_select=k_select,
        tol_gap=tol_gap,
        max_elapsed_sec=max_elapsed_sec,
        selection_cfg=selection_cfg,
    )
    elapsed = time.time() - t0

    os.makedirs(out_dir, exist_ok=True)
    steps_path = os.path.join(out_dir, "steps.jsonl")
    with open(steps_path, "w", encoding="utf-8") as fh:
        for row in step_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_out = {
        "policy_path": policy_path,
        "base_dir": base_dir,
        "k_select": k_select,
        "seed": seed,
        "tol_gap": tol_gap,
        "reward_name": reward_name,
        "device": str(device),
        "elapsed_sec": elapsed,
        **summary,
        "max_iters": max_iters,
        "max_elapsed_sec": float(max_elapsed_sec) if max_elapsed_sec > 0.0 else None,
        "run_name": run_name,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_out, fh, ensure_ascii=False, indent=2)
    return summary_out


def _build_reward_overrides(args: argparse.Namespace) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    direct_keys = [
        "reward_eps",
        "reward_alpha",
        "reward_beta",
        "reward_tref",
        "reward_lambda_penalty",
        "reward_vl_ref",
        "reward_c",
        "reward_gamma",
        "reward_eta",
    ]
    for key in direct_keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = float(value)

    for item in getattr(args, "reward_param", []) or []:
        if "=" not in str(item):
            raise ValueError(f"invalid --reward_param {item!r}, expected KEY=VALUE")
        key, value = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --reward_param {item!r}, empty key")
        try:
            overrides[key] = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --reward_param {item!r}, value must be float") from exc

    return overrides


def _selection_config_from_args(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        mode=str(args.selection_mode),
        k_select=int(args.k_select),
        adaptive_k_min=int(args.adaptive_k_min),
        adaptive_k_max=int(args.adaptive_k_max),
        adaptive_tau_base=float(args.adaptive_tau_base),
        adaptive_tau_min=float(args.adaptive_tau_min),
        adaptive_tau_max=float(args.adaptive_tau_max),
        adaptive_gap_ref=float(args.adaptive_gap_ref),
        adaptive_stagnation_tol=float(args.adaptive_stagnation_tol),
        adaptive_time_ref=float(args.adaptive_time_ref),
        adaptive_a_gap=float(args.adaptive_a_gap),
        adaptive_b_stag=float(args.adaptive_b_stag),
        adaptive_c_time=float(args.adaptive_c_time),
    )


_SUMMARY_CSV_FIELDS = [
    "policy_name",
    "policy_path",
    "selection_mode",
    "avg_selected_cuts_per_iter",
    "min_selected_cuts_per_iter",
    "max_selected_cuts_per_iter",
    "final_gap",
    "num_steps",
    "converged",
    "final_LB",
    "final_UB",
    "total_master_time",
    "total_sub_time",
    "elapsed_sec",
    "total_selected_cuts",
    "adaptive_k_min",
    "adaptive_k_max",
    "final_total_cuts_in_master",
    "unique_selected_scenarios_count",
    "unique_selected_scenarios_ratio",
    "total_reward",
    "reward_name",
]


def _summary_csv_row(
    *,
    policy_name: str,
    policy_path: str,
    summary_out: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "policy_name": policy_name,
        "policy_path": policy_path,
        "selection_mode": summary_out.get("selection_mode"),
        "avg_selected_cuts_per_iter": summary_out.get("avg_selected_cuts_per_iter"),
        "min_selected_cuts_per_iter": summary_out.get("min_selected_cuts_per_iter"),
        "max_selected_cuts_per_iter": summary_out.get("max_selected_cuts_per_iter"),
        "final_gap": summary_out.get("final_gap"),
        "num_steps": summary_out.get("num_steps"),
        "converged": summary_out.get("converged"),
        "final_LB": summary_out.get("final_LB"),
        "final_UB": summary_out.get("final_UB"),
        "total_master_time": summary_out.get("total_master_time"),
        "total_sub_time": summary_out.get("total_sub_time"),
        "elapsed_sec": summary_out.get("elapsed_sec"),
        "total_selected_cuts": summary_out.get("total_selected_cuts"),
        "adaptive_k_min": summary_out.get("adaptive_k_min"),
        "adaptive_k_max": summary_out.get("adaptive_k_max"),
        "final_total_cuts_in_master": summary_out.get("final_total_cuts_in_master"),
        "unique_selected_scenarios_count": summary_out.get("unique_selected_scenarios_count"),
        "unique_selected_scenarios_ratio": summary_out.get("unique_selected_scenarios_ratio"),
        "total_reward": summary_out.get("total_reward"),
        "reward_name": summary_out.get("reward_name"),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained policy with greedy top-k cut selection")
    default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset test Feb")
    p.add_argument("--policy_path", type=str, default=None)
    p.add_argument("--policy_dir", type=str, default=None)
    p.add_argument("--base_dir", type=str, default=default_base)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument(
        "--eval_max_elapsed_sec",
        type=float,
        default=0.0,
        help=(
            "Cap on cumulative master+sub solve seconds in the greedy loop "
            "(same quantity as benchmark solve_time_sec = total_master_time + total_sub_time); "
            "0 disables. Last full step may slightly exceed the cap."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k_select", type=int, default=5)
    p.add_argument(
        "--selection_mode",
        type=str,
        default="fixed_k",
        choices=["fixed_k", "adaptive_score"],
        help="Cut selection mode: fixed top-k (default) or adaptive score-based K_t.",
    )
    p.add_argument("--adaptive_k_min", type=int, default=3)
    p.add_argument("--adaptive_k_max", type=int, default=20)
    p.add_argument("--adaptive_tau_base", type=float, default=1.0)
    p.add_argument("--adaptive_tau_min", type=float, default=0.25)
    p.add_argument("--adaptive_tau_max", type=float, default=2.5)
    p.add_argument("--adaptive_gap_ref", type=float, default=0.20)
    p.add_argument("--adaptive_stagnation_tol", type=float, default=0.005)
    p.add_argument("--adaptive_time_ref", type=float, default=0.05)
    p.add_argument("--adaptive_a_gap", type=float, default=0.75)
    p.add_argument("--adaptive_b_stag", type=float, default=0.50)
    p.add_argument("--adaptive_c_time", type=float, default=0.50)
    p.add_argument("--out_dir", type=str, default="./eval_runs")
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subdirectory name under --out_dir for this run (default: timestamp YYYYMMDD_HHMMSS).",
    )
    p.add_argument(
        "--use_out_dir_as_run_dir",
        action="store_true",
        help="Write all outputs directly under --out_dir (no timestamp / --run_name subfolder).",
    )
    p.add_argument("--tol_gap", type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--reward_name", type=str, default="log_gap_rmp_time_penalty")
    p.add_argument("--reward_eps", type=float, default=None)
    p.add_argument("--reward_alpha", type=float, default=None)
    p.add_argument("--reward_beta", type=float, default=None)
    p.add_argument("--reward_tref", type=float, default=None)
    p.add_argument("--reward_lambda_penalty", type=float, default=None)
    p.add_argument("--reward_vl_ref", type=float, default=None)
    p.add_argument("--reward_c", type=float, default=None)
    p.add_argument("--reward_gamma", type=float, default=None)
    p.add_argument("--reward_eta", type=float, default=None)
    p.add_argument(
        "--reward_param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional reward override, repeatable (same as mdp_commander)",
    )
    p.add_argument(
        "--write_benchmark_metrics",
        action="store_true",
        help="Write run_dir/benchmark_time_summary/greedy_checkpoint_metrics.json (same schema as run_experiment).",
    )
    p.add_argument(
        "--eval_scenario_seed",
        type=int,
        default=None,
        help="Label for greedy_checkpoint_metrics records; default: same as --seed",
    )
    p.add_argument(
        "--eval_index",
        type=int,
        default=0,
        help="eval_index field in greedy_checkpoint_metrics records (default 0).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    reward_overrides = _build_reward_overrides(args)
    selection_cfg = _selection_config_from_args(args)
    eff_eval_seed = int(args.eval_scenario_seed) if args.eval_scenario_seed is not None else int(args.seed)
    if not args.policy_path and not args.policy_dir:
        raise ValueError("Provide at least one of --policy_path or --policy_dir")

    out_base = os.path.abspath(args.out_dir)
    if args.use_out_dir_as_run_dir:
        run_dir = out_base
        run_name_str = (
            str(args.run_name).strip()
            if (args.run_name is not None and str(args.run_name).strip() != "")
            else "flat"
        )
        os.makedirs(run_dir, exist_ok=True)
    elif args.run_name is not None and str(args.run_name).strip() != "":
        run_name_str = str(args.run_name).strip()
        run_dir = os.path.join(out_base, run_name_str)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_name_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(out_base, run_name_str)
        os.makedirs(run_dir, exist_ok=True)

    if args.policy_dir:
        discovered = discover_policies_in_dir(args.policy_dir)
        if not discovered:
            raise FileNotFoundError(f"No policy files found in: {args.policy_dir}")

        rows: List[Dict[str, Any]] = []
        metrics_records: List[Dict[str, Any]] = []
        for policy_name, policy_path in discovered:
            policy_out_dir = os.path.join(run_dir, policy_name)
            summary_out = evaluate_one_policy(
                policy_path=policy_path,
                base_dir=args.base_dir,
                seed=args.seed,
                max_iters=args.max_iters,
                k_select=args.k_select,
                tol_gap=args.tol_gap,
                device=device,
                out_dir=policy_out_dir,
                run_name=run_name_str,
                reward_name=args.reward_name,
                reward_overrides=reward_overrides,
                max_elapsed_sec=float(args.eval_max_elapsed_sec),
                selection_cfg=selection_cfg,
            )
            metrics_records.append(
                _greedy_checkpoint_record(
                    eval_scenario_seed=eff_eval_seed,
                    eval_index=int(args.eval_index),
                    dataset_dir=args.base_dir,
                    policy_name=policy_name,
                    policy_path=policy_path,
                    summary_row=summary_out,
                )
            )
            rows.append(_summary_csv_row(
                policy_name=policy_name,
                policy_path=policy_path,
                summary_out=summary_out,
            ))

        summary_csv_path = os.path.join(run_dir, "summary.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_SUMMARY_CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        if args.write_benchmark_metrics:
            bm_path = _write_greedy_checkpoint_metrics_json(
                run_dir=run_dir,
                eval_scenario_seed=eff_eval_seed,
                records=metrics_records,
            )
            print("Greedy checkpoint metrics:", bm_path)

        print("Greedy batch evaluation finished.")
        print("Run dir:", run_dir)
        print("Batch summary:", summary_csv_path)
        print("Policies evaluated:", len(rows))
        return

    summary_out = evaluate_one_policy(
        policy_path=args.policy_path,
        base_dir=args.base_dir,
        seed=args.seed,
        max_iters=args.max_iters,
        k_select=args.k_select,
        tol_gap=args.tol_gap,
        device=device,
        out_dir=run_dir,
        run_name=run_name_str,
        reward_name=args.reward_name,
        reward_overrides=reward_overrides,
        max_elapsed_sec=float(args.eval_max_elapsed_sec),
        selection_cfg=selection_cfg,
    )

    if args.write_benchmark_metrics:
        bm_path = _write_greedy_checkpoint_metrics_json(
            run_dir=run_dir,
            eval_scenario_seed=eff_eval_seed,
            records=[
                _greedy_checkpoint_record(
                    eval_scenario_seed=eff_eval_seed,
                    eval_index=int(args.eval_index),
                    dataset_dir=args.base_dir,
                    policy_name="policy_single",
                    policy_path=args.policy_path,
                    summary_row=summary_out,
                )
            ],
        )
        print("Greedy checkpoint metrics:", bm_path)

    print("Greedy evaluation finished.")
    print("Run dir:", run_dir)
    print("Steps log:", os.path.join(run_dir, "steps.jsonl"))
    print("Summary:", os.path.join(run_dir, "summary.json"))
    print("Final gap:", summary_out["final_gap"])
    print("Converged:", summary_out["converged"])
    print("Total reward:", summary_out.get("total_reward"))


if __name__ == "__main__":
    main()
