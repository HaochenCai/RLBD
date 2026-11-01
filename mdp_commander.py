from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benders_solver_mdp import BendersMDPSolver
from env_benders_mdp import BendersMDPEnv
# from reward_plugins import gap_reduction_reward
# from reward_log_gap import log_gap_reward
# from reward_log_gap_vl import  log_gap_vl_reward
# from reward_log_gap_step_penalty import log_gap_step_penalty_reward
# from reward_log_gap_step_penalty_baseline import log_gap_step_penalty_baseline_reward
# from reward_log_gap_time_weighted import log_gap_time_weighted_reward
# from reward_abs_gap_step_penalty import abs_gap_step_penalty_reward
# from reward_phase_adaptive_log_gap import phase_adaptive_log_gap_reward

# from reward_log_gap_log_vl_finalexp import log_gap_log_vl_finalexp_reward
# from reward_log_gap_norm_vl_finalexp import log_gap_norm_vl_finalexp_reward
#from reward_phase_log_gap_log_vl_finalexp import phase_log_gap_log_vl_finalexp_reward
# from reward_phase_log_gap_norm_vl_finalexp import phase_log_gap_norm_vl_finalexp_reward

from cut_selection import compute_cut_features_batch, sample_indices_from_logits_sequential
from nn_policy import CutPolicyNet
from reward_registry import get_reward_fn, with_reward_overrides

_EPS = 1e-12
def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_step_debug_metrics(
    info: Dict[str, Any],
    initial_probs_np: Optional[np.ndarray],
) -> Dict[str, Any]:
    eps = 1e-8
    gap_old = info.get("gap_old")
    gap_new = info.get("gap_new")
    gap_term = None
    try:
        gap_old_val = float(gap_old)
        gap_new_val = float(gap_new)
        if (
            not math.isnan(gap_old_val)
            and not math.isnan(gap_new_val)
            and gap_old_val >= 0.0
            and gap_new_val >= 0.0
        ):
            gap_term = math.log(gap_old_val + eps) - math.log(gap_new_val + eps)
    except (TypeError, ValueError):
        gap_term = None

    vl_old = info.get("violation_old_mean", info.get("vl_old", None))
    vl_new = info.get("violation_new_mean", info.get("vl_new", None))
    delta_vl = None
    try:
        if vl_old is not None and vl_new is not None:
            delta_vl = float(vl_old) - float(vl_new)
    except (TypeError, ValueError):
        delta_vl = None

    vl_term = None
    try:
        if vl_old is not None and vl_new is not None:
            vl_old_val = float(vl_old)
            vl_new_val = float(vl_new)
            if (
                not math.isnan(vl_old_val)
                and not math.isnan(vl_new_val)
                and vl_old_val >= 0.0
                and vl_new_val >= 0.0
            ):
                vl_term = (vl_old_val - vl_new_val) / (vl_old_val + eps)
    except (TypeError, ValueError):
        vl_term = None

    initial_probs = initial_probs_np.tolist() if initial_probs_np is not None else []
    initial_entropy = 0.0
    for p in initial_probs:
        initial_entropy -= float(p) * math.log(float(p) + 1e-8)

    return {
        "gap_term": gap_term,
        "vl_old": vl_old,
        "vl_new": vl_new,
        "delta_vl": delta_vl,
        "vl_term": vl_term,
        "initial_entropy": initial_entropy,
        "initial_probs": initial_probs,
    }


def _run_episodic(
    env: BendersMDPEnv,
    policy: CutPolicyNet,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    seed: int,
    max_iters: int,
    temperature: float,
    log_every: int,
    debug_state: bool,
    state_vec_history: Optional[list],
    steps_path: str,
    run_dir: str,
    num_episodes: int = 10,
    k_select: int = 5,
    reward_ma_window: int = 10,
    ckpt_every: int = 100,
    max_elapsed_sec: float = 0.0,
) -> Tuple[List[float], List[float], str, float]:
    """Episodic REINFORCE: rollout per episode, then one update at episode end."""
    train_t0 = time.perf_counter()
    gamma = 0.99
    all_rewards: List[float] = []
    episode_total_rewards: List[float] = []
    episode_solve_times_sec: List[float] = []
    returns_debug_path = os.path.join(run_dir, "returns_debug.json")
    returns_debug_data = []

    # Pre-update weights (before any episodic optimizer.step); policy_ep0 sorts first in greedy eval.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    _init_ckpt = os.path.join(ckpt_dir, "policy_ep0.pt")
    torch.save(policy.state_dict(), _init_ckpt)
    print(f"[INFO] Saved pre-update (episodic) policy to: {_init_ckpt}")

    with open(steps_path, "w", encoding="utf-8") as fh:
        stop_training_time = False
        for episode in range(1, num_episodes + 1):
            episode_t0 = time.time()
            state = env.reset()
            theta_init = env.get_theta_current()
            if theta_init is not None and len(theta_init) > 0:
                num_scenarios = int(len(theta_init))
            else:
                cuts_init = env.get_candidate_cuts()
                if cuts_init:
                    num_scenarios = max(int(c["k"]) for c in cuts_init) + 1
                else:
                    num_scenarios = 0
            nc_added = np.zeros(num_scenarios, dtype=np.int32)
            trajectory: List[Dict[str, Any]] = []
            episode_rows: List[Dict[str, Any]] = []
            episode_reward = 0.0
            episode_steps = 0

            for step in range(1, max_iters + 1):
                cuts = env.get_candidate_cuts()
                z = env.get_z_current()
                theta = env.get_theta_current()
                if z is None:
                    z = np.zeros(1, dtype=np.float64)
                if theta is None:
                    theta = np.zeros(1, dtype=np.float64)

                state_vec = state.to_vector()
                initial_probs_np = None
                cut_features = None
                if len(cuts) == 0:
                    action = {"selected_indices": []}
                    log_prob = torch.tensor(0.0, device=device)
                    log_prob.requires_grad_(False)
                    nc_per_cut = np.zeros(0, dtype=np.float64)
                else:
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
                    cut_features = compute_cut_features_batch(cuts, z, theta, nc_per_cut=nc_per_cut)
                    cf_t = torch.from_numpy(cut_features).float().to(device)
                    sv_t = torch.from_numpy(state_vec).float().to(device)
                    if sv_t.dim() == 1:
                        sv_t = sv_t.unsqueeze(0)
                    policy.train()
                    logits = policy(cf_t, sv_t)
                    logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                    logits = logits / max(_EPS, temperature)
                    probs_torch = torch.softmax(logits, dim=0)
                    initial_probs_np = probs_torch.detach().cpu().numpy()
                    step_seed = int(seed + episode * 100000 + step)
                    selected_indices, log_prob = sample_indices_from_logits_sequential(logits, k=k_select, seed=step_seed)
                    action = {"selected_indices": selected_indices}

                next_state, reward, done, info = env.step(action)
                nc_selected_values: List[float] = []
                for idx in action["selected_indices"]:
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
                debug_metrics = _build_step_debug_metrics(info, initial_probs_np)
                episode_reward += reward
                episode_steps = step
                all_rewards.append(reward)

                trajectory.append({
                    "log_prob": log_prob,
                    "reward": reward,
                    "done": done,
                    "selected_indices": action["selected_indices"],
                    "info": info,
                    "action": action,
                })

                row = {
                    "episode": episode,
                    "iter": step,
                    "gap_old": info.get("gap_old"),
                    "gap_new": info.get("gap_new"),
                    "solve_time_master": (
                        info.get("solve_time_master")
                        or info.get("rmp_solve_time")
                        or info.get("master_solve_time")
                    ),
                    "reward": reward,
                    "done": done,
                    "selected": info.get("selected_cut_count"),
                    "candidate": info.get("candidate_cut_count"),
                    "LB_new": info.get("LB_new"),
                    "UB_new": info.get("UB_new"),
                    "selected_indices": action["selected_indices"],
                    "loss": None,
                    "log_prob": float(log_prob.item()),
                    "debug_gap_term": debug_metrics["gap_term"],
                    "debug_vl_old": debug_metrics["vl_old"],
                    "debug_vl_new": debug_metrics["vl_new"],
                    "debug_delta_vl": debug_metrics["delta_vl"],
                    "debug_vl_term": debug_metrics["vl_term"],
                    "debug_initial_entropy": debug_metrics["initial_entropy"],
                    "nc_selected_mean": float(np.mean(nc_selected_values)) if len(nc_selected_values) > 0 else 0.0,
                    "nc_selected_max": float(np.max(nc_selected_values)) if len(nc_selected_values) > 0 else 0.0,
                    "nc_vector_nonzero_count": int(np.count_nonzero(nc_added)),
                }
                reward_debug = info.get("reward_debug", {})
                if isinstance(reward_debug, dict) and reward_debug:
                    row.update(reward_debug)
                print(f"iter={step}")
                print("  initial_probs:", debug_metrics["initial_probs"])
                print("  selected:", action["selected_indices"])
                print("  gap_term:", debug_metrics["gap_term"])
                print("  vl_old:", debug_metrics["vl_old"], "vl_new:", debug_metrics["vl_new"], "delta_vl:", debug_metrics["delta_vl"])
                print("  vl_term:", debug_metrics["vl_term"])
                print("  reward:", reward)
                print("  initial_entropy:", debug_metrics["initial_entropy"])
                if log_every > 0 and step % log_every == 0:
                    scenarios_str = f"scenarios={info.get('selected_scenarios', [])}" if info.get("selected_scenarios") else "scenarios=[]"
                    rmp_str = f"rmp_cuts={info.get('total_cuts_in_rmp', 0)}"
                    print(f"ep={episode} iter={step} reward={reward:.6f} gap={info.get('gap_new', 0):.6f} {scenarios_str} {rmp_str}")

                episode_rows.append(row)
                if debug_state and state_vec_history is not None:
                    state_vec_history.append(state.to_vector())
                state = next_state
                if done:
                    break
                if max_elapsed_sec > 0.0 and (time.perf_counter() - train_t0) >= max_elapsed_sec:
                    stop_training_time = True
                    break

            returns: List[float] = []
            G = 0.0
            for t in reversed(range(len(trajectory))):
                G = trajectory[t]["reward"] + gamma * G
                returns.insert(0, G)

            returns_np = np.array(returns, dtype=np.float64)

            mean = returns_np.mean()
            std = returns_np.std() + 1e-8  

            returns = ((returns_np - mean) / (std + 1e-8)).tolist()    
            # compute statistics
            returns_np = np.array(returns, dtype=np.float64)
            returns_mean = float(np.mean(returns_np))
            returns_std = float(np.std(returns_np))
            returns_max = float(np.max(returns_np))
            returns_min = float(np.min(returns_np))

            # append to debug data
            returns_debug_data.append({
                "episode": episode,
                "returns": [float(x) for x in returns],
                "mean": returns_mean,
                "std": returns_std,
                "max": returns_max,
                "min": returns_min,
            })
            if episode == 1:
                print("\n===== G (returns) DEBUG =====")
                print("mean:", returns_mean)
                print("std:", returns_std)
                print("max:", returns_max)
                print("min:", returns_min)
                print("first 10:", returns[:10])
                print("last 10:", returns[-10:])
                print("================================\n")

            # Episode loss: only include differentiable log_prob terms
            loss_terms: List[torch.Tensor] = []
            for t, rec in enumerate(trajectory):
                if rec["log_prob"].requires_grad:
                    advantage_t = returns[t]
                    loss_terms.append(-rec["log_prob"] * advantage_t)

            if loss_terms:
                loss = sum(loss_terms)
                optimizer.zero_grad()
                # print gradient norm 
                

                loss.backward()
                optimizer.step()
                episode_loss = float(loss.item())
            else:
                episode_loss = float("nan")
                print(f"Warning: episode {episode} had no differentiable policy decisions, skipping optimizer update")

            for row in episode_rows:
                row["loss"] = episode_loss
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

            final_gap = float(state.optimality_gap)
            episode_total_rewards.append(float(episode_reward))
            episode_solve_time_sec = float(time.time() - episode_t0)
            episode_solve_times_sec.append(episode_solve_time_sec)
            print(f"Episode {episode}: steps={episode_steps} total_reward={episode_reward:.6f} final_gap={final_gap:.6f} loss={episode_loss:.6f}")
            print(f"Episode {episode} summary:")
            print("  total_reward:", episode_reward)
            print("  final_gap:", final_gap)
            print("  iterations:", episode_steps)
            print("  episode_loss:", episode_loss)
            print("  episode_solve_time_sec:", episode_solve_time_sec)
            if ckpt_every > 0 and episode >= ckpt_every and episode % ckpt_every == 0:
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"policy_ep{episode}.pt")
                torch.save(policy.state_dict(), ckpt_path)

            if stop_training_time:
                break

    # Reward vs steps line plot
    plot_path = os.path.join(run_dir, "reward_steps.png")
    plt.figure()
    plt.plot(range(1, len(all_rewards) + 1), all_rewards, color="blue")
    plt.xlabel("Global training step")
    plt.ylabel("Reward")
    plt.title("Reward vs steps (episodic REINFORCE)")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Reward plot saved to {plot_path}")

    # Standard RL training curve: episode accumulated reward (+ moving average)
    episode_reward_sums: Dict[int, float] = {}
    with open(steps_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = rec.get("episode")
            rew = rec.get("reward")
            try:
                ep_i = int(ep)
                rew_f = float(rew)
            except (TypeError, ValueError):
                continue
            if ep_i not in episode_reward_sums:
                episode_reward_sums[ep_i] = 0.0
            episode_reward_sums[ep_i] += rew_f

    episodes = sorted(episode_reward_sums.keys())
    acc_rewards = [float(episode_reward_sums[e]) for e in episodes]
    plot_ep_path = os.path.join(run_dir, "reward_episodes.png")

    if episodes:
        plt.figure()
        plt.plot(episodes, acc_rewards, alpha=0.35, label="Episode reward")
        window = max(1, int(reward_ma_window))
        if len(acc_rewards) >= window and window > 1:
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smooth = np.convolve(np.asarray(acc_rewards, dtype=np.float64), kernel, mode="valid")
            smooth_x = episodes[window - 1 :]
            plt.plot(smooth_x, smooth, linewidth=2.0, label=f"Moving average ({window})")
        plt.title("Episode Reward (REINFORCE)")
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_ep_path, dpi=100)
        plt.close()
        print(
            f"[episode_reward] episodes={len(acc_rewards)} "
            f"min={min(acc_rewards):.6f} max={max(acc_rewards):.6f} mean={float(np.mean(acc_rewards)):.6f}"
        )
    else:
        plt.figure()
        plt.title("Episode Reward (REINFORCE)")
        plt.xlabel("Episode")
        plt.ylabel("Accumulated Reward")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_ep_path, dpi=100)
        plt.close()
        print("[episode_reward] episodes=0 min=nan max=nan mean=nan")
    print(f"Episode reward plot saved to {plot_ep_path}")

    plot_time_path = os.path.join(run_dir, "episode_solve_time.png")
    if episodes:
        plt.figure()
        plt.plot(episodes, episode_solve_times_sec, color="green")
        plt.title("Episode Solve Time")
        plt.xlabel("Episode")
        plt.ylabel("Solve Time (sec)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_time_path, dpi=100)
        plt.close()
    else:
        plt.figure()
        plt.title("Episode Solve Time")
        plt.xlabel("Episode")
        plt.ylabel("Solve Time (sec)")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(plot_time_path, dpi=100)
        plt.close()
    print(f"Episode solve-time plot saved to {plot_time_path}")

    with open(returns_debug_path, "w", encoding="utf-8") as f:
        json.dump(returns_debug_data, f, ensure_ascii=False, indent=2)
    training_wall_elapsed_sec = float(time.perf_counter() - train_t0)
    episodic_stop_reason = "time" if stop_training_time else "num_episodes"
    return episode_total_rewards, episode_solve_times_sec, episodic_stop_reason, training_wall_elapsed_sec


def run_episode(
    base_dir: str,
    seed: int = 0,
    max_iters: int = 2000,
    out_dir: str = "./runs",
    run_name: Optional[str] = None,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_ckpt: Optional[str] = None,
    load_ckpt: Optional[str] = None,
    log_every: int = 1,
    debug_state: bool = False,
    episodic: bool = False,
    num_episodes: int = 10,
    k_select: int = 5,
    reward_ma_window: int = 10,
    reward_name: str = "log_gap_rmp_time_penalty",
    tol_gap: float = 1e-2,
    reward_overrides: Optional[Dict[str, float]] = None,
    ckpt_every: int = 100,
    max_elapsed_sec: float = 0.0,
    solver: Optional[Any] = None,
) -> Dict[str, Any]:
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if solver is None:
        solver = BendersMDPSolver(base_dir=base_dir, seed=seed)
    # env = BendersMDPEnv(solver=solver, reward_fn=gap_reduction_reward, max_iters=max_iters, tol_gap=1e-4)
    # env = BendersMDPEnv(solver=solver, reward_fn=log_gap_reward, max_iters=max_iters, tol_gap=1e-4)
    # env = BendersMDPEnv(solver=solver, reward_fn=log_gap_vl_reward, max_iters=max_iters, tol_gap=1e-2)
    # env = BendersMDPEnv(solver=solver, reward_fn=log_gap_step_penalty_reward, max_iters=max_iters, tol_gap=1e-2)
    # env = BendersMDPEnv(solver=solver, reward_fn=log_gap_step_penalty_baseline_reward, max_iters=max_iters, tol_gap=1e-2)
    #env = BendersMDPEnv(solver=solver, reward_fn=log_gap_time_weighted_reward, max_iters=max_iters, tol_gap=1e-2)
    # env = BendersMDPEnv(solver=solver, reward_fn=abs_gap_step_penalty_reward, max_iters=max_iters, tol_gap=1e-2)
    # env = BendersMDPEnv(solver=solver, reward_fn=phase_adaptive_log_gap_reward, max_iters=max_iters, tol_gap=1e-2)
    #env = BendersMDPEnv(solver=solver, reward_fn=log_gap_log_vl_finalexp_reward, max_iters=max_iters, tol_gap=1e-2)
    reward_fn = get_reward_fn(reward_name)
    reward_fn = with_reward_overrides(reward_fn, reward_overrides)
    step_cap = max(max_iters, 10**9) if max_elapsed_sec > 0.0 else max_iters
    env = BendersMDPEnv(solver=solver, reward_fn=reward_fn, max_iters=step_cap, tol_gap=tol_gap)

    device = torch.device("cpu")
    policy = CutPolicyNet().to(device)
    if load_ckpt and os.path.isfile(load_ckpt):
        policy.load_state_dict(torch.load(load_ckpt, map_location=device))
        print(f"Loaded policy from {load_ckpt}")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    t0 = time.time()
    steps_path = os.path.join(run_dir, "steps.jsonl")
    iters_used = 0
    total_reward = 0.0
    total_selected_cuts = 0

    # debug_state: collect per-dim min/max; at episode end assert >=16 dims vary
    state_vec_history: Optional[list] = None
    if debug_state:
        state_vec_history = []

    if episodic:
        episode_total_rewards, episode_solve_times_sec, ep_stop_reason, training_wall_elapsed_sec = _run_episodic(
            env=env,
            policy=policy,
            optimizer=optimizer,
            device=device,
            seed=seed,
            max_iters=step_cap,
            temperature=temperature,
            log_every=log_every,
            debug_state=debug_state,
            state_vec_history=state_vec_history,
            steps_path=steps_path,
            run_dir=run_dir,
            num_episodes=num_episodes,
            k_select=k_select,
            reward_ma_window=reward_ma_window,
            ckpt_every=ckpt_every,
            max_elapsed_sec=max_elapsed_sec,
        )
        if debug_state and state_vec_history is not None and len(state_vec_history) >= 1:
            arr = np.array(state_vec_history, dtype=np.float32)
            assert arr.ndim == 2 and arr.shape[1] == 24, "state must be 24-dim"
            assert np.all(np.isfinite(arr)), "state must have no NaN/Inf"
            varying_dims = np.sum(np.ptp(arr, axis=0) > 1e-8)
            min_varying = 16 if len(state_vec_history) >= 10 else 12
            if len(state_vec_history) >= 2:
                assert varying_dims >= min_varying, (
                    f"state should have >= {min_varying} varying dims, got {varying_dims}"
                )
            print(f"[debug_state] {varying_dims}/24 dims varied, range OK")
        total_time_sec = time.time() - t0
        summary = {
            "base_dir": base_dir,
            "seed": seed,
            "max_iters": max_iters,
            "max_elapsed_sec": float(max_elapsed_sec) if max_elapsed_sec > 0.0 else None,
            "training_wall_elapsed_sec": training_wall_elapsed_sec,
            "stop_reason": ep_stop_reason,
            "episodic": True,
            "num_episodes": num_episodes,
            "total_time_sec": total_time_sec,
            "run_name": run_name,
        }
        summary_path = os.path.join(run_dir, "summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=summary.keys())
            w.writeheader()
            w.writerow(summary)
        summary_episodic_path = os.path.join(run_dir, "summary_episodic.json")
        with open(summary_episodic_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    **summary,
                    "episode_total_rewards": episode_total_rewards,
                    "episode_solve_times_sec": episode_solve_times_sec,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        model_path = os.path.join(run_dir, "policy_final.pt")
        torch.save(policy.state_dict(), model_path)
        print(f"[INFO] Saved trained policy to: {model_path}")
        if save_ckpt:
            ckpt_path = os.path.join(run_dir, "policy.pt") if save_ckpt in ("", ".") else save_ckpt
            torch.save(policy.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        return summary

    train_t0 = time.perf_counter()
    stop_reason = "max_iters"
    state = env.reset()
    with open(steps_path, "w", encoding="utf-8") as fh:
        for step in range(1, step_cap + 1):
            cuts = env.get_candidate_cuts()
            z = env.get_z_current()
            theta = env.get_theta_current()
            if z is None:
                z = np.zeros(1, dtype=np.float64)
            if theta is None:
                theta = np.zeros(1, dtype=np.float64)

            state_vec = state.to_vector()
            initial_probs_np = None
            cut_features = None
            if len(cuts) == 0:
                action = {"selected_indices": []}
                log_prob = torch.tensor(0.0, device=device)
                log_prob.requires_grad_(False)
            else:
                cut_features = compute_cut_features_batch(cuts, z, theta)
                cf_t = torch.from_numpy(cut_features).float().to(device)
                sv_t = torch.from_numpy(state_vec).float().to(device)
                if sv_t.dim() == 1:
                    sv_t = sv_t.unsqueeze(0)

                policy.train()
                logits = policy(cf_t, sv_t)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                logits = logits / max(_EPS, temperature)
                probs_torch = torch.softmax(logits, dim=0)
                initial_probs_np = probs_torch.detach().cpu().numpy()

                assert abs(float(probs_torch.sum()) - 1.0) < 1e-5, "probs must sum to 1"

                step_seed = int(seed + step * 100000)
                selected_indices, log_prob = sample_indices_from_logits_sequential(logits, k=k_select, seed=step_seed)
                for idx in selected_indices:
                    assert 0 <= idx < len(cuts), f"selected_idx={idx} out of [0, {len(cuts)-1}]"
                assert len(selected_indices) == len(set(selected_indices)), "selected_indices must be unique"

                action = {"selected_indices": selected_indices}

            next_state, reward, done, info = env.step(action)
            debug_metrics = _build_step_debug_metrics(info, initial_probs_np)

            assert info["selected_indices"] == action["selected_indices"], "selected_indices mismatch (double sampling)"

            advantage = float(reward)
            loss = -(log_prob * advantage)
            if log_prob.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_reward += reward
            total_selected_cuts += info.get("selected_cut_count", 0)
            iters_used = step

            row = {
                "iter": step,
                "gap_old": info.get("gap_old"),
                "gap_new": info.get("gap_new"),
                "solve_time_master": (
                    info.get("solve_time_master")
                    or info.get("rmp_solve_time")
                    or info.get("master_solve_time")
                ),
                "reward": reward,
                "done": done,
                "selected": info.get("selected_cut_count"),
                "candidate": info.get("candidate_cut_count"),
                "LB_new": info.get("LB_new"),
                "UB_new": info.get("UB_new"),
                "selected_indices": action["selected_indices"],
                "loss": float(loss.item()),
                "log_prob": float(log_prob.item()),
                "debug_gap_term": debug_metrics["gap_term"],
                "debug_vl_old": debug_metrics["vl_old"],
                "debug_vl_new": debug_metrics["vl_new"],
                "debug_delta_vl": debug_metrics["delta_vl"],
                "debug_vl_term": debug_metrics["vl_term"],
                "debug_initial_entropy": debug_metrics["initial_entropy"],
            }
            reward_debug = info.get("reward_debug", {})
            if isinstance(reward_debug, dict) and reward_debug:
                row.update(reward_debug)
            print(f"iter={step}")
            print("  initial_probs:", debug_metrics["initial_probs"])
            print("  selected:", action["selected_indices"])
            print("  gap_term:", debug_metrics["gap_term"])
            print("  vl_old:", debug_metrics["vl_old"], "vl_new:", debug_metrics["vl_new"], "delta_vl:", debug_metrics["delta_vl"])
            print("  vl_term:", debug_metrics["vl_term"])
            print("  reward:", reward)
            print("  initial_entropy:", debug_metrics["initial_entropy"])
            if log_every > 0 and step % log_every == 0:
                scenarios_str = f"scenarios={info.get('selected_scenarios', [])}" if info.get("selected_scenarios") else "scenarios=[]"
                rmp_str = f"rmp_cuts={info.get('total_cuts_in_rmp', 0)}"
                print(f"iter={step} reward={reward:.6f} loss={row['loss']:.6f} gap={info.get('gap_new', 0):.6f} {scenarios_str} {rmp_str}")
                if cut_features is not None:
                    print("cut_features stats")
                    print(cut_features.min(), cut_features.max())
                print("state stats")
                print(state_vec.min(), state_vec.max())
                if initial_probs_np is not None:
                    probs_str = " ".join(f"{p:.4f}" for p in initial_probs_np)
                    print(f"  initial_probs: [{probs_str}]")

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if debug_state and state_vec_history is not None:
                state_vec_history.append(state.to_vector())
            state = next_state
            if done:
                stop_reason = "gap"
                break
            if max_elapsed_sec > 0.0 and (time.perf_counter() - train_t0) >= max_elapsed_sec:
                stop_reason = "time"
                break

    if debug_state and state_vec_history is not None and len(state_vec_history) >= 1:
        state_vec_history.append(state.to_vector())
        arr = np.array(state_vec_history, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == 24, "state must be 24-dim"
        assert np.all(np.isfinite(arr)), "state must have no NaN/Inf"
        varying_dims = np.sum(np.ptp(arr, axis=0) > 1e-8)
        min_varying = 16 if len(state_vec_history) >= 10 else 12
        if len(state_vec_history) >= 2:
            assert varying_dims >= min_varying, (
                f"state should have >= {min_varying} varying dims, got {varying_dims}; "
                f"min/max per dim: {np.min(arr, axis=0)}, {np.max(arr, axis=0)}"
            )
        print(f"[debug_state] {varying_dims}/24 dims varied, range OK")

    total_time_sec = time.time() - t0
    training_wall_elapsed_sec = float(time.perf_counter() - train_t0)
    final_gap = float(state.optimality_gap)

    if save_ckpt:
        ckpt_path = os.path.join(run_dir, "policy.pt") if save_ckpt in ("", ".") else save_ckpt
        torch.save(policy.state_dict(), ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    summary = {
        "base_dir": base_dir,
        "seed": seed,
        "max_iters": max_iters,
        "max_elapsed_sec": float(max_elapsed_sec) if max_elapsed_sec > 0.0 else None,
        "training_wall_elapsed_sec": training_wall_elapsed_sec,
        "stop_reason": stop_reason,
        "iters_used": iters_used,
        "final_gap": final_gap,
        "total_reward": total_reward,
        "total_time_sec": total_time_sec,
        "total_selected_cuts": total_selected_cuts,
        "run_name": run_name,
    }
    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=summary.keys())
        w.writeheader()
        w.writerow(summary)
    model_path = os.path.join(run_dir, "policy_final.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"[INFO] Saved trained policy to: {model_path}")

    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MDP Commander: RL training (REINFORCE)")
    _default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset test Feb")
    p.add_argument("--base_dir", type=str, default=_default_base)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--out_dir", type=str, default="./runs")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--save_ckpt", type=str, default=None, nargs="?", const=".")
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--sanity", action="store_true", help="Run short episode (max_iters=3) and exit")
    p.add_argument("--debug_state", action="store_true", help="Assert >= 16 state dims vary, no NaN/Inf")
    p.add_argument("--episodic", action="store_true", help="Episodic REINFORCE: 10 episodes, update at episode end")
    p.add_argument("--num_episodes", type=int, default=10, help="Number of episodes when --episodic")
    p.add_argument("--k_select", type=int, default=5, help="Number of cuts to sample each step")
    p.add_argument("--reward_ma_window", type=int, default=10, help="Moving-average window for episode reward plot")
    p.add_argument("--reward_name", type=str, default="log_gap_rmp_time_penalty")
    p.add_argument("--tol_gap", type=float, default=1e-2)
    p.add_argument(
        "--train_max_elapsed_sec",
        type=float,
        default=0.0,
        help="Wall-clock cap (perf_counter) on RL training; 0 disables. gap<=tol still ends early.",
    )
    p.add_argument("--ckpt_every", type=int, default=100, help="Save episodic checkpoint every N episodes")
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
        help="Additional reward override, repeatable",
    )
    return p.parse_args()


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


def _sanity_test(base_dir: str, debug_state: bool = False) -> None:
    """Short episode (max_iters=3) to confirm no exceptions and finite loss."""
    if not os.path.isdir(base_dir):
        print("Skip sanity test: dataset not found")
        return
    summary = run_episode(
        base_dir=base_dir,
        seed=42,
        max_iters=3,
        out_dir="./runs",
        run_name="sanity_test",
        debug_state=debug_state,
    )
    assert summary["iters_used"] >= 1
    assert np.isfinite(summary["total_reward"])
    print("Sanity test OK")


if __name__ == "__main__":
    args = _parse_args()
    _set_global_seed(args.seed)
    reward_overrides = _build_reward_overrides(args)
    if args.sanity:
        _sanity_test(args.base_dir, debug_state=args.debug_state)
        exit(0)
    summary = run_episode(
        base_dir=args.base_dir,
        seed=args.seed,
        max_iters=args.max_iters,
        out_dir=args.out_dir,
        run_name=args.run_name,
        lr=args.lr,
        temperature=args.temperature,
        save_ckpt=args.save_ckpt,
        load_ckpt=args.load_ckpt,
        log_every=args.log_every,
        debug_state=args.debug_state,
        episodic=args.episodic,
        num_episodes=args.num_episodes,
        k_select=args.k_select,
        reward_ma_window=args.reward_ma_window,
        reward_name=args.reward_name,
        tol_gap=args.tol_gap,
        reward_overrides=reward_overrides,
        ckpt_every=args.ckpt_every,
        max_elapsed_sec=float(args.train_max_elapsed_sec),
    )
    print("Run finished:", summary)
    out_path = os.path.join(args.out_dir, summary["run_name"])
    print("Outputs: steps.jsonl + summary.csv under", out_path)
    if args.episodic:
        print("Reward plot: reward_steps.png under", out_path)
