"""
SVM-based Benders benchmark (3 phases) aligned with current pipeline math.

This file preserves the original callable entry `learn_bd(...)` while aligning
the master/subproblem formulation with `vanilla_benders.py`.

Phase 1 (label collection) may use a separate training instance via
``train_payload`` (e.g. RLBD ``dataset_train``). Phases 2–3 always use the
primary ``inp`` built from the eval instance (e.g. ``dataset_eval/seed_*``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from sklearn import svm

from vanilla_benders import solve_subproblem_duals

_EPS = 1e-12
_VIOL_TOL = 1e-9
_TRAINING_CACHE: List[Dict[str, float]] = []


class _AlwaysPositiveClassifier:
    def predict(self, X: np.ndarray) -> np.ndarray:
        n = int(np.asarray(X).shape[0])
        return np.ones(n, dtype=int)


@dataclass
class _AlignedInputs:
    scenarios: List[int]
    stations: List[int]
    customers: List[int]
    setup: np.ndarray
    charge_cost: np.ndarray
    max_chargers: np.ndarray
    capacity: np.ndarray
    probability: np.ndarray
    demand: np.ndarray
    c: np.ndarray
    p: np.ndarray
    r: np.ndarray


def _as_vector(values: Any, keys: Sequence[int], default: float = 0.0) -> np.ndarray:
    if isinstance(values, np.ndarray):
        out = np.asarray(values, dtype=float).reshape(-1)
        if out.size != len(keys):
            raise ValueError(f"vector length mismatch: {out.size} != {len(keys)}")
        return out
    if isinstance(values, (list, tuple)):
        out = np.asarray(values, dtype=float).reshape(-1)
        if out.size != len(keys):
            raise ValueError(f"vector length mismatch: {out.size} != {len(keys)}")
        return out
    if isinstance(values, dict):
        return np.asarray([float(values.get(k, default)) for k in keys], dtype=float)
    raise TypeError(f"unsupported vector container: {type(values)}")


def _as_matrix(
    values: Any,
    row_keys: Sequence[int],
    col_keys: Sequence[int],
    pair_keys: Iterable[Tuple[int, int]] | None = None,
) -> np.ndarray:
    if isinstance(values, np.ndarray):
        mat = np.asarray(values, dtype=float)
        if mat.shape != (len(row_keys), len(col_keys)):
            raise ValueError(f"matrix shape mismatch: {mat.shape} != {(len(row_keys), len(col_keys))}")
        return mat
    if isinstance(values, list):
        mat = np.asarray(values, dtype=float)
        if mat.shape != (len(row_keys), len(col_keys)):
            raise ValueError(f"matrix shape mismatch: {mat.shape} != {(len(row_keys), len(col_keys))}")
        return mat
    if isinstance(values, dict):
        row_pos = {k: i for i, k in enumerate(row_keys)}
        col_pos = {k: j for j, k in enumerate(col_keys)}
        mat = np.zeros((len(row_keys), len(col_keys)), dtype=float)
        if pair_keys is not None:
            for i_key, j_key in pair_keys:
                mat[row_pos[i_key], col_pos[j_key]] = float(values[(i_key, j_key)])
            return mat
        for (i_key, j_key), v in values.items():
            if i_key in row_pos and j_key in col_pos:
                mat[row_pos[i_key], col_pos[j_key]] = float(v)
        return mat
    raise TypeError(f"unsupported matrix container: {type(values)}")


def _normalize_probability(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError("empty scenario probability")
    if np.any(p < 0.0):
        raise ValueError("scenario probability must be non-negative")
    s = float(np.sum(p))
    if s <= _EPS:
        return np.full_like(p, 1.0 / float(p.size))
    return p / s


def _aligned_inputs_from_payload(payload: Dict[str, Any]) -> _AlignedInputs:
    """Build _AlignedInputs from the same dict layout as run_svm_eval._build_problem_payload."""
    return _prepare_inputs(
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
        charge_cost=payload.get("charge_cost"),
        max_chargers=payload.get("max_chargers"),
        revenue=payload.get("revenue"),
    )


def _prepare_inputs(
    Warehouse,
    setup,
    capacity,
    Factory,
    lostsalecost,
    Pair,
    cost,
    Scenario,
    probability,
    Demand,
    charge_cost=None,
    max_chargers=None,
    revenue=None,
) -> _AlignedInputs:
    scenarios = list(Scenario)
    stations = list(Warehouse)
    customers = list(Factory)

    f = _as_vector(setup, stations)
    b = _as_vector(charge_cost if charge_cost is not None else np.zeros_like(f), stations)
    M = _as_vector(max_chargers if max_chargers is not None else np.full_like(f, 10.0), stations)
    C = _as_vector(capacity, stations)
    p = _as_vector(lostsalecost, customers)
    r = _as_vector(revenue if revenue is not None else np.zeros_like(p), customers)
    prob = _normalize_probability(_as_vector(probability, scenarios, default=0.0))
    c = _as_matrix(cost, stations, customers, pair_keys=Pair)
    d = _as_matrix(Demand, scenarios, customers)

    return _AlignedInputs(
        scenarios=scenarios,
        stations=stations,
        customers=customers,
        setup=f,
        charge_cost=b,
        max_chargers=M,
        capacity=C,
        probability=prob,
        demand=d,
        c=c,
        p=p,
        r=r,
    )


def _build_master(inp: _AlignedInputs, output_flag: bool = False):
    I = len(inp.stations)
    K = len(inp.scenarios)
    model = gp.Model("MP_aligned")
    model.Params.OutputFlag = int(bool(output_flag))
    model.Params.Threads = 1
    y = model.addVars(I, vtype=GRB.BINARY, name="y")
    z = model.addVars(I, vtype=GRB.INTEGER, lb=0, name="z")
    theta = model.addVars(K, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")

    const_k = -np.einsum("kj,j->k", inp.demand, inp.r)
    for k in range(K):
        model.addConstr(theta[k] >= float(const_k[k]), name=f"theta_lb_{k}")
    for i in range(I):
        model.addConstr(z[i] <= float(inp.max_chargers[i]) * y[i], name=f"link_{i}")

    model.setObjective(
        gp.quicksum(float(inp.setup[i]) * y[i] + float(inp.charge_cost[i]) * z[i] for i in range(I))
        + gp.quicksum(float(inp.probability[k]) * theta[k] for k in range(K)),
        GRB.MINIMIZE,
    )
    return model, y, z, theta


def _collect_subproblem_info(inp: _AlignedInputs, z_val: np.ndarray, theta_val: np.ndarray):
    K = len(inp.scenarios)
    I = len(inp.stations)
    q_vals = np.zeros(K, dtype=float)
    cuts = []
    for k in range(K):
        qk, alpha_k, beta_k, _, _sub_stats = solve_subproblem_duals(
            c=inp.c,
            d_k=inp.demand[k, :],
            p=inp.p,
            r=inp.r,
            C=inp.capacity,
            z_val=z_val,
            OutputFlag=0,
        )
        coef_z = np.asarray([beta_k[i] * float(inp.capacity[i]) for i in range(I)], dtype=float)
        rhs_const = float(qk - np.dot(coef_z, z_val))
        violation = max(0.0, rhs_const + float(np.dot(coef_z, z_val)) - float(theta_val[k]))
        q_vals[k] = float(qk)
        cuts.append(
            {
                "k": k,
                "rhs_const": rhs_const,
                "coef_z": coef_z,
                "violation": float(violation),
                "alpha": alpha_k,
                "beta": beta_k,
            }
        )
    return cuts, q_vals


def _train_classifier_from_records(records: List[Dict[str, Any]], key: float):
    if not records:
        return _AlwaysPositiveClassifier(), 0.0
    df = pd.DataFrame(records).copy()
    for col in ("record_idx", "path_id", "iter_idx", "violation", "n_prev_cuts", "obj_change"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    labels = []
    for _, grp in df.groupby("path_id", sort=True):
        grp = grp.sort_values(["iter_idx", "record_idx"]).reset_index(drop=True)
        obj_changes = grp["obj_change"].to_numpy(dtype=float)
        grp_labels = np.ones(len(grp), dtype=int)
        for i in range(len(grp) - 1):
            grp_labels[i] = -1 if obj_changes[i] < key * obj_changes[i + 1] else 1
        labels.extend(grp_labels.tolist())
    df["label"] = labels
    df = df.drop_duplicates(subset=["violation", "n_prev_cuts"], keep="first")
    X = df[["violation", "n_prev_cuts"]].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return _AlwaysPositiveClassifier(), 0.0

    clf = svm.SVC()
    t0 = time.perf_counter()
    clf.fit(X, y)
    t1 = time.perf_counter()
    return clf, float(t1 - t0)


def learn_bd(
    Warehouse,
    setup,
    capacity,
    Factory,
    lostsalecost,
    Pair,
    cost,
    Scenario,
    probability,
    Demand,
    gapv,
    K,
    N,
    training_fileposition,
    fileposition,
    key,
    key_low,
    key_change_time,
    itelimit,
    pretrain,
    timelimit,  # Phase-3: max cumulative master+sub solve seconds (<=0 disables).
    charge_cost=None,
    max_chargers=None,
    revenue=None,
    rng_seed: int = 1234,
    train_payload: Dict[str, Any] | None = None,
    training_records_path: str | None = None,
):
    global _TRAINING_CACHE
    inp = _prepare_inputs(
        Warehouse=Warehouse,
        setup=setup,
        capacity=capacity,
        Factory=Factory,
        lostsalecost=lostsalecost,
        Pair=Pair,
        cost=cost,
        Scenario=Scenario,
        probability=probability,
        Demand=Demand,
        charge_cost=charge_cost,
        max_chargers=max_chargers,
        revenue=revenue,
    )
    # Phase 1 (SVM label collection): optional separate training instance (e.g. RLBD dataset_train).
    # Phase 2–3: classifier fit + Benders loop always use `inp` (eval / stage-3 test instance).
    inp_train = _aligned_inputs_from_payload(train_payload) if train_payload is not None else inp

    gaplimit = float(gapv[-1] if isinstance(gapv, (list, tuple, np.ndarray)) and len(gapv) > 0 else gapv)
    rng = np.random.default_rng(int(rng_seed))
    phase1_master_times: List[float] = []
    phase1_sub_times: List[float] = []
    training_records: List[Dict[str, Any]] = []

    print("##########################################")
    print("Learn-Stage1")
    print("##########################################")
    if training_records_path and os.path.isfile(training_records_path):
        df_loaded = pd.read_csv(training_records_path)
        training_records = df_loaded.to_dict("records")
        print(f"Loaded Phase-1 SVM training records from: {training_records_path}")
    elif int(pretrain) != 0:
        path_hint = training_records_path if training_records_path else "(no --training_records_path given)"
        raise FileNotFoundError(
            f"pretrain={pretrain} requires existing Phase-1 SVM training records, but no file was found. "
            f"Expected CSV at {path_hint!r}. Run once with pretrain=0 and the same --training_records_path "
            f"to generate svm_training_records.csv, or point to an existing file."
        )
    elif int(pretrain) == 0:
        for path_id in range(int(K)):
            model, y, z, theta = _build_master(inp_train, output_flag=False)
            master_time = 0.0
            sub_time = 0.0
            cut_count_by_sce = np.zeros(len(inp_train.scenarios), dtype=np.int32)
            record_idx = 0

            for iter_idx in range(1, int(N) + 1):
                t0 = time.perf_counter()
                model.optimize()
                t1 = time.perf_counter()
                master_time += t1 - t0
                if model.Status != GRB.OPTIMAL:
                    break
                obj_before = float(model.ObjVal)
                y_val = np.array([y[i].X for i in range(len(inp_train.stations))], dtype=float)
                z_val = np.array([z[i].X for i in range(len(inp_train.stations))], dtype=float)
                theta_val = np.array([theta[k].X for k in range(len(inp_train.scenarios))], dtype=float)

                order = rng.permutation(len(inp_train.scenarios))
                new_cut_added = False
                I = len(inp_train.stations)
                for k in order:
                    k_int = int(k)
                    t2 = time.perf_counter()
                    qk, alpha_k, beta_k, _, _sub_stats = solve_subproblem_duals(
                        c=inp_train.c,
                        d_k=inp_train.demand[k_int, :],
                        p=inp_train.p,
                        r=inp_train.r,
                        C=inp_train.capacity,
                        z_val=z_val,
                        OutputFlag=0,
                    )
                    t3 = time.perf_counter()
                    sub_time += t3 - t2
                    coef_z = np.asarray([beta_k[i] * float(inp_train.capacity[i]) for i in range(I)], dtype=float)
                    rhs_const = float(qk - np.dot(coef_z, z_val))
                    violation = max(0.0, rhs_const + float(np.dot(coef_z, z_val)) - float(theta_val[k_int]))
                    if violation <= _VIOL_TOL:
                        continue
                    cut = {
                        "rhs_const": rhs_const,
                        "coef_z": coef_z,
                        "violation": float(violation),
                    }
                    model.addConstr(
                        theta[int(k_int)]
                        >= float(cut["rhs_const"])
                        + gp.quicksum(float(cut["coef_z"][i]) * z[i] for i in range(len(inp_train.stations))),
                        name=f"svm_train_path{path_id}_it{iter_idx}_k{int(k_int)}",
                    )
                    cut_count_by_sce[k_int] += 1
                    t4 = time.perf_counter()
                    model.optimize()
                    t5 = time.perf_counter()
                    master_time += t5 - t4
                    if model.Status != GRB.OPTIMAL:
                        continue
                    obj_change = abs(float(model.ObjVal - obj_before))
                    training_records.append(
                        {
                            "record_idx": float(record_idx),
                            "path_id": float(path_id),
                            "iter_idx": float(iter_idx),
                            "scenario_id": float(k_int),
                            "violation": float(cut["violation"]),
                            "n_prev_cuts": float(cut_count_by_sce[k_int] - 1),
                            "obj_change": float(obj_change),
                        }
                    )
                    record_idx += 1
                    new_cut_added = True
                    break

                if not new_cut_added:
                    break

            phase1_master_times.append(master_time)
            phase1_sub_times.append(sub_time)
            print(f"K={path_id + 1}")
            print("mastertime-", master_time)
            print("subtime-", sub_time)

        if training_records_path:
            parent = os.path.dirname(training_records_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            cols = [
                "record_idx",
                "path_id",
                "iter_idx",
                "scenario_id",
                "violation",
                "n_prev_cuts",
                "obj_change",
            ]
            df_save = pd.DataFrame(training_records)
            df_save = df_save[[c for c in cols if c in df_save.columns]]
            df_save.to_csv(training_records_path, index=False)
            print(f"Saved Phase-1 SVM training records to: {training_records_path}")

    _TRAINING_CACHE = training_records

    print("##########################################")
    print("Learn-Stage2")
    print("##########################################")
    clf, _ = _train_classifier_from_records(_TRAINING_CACHE, key=float(key))

    print("##########################################")
    print("Learn-Stage3")
    print("##########################################")
    model, y, z, theta = _build_master(inp, output_flag=False)
    cut_count_by_sce = np.zeros(len(inp.scenarios), dtype=np.int32)
    retrain_iteration: List[int] = []
    trainingtime_v: List[float] = []
    mastertime = 0.0
    subtime = 0.0
    mastertimev: List[float] = []
    subtimev: List[float] = []
    walltimev: List[float] = []
    ratiobindv: List[float] = []
    iterationv: List[int] = []
    realgapv: List[float] = []
    cutnv: List[int] = []
    lowerboundv: List[float] = []
    pos_bd_flag: List[Any] = []
    bd_flag = 0
    key_now = float(key)
    key_range = max(0.0, float(key) - float(key_low))
    key_change_time = max(1.0, float(key_change_time))
    ub_best = float("inf")

    t0_total = time.perf_counter()
    t = 1
    while t <= int(itelimit):
        t0 = time.perf_counter()
        model.optimize()
        t1 = time.perf_counter()
        mastertime += t1 - t0
        if model.Status != GRB.OPTIMAL:
            break

        lb = float(model.ObjVal)
        y_val = np.array([y[i].X for i in range(len(inp.stations))], dtype=float)
        z_val = np.array([z[i].X for i in range(len(inp.stations))], dtype=float)
        theta_val = np.array([theta[k].X for k in range(len(inp.scenarios))], dtype=float)

        t2 = time.perf_counter()
        cuts, q_vals = _collect_subproblem_info(inp, z_val=z_val, theta_val=theta_val)
        t3 = time.perf_counter()
        subtime += t3 - t2

        fs_cost = float(np.dot(inp.setup, y_val) + np.dot(inp.charge_cost, z_val))
        ub_candidate = fs_cost + float(np.dot(inp.probability, q_vals))
        ub_best = min(ub_best, ub_candidate)
        gap = (ub_best - lb) / (abs(ub_best) + _EPS)

        print(f"Iteration {t}")
        ncut = 0
        for cut in cuts:
            k = int(cut["k"])
            if cut["violation"] <= _VIOL_TOL:
                continue
            if bd_flag == 0:
                feat = np.array([[float(cut["violation"]), float(cut_count_by_sce[k])]], dtype=float)
                score = int(clf.predict(feat)[0])
            else:
                score = 1
            if score != 1:
                continue
            model.addConstr(
                theta[k]
                >= float(cut["rhs_const"])
                + gp.quicksum(float(cut["coef_z"][i]) * z[i] for i in range(len(inp.stations))),
                name=f"svm_eval_it{t}_k{k}_{ncut}",
            )
            cut_count_by_sce[k] += 1
            ncut += 1

        pos_bd_flag.append(key_now if bd_flag == 0 else "tradi")
        if ncut == 0 and bd_flag == 0:
            if key_now > float(key_low):
                key_now = key_now - key_range / key_change_time
                clf, trainingtime = train_svm(training_fileposition, fileposition, key_now, t, K, N)
                retrain_iteration.append(t)
                trainingtime_v.append(float(trainingtime))
            else:
                bd_flag = 1
                trainingtime_v.append(0.0)
                print("##########################################")
                print("TraBD----Iteration------", t)
                print("##########################################")
        else:
            trainingtime_v.append(0.0)

        print("UB", ub_best, "     LB", lb, "    dif", gap, "   mastertime", mastertime)
        slack = np.asarray(model.getAttr("Slack"), dtype=float)
        n_bind = int(np.sum(np.abs(slack) <= 1e-8)) if slack.size > 0 else 0
        ratio_bind = float(n_bind / slack.size) if slack.size > 0 else math.inf
        # Cumulative wall since Phase-3 start (includes retrain when it happens this iter).
        wall_clock_sec = float(time.perf_counter() - t0_total)
        cutnv.append(int(slack.size))
        mastertimev.append(float(mastertime))
        subtimev.append(float(subtime))
        walltimev.append(wall_clock_sec)
        iterationv.append(int(t))
        ratiobindv.append(float(ratio_bind))
        realgapv.append(float(gap))
        lowerboundv.append(float(lb))

        if gap <= gaplimit:
            break
        # timelimit: cumulative Phase-3 master+sub solve time (same as phase3_solve_time_sec / svm_stage3_solve_time_sec).
        if float(timelimit) > 0.0 and (mastertime + subtime) >= float(timelimit):
            break
        t += 1

    total_elapsed = float(time.perf_counter() - t0_total)
    print("##########################################")
    print("retrain---iterations----", retrain_iteration)
    print("phase3_total_elapsed----", total_elapsed)
    print("##########################################")
    return (
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
    )


def train_svm(train_fileposition, fileposition, key, t, K, N):
    print("##########################################")
    print("Training----Iteration------", t)
    print("key-update", key)
    print("##########################################")
    clf, trainingtime = _train_classifier_from_records(_TRAINING_CACHE, key=float(key))
    print("!! -- Phase 2 training time", trainingtime)
    return clf, float(trainingtime)