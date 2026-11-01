"""
Read two-stage facility location + stochastic demand from Excel.

Full vanilla multicut loop removed; incremental Benders + top-k live in benders_solver_mdp / mdp_commander.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


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

    if df.shape == (I, J) and np.issubdtype(df.dtypes.iloc[0], np.number):
        return df.to_numpy(dtype=float)

    first_col = df.columns[0]
    if not np.issubdtype(df[first_col].dtype, np.number):
        df = df.set_index(first_col)

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

    if df.shape[1] == J and all(np.issubdtype(t, np.number) for t in df.dtypes):
        return df.to_numpy(dtype=float)

    first_col = df.columns[0]
    if not np.issubdtype(df[first_col].dtype, np.number):
        df = df.set_index(first_col)

    df.columns = df.columns.map(str)
    cid_str = list(map(str, customer_ids))

    missing = [c for c in cid_str if c not in df.columns]
    if missing:
        cust_cols = [f"cust{c}" for c in cid_str]
        if all(col in df.columns for col in cust_cols):
            df = df.loc[:, cust_cols]
            return df.to_numpy(dtype=float)
        raise ValueError(
            f"Demand alignment failed. Missing customers in scenario sheet: {missing[:8]} ..."
        )

    df = df.loc[:, cid_str]
    return df.to_numpy(dtype=float)


def _resolve_file_suffix(base_dir: str) -> Optional[str]:
    """
    If dataset_config.json exists with int "seed", return str(seed).
    Else parse seed_<digits> from basename. None -> legacy *_episode_test.xlsx names.
    """
    base_dir = os.path.normpath(base_dir)
    cfg_path = os.path.join(base_dir, "dataset_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        s = cfg.get("seed")
        if isinstance(s, int):
            return str(s)
    m = re.search(r"seed_(\d+)$", os.path.basename(base_dir), flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def read_inputs(base_dir: str):
    """
    Load four workbooks; return arrays and dimensions.

    Layouts:
      - New: base_dir like .../dataset/seed_110 with station_side_110.xlsx etc.
      - Legacy: station_side_episode_test.xlsx under base_dir when suffix unknown.
    """
    suffix = _resolve_file_suffix(base_dir)
    if suffix is not None:
        station_path = os.path.join(base_dir, f"station_side_{suffix}.xlsx")
        customer_path = os.path.join(base_dir, f"customer_side_{suffix}.xlsx")
        dist_path = os.path.join(base_dir, f"distance_{suffix}.xlsx")
        demand_path = os.path.join(base_dir, f"demand_scenario_{suffix}.xlsx")
    else:
        station_path = os.path.join(base_dir, "station_side_episode_test.xlsx")
        customer_path = os.path.join(base_dir, "customer_side_episode_test.xlsx")
        dist_path = os.path.join(base_dir, "distance_episode_test.xlsx")
        demand_path = os.path.join(base_dir, "demand_scenario_episode_test.xlsx")

    st = read_first_sheet(station_path)
    cu = read_first_sheet(customer_path)
    dist = read_first_sheet(dist_path)
    dem = read_first_sheet(demand_path)

    sid_col = _col(st, ["station_id", "station", "id"])
    if sid_col is None:
        raise ValueError("station table must contain a station id column like station_id.")

    station_ids = st[sid_col].tolist()
    I = len(station_ids)

    f_col = _col(st, ["fixed_cost_f", "f", "fixed_cost", "open_cost"])
    b_col = _col(st, ["charge_cost_b", "charger_cost_b", "b", "pole_cost", "charger_cost"])
    M_col = _col(st, ["max_chargers_M", "m", "M", "max_poles", "max_chargers"])
    C_col = _col(
        st,
        [
            "capacity_per_charger_C",
            "cap_per_charger_C",
            "c",
            "C",
            "cap",
            "capacity",
            "capacity_per_pole",
        ],
    )

    if f_col is None or b_col is None or M_col is None:
        raise ValueError("station table must contain columns for f, b, M (fixed cost, pole cost, max poles).")

    f = st[f_col].to_numpy(dtype=float)
    b = st[b_col].to_numpy(dtype=float)
    M = st[M_col].to_numpy(dtype=float)
    if C_col is None:
        C = np.full(I, 100.0, dtype=float)
    else:
        C = st[C_col].to_numpy(dtype=float)

    cid_col = _col(cu, ["customer_id", "customer", "id"])
    if cid_col is None:
        raise ValueError("customer table must contain a customer id column like customer_id.")

    customer_ids = cu[cid_col].tolist()
    J = len(customer_ids)

    p_col = _col(cu, ["shortage_penalty_p", "penalty_p", "p", "penalty"])
    r_col = _col(cu, ["revenue_r", "revenue", "r", "reward"])

    if p_col is None or r_col is None:
        raise ValueError("customer table must contain columns for p and r (penalty, revenue).")

    p = cu[p_col].to_numpy(dtype=float)
    r = cu[r_col].to_numpy(dtype=float)

    c = build_cost_matrix(dist, station_ids, customer_ids)
    d = build_demand_matrix(dem, customer_ids)
    N = d.shape[0]

    return station_ids, customer_ids, f, b, M, C, p, r, c, d, I, J, N


__all__ = [
    "read_first_sheet",
    "read_inputs",
    "build_cost_matrix",
    "build_demand_matrix",
]
