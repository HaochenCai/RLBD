from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    num_stations: int = 10
    num_customers: int = 20
    num_scenarios: int = 30
    seed: int = 0
    fixed_cost_min: int = 80
    fixed_cost_max: int = 300
    charge_cost_min: int = 5
    charge_cost_max: int = 40
    max_chargers_min: int = 10
    max_chargers_max: int = 80
    capacity_min: int = 40
    capacity_max: int = 120
    penalty_min: int = 30
    penalty_max: int = 120
    revenue_min: int = 5
    revenue_max: int = 60
    distance_min: int = 1
    distance_max: int = 80
    demand_min: int = 5
    demand_max: int = 120
    demand_mu_min: int = 20
    demand_mu_max: int = 100
    demand_sigma_ratio: float = 0.1
    truncate_negative_demand: bool = True
    demand_distribution: str = "normal"
    demand_skew_alpha: float = 5.0


_SUPPORTED_DEMAND_DISTRIBUTIONS = ("normal", "left_skewed", "right_skewed")


def _randint_matrix(rng: np.random.Generator, low: int, high: int, shape: tuple[int, ...]) -> np.ndarray:
    if low > high:
        raise ValueError(f"invalid integer range: low={low}, high={high}")
    return rng.integers(low, high + 1, size=shape, endpoint=False)


def build_static_dataframes(
    cfg: DatasetConfig, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    station_ids = np.arange(1, cfg.num_stations + 1, dtype=int)
    customer_ids = np.arange(1, cfg.num_customers + 1, dtype=int)

    station_df = pd.DataFrame(
        {
            "station_id": station_ids,
            "fixed_cost_f": _randint_matrix(rng, cfg.fixed_cost_min, cfg.fixed_cost_max, (cfg.num_stations,)),
            "charge_cost_b": _randint_matrix(rng, cfg.charge_cost_min, cfg.charge_cost_max, (cfg.num_stations,)),
            "max_chargers_M": _randint_matrix(rng, cfg.max_chargers_min, cfg.max_chargers_max, (cfg.num_stations,)),
            "capacity_per_charger_C": _randint_matrix(rng, cfg.capacity_min, cfg.capacity_max, (cfg.num_stations,)),
        }
    )

    customer_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "shortage_penalty_p": _randint_matrix(rng, cfg.penalty_min, cfg.penalty_max, (cfg.num_customers,)),
            "revenue_r": _randint_matrix(rng, cfg.revenue_min, cfg.revenue_max, (cfg.num_customers,)),
        }
    )

    distance_mat = _randint_matrix(
        rng,
        cfg.distance_min,
        cfg.distance_max,
        (cfg.num_stations, cfg.num_customers),
    )
    distance_df = pd.DataFrame(distance_mat, columns=[str(c) for c in customer_ids])
    return station_df, customer_df, distance_df, station_ids, customer_ids


def sample_demand_mu(
    cfg: DatasetConfig, rng: np.random.Generator, num_customers: Optional[int] = None
) -> np.ndarray:
    n_customers = int(cfg.num_customers if num_customers is None else num_customers)
    return _randint_matrix(rng, cfg.demand_mu_min, cfg.demand_mu_max, (n_customers,))


def _standardized_skew_normal_noise(
    rng: np.random.Generator,
    alpha: float,
    shape: tuple[int, ...],
) -> np.ndarray:
    a = float(alpha)
    if a == 0.0:
        return rng.normal(0.0, 1.0, size=shape)
    delta = a / np.sqrt(1.0 + a * a)
    u = rng.normal(0.0, 1.0, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    z = delta * np.abs(u) + np.sqrt(1.0 - delta * delta) * v
    mean = delta * np.sqrt(2.0 / np.pi)
    var = 1.0 - 2.0 * delta * delta / np.pi
    return (z - mean) / np.sqrt(var)


def _sample_demand_noise(
    cfg: DatasetConfig,
    rng: np.random.Generator,
    shape: tuple[int, ...],
) -> np.ndarray:
    d = str(cfg.demand_distribution)
    if d == "normal":
        return rng.normal(0.0, 1.0, size=shape)
    if d == "right_skewed":
        return _standardized_skew_normal_noise(rng, abs(float(cfg.demand_skew_alpha)), shape)
    if d == "left_skewed":
        return _standardized_skew_normal_noise(rng, -abs(float(cfg.demand_skew_alpha)), shape)
    raise ValueError(
        f"unsupported demand_distribution {d!r}; "
        f"expected one of {list(_SUPPORTED_DEMAND_DISTRIBUTIONS)}"
    )


def build_demand_dataframe(
    cfg: DatasetConfig,
    rng: np.random.Generator,
    customer_ids: np.ndarray,
    demand_mu: np.ndarray,
) -> pd.DataFrame:
    scenario_ids = np.arange(1, cfg.num_scenarios + 1, dtype=int)
    demand_mu = np.asarray(demand_mu, dtype=float).reshape(-1)
    if demand_mu.size != int(cfg.num_customers):
        raise ValueError(
            f"demand_mu length mismatch: {demand_mu.size} != num_customers={cfg.num_customers}"
        )
    #   Var_j = scale_sigma * mu_j
    # so std is sqrt(Var_j).
    demand_var = cfg.demand_sigma_ratio * demand_mu
    demand_var = np.clip(demand_var, 0.0, None)
    demand_std = np.sqrt(demand_var)
    demand_noise = _sample_demand_noise(
        cfg,
        rng,
        shape=(cfg.num_scenarios, cfg.num_customers),
    )
    demand_mat = np.rint(demand_mu + demand_noise * demand_std).astype(int)
    if cfg.truncate_negative_demand:
        demand_mat = np.clip(demand_mat, 0, None)
    demand_df = pd.DataFrame(demand_mat, columns=[str(c) for c in customer_ids])
    demand_df.insert(0, "scenario_id", scenario_ids)
    return demand_df


def write_dataset_files(
    base_dir: str,
    cfg: DatasetConfig,
    station_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)

    seed_suffix = str(int(cfg.seed))
    station_path = os.path.join(base_dir, f"station_side_{seed_suffix}.xlsx")
    customer_path = os.path.join(base_dir, f"customer_side_{seed_suffix}.xlsx")
    distance_path = os.path.join(base_dir, f"distance_{seed_suffix}.xlsx")
    demand_path = os.path.join(base_dir, f"demand_scenario_{seed_suffix}.xlsx")

    station_df.to_excel(station_path, index=False)
    customer_df.to_excel(customer_path, index=False)
    distance_df.to_excel(distance_path, index=False)
    demand_df.to_excel(demand_path, index=False)

    config_path = os.path.join(base_dir, "dataset_config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        cfg_dict = asdict(cfg)
        if extra_config:
            cfg_dict.update(extra_config)
        json.dump(cfg_dict, fh, ensure_ascii=False, indent=2)

    return {
        "station_path": station_path,
        "customer_path": customer_path,
        "distance_path": distance_path,
        "demand_path": demand_path,
        "config_path": config_path,
    }


def generate_dataset(base_dir: str, cfg: DatasetConfig) -> Dict[str, str]:
    rng = np.random.default_rng(int(cfg.seed))
    station_df, customer_df, distance_df, _station_ids, customer_ids = build_static_dataframes(cfg, rng)
    demand_mu = sample_demand_mu(cfg, rng, cfg.num_customers)
    demand_df = build_demand_dataframe(cfg, rng, customer_ids, demand_mu)
    return write_dataset_files(
        base_dir=base_dir,
        cfg=cfg,
        station_df=station_df,
        customer_df=customer_df,
        distance_df=distance_df,
        demand_df=demand_df,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic Excel dataset for RL+Benders pipeline")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
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
        "--demand_distribution",
        type=str,
        default="normal",
        choices=["normal", "left_skewed", "right_skewed"],
    )
    p.add_argument("--demand_skew_alpha", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = DatasetConfig(
        num_stations=args.num_stations,
        num_customers=args.num_customers,
        num_scenarios=args.num_scenarios,
        seed=args.seed,
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
        demand_distribution=args.demand_distribution,
        demand_skew_alpha=args.demand_skew_alpha,
    )
    paths = generate_dataset(args.out_dir, cfg)
    print("Dataset generated.")
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
