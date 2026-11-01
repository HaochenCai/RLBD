"""
Central seed control for RLBD EV experiments.

Master ``--seed S`` derives non-overlapping role seeds:

  dataset_structure_seed          = S + 11
  dataset_train_scenario_seed     = S + 13
  dataset_eval_scenario_seed_start= S + 101   (unless overridden)
  demand_mu_seed                  = S + 17
  vanilla_seed                    = S + 23
  train_seed                      = S + 37
  eval_seed                       = S + 53
  svm_eval_seed                   = S + 71

Per eval instance index ``i`` (i=0 → scenario 110 when S=9 and eval_start=110):

  eval_scenario_seed[i] = eval_seed_start + i
  vanilla Gurobi seed           = vanilla_seed + i
  greedy / Top-K / Random-K seed= eval_seed + i
  LearnBD / SVM seed            = svm_eval_seed + i

The reference setting uses ``S=9``, ``eval_scenario_seed_start=110``,
``num_eval_datasets=2`` → scenarios {110,111} with solver seeds
vanilla {32,33}, eval {62,63}, svm {80,81}.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def derive_seeds(seed: int) -> Dict[str, int]:
    """Derive role seeds from master seed S."""
    base = int(seed)
    return {
        "dataset_structure_seed": base + 11,
        "dataset_train_scenario_seed": base + 13,
        "dataset_eval_scenario_seed_start": base + 101,
        "demand_mu_seed": base + 17,
        "vanilla_seed": base + 23,
        "train_seed": base + 37,
        "eval_seed": base + 53,
        "svm_eval_seed": base + 71,
    }


def resolve_eval_seed_start(
    master_seed: int,
    *,
    eval_scenario_seed_start: Optional[int] = None,
    seeds: Optional[Mapping[str, int]] = None,
) -> int:
    """Eval scenario stream base; override wins, else derived S+101."""
    if eval_scenario_seed_start is not None:
        return int(eval_scenario_seed_start)
    derived = seeds if seeds is not None else derive_seeds(master_seed)
    return int(derived["dataset_eval_scenario_seed_start"])


def eval_scenario_seeds(
    master_seed: int,
    *,
    num_eval_datasets: int = 1,
    eval_scenario_seed_start: Optional[int] = None,
) -> list[int]:
    start = resolve_eval_seed_start(
        master_seed, eval_scenario_seed_start=eval_scenario_seed_start
    )
    n = max(1, int(num_eval_datasets))
    return [int(start) + i for i in range(n)]


def instance_seeds(
    master_seed: int,
    eval_index: int = 0,
    *,
    eval_scenario_seed_start: Optional[int] = None,
    train_seed_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    All seeds needed to reproduce one eval instance.

    Returns keys used by standalone CLIs:
      master_seed, eval_index, eval_scenario_seed,
      vanilla_seed, eval_seed (greedy/topk/random_k), svm_eval_seed,
      train_seed, plus the full derived ``seeds`` map.
    """
    seeds = derive_seeds(master_seed)
    i = int(eval_index)
    if i < 0:
        raise ValueError(f"eval_index must be >= 0, got {eval_index}")
    sce_start = resolve_eval_seed_start(
        master_seed, eval_scenario_seed_start=eval_scenario_seed_start, seeds=seeds
    )
    train_seed = (
        int(train_seed_override)
        if train_seed_override is not None
        else int(seeds["train_seed"])
    )
    return {
        "master_seed": int(master_seed),
        "eval_index": i,
        "eval_scenario_seed": int(sce_start) + i,
        "vanilla_seed": int(seeds["vanilla_seed"]) + i,
        "eval_seed": int(seeds["eval_seed"]) + i,
        "svm_eval_seed": int(seeds["svm_eval_seed"]) + i,
        "train_seed": train_seed,
        "train_seed_derived_default": int(seeds["train_seed"]),
        "seeds": seeds,
        "eval_scenario_seed_start": int(sce_start),
    }


# Reference experiment: master seed 9, first eval scenario seed 110.
REFERENCE_MASTER_SEED = 9
REFERENCE_EVAL_SCENARIO_SEED_START = 110


def reference_instance(eval_index: int = 0) -> Dict[str, Any]:
    """Seeds for the reference experiment at eval index i."""
    return instance_seeds(
        REFERENCE_MASTER_SEED,
        eval_index,
        eval_scenario_seed_start=REFERENCE_EVAL_SCENARIO_SEED_START,
    )


__all__ = [
    "derive_seeds",
    "resolve_eval_seed_start",
    "eval_scenario_seeds",
    "instance_seeds",
    "reference_instance",
    "REFERENCE_MASTER_SEED",
    "REFERENCE_EVAL_SCENARIO_SEED_START",
]


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Print derived seeds.")
    p.add_argument("--master_seed", type=int, default=REFERENCE_MASTER_SEED)
    p.add_argument("--eval_index", type=int, default=0)
    p.add_argument(
        "--eval_scenario_seed_start",
        type=int,
        default=REFERENCE_EVAL_SCENARIO_SEED_START,
        help="Override the first eval scenario seed. The reference setting uses 110.",
    )
    p.add_argument("--num_eval_datasets", type=int, default=2)
    args = p.parse_args()
    payload = {
        "derived": derive_seeds(args.master_seed),
        "eval_scenario_seeds": eval_scenario_seeds(
            args.master_seed,
            num_eval_datasets=args.num_eval_datasets,
            eval_scenario_seed_start=args.eval_scenario_seed_start,
        ),
        "instance": instance_seeds(
            args.master_seed,
            args.eval_index,
            eval_scenario_seed_start=args.eval_scenario_seed_start,
        ),
    }
    print(json.dumps(payload, indent=2))
