#!/usr/bin/env bash
# Launch one EV experiment.
#
# Python on PATH must provide gurobipy, numpy, pandas, openpyxl, torch,
# and scikit-learn. Scheduler account, modules, and partitions stay outside
# this file. Example:
#   sbatch --time=48:00:00 --cpus-per-task=16 --mem=64G run_ev.sh
#
set -euo pipefail

cd "$(dirname "$0")"

# ---------- experiment switches ----------
# full | eval | eval_topk | topk | randomk
METHOD=full
RUN_NAME=ev_run

# Data seeds follow master seed 9 (structure 20, train scenarios 22).
# TRAIN_SEED is the RL seed used for the paper run, not S+37.
MASTER_SEED=9
TRAIN_SEED=19000

# Paper training instance. Other Table 1 cells change the size
# and EVAL_DEMAND_DISTRIBUTION; training demand stays normal.
NUM_STATIONS=8
NUM_CUSTOMERS=12
NUM_SCENARIOS=100

# normal | left_skewed | right_skewed
TRAIN_DEMAND_DISTRIBUTION=normal
EVAL_DEMAND_DISTRIBUTION=normal
DEMAND_SKEW_ALPHA=5.0

EVAL_SCENARIO_SEED_START=110
NUM_EVAL_DATASETS=5

# Used by eval and eval_topk. The shipped checkpoint is episode 600.
# Copied to experiment_runs/${RUN_NAME}/train/policy_final.pt
POLICY_PATH=./policy_ep600.pt

# Episodic REINFORCE. K=10 is the training cut budget.
# full uses these; eval, topk, and randomk skip training.
TRAIN_MAX_ITERS=2000
NUM_EPISODES=600
K_SELECT=10
LR=0.001
TEMPERATURE=1.0
CKPT_EVERY=10
REWARD_MA_WINDOW=10

# One-hour cap and 1% gap for Vanilla, LearnBD, and policy evaluation.
# Top-K and Random-K keep K=10 and gap 0.01 inside baselines/.
VANILLA_MAX_ITERS=999999999
VANILLA_MAX_ELAPSED_SEC=3600
VANILLA_TOL_REL_GAP=0.01
EVAL_MAX_ITERS=9999999999
EVAL_MAX_ELAPSED_SEC=3600
EVAL_K_SELECT=10
EVAL_TOL_GAP=0.01
SVM_ITELIMIT=9999999999
SVM_TIMELIMIT=3600
REWARD_NAME=log_gap_rmp_time_penalty
REWARD_ALPHA=0.01
REWARD_BETA=0.001
REWARD_TREF=0.1
REWARD_LAMBDA_PENALTY=0.001
REWARD_EPS=1e-8
TOL_GAP=0.01
# -----------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "${METHOD}" in
  full)
    MODE_ARGS=()
    ;;
  eval)
    MODE_ARGS=(--eval_only)
    ;;
  eval_topk)
    MODE_ARGS=(--eval_then_topk)
    ;;
  topk)
    MODE_ARGS=(--top_k_only)
    ;;
  randomk)
    MODE_ARGS=(--random_k_only)
    ;;
  *)
    echo "Unknown METHOD=${METHOD}. Use full, eval, eval_topk, topk, or randomk." >&2
    exit 1
    ;;
esac

if [[ "${METHOD}" == "eval" || "${METHOD}" == "eval_topk" ]]; then
  if [[ ! -f "${POLICY_PATH}" ]]; then
    echo "POLICY_PATH must point to an existing policy file for ${METHOD}." >&2
    exit 1
  fi
  mkdir -p "experiment_runs/${RUN_NAME}/train"
  cp "${POLICY_PATH}" "experiment_runs/${RUN_NAME}/train/policy_final.pt"
fi

python3 -u run_experiment.py \
  --seed "${MASTER_SEED}" \
  --out_dir "./experiment_runs" \
  --run_name "${RUN_NAME}" \
  --num_stations "${NUM_STATIONS}" \
  --num_customers "${NUM_CUSTOMERS}" \
  --num_scenarios "${NUM_SCENARIOS}" \
  --train_demand_distribution "${TRAIN_DEMAND_DISTRIBUTION}" \
  --eval_demand_distribution "${EVAL_DEMAND_DISTRIBUTION}" \
  --demand_skew_alpha "${DEMAND_SKEW_ALPHA}" \
  --eval_scenario_seed_start "${EVAL_SCENARIO_SEED_START}" \
  --num_eval_datasets "${NUM_EVAL_DATASETS}" \
  --train_seed "${TRAIN_SEED}" \
  --train_max_iters "${TRAIN_MAX_ITERS}" \
  --episodic \
  --num_episodes "${NUM_EPISODES}" \
  --k_select "${K_SELECT}" \
  --lr "${LR}" \
  --temperature "${TEMPERATURE}" \
  --ckpt_every "${CKPT_EVERY}" \
  --reward_ma_window "${REWARD_MA_WINDOW}" \
  --vanilla_max_iters "${VANILLA_MAX_ITERS}" \
  --vanilla_max_elapsed_sec "${VANILLA_MAX_ELAPSED_SEC}" \
  --vanilla_tol_rel_gap "${VANILLA_TOL_REL_GAP}" \
  --eval_max_iters "${EVAL_MAX_ITERS}" \
  --eval_max_elapsed_sec "${EVAL_MAX_ELAPSED_SEC}" \
  --eval_all_checkpoints \
  --eval_k_select "${EVAL_K_SELECT}" \
  --eval_tol_gap "${EVAL_TOL_GAP}" \
  --svm_itelimit "${SVM_ITELIMIT}" \
  --svm_timelimit "${SVM_TIMELIMIT}" \
  --reward_name "${REWARD_NAME}" \
  --reward_alpha "${REWARD_ALPHA}" \
  --reward_beta "${REWARD_BETA}" \
  --reward_tref "${REWARD_TREF}" \
  --reward_lambda_penalty "${REWARD_LAMBDA_PENALTY}" \
  --reward_eps "${REWARD_EPS}" \
  --tol_gap "${TOL_GAP}" \
  "${MODE_ARGS[@]}"
