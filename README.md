# EV Benders

Two-stage stochastic EV charging location with multi-cut Benders decomposition.
The learned policy, Vanilla multi-cut, LearnBD, violation Top-K, and uniform Random-K share one entry point.

## Requirements

Python 3.10 or newer, with `numpy`, `pandas`, `openpyxl`, `torch`, `matplotlib`, `scikit-learn`, and a licensed `gurobipy`. Training writes reward plots, so `matplotlib` is required for `full`.

## Run

Edit the switches at the top of `run_ev.sh`, then:

```bash
bash run_ev.sh
```

`METHOD` selects the experiment:

| Value | What runs |
| --- | --- |
| `full` | Data, RL training, Vanilla, policy evaluation, LearnBD |
| `eval` | Data, then policy evaluation only |
| `eval_topk` | Data, policy evaluation, then violation Top-K |
| `topk` | Data, then violation Top-K only |
| `randomk` | Data, then uniform Random-K only |

`full` trains with episodic REINFORCE on one complete Benders trajectory per episode. The launcher defaults are the paper training run: 8 stations, 12 customers, 100 normal scenarios, 600 episodes, at most 2000 iterations per episode, and 10 cuts sampled each iteration. The learning rate is `0.001`. The reward is `log_gap_rmp_time_penalty` with `alpha=0.01`, `beta=0.001`, `Tref=0.1`, `lambda=0.001`, and `eps=1e-8`. Training stops an episode at relative gap `0.01`. Other Table 1 cells change `NUM_STATIONS`, `NUM_CUSTOMERS`, and `EVAL_DEMAND_DISTRIBUTION`. Training demand stays `normal`.

`full` evaluates five test sets, seeds `110` through `114`. Vanilla, LearnBD, and policy evaluation stop at relative gap `0.01` or after 3600 seconds. Policy evaluation keeps the 10 highest-scoring cuts. Checkpoints are written every 10 episodes and all of them are evaluated.

`eval` and `eval_topk` read `POLICY_PATH`. The shipped checkpoint is `policy_ep600.pt` in the repository root. The launcher copies it to `experiment_runs/<RUN_NAME>/train/policy_final.pt` before evaluation.

Top-K keeps the 10 cuts with the largest violation. Random-K draws 10 cuts uniformly without replacement. Both stop at relative gap `0.01`. Those two values are fixed in `baselines/`. Training and policy evaluation set a Gurobi seed. Top-K and Random-K leave the Gurobi seed at its default.

Numeric libraries are pinned to one thread in `run_ev.sh`. Gurobi models inside the solvers also use one thread.

## Seeds

`seed_control.py` derives role seeds from the master seed `S`:

| Role | Formula |
| --- | --- |
| Structure | `S + 11` |
| Train demand scenarios | `S + 13` |
| Demand means | `S + 17` |
| Vanilla | `S + 23` |
| RL training | `S + 37`, overridden by `TRAIN_SEED` |
| Policy evaluation, Top-K, Random-K | `S + 53` |
| LearnBD | `S + 71` |

Eval scenario ids start at `EVAL_SCENARIO_SEED_START` and increase by one. The reference setting is master seed `9` and scenario start `110`. `run_ev.sh` sets `TRAIN_SEED=19000`, so RL training does not use `9 + 37`.

## Layout

LearnBD remains `Jia&Shen_SVM.py`. Limited-cut baselines live in `baselines/`.
