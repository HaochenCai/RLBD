# EV Benders

Two-stage stochastic EV charging location with multi-cut Benders decomposition.
The learned policy, Vanilla multi-cut, LearnBD, violation Top-K, and uniform Random-K share one entry point.

## Requirements

Python 3.10 or newer, with `numpy`, `pandas`, `openpyxl`, `torch`, `scikit-learn`, and a licensed `gurobipy`.

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
| RL training | `S + 37` |
| Policy evaluation, Top-K, Random-K | `S + 53` |
| LearnBD | `S + 71` |

Eval scenario ids start at `EVAL_SCENARIO_SEED_START` and increase by one. The reference setting is master seed `9` and scenario start `110`.

## Layout

LearnBD remains `Jia&Shen_SVM.py`. Limited-cut baselines live in `baselines/`.
