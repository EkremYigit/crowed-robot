# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for the ICRA 2019 paper "Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning." Trains and evaluates value-network-based RL navigation policies (CADRL, LSTM-RL, SARL, OM-SARL) in a 2D OpenAI Gym environment where a robot navigates through ORCA-driven humans. A second PPO pipeline using Stable-Baselines3 is also included.

## Setup

1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) — **not on PyPI**, must be built from source. `crowd_sim` imports `rvo2` at module load; the package will not import without it.
2. `pip install -e .` — installs `crowd_nav` + `crowd_sim` and their dependencies (gym, torch, numpy, matplotlib, gitpython).
3. PPO pipeline only: `pip install stable-baselines3 gymnasium imageio imageio-ffmpeg`

## Common Commands

**All commands must be run from inside `crowd_nav/`** — configs and output paths are relative to that cwd.

```bash
# Train (policy-factory keys: cadrl | lstm_rl | sarl). SARL with with_om=true becomes OM-SARL.
python train.py --policy sarl
python train.py --policy sarl --gpu --output_dir data/my_run
python train.py --policy sarl --resume           # continues RL from data/output/rl_model.pth
python train.py --policy sarl --debug            # DEBUG-level logs

# Evaluate trainable policies against 500 test cases
python test.py --policy sarl --model_dir data/output --phase test
python test.py --policy orca --phase test        # ORCA baseline, no model_dir

# Single-episode visualization (matplotlib; --video_file saves mp4 via ffmpeg)
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --traj   # trajectory plot

# Plot training curves from the log file
python utils/plot.py data/output/output.log --plot_sr --plot_cr --plot_time --plot_reward
```

## Architecture

### Two Packages, One Pipeline

- **`crowd_sim/`** — the environment. Registers `CrowdSim-v0` with gym. Owns the simulation loop, reward function, human generation rules, and all shared data primitives (`Agent`, `Human`, `Robot`, state classes, `ActionXY`/`ActionRot`, non-trainable policies `Linear` and `ORCA`).
- **`crowd_nav/`** — the learning side. Contains trainable policies, training/evaluation loop, replay memory, and log plotting. Imports from `crowd_sim` but not vice-versa.

### Policy Class Hierarchy (`crowd_nav/policy/`)

```
Policy (crowd_sim/envs/policy/policy.py)           # abstract base
├── Linear, ORCA                                    # non-trainable, used for humans + IL demonstrator
└── CADRL (cadrl.py)                                # value net + action-space sampling + rotate()
    └── MultiHumanRL (multi_human_rl.py)            # overrides predict() for joint multi-human state
        ├── LstmRL (lstm_rl.py)                     # sorts humans by distance, encodes via LSTM
        └── SARL (sarl.py)                          # self-attention over humans; OM-SARL when with_om=True
```

`CADRL.rotate()` transforms world-frame joint state into robot-centric state and is reused by subclasses. Every trainable policy produces a `ValueNetwork` assigned to `self.model`; `predict()` enumerates actions, propagates one step, queries the network, and picks argmax (epsilon-greedy during training).

### Policy Factory Extension Pattern

`crowd_nav/policy/policy_factory.py` **mutates** the dict defined in `crowd_sim/envs/policy/policy_factory.py`, adding `cadrl`, `lstm_rl`, `sarl` keys on top of `linear`, `orca`, `none`. This means `crowd_nav` must be imported before policies like `sarl` are resolvable by the factory — train.py/test.py import it transitively. New trainable policies are registered here.

### Training Flow (`crowd_nav/train.py`)

Two-stage pipeline, controlled entirely by `configs/train.config`:

1. **Imitation learning**: if `il_model.pth` exists, load it. Otherwise run `il_episodes` episodes with robot controlled by `il_policy` (ORCA), collect trajectories into `ReplayMemory`, train the value network for `il_epochs` with MSE loss against discounted cumulative reward. Save `il_model.pth`.
2. **Reinforcement learning**: epsilon-decay over `epsilon_decay` episodes, sample episodes into memory, optimize via `train_batches` minibatches per episode. Evaluate on val split every `evaluation_interval`, update target model every `target_update_interval`, checkpoint `rl_model.pth` every `checkpoint_interval`.

`Explorer` (`crowd_nav/utils/explorer.py`) is the single entry point for running episodes — it calls `env.step()`, records states/actions/rewards, and either writes to memory (with IL or RL value targets) or just logs metrics (success / collision / timeout rate, nav time).

### Config Snapshot Convention

`train.py` **copies** `env.config`, `policy.config`, `train.config` into `args.output_dir` at run start. `test.py --model_dir <dir>` then reads configs from that directory, **not** from the repo. This pins each trained model to the exact config it was trained with — editing repo configs does not affect evaluation of existing models. When resuming or re-evaluating, always point at the `output_dir`, not the repo configs.

### State / Action Types (`crowd_sim/envs/utils/`)

- `ObservableState` — what one agent sees of another: `(px, py, vx, vy, radius)`. 5-dim.
- `FullState` — an agent's own complete state: adds `gx, gy, v_pref, theta`. 9-dim.
- `JointState` — `(self_state: FullState, human_states: List[ObservableState])`. Input to policies.
- `ActionXY(vx, vy)` for `kinematics='holonomic'`; `ActionRot(v, r)` for `'unicycle'`. The policy's `kinematics` attribute dictates which the agent produces; `Agent.check_validity()` asserts the match.

`FullState.__add__` and `ObservableState.__add__` are overridden to let `self_state + human_state` produce a concatenated tuple — this is how joint-state tensors are built throughout the policy code. Don't replace these with regular list concat; the rest of the codebase relies on the operator.

### Environment Step (`crowd_sim/envs/crowd_sim.py`)

`CrowdSim.step()` computes human actions via each human's ORCA policy, checks robot-human and human-human collisions, checks goal reach, computes reward, and returns `(ob, reward, done, info)`. `info` is one of `Timeout / ReachGoal / Collision / Danger(min_dist) / Nothing` (from `envs/utils/info.py`) — Explorer discriminates on type via `isinstance()`, so new terminal categories must be added there too.

`onestep_lookahead(action)` = `step(action, update=False)`: used by `CADRL.predict()` / `MultiHumanRL.predict()` when `query_env=true` to peek at the next state without committing. If `query_env=false`, policies compute the reward themselves via `compute_reward()`.

Reward constants live in `env.config [reward]` (`success_reward`, `collision_penalty`, `discomfort_dist`, `discomfort_penalty_factor`). The same function is duplicated in `MultiHumanRL.compute_reward()` — keep the two in sync when tuning.

### Output Directory Layout

Everything below `data/` is git-ignored. A training run produces:

```
data/output/
├── env.config, policy.config, train.config   # snapshot of configs used
├── output.log                                 # plot.py parses this
├── il_model.pth                               # post-imitation-learning weights
├── rl_model.pth                               # latest RL checkpoint
└── resumed_rl_model.pth                       # exists only if --resume was used
```

`test.py` prefers `resumed_rl_model.pth` over `rl_model.pth` when both exist.

## PPO Pipeline (Stable-Baselines3)

A second training stack using Stable-Baselines3 PPO instead of the custom `Trainer`.

`ppo_env.py` uses `gymnasium` (the maintained fork), not OpenAI `gym` — the two packages are not interchangeable. The original `CrowdSim` still registers with `gym`; mixing them in a single process requires care.

### PPO Commands

```bash
# Train (single env)
python train_ppo.py

# Train with 8 parallel envs (SubprocVecEnv) — recommended
python train_ppo.py --n_envs 8 --output_dir data/ppo_parallel

# Evaluate PPO model (500 episodes, social metrics)
python test_ppo.py --model_path data/ppo_parallel/best_model

# Evaluate ORCA baseline with social metrics
python test_ppo.py --policy orca

# Visualize a single episode
python test_ppo.py --model_path data/ppo_parallel/best_model --visualize --test_case 0
```

### PPO Architecture (`ppo_env.py`, `train_ppo.py`, `test_ppo.py`)

`CrowdNavPPOEnv` wraps `CrowdSim` as a `gymnasium.Env`:

- **Observation** (49-dim float32): `[robot(8)] + [5 humans × 5](25) + [5 humans × 3 relative](15) + [goal_dist](1)`. Missing humans are zero-padded.
- **Action** (2-dim, [-1, 1]): scaled by `robot.v_pref` → `(vx, vy)`, magnitude clipped to `v_pref`.
- **Reward**: `+1.0` (ReachGoal), `-1.0` (Collision), else `0.4*(prev_dist − new_dist) − social_penalty`. Social penalty: `-0.2 * (0.6 − clearance) / 0.6` per human with clearance < 0.6 m. This differs from the SARL reward in `env.config` — keep this in mind when comparing metrics across pipelines.

`test_ppo.py` reports two social metrics not in the original `test.py`: *avg zone time/episode* (seconds inside the 0.6 m personal zone) and *avg violations/episode* (number of entries into the zone).

`train_ppo.py` accepts a `--reward` flag but only the default `circular` variant (`ppo_env.py`) is included in this repo.

## Conventions

- Lint: `pylint` with `.pylintrc` (max line length 120; single-letter names like `dx, dy, px, py, vx, vy, th` are whitelisted). No test suite in the repo despite `extras_require['test']`.
- Python 2-era idioms present in places (`object` base class, `super().__init__()` with no args). Don't "modernize" unless there's a reason.
- Git: `data/`, `*.mp4`, `videos/`, `report/` are ignored — training outputs and generated media stay local.
