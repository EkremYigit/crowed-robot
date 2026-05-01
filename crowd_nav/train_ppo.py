"""
Train a PPO navigation policy for CrowdNav using Stable-Baselines3.

Usage (run from crowd_nav/):
    python train_ppo.py                          # single env
    python train_ppo.py --n_envs 8               # parallel (SubprocVecEnv)
    python train_ppo.py --n_envs 8 --output_dir data/ppo_parallel
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ppo_env import CrowdNavPPOEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps',   type=int,   default=1_000_000)
    parser.add_argument('--output_dir',  type=str,   default='data/ppo_output')
    parser.add_argument('--n_envs',      type=int,   default=1,
                        help='Number of parallel environments (>1 uses SubprocVecEnv)')
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--n_steps',     type=int,   default=2048)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--gamma',       type=float, default=0.99)
    parser.add_argument('--gae_lambda',  type=float, default=0.95)
    parser.add_argument('--clip_range',  type=float, default=0.2)
    parser.add_argument('--reward',      type=str,   default='circular',
                        choices=['circular', 'gaussian', 'v2', 'v3a', 'v3b', 'v3c', 'v3d'],
                        help='Reward function variant')
    args = parser.parse_args()

    if args.reward == 'gaussian':
        from ppo_env_gaussian import CrowdNavPPOEnvGaussian
        EnvClass = CrowdNavPPOEnvGaussian
    elif args.reward == 'v2':
        from ppo_env_v2 import CrowdNavPPOEnvV2
        EnvClass = CrowdNavPPOEnvV2
    elif args.reward == 'v3a':
        from ppo_env_v3a import CrowdNavPPOEnvV3a
        EnvClass = CrowdNavPPOEnvV3a
    elif args.reward == 'v3b':
        from ppo_env_v3b import CrowdNavPPOEnvV3b
        EnvClass = CrowdNavPPOEnvV3b
    elif args.reward == 'v3c':
        from ppo_env_v3c import CrowdNavPPOEnvV3c
        EnvClass = CrowdNavPPOEnvV3c
    elif args.reward == 'v3d':
        from ppo_env_v3d import CrowdNavPPOEnvV3d
        EnvClass = CrowdNavPPOEnvV3d
    else:
        EnvClass = CrowdNavPPOEnv

    os.makedirs(args.output_dir, exist_ok=True)

    # --- training env ---
    if args.n_envs > 1:
        print(f'Using SubprocVecEnv with {args.n_envs} parallel environments')
        train_env = make_vec_env(
            EnvClass,
            n_envs=args.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={'phase': 'train'},
            monitor_dir=os.path.join(args.output_dir, 'monitor'),
        )
    else:
        print('Using single environment (DummyVecEnv)')
        train_env = Monitor(EnvClass(phase='train'))

    # --- eval env (always single, deterministic) ---
    eval_env = Monitor(EnvClass(phase='val'))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.output_dir,
        log_path=args.output_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=100,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(args.output_dir, 'checkpoints'),
        name_prefix='ppo',
        verbose=0,
    )

    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        device='cpu',
        tensorboard_log=os.path.join(args.output_dir, 'tb_logs'),
    )

    print(f'Device      : {model.device}')
    print(f'Parallel envs: {args.n_envs}')
    print(f'Output dir  : {args.output_dir}\n')

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=False,
    )

    final_path = os.path.join(args.output_dir, 'ppo_final')
    model.save(final_path)
    print(f'\nModel saved: {final_path}.zip')


if __name__ == '__main__':
    main()
