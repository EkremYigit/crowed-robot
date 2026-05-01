"""
Fine-tune an existing PPO model with a new reward environment.

Usage (from crowd_nav/):
    python finetune_ppo.py \\
        --pretrain_from data/ppo_gaussian/best_model \\
        --reward v4a \\
        --output_dir data/ppo_v4a \\
        --timesteps 500000 \\
        --n_envs 8
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def get_env_class(reward):
    if reward == 'v4c':
        from ppo_env_v4c import CrowdNavPPOEnvV4c
        return CrowdNavPPOEnvV4c
    if reward == 'v4b':
        from ppo_env_v4b import CrowdNavPPOEnvV4b
        return CrowdNavPPOEnvV4b
    if reward == 'v4a':
        from ppo_env_v4a import CrowdNavPPOEnvV4a
        return CrowdNavPPOEnvV4a
    if reward == 'v3a':
        from ppo_env_v3a import CrowdNavPPOEnvV3a
        return CrowdNavPPOEnvV3a
    if reward == 'v3b':
        from ppo_env_v3b import CrowdNavPPOEnvV3b
        return CrowdNavPPOEnvV3b
    if reward == 'v3c':
        from ppo_env_v3c import CrowdNavPPOEnvV3c
        return CrowdNavPPOEnvV3c
    if reward == 'v3d':
        from ppo_env_v3d import CrowdNavPPOEnvV3d
        return CrowdNavPPOEnvV3d
    if reward == 'gaussian':
        from ppo_env_gaussian import CrowdNavPPOEnvGaussian
        return CrowdNavPPOEnvGaussian
    raise ValueError(f'Unknown reward: {reward}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_from', type=str, required=True,
                        help='Path to existing model to fine-tune from (no .zip)')
    parser.add_argument('--reward',        type=str, required=True,
                        choices=['gaussian', 'v3a', 'v3b', 'v3c', 'v3d', 'v4a', 'v4b', 'v4c'])
    parser.add_argument('--output_dir',    type=str, default='data/ppo_finetune')
    parser.add_argument('--timesteps',     type=int, default=500_000)
    parser.add_argument('--n_envs',        type=int, default=8)
    parser.add_argument('--lr',            type=float, default=1e-4,
                        help='Lower LR than from-scratch to avoid catastrophic forgetting')
    parser.add_argument('--n_steps',       type=int, default=2048)
    parser.add_argument('--batch_size',    type=int, default=256)
    args = parser.parse_args()

    EnvClass = get_env_class(args.reward)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.n_envs > 1:
        train_env = make_vec_env(
            EnvClass,
            n_envs=args.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={'phase': 'train'},
            monitor_dir=os.path.join(args.output_dir, 'monitor'),
        )
    else:
        train_env = Monitor(EnvClass(phase='train'))

    eval_env = Monitor(EnvClass(phase='val'))

    print(f'Loading pretrained model: {args.pretrain_from}')
    model = PPO.load(
        args.pretrain_from,
        env=train_env,
        device='cpu',
        # Override hyperparams for fine-tuning
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )

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
        name_prefix='ppo_ft',
        verbose=0,
    )

    print(f'Reward       : {args.reward}')
    print(f'Pretrain from: {args.pretrain_from}')
    print(f'Output dir   : {args.output_dir}')
    print(f'Timesteps    : {args.timesteps}')
    print(f'LR           : {args.lr}  (reduced for fine-tuning)\n')

    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=False,
        reset_num_timesteps=False,
    )

    final_path = os.path.join(args.output_dir, 'ppo_ft_final')
    model.save(final_path)
    print(f'\nModel saved: {final_path}.zip')


if __name__ == '__main__':
    main()
