"""
Evaluate or visualize a trained PPO model. Also supports ORCA baseline evaluation.

Usage (run from crowd_nav/):
    python test_ppo.py --model_path data/ppo_parallel/best_model
    python test_ppo.py --policy orca
    python test_ppo.py --model_path data/ppo_parallel/best_model --visualize --test_case 0
"""

import argparse
import configparser
import os
import numpy as np

from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout


SOCIAL_ZONE = 0.6   # metre — kişisel alan sınırı
TIME_STEP   = 0.25  # saniye


def _social_check(crowdsim_env):
    """
    Her insan için boundary_clearance hesaplar.
    Dönüş: list[float] — her insanın clearance değeri
    """
    robot = crowdsim_env.robot
    clearances = []
    for h in crowdsim_env.humans:
        dist = np.hypot(robot.px - h.px, robot.py - h.py)
        clearances.append(dist - robot.radius - h.radius)
    return clearances


def _print_report(label, k, success, collision, timeout,
                  rewards, nav_times, total_zone_time, total_violations):
    print(f'\n{"="*55}')
    print(f'  {label}')
    print(f'{"="*55}')
    print(f'  Episodes              : {k}')
    print(f'  Success rate          : {success/k:.3f}  ({success}/{k})')
    print(f'  Collision rate        : {collision/k:.3f}  ({collision}/{k})')
    print(f'  Timeout rate          : {timeout/k:.3f}  ({timeout}/{k})')
    print(f'  Avg reward            : {np.mean(rewards):.4f}')
    if nav_times:
        print(f'  Avg completion time   : {np.mean(nav_times):.2f}s  (basarili episodeler)')
    print(f'  Avg zone time/episode : {total_zone_time/k:.2f}s  (kisisel alanda gecirilen sure)')
    print(f'  Avg violations/episode: {total_violations/k:.2f}  (kisisel alan ihlali sayisi)')
    print(f'{"="*55}')


# ---------------------------------------------------------------
# PPO evaluation
# ---------------------------------------------------------------
def evaluate_ppo(model, n_episodes):
    from stable_baselines3.common.monitor import Monitor
    from ppo_env import CrowdNavPPOEnv

    eval_env = Monitor(CrowdNavPPOEnv(phase='test'))
    ppo_env  = eval_env.env          # CrowdNavPPOEnv
    csim     = ppo_env.env           # CrowdSim

    success, collision, timeout = 0, 0, 0
    rewards, nav_times = [], []
    total_zone_time   = 0.0
    total_violations  = 0
    disc_steps_success = 0

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        prev_in_zone = [False] * len(csim.humans)
        ep_zone_steps = 0
        ep_violations = 0
        ep_disc_steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info_dict = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

            clearances = _social_check(csim)
            if any(cl < SOCIAL_ZONE for cl in clearances):
                ep_disc_steps += 1
            for i, cl in enumerate(clearances):
                in_zone = cl < SOCIAL_ZONE
                if in_zone:
                    ep_zone_steps += 1
                if in_zone and not prev_in_zone[i]:
                    ep_violations += 1
                prev_in_zone[i] = in_zone

        info_str = info_dict.get('info', '')
        if 'Reaching goal' in info_str:
            success += 1
            nav_times.append(csim.global_time)
            disc_steps_success += ep_disc_steps
        elif 'Collision' in info_str:
            collision += 1
        else:
            timeout += 1

        rewards.append(ep_reward)
        total_zone_time  += ep_zone_steps * TIME_STEP
        total_violations += ep_violations

    _print_report('PPO Parallel (8 env)', n_episodes,
                  success, collision, timeout,
                  rewards, nav_times, total_zone_time, total_violations)
    denom = success + collision
    t_limit = csim.time_limit
    t_weighted_nav = (sum(nav_times) + disc_steps_success * 0.5 * TIME_STEP + collision * t_limit) / denom \
        if denom > 0 else t_limit
    print(f'  Weighted nav time     : {t_weighted_nav:.2f}s  (succ={success}, coll={collision}, disc_steps={disc_steps_success})')


# ---------------------------------------------------------------
# ORCA evaluation
# ---------------------------------------------------------------
def evaluate_orca(n_episodes, env_config_path='configs/env.config'):
    from crowd_sim.envs.crowd_sim import CrowdSim
    from crowd_sim.envs.utils.robot import Robot
    from crowd_sim.envs.utils.state import JointState
    from crowd_sim.envs.policy.orca import ORCA

    if not os.path.isabs(env_config_path):
        env_config_path = os.path.join(os.path.dirname(__file__), env_config_path)

    env_config = configparser.RawConfigParser()
    env_config.read(env_config_path)

    env   = CrowdSim()
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    orca  = ORCA()
    orca.time_step = env_config.getfloat('env', 'time_step')
    orca.multiagent_training = True
    robot.set_policy(orca)
    env.set_robot(robot)

    success, collision, timeout = 0, 0, 0
    rewards, nav_times = [], []
    total_zone_time  = 0.0
    total_violations = 0
    disc_steps_success = 0

    for ep in range(n_episodes):
        ob   = env.reset('test')
        done = False
        ep_reward = 0.0
        prev_in_zone = [False] * len(env.humans)
        ep_zone_steps = 0
        ep_violations = 0
        ep_disc_steps = 0

        while not done:
            state  = JointState(robot.get_full_state(), ob)
            action = orca.predict(state)
            ob, reward, done, info = env.step(action)
            ep_reward += reward

            clearances = _social_check(env)
            if any(cl < SOCIAL_ZONE for cl in clearances):
                ep_disc_steps += 1
            for i, cl in enumerate(clearances):
                in_zone = cl < SOCIAL_ZONE
                if in_zone:
                    ep_zone_steps += 1
                if in_zone and not prev_in_zone[i]:
                    ep_violations += 1
                prev_in_zone[i] = in_zone

        if isinstance(info, ReachGoal):
            success += 1
            nav_times.append(env.global_time)
            disc_steps_success += ep_disc_steps
        elif isinstance(info, Collision):
            collision += 1
        else:
            timeout += 1

        rewards.append(ep_reward)
        total_zone_time  += ep_zone_steps * TIME_STEP
        total_violations += ep_violations

    _print_report('ORCA (baseline)', n_episodes,
                  success, collision, timeout,
                  rewards, nav_times, total_zone_time, total_violations)
    denom = success + collision
    t_limit = env.time_limit
    t_weighted_nav = (sum(nav_times) + disc_steps_success * 0.5 * TIME_STEP + collision * t_limit) / denom \
        if denom > 0 else t_limit
    print(f'  Weighted nav time     : {t_weighted_nav:.2f}s  (succ={success}, coll={collision}, disc_steps={disc_steps_success})')


# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
def visualize_episode(model, test_case, output_file=None):
    from ppo_env import CrowdNavPPOEnv

    env = CrowdNavPPOEnv(phase='test')
    obs, _ = env.reset()
    env.env.case_counter['test'] = test_case
    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.render(mode='video', output_file=output_file)


# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--policy',     type=str, default=None, choices=['orca'])
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--visualize',  action='store_true')
    parser.add_argument('--test_case',  type=int, default=0)
    parser.add_argument('--video_file', type=str, default=None)
    args = parser.parse_args()

    if args.policy == 'orca':
        evaluate_orca(args.n_episodes)
        return

    if args.model_path is None:
        raise ValueError('--model_path veya --policy orca gerekli')

    from stable_baselines3 import PPO
    model = PPO.load(args.model_path)
    print(f'Model: {args.model_path}')

    if args.visualize:
        visualize_episode(model, args.test_case, args.video_file)
    else:
        evaluate_ppo(model, args.n_episodes)


if __name__ == '__main__':
    main()
