import os
import configparser
import numpy as np
import gymnasium
from gymnasium import spaces

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import ReachGoal, Collision, Timeout


class _DummyPolicy:
    """Minimal policy stub so env.reset() can read multiagent_training."""
    multiagent_training = True
    kinematics = 'holonomic'
    time_step = 0.25

    def set_phase(self, phase): pass
    def set_env(self, env): pass


class CrowdNavPPOEnv(gymnasium.Env):
    """
    Gymnasium wrapper around CrowdSim for Stable-Baselines3 PPO.

    Observation (49-dim float32):
        [0:8]   robot  : px, py, vx, vy, gx, gy, v_pref, radius
        [8:33]  humans : (px, py, vx, vy, radius) × 5  — zeros if fewer humans
        [33:48] relative: (dx, dy, dist) × 5  — robot→human vector
        [48]    goal_dist: ||robot - goal||

    Action (2-dim float32 in [-1, 1]):
        Scaled by robot.v_pref → (vx, vy), magnitude clipped to v_pref.
    """

    metadata = {'render_modes': ['human']}

    # 8 robot + 5*5 humans + 5*3 relative + 1 goal_dist
    OBS_DIM = 49
    HUMAN_NUM = 5

    def __init__(self, phase='train', env_config_path='configs/env.config'):
        super().__init__()

        self.phase = phase

        # resolve path relative to this file so it works from any cwd
        if not os.path.isabs(env_config_path):
            env_config_path = os.path.join(os.path.dirname(__file__), env_config_path)

        env_config = configparser.RawConfigParser()
        env_config.read(env_config_path)

        self.env = CrowdSim()
        self.env.configure(env_config)

        robot = Robot(env_config, 'robot')
        robot.set_policy(_DummyPolicy())
        self.env.set_robot(robot)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self._last_ob = None

    @property
    def robot(self):
        return self.env.robot

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        ob = self.env.reset(self.phase)
        self._last_ob = ob
        return self._get_obs(ob), {}

    def step(self, action):
        v_pref = self.robot.v_pref
        vx, vy = float(action[0]) * v_pref, float(action[1]) * v_pref

        # clip to preferred speed
        speed = np.hypot(vx, vy)
        if speed > v_pref:
            vx, vy = vx / speed * v_pref, vy / speed * v_pref

        prev_dist = self._goal_dist()
        ob, _, _, info = self.env.step(ActionXY(vx, vy))
        self._last_ob = ob

        reward = self._compute_reward(info, prev_dist, ob)
        terminated = isinstance(info, (ReachGoal, Collision))
        truncated = isinstance(info, Timeout)

        return self._get_obs(ob), reward, terminated, truncated, {'info': str(info)}

    def render(self, mode='video', output_file=None):
        self.env.render(mode=mode, output_file=output_file)

    # ------------------------------------------------------------------
    def _get_obs(self, ob):
        rs = self.robot.get_full_state()

        obs = [rs.px, rs.py, rs.vx, rs.vy, rs.gx, rs.gy, rs.v_pref, rs.radius]

        for i in range(self.HUMAN_NUM):
            if i < len(ob):
                h = ob[i]
                obs += [h.px, h.py, h.vx, h.vy, h.radius]
            else:
                obs += [0.0, 0.0, 0.0, 0.0, 0.0]

        for i in range(self.HUMAN_NUM):
            if i < len(ob):
                h = ob[i]
                dx, dy = h.px - rs.px, h.py - rs.py
                obs += [dx, dy, np.hypot(dx, dy)]
            else:
                obs += [0.0, 0.0, 0.0]

        obs.append(self._goal_dist())

        return np.array(obs, dtype=np.float32)

    def _goal_dist(self):
        r = self.robot
        return float(np.hypot(r.px - r.gx, r.py - r.gy))

    def _compute_reward(self, info, prev_dist, ob):
        """
        reward = goal_reward + collision_reward + progress_reward + social_penalty

        Terminal:
            ReachGoal  → +1.0
            Collision  → -1.0
        Step-wise:
            progress   = 0.4 * (prev_dist - new_dist)
            social_pen = sum over humans of -0.2 * ((0.6 - clearance) / 0.6)
                         when clearance = center_dist - r_robot - r_human < 0.6
        """
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        new_dist = self._goal_dist()
        progress = 0.4 * (prev_dist - new_dist)

        social_pen = 0.0
        r = self.robot
        for h in ob:
            center_dist = np.hypot(r.px - h.px, r.py - h.py)
            clearance = center_dist - r.radius - h.radius
            if clearance < 0.6:
                social_pen += -0.2 * ((0.6 - clearance) / 0.6)

        return float(progress + social_pen)
