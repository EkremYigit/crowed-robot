"""
V3c: Tighter social zones (LAMBDA=0.12, SIGMA=0.6) + time penalty (-0.004/step).
Less avoidance area -> robot squeezes through crowds faster.
Trade-off: reduced social safety vs faster navigation.
"""

import numpy as np
from ppo_env import CrowdNavPPOEnv
from crowd_sim.envs.utils.info import ReachGoal, Collision

K_V           = 1.0
LAMBDA_SOCIAL = 0.12   # down from 0.2
SIGMA_BASE    = 0.6    # down from 0.8
PROGRESS_COEFF = 0.5
TIME_PENALTY  = -0.004


class CrowdNavPPOEnvV3c(CrowdNavPPOEnv):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        progress = PROGRESS_COEFF * (prev_dist - self._goal_dist())

        social_pen = 0.0
        r = self.robot
        for h in ob:
            dx = r.px - h.px
            dy = r.py - h.py
            d  = np.hypot(dx, dy)

            speed   = np.hypot(h.vx, h.vy)
            sigma_x = SIGMA_BASE * (1 + K_V * speed)
            sigma_y = SIGMA_BASE

            delta = np.arctan2(dy, dx) - np.arctan2(h.vy, h.vx)
            x_dir = d * np.cos(delta)
            y_dir = d * np.sin(delta)

            G = np.exp(-((x_dir ** 2 / (2 * sigma_x ** 2)) + (y_dir ** 2 / (2 * sigma_y ** 2))))
            social_pen += -LAMBDA_SOCIAL * G

        social_pen = max(social_pen, -1.0)
        return float(progress + social_pen + TIME_PENALTY)
