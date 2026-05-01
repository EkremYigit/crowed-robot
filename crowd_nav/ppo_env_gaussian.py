import numpy as np

from ppo_env import CrowdNavPPOEnv
from crowd_sim.envs.utils.info import ReachGoal, Collision

K_V           = 1.0
LAMBDA_SOCIAL = 0.2
SIGMA_BASE    = 0.8


class CrowdNavPPOEnvGaussian(CrowdNavPPOEnv):
    """
    CrowdNavPPOEnv with asymmetric Gaussian social reward instead of the
    fixed 0.6 m circular penalty. The social zone is elongated in the
    direction the human is moving; faster humans produce a wider zone.
    """

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        progress = 0.4 * (prev_dist - self._goal_dist())

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
        return float(progress + social_pen)
