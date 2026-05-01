"""
V3a: Stronger progress (0.4->0.7) + time penalty (-0.005/step).
Two-pronged frozen-robot fix: stronger forward incentive + cost per wasted step.
Social zone (Gaussian asymmetric) unchanged.
"""

import numpy as np
from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

PROGRESS_DELTA = 0.3   # new_coeff(0.7) - old_coeff(0.4)
TIME_PENALTY   = -0.005


class CrowdNavPPOEnvV3a(CrowdNavPPOEnvGaussian):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        gaussian_r = super()._compute_reward(info, prev_dist, ob)
        # Boost progress coefficient: add the extra (0.3 * delta_dist) on top
        extra_progress = PROGRESS_DELTA * (prev_dist - self._goal_dist())
        return float(gaussian_r + extra_progress + TIME_PENALTY)
