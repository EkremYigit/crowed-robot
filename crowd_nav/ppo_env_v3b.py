"""
V3b: Pure time penalty only (-0.006/step), no other changes.
Minimal intervention hypothesis: making every idle step costly is sufficient.
"""

from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

TIME_PENALTY = -0.006


class CrowdNavPPOEnvV3b(CrowdNavPPOEnvGaussian):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        gaussian_r = super()._compute_reward(info, prev_dist, ob)
        return float(gaussian_r + TIME_PENALTY)
