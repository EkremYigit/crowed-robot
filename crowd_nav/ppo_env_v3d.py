"""
V3d: Time penalty + conditional velocity reward (only when nearest human > 1.2m).
Speed bonus is active only when path is clear — avoids aggressive crowded-space rushing.
"""

import numpy as np
from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

TIME_PENALTY  = -0.003
V_REWARD_MAX  = 0.015
CLEAR_DIST    = 1.2    # minimum distance to nearest human for velocity reward


class CrowdNavPPOEnvV3d(CrowdNavPPOEnvGaussian):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        gaussian_r = super()._compute_reward(info, prev_dist, ob)

        # Velocity reward only when path is clear
        min_human_dist = min(
            (np.hypot(self.robot.px - h.px, self.robot.py - h.py) for h in ob),
            default=float('inf')
        )
        if min_human_dist > CLEAR_DIST:
            speed = np.hypot(self.robot.vx, self.robot.vy)
            if speed > 1e-6:
                goal_dx = self.robot.gx - self.robot.px
                goal_dy = self.robot.gy - self.robot.py
                goal_norm = np.hypot(goal_dx, goal_dy) + 1e-8
                alignment = max(0.0, (self.robot.vx * goal_dx + self.robot.vy * goal_dy) / (speed * goal_norm))
                velocity_r = V_REWARD_MAX * (speed / self.robot.v_pref) * alignment
            else:
                velocity_r = 0.0
        else:
            velocity_r = 0.0

        return float(gaussian_r + TIME_PENALTY + velocity_r)
