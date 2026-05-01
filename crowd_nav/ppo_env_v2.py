"""
CrowdNavPPOEnvV2: Gaussian sosyal alan + frozen-robot cezalari.

Eklenenler (Gaussian'a gore):
  1. Alive bonus (+ALIVE_BONUS/adim) — beklemenin her adiminda maliyet
  2. Goal-directed velocity reward — sadece hedefe dogru hareket oduller;
     cevresel/geri hareket icin sifir (circle-of-death hack'ini kapatir)

Literatur:
  - Alive bonus pattern: arxiv:2403.09793 (Socially Integrated Navigation, 2024)
  - Velocity reward (goal-aligned): DRL-VO, Xie & Dames ICRA 2022
  - Yon filtreleme: circle-of-death kapatma (critic review, 2025)

Critic bulgusu — robot.vx/vy staleness:
  agent.py:130 `self.vx = action.vx` CrowdSim.step() icinde senkron guncelleniyor;
  _compute_reward cagrildiginda degerler gunceldir.
"""

import numpy as np

from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

ALIVE_BONUS  = 0.008   # beklemenin adim maliyeti; waiting << navigating
V_REWARD_MAX = 0.025   # max bonus; yalnizca v=v_pref & tam hedefe dogru harekette


class CrowdNavPPOEnvV2(CrowdNavPPOEnvGaussian):
    """
    Gaussian sosyal ceza + alive bonus + goal-directed velocity reward.

    circle-of-death korumalari:
      - Tangential hareket: alignment~=0 -> velocity_r~=0
      - Geri hareket:       alignment<0  -> velocity_r=0 (max(0, ...) ile)
    """

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        # Gaussian progress + asimetrik sosyal ceza (parent'tan)
        gaussian_r = super()._compute_reward(info, prev_dist, ob)

        # Alive bonus — bos beklemeyi her adimda maliyetli yapar
        alive = ALIVE_BONUS

        # Goal-directed velocity reward
        speed = np.hypot(self.robot.vx, self.robot.vy)
        if speed > 1e-6:
            goal_dx = self.robot.gx - self.robot.px
            goal_dy = self.robot.gy - self.robot.py
            goal_norm = np.hypot(goal_dx, goal_dy) + 1e-8
            # cos(theta): 1.0 tam hedefe, 0 dik, <0 uzaklasirken
            alignment = max(0.0, (self.robot.vx * goal_dx + self.robot.vy * goal_dy) / (speed * goal_norm))
            velocity_r = V_REWARD_MAX * (speed / self.robot.v_pref) * alignment
        else:
            velocity_r = 0.0

        return float(gaussian_r + alive + velocity_r)
