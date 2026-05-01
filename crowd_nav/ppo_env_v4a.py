"""
V4a: Gaussian sosyal alan + hiz-orantili dondurma cezasi.

Sabit zaman cezasindan farki:
  - Sabit ceza: her adimda sabit maliyet -> rushing ogrenir
  - Hiz-orantili: sadece durma cezalandirilir, yavas ama temkinli
    navigasyon serbest -> sosyal uyumu korur

Fine-tuning ile kullanilmasi hedeflenmistir:
  python finetune_ppo.py --pretrain_from data/ppo_gaussian/best_model
"""

import numpy as np
from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

FREEZE_PENALTY_MAX = -0.04   # tamamen dururken max ceza


class CrowdNavPPOEnvV4a(CrowdNavPPOEnvGaussian):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        gaussian_r = super()._compute_reward(info, prev_dist, ob)

        speed = np.hypot(self.robot.vx, self.robot.vy)
        speed_ratio = min(speed / self.robot.v_pref, 1.0)
        freeze_pen = FREEZE_PENALTY_MAX * (1.0 - speed_ratio)

        return float(gaussian_r + freeze_pen)
