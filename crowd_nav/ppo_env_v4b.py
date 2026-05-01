"""
V4b: Gaussian sosyal alan + yol-sartli dondurma cezasi.

V4a'dan farki:
  - V4a: her durumda dondurma cezasi -> insan yakininda bile -> collision artiyor
  - V4b: sadece yol acikken dondurma cezasi (min_human > CLEAR_DIST)
    -> insan yaninda sabırla beklemek serbes
    -> bos yolda yavaslama cezalandirilir

Fine-tuning: python finetune_ppo.py --pretrain_from data/ppo_gaussian/best_model
             --reward v4b --output_dir data/ppo_v4b --timesteps 1000000
"""

import numpy as np
from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

FREEZE_PENALTY_MAX = -0.02  # V4a'dan kucuk (0.04 yerine 0.02)
CLEAR_DIST         = 1.5    # m — bu mesafenin altinda penalty yok


class CrowdNavPPOEnvV4b(CrowdNavPPOEnvGaussian):

    def _compute_reward(self, info, prev_dist, ob):
        if isinstance(info, ReachGoal):
            return 1.0
        if isinstance(info, Collision):
            return -1.0

        gaussian_r = super()._compute_reward(info, prev_dist, ob)

        min_human_dist = min(
            (np.hypot(self.robot.px - h.px, self.robot.py - h.py) for h in ob),
            default=float('inf')
        )

        if min_human_dist > CLEAR_DIST:
            speed = np.hypot(self.robot.vx, self.robot.vy)
            speed_ratio = min(speed / self.robot.v_pref, 1.0)
            freeze_pen = FREEZE_PENALTY_MAX * (1.0 - speed_ratio)
        else:
            freeze_pen = 0.0

        return float(gaussian_r + freeze_pen)
