"""
V4c: V4b'den daha hafif kosullu dondurma cezasi.

V4b sorunu: collision Gaussian'in 0.116 seviyesine geri donemiyor (0.156'da kaliyor).
V4c fark:
  - FREEZE_PENALTY_MAX: 0.02 -> 0.01  (yarim)
  - CLEAR_DIST: 1.5m -> 2.0m          (daha genis bos alan gerekiyor)
  -> Robot sadece gercekten bos yolda ve cok yavaslayinca cezalandirilir
  -> Collision'in azalmasi beklenir; az timeout kaybedilebilir

Fine-tuning: --pretrain_from data/ppo_v4b2/checkpoints/ppo_ft_3200000_steps
             --reward v4c --output_dir data/ppo_v4c --timesteps 1000000
"""

import numpy as np
from ppo_env_gaussian import CrowdNavPPOEnvGaussian
from crowd_sim.envs.utils.info import ReachGoal, Collision

FREEZE_PENALTY_MAX = -0.01  # V4b'nin yarisi
CLEAR_DIST         = 2.0    # V4b'den daha genis esik


class CrowdNavPPOEnvV4c(CrowdNavPPOEnvGaussian):

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
