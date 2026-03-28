"""
Wrapper that normalizes observations to [0, 1] range using min-max scaling.

Each feature is mapped from its known physical range to [0, 1].
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.engine.constants import (
    GROUND_WIDTH,
    PLAYER_TOUCHING_GROUND_Y_COORD,
)
from pika_zoo.env.observations import OBSERVATION_SIZE

# Min/max for each of the 35 observation features.
# Player features (13 each, repeated for self and opponent):
#   [0] x: [0, GROUND_WIDTH]
#   [1] y: [0, PLAYER_TOUCHING_GROUND_Y_COORD]
#   [2] y_velocity: [-16, 16]  (jump velocity = -16, gravity bounded)
#   [3] diving_direction: [-1, 1]
#   [4] lying_down_duration_left: [-1, 3]
#   [5] frame_number: [0, 4]
#   [6] delay_before_next_frame: [0, 5]
#   [7-11] one-hot state: [0, 1] each
#   [12] prev_power_hit: [0, 1]
# Ball features (9):
#   [26] x: [0, GROUND_WIDTH]
#   [27] y: [0, BALL_TOUCHING_GROUND_Y_COORD]  (use 304 for safety)
#   [28-31] previous x/y: same ranges
#   [32] x_velocity: [-20, 20]
#   [33] y_velocity: [-30, 30]  (generous bound)
#   [34] is_power_hit: [0, 1]

# fmt: off
_PLAYER_LOW = np.array([
    0, 0, -16, -1, -1, 0, 0,
    0, 0, 0, 0, 0,
    0,
], dtype=np.float32)

_PLAYER_HIGH = np.array([
    GROUND_WIDTH, PLAYER_TOUCHING_GROUND_Y_COORD, 16, 1, 3, 4, 5,
    1, 1, 1, 1, 1,
    1,
], dtype=np.float32)

_BALL_LOW = np.array([
    0, 0, 0, 0, 0, 0, -20, -30, 0,
], dtype=np.float32)

_BALL_HIGH = np.array([
    GROUND_WIDTH, 304, GROUND_WIDTH, 304, GROUND_WIDTH, 304, 20, 30, 1,
], dtype=np.float32)
# fmt: on

OBS_LOW = np.concatenate([_PLAYER_LOW, _PLAYER_LOW, _BALL_LOW])
OBS_HIGH = np.concatenate([_PLAYER_HIGH, _PLAYER_HIGH, _BALL_HIGH])
OBS_RANGE = OBS_HIGH - OBS_LOW
# Avoid division by zero for constant features
OBS_RANGE[OBS_RANGE == 0] = 1.0


class NormalizeObservation(BaseParallelWrapper):
    """Normalize observations to [0, 1] using known physical ranges."""

    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        return {agent: self._normalize(obs) for agent, obs in observations.items()}, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        return (
            {agent: self._normalize(obs) for agent, obs in observations.items()},
            rewards,
            terminations,
            truncations,
            infos,
        )

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((obs - OBS_LOW) / OBS_RANGE, 0.0, 1.0).astype(np.float32)
