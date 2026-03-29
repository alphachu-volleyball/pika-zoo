"""
Wrapper that mirrors x-axis observations for player_2.

Both players' observations are already agent-centric (self first, opponent second),
but x-coordinates are absolute. Player 1 is on the left (x near 0), player 2 on
the right (x near 432). This wrapper mirrors player_2's x-axis so both players
see themselves as "on the left side of the court."

Player_1 observations pass through unchanged.

Must be applied BEFORE NormalizeObservation in the wrapper chain — mirroring
operates on raw coordinates, normalization maps them to [0, 1] afterward.

Mirrored features (player_2 only):
  Positions (432 - x):  self.x, opponent.x, ball.x, ball.prev_x, ball.prev_prev_x
  Directions (negate):  self.diving_direction, opponent.diving_direction, ball.x_velocity
"""

from __future__ import annotations

import numpy as np
from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.engine.constants import GROUND_WIDTH

# Indices where x-position must be mirrored (GROUND_WIDTH - x)
_POSITION_INDICES = np.array([0, 13, 26, 28, 30], dtype=np.intp)

# Indices where x-direction/velocity must be negated
_DIRECTION_INDICES = np.array([3, 16, 32], dtype=np.intp)


class SimplifyObservation(BaseParallelWrapper):
    """Mirror player_2 x-axis observations so both players see the left-side perspective."""

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        observations["player_2"] = _mirror(observations["player_2"])
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        observations["player_2"] = _mirror(observations["player_2"])
        return observations, rewards, terminations, truncations, infos


def _mirror(obs: np.ndarray) -> np.ndarray:
    """Mirror x-axis values so player sees themselves on the left side."""
    mirrored = obs.copy()
    mirrored[_POSITION_INDICES] = GROUND_WIDTH - mirrored[_POSITION_INDICES]
    mirrored[_DIRECTION_INDICES] = -mirrored[_DIRECTION_INDICES]
    return mirrored
