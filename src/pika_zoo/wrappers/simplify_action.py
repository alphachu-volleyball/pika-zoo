"""
Wrapper that reduces the action space from 18 to 13 using relative directions.

The 18-action space uses absolute directions (left/right), but from each player's
perspective, "toward net" and "away from net" are more meaningful. This wrapper
maps 13 relative actions to the appropriate 18 absolute actions per player.

Removed actions: simultaneous left+right or up+down (physically impossible),
and redundant diagonal combinations.

Relative actions:
  0: NOOP
  1: FIRE
  2: UP
  3: TOWARD_NET
  4: AWAY_FROM_NET
  5: DOWN
  6: UP+TOWARD_NET
  7: UP+AWAY_FROM_NET
  8: DOWN+TOWARD_NET
  9: DOWN+AWAY_FROM_NET
 10: UP+FIRE
 11: TOWARD_NET+FIRE
 12: AWAY_FROM_NET+FIRE
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper

NUM_SIMPLIFIED_ACTIONS: int = 13

# Player 1 (left side): toward_net = RIGHT, away = LEFT
# fmt: off
_P1_MAP: list[int] = [
    0,   # NOOP
    1,   # FIRE
    2,   # UP
    3,   # RIGHT (toward net)
    4,   # LEFT (away from net)
    5,   # DOWN
    6,   # UP+RIGHT
    7,   # UP+LEFT
    8,   # DOWN+RIGHT
    9,   # DOWN+LEFT
    10,  # UP+FIRE
    11,  # RIGHT+FIRE
    12,  # LEFT+FIRE
]

# Player 2 (right side): toward_net = LEFT, away = RIGHT
_P2_MAP: list[int] = [
    0,   # NOOP
    1,   # FIRE
    2,   # UP
    4,   # LEFT (toward net)
    3,   # RIGHT (away from net)
    5,   # DOWN
    7,   # UP+LEFT
    6,   # UP+RIGHT
    9,   # DOWN+LEFT
    8,   # DOWN+RIGHT
    10,  # UP+FIRE
    12,  # LEFT+FIRE
    11,  # RIGHT+FIRE
]
# fmt: on


class SimplifyAction(BaseParallelWrapper):
    """Reduce action space from 18 absolute to 13 relative actions."""

    def __init__(self, env):
        super().__init__(env)
        self._maps = {
            "player_1": np.array(_P1_MAP, dtype=np.int32),
            "player_2": np.array(_P2_MAP, dtype=np.int32),
        }

    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_SIMPLIFIED_ACTIONS)

    def step(self, actions):
        converted = {agent: int(self._maps[agent][action]) for agent, action in actions.items()}
        return super().step(converted)
