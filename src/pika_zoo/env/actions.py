"""
Action space definition and conversion for the Pikachu Volleyball environment.

18 discrete actions covering all combinations of:
- 3 x-directions (left, neutral, right)
- 3 y-directions (up, neutral, down)
- 2 power_hit (off, on)

Each action maps to a 5-element binary vector: [left, right, up, down, power_hit].

Reference: helpingstar/pika-zoo action_key_map
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pika_zoo.engine.types import UserInput

# fmt: off
# Action table: each row is [left, right, up, down, power_hit]
ACTION_TABLE: NDArray[np.int32] = np.array([
    [0, 0, 0, 0, 0],  #  0: NOOP
    [0, 0, 0, 0, 1],  #  1: FIRE
    [0, 0, 1, 0, 0],  #  2: UP
    [0, 1, 0, 0, 0],  #  3: RIGHT
    [1, 0, 0, 0, 0],  #  4: LEFT
    [0, 0, 0, 1, 0],  #  5: DOWN
    [0, 1, 1, 0, 0],  #  6: UP+RIGHT
    [1, 0, 1, 0, 0],  #  7: UP+LEFT
    [0, 1, 0, 1, 0],  #  8: DOWN+RIGHT
    [1, 0, 0, 1, 0],  #  9: DOWN+LEFT
    [0, 0, 1, 0, 1],  # 10: UP+FIRE
    [0, 1, 0, 0, 1],  # 11: RIGHT+FIRE
    [1, 0, 0, 0, 1],  # 12: LEFT+FIRE
    [0, 0, 0, 1, 1],  # 13: DOWN+FIRE
    [0, 1, 1, 0, 1],  # 14: UP+RIGHT+FIRE
    [1, 0, 1, 0, 1],  # 15: UP+LEFT+FIRE
    [0, 1, 0, 1, 1],  # 16: DOWN+RIGHT+FIRE
    [1, 0, 0, 1, 1],  # 17: DOWN+LEFT+FIRE
], dtype=np.int32)
# fmt: on

NUM_ACTIONS: int = len(ACTION_TABLE)

# Reverse mapping: (x_direction, y_direction, power_hit) → action index
_INPUT_TO_ACTION: dict[tuple[int, int, int], int] = {}
for _i, _keys in enumerate(ACTION_TABLE):
    _xd = -int(_keys[0]) + int(_keys[1])
    _yd = -int(_keys[2]) + int(_keys[3])
    _ph = int(_keys[4])
    _INPUT_TO_ACTION[(_xd, _yd, _ph)] = _i


def user_input_to_action(x_direction: int, y_direction: int, power_hit: int) -> int:
    """Convert UserInput fields to action index (0-17). Returns 0 (NOOP) if no match."""
    return _INPUT_TO_ACTION.get((x_direction, y_direction, power_hit), 0)


class ActionConverter:
    """Converts discrete action index to UserInput with power_hit debouncing.

    The original game requires power_hit to be a fresh press (not auto-repeated).
    This converter tracks the previous power_hit key state and only sets
    power_hit=1 on the rising edge (key was not pressed → now pressed).
    """

    __slots__ = ("_prev_power_hit",)

    def __init__(self) -> None:
        self._prev_power_hit: int = 0

    def convert(self, action: int | NDArray[np.int32]) -> UserInput:
        """Convert a discrete action index (or key array) to UserInput."""
        if isinstance(action, (int, np.integer)):
            keys = ACTION_TABLE[action]
        else:
            keys = action

        user_input = UserInput()

        # x_direction: left=-1, right=+1
        user_input.x_direction = -keys[0] + keys[1]
        # y_direction: up=-1, down=+1
        user_input.y_direction = -keys[2] + keys[3]

        # power_hit: only trigger on rising edge
        current_power_hit = int(keys[4])
        if current_power_hit == 1 and self._prev_power_hit == 0:
            user_input.power_hit = 1
        else:
            user_input.power_hit = 0
        self._prev_power_hit = current_power_hit

        return user_input

    def reset(self) -> None:
        self._prev_power_hit = 0
