"""
Type definitions for the physics engine.

Source: https://github.com/gorisanson/pikachu-volleyball/blob/main/src/resources/js/physics.js
"""

from __future__ import annotations

from enum import IntEnum


class PlayerState(IntEnum):
    """Player states from the original game.

    0: normal, 1: jumping, 2: jumping_and_power_hitting, 3: diving,
    4: lying_down_after_diving, 5: win, 6: lose
    """

    NORMAL = 0
    JUMPING = 1
    JUMPING_POWER_HIT = 2
    DIVING = 3
    LYING_DOWN = 4
    WIN = 5
    LOSE = 6


class UserInput:
    """User input for a single player (from keyboard, AI, or RL agent).

    Attributes:
        x_direction: 0 = no input, -1 = left, 1 = right
        y_direction: 0 = no input, -1 = up, 1 = down
        power_hit: 0 = no input or auto-repeated, 1 = power hit triggered
    """

    __slots__ = ("x_direction", "y_direction", "power_hit")

    def __init__(self) -> None:
        self.x_direction: int = 0
        self.y_direction: int = 0
        self.power_hit: int = 0

    def reset(self) -> None:
        self.x_direction = 0
        self.y_direction = 0
        self.power_hit = 0
