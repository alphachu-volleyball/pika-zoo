"""
Random AI — takes completely random actions every frame.

Useful as a baseline opponent for training and evaluation.
"""

from __future__ import annotations

from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput


class RandomAI:
    """AI that picks random x_direction, y_direction, and power_hit each frame."""

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        user_input = UserInput()
        user_input.x_direction = int(rng.integers(-1, 2))  # -1, 0, 1
        user_input.y_direction = int(rng.integers(-1, 2))
        user_input.power_hit = int(rng.integers(0, 2))  # 0 or 1
        return user_input

    def reset(self, rng: Generator) -> None:
        pass
