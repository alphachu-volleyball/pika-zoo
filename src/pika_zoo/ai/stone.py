"""
Stone AI — stands still and does nothing.

Useful as a minimal baseline for training. Named after the Korean
expression "망부석" (a stone that waits forever).

With `random_position=True`, the stone spawns at a random x position
at the start of each round.
"""

from __future__ import annotations

from numpy.random import Generator

from pika_zoo.engine.constants import GROUND_HALF_WIDTH, GROUND_WIDTH, PLAYER_HALF_LENGTH
from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput

# Valid x ranges per side
_P1_X_MIN = PLAYER_HALF_LENGTH  # 32
_P1_X_MAX = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH  # 184
_P2_X_MIN = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH  # 248
_P2_X_MAX = GROUND_WIDTH - PLAYER_HALF_LENGTH  # 400


class StoneAI:
    """AI that never moves or acts — just stands in place.

    Args:
        random_position: If True, spawn at a random x position each round.
    """

    def __init__(self, random_position: bool = False) -> None:
        self.random_position = random_position

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        return UserInput()

    def reset(self, rng: Generator) -> None:
        pass

    def get_initial_x(self, is_player2: bool, rng: Generator) -> int | None:
        """Return initial x position, or None for default."""
        if not self.random_position:
            return None
        if is_player2:
            return _P2_X_MIN + int(rng.random() * (_P2_X_MAX - _P2_X_MIN))
        return _P1_X_MIN + int(rng.random() * (_P1_X_MAX - _P1_X_MIN))
