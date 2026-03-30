"""
Stone AI — stands still and does nothing.

Useful as a minimal baseline for training. Named after the Korean
expression "망부석" (a stone that waits forever).
"""

from __future__ import annotations

from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput


class StoneAI:
    """AI that never moves or acts — just stands in place."""

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
