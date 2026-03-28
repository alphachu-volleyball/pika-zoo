"""
AI policy protocol — the interface that all AI implementations must follow.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput


@runtime_checkable
class AIPolicy(Protocol):
    """Protocol for any entity that can decide actions for a player.

    Implementations include:
    - BuiltinAI: the original gorisanson AI (with intentional bugs)
    - SuperAI (future): from duckll/pikachu-volleyball
    - RL model wrappers: adapters around trained policies
    """

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        """Given game state, return the input to apply for this player."""
        ...

    def reset(self, rng: Generator) -> None:
        """Called at the start of each round. Optional initialization."""
        ...
