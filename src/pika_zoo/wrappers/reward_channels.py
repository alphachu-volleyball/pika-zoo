"""
Built-in reward channel functions for RewardShaping.

Each channel extends RewardChannel, implementing __call__ and __repr__.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pika_zoo.engine.constants import GROUND_HALF_WIDTH
from pika_zoo.engine.physics import PikaPhysics


class RewardChannel(ABC):
    """Base class for reward channels.

    Subclasses must implement __call__ and __repr__.
    __call__ receives PikaPhysics and returns (p1_reward, p2_reward).
    __repr__ should return a reconstructible string for W&B logging.
    """

    @abstractmethod
    def __call__(self, physics: PikaPhysics) -> tuple[float, float]: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class LinearBallPosition(RewardChannel):
    """Reward based on ball x position. Positive when ball is on opponent's side.

    Zero-sum: p1 bonus = -p2 bonus.
    """

    def __call__(self, physics: PikaPhysics) -> tuple[float, float]:
        p1_bonus = (physics.ball.x - GROUND_HALF_WIDTH) / GROUND_HALF_WIDTH
        return (p1_bonus, -p1_bonus)

    def __repr__(self) -> str:
        return "LinearBallPosition()"


class QuadrantBallPosition(RewardChannel):
    """Quadrant-based ball position reward.

    Divides the court into 4 zones and assigns fixed rewards per zone.

    Args:
        x_line: Vertical boundary (default: net at 216).
        y_line: Horizontal boundary (default: 192, net pillar top).
        rewards: 8 values — (p1_z0, p1_z1, p1_z2, p1_z3, p2_z0, p2_z1, p2_z2, p2_z3).

    Zones::

            P1 side          P2 side
          [0, x_line)     [x_line, 432]
        ┌──────────┬──────────┐
        │  zone 0  │  zone 2  │  y < y_line (upper)
        ├──────────┼──────────┤
        │  zone 1  │  zone 3  │  y >= y_line (lower)
        └──────────┴──────────┘
    """

    def __init__(
        self,
        x_line: int = GROUND_HALF_WIDTH,
        y_line: int = 192,
        rewards: tuple[float, ...] = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0),
    ) -> None:
        if len(rewards) != 8:
            raise ValueError(f"rewards must have 8 values, got {len(rewards)}")
        self.x_line = x_line
        self.y_line = y_line
        self.rewards = rewards

    def __call__(self, physics: PikaPhysics) -> tuple[float, float]:
        zone = (0 if physics.ball.x < self.x_line else 2) + (0 if physics.ball.y < self.y_line else 1)
        return (self.rewards[zone], self.rewards[zone + 4])

    def __repr__(self) -> str:
        return f"QuadrantBallPosition(x_line={self.x_line}, y_line={self.y_line}, rewards={self.rewards})"
