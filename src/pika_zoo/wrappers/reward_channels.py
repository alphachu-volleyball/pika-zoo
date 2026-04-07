"""
Built-in reward channel functions for RewardShaping.

Each channel is a callable: (PikaPhysics) -> (p1_reward, p2_reward).
The returned values are multiplied by the channel's coefficient.
"""

from __future__ import annotations

from collections.abc import Callable

from pika_zoo.engine.constants import GROUND_HALF_WIDTH
from pika_zoo.engine.physics import PikaPhysics

BALL_TOUCHING_GROUND_Y_COORD = 252


def linear_ball_position(physics: PikaPhysics) -> tuple[float, float]:
    """Reward based on ball x position. Positive when ball is on opponent's side.

    Zero-sum: p1 bonus = -p2 bonus.
    """
    p1_bonus = (physics.ball.x - GROUND_HALF_WIDTH) / GROUND_HALF_WIDTH
    return (p1_bonus, -p1_bonus)


def quadrant_ball_position(
    rewards: tuple[float, ...] = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0),
    x_line: int = GROUND_HALF_WIDTH,
    y_line: int = 192,
) -> Callable[[PikaPhysics], tuple[float, float]]:
    """Quadrant-based ball position reward.

    Divides the court into 4 zones and assigns fixed rewards per zone.

    Args:
        rewards: 8 values — (p1_z0, p1_z1, p1_z2, p1_z3, p2_z0, p2_z1, p2_z2, p2_z3).
        x_line: Vertical boundary (default: net at 216).
        y_line: Horizontal boundary (default: 192, net pillar top).

    Zones::

            P1 side          P2 side
          [0, x_line)     [x_line, 432]
        ┌──────────┬──────────┐
        │  zone 0  │  zone 2  │  y < y_line (upper)
        ├──────────┼──────────┤
        │  zone 1  │  zone 3  │  y >= y_line (lower)
        └──────────┴──────────┘
    """
    if len(rewards) != 8:
        raise ValueError(f"rewards must have 8 values, got {len(rewards)}")

    def channel(physics: PikaPhysics) -> tuple[float, float]:
        zone = (0 if physics.ball.x < x_line else 2) + (0 if physics.ball.y < y_line else 1)
        return (rewards[zone], rewards[zone + 4])

    channel.__name__ = "quadrant_ball_position"
    return channel
