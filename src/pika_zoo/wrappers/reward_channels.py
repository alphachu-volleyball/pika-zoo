"""
Built-in reward channel functions for RewardShaping.

Each channel is a callable: (PikaPhysics) -> (p1_reward, p2_reward).
The returned values are multiplied by the channel's coefficient.
"""

from __future__ import annotations

from pika_zoo.engine.constants import GROUND_HALF_WIDTH
from pika_zoo.engine.physics import PikaPhysics


def linear_ball_position(physics: PikaPhysics) -> tuple[float, float]:
    """Reward based on ball x position. Positive when ball is on opponent's side.

    Zero-sum: p1 bonus = -p2 bonus.
    """
    p1_bonus = (physics.ball.x - GROUND_HALF_WIDTH) / GROUND_HALF_WIDTH
    return (p1_bonus, -p1_bonus)


def normal_state_bonus(physics: PikaPhysics) -> tuple[float, float]:
    """Small reward for being in normal (idle/ready) state.

    Not zero-sum: each player gets rewarded independently.
    """
    p1 = 1.0 if physics.player1.state == 0 else 0.0
    p2 = 1.0 if physics.player2.state == 0 else 0.0
    return (p1, p2)
