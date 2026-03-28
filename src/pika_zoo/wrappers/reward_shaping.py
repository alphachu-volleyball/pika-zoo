"""
Wrapper that adds shaped rewards based on ball position and player state.

Combines two reward shaping strategies:
1. Ball position reward: bonus/penalty based on which side of the court the ball is on
2. Normal state reward: small reward for being in ready (normal) state when idle
"""

from __future__ import annotations

from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.engine.constants import GROUND_HALF_WIDTH


class RewardShaping(BaseParallelWrapper):
    """Add shaped rewards to the base environment.

    Args:
        env: The base PikachuVolleyballEnv.
        ball_position_coeff: Reward coefficient for ball position.
            Positive when ball is on opponent's side, negative on own side.
            Applied per-frame (not just on scoring). Default 0.01.
        normal_state_coeff: Reward for being in normal (idle) state when
            no scoring happens. Encourages staying ready. Default 0.0 (off).
    """

    def __init__(
        self,
        env,
        ball_position_coeff: float = 0.01,
        normal_state_coeff: float = 0.0,
    ) -> None:
        super().__init__(env)
        self._ball_position_coeff = ball_position_coeff
        self._normal_state_coeff = normal_state_coeff

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        physics = self.env._physics
        if physics is None:
            return observations, rewards, terminations, truncations, infos

        ball_x = physics.ball.x

        # Ball position reward: positive when ball is on opponent's side
        if self._ball_position_coeff != 0.0:
            # Player 1: ball on right side (opponent) is good
            p1_ball_bonus = (ball_x - GROUND_HALF_WIDTH) / GROUND_HALF_WIDTH * self._ball_position_coeff
            rewards["player_1"] += p1_ball_bonus
            rewards["player_2"] -= p1_ball_bonus

        # Normal state reward: small bonus for being in ready state
        if self._normal_state_coeff != 0.0 and rewards["player_1"] == 0.0:
            if physics.player1.state == 0:
                rewards["player_1"] += self._normal_state_coeff
            if physics.player2.state == 0:
                rewards["player_2"] += self._normal_state_coeff

        return observations, rewards, terminations, truncations, infos
