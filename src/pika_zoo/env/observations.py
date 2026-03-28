"""
Observation space definition and construction for the Pikachu Volleyball environment.

35-element agent-centric observation vector:
- [0:13]  Self player state (13 features)
- [13:26] Opponent player state (13 features)
- [26:35] Ball state (9 features)

Reference: helpingstar/pika-zoo observation format
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from pika_zoo.engine.physics import Ball, Player

OBSERVATION_SIZE: int = 35


def build_observation_space() -> spaces.Box:
    """Build the observation space for a single agent."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)


def build_observation(
    player: Player,
    opponent: Player,
    ball: Ball,
    prev_power_hit: int,
    opponent_prev_power_hit: int,
) -> NDArray[np.float32]:
    """Build a 35-element observation vector from the agent's perspective.

    Each agent sees: [self_state(13), opponent_state(13), ball_state(9)]
    """
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # Self player (13 features)
    _fill_player_obs(obs, 0, player, prev_power_hit)

    # Opponent player (13 features)
    _fill_player_obs(obs, 13, opponent, opponent_prev_power_hit)

    # Ball (9 features)
    obs[26] = ball.x
    obs[27] = ball.y
    obs[28] = ball.previous_x
    obs[29] = ball.previous_y
    obs[30] = ball.previous_previous_x
    obs[31] = ball.previous_previous_y
    obs[32] = ball.x_velocity
    obs[33] = ball.y_velocity
    obs[34] = float(ball.is_power_hit)

    return obs


def _fill_player_obs(obs: NDArray[np.float32], offset: int, player: Player, prev_power_hit: int) -> None:
    """Fill 13 player features into obs starting at offset.

    Features:
    [0] x, [1] y, [2] y_velocity, [3] diving_direction, [4] lying_down_duration_left,
    [5] frame_number, [6] delay_before_next_frame,
    [7-11] one-hot state (normal, jumping, power_hitting, diving, lying_down),
    [12] previous power_hit key status
    """
    obs[offset + 0] = player.x
    obs[offset + 1] = player.y
    obs[offset + 2] = player.y_velocity
    obs[offset + 3] = player.diving_direction
    obs[offset + 4] = player.lying_down_duration_left
    obs[offset + 5] = player.frame_number
    obs[offset + 6] = player.delay_before_next_frame

    # One-hot state encoding (5 dimensions: states 0-4)
    # States 5 (win) and 6 (lose) only occur at game end
    state = player.state
    if state <= 4:
        obs[offset + 7 + state] = 1.0

    obs[offset + 12] = prev_power_hit
