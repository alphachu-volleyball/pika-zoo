"""
Adapter that wraps an SB3 model as an AIPolicy.

SB3 (stable-baselines3) is an optional dependency — only imported when used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput
from pika_zoo.env.actions import ACTION_TABLE
from pika_zoo.wrappers.normalize_observation import OBS_LOW, OBS_RANGE
from pika_zoo.wrappers.simplify_action import P1_MAP, P2_MAP
from pika_zoo.wrappers.simplify_observation import _mirror

_SIMPLIFIED_MAPS: dict[str, list[int]] = {
    "player_1": P1_MAP,
    "player_2": P2_MAP,
}


class SB3ModelPolicy:
    """Wraps an SB3 model (PPO, etc.) as an AIPolicy.

    The model receives a processed observation and returns a discrete action.
    This adapter converts between the engine's Player/Ball state and the
    model's expected observation format.

    Args:
        model_path: Path to saved SB3 model (.zip).
        deterministic: Use deterministic predictions (default True).
        action_simplified: If True, treat model output as simplified 13-action
            indices and remap through the per-player SimplifyAction table (default True).
        observation_simplified: If True, mirror player_2's x-axis observations
            using the same transform as SimplifyObservation (default False).
        observation_normalized: If True, normalize raw observations to [0, 1]
            using the same min-max scaling as NormalizeObservation (default True).
        agent: Agent name ("player_1" or "player_2").
            Required when action_simplified=True or observation_simplified=True.
    """

    def __init__(
        self,
        model_path: str | Path,
        deterministic: bool = True,
        action_simplified: bool = True,
        observation_simplified: bool = False,
        observation_normalized: bool = True,
        agent: str | None = None,
    ) -> None:
        if (action_simplified or observation_simplified) and agent is None:
            raise ValueError("agent must be specified when action_simplified=True or observation_simplified=True")
        if action_simplified and agent not in _SIMPLIFIED_MAPS:
            raise ValueError(f"Unknown agent: {agent!r}. Must be 'player_1' or 'player_2'.")
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required to load SB3 models. Install with: pip install stable-baselines3"
            ) from None

        self._model = PPO.load(str(model_path), device="cpu")
        self._deterministic = deterministic
        self._observation_simplified = observation_simplified and agent == "player_2"
        self._observation_normalized = observation_normalized
        self._action_map: list[int] | None = _SIMPLIFIED_MAPS.get(agent) if action_simplified else None
        self._last_obs = None

    def set_observation(self, obs) -> None:
        """Set the latest observation for this player (called by play script)."""
        self._last_obs = obs

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        """Predict action from the model using the stored observation."""
        if self._last_obs is None:
            return UserInput()

        obs = self._last_obs
        if self._observation_simplified:
            obs = _mirror(obs)
        if self._observation_normalized:
            obs = np.clip((obs - OBS_LOW) / OBS_RANGE, 0.0, 1.0).astype(np.float32)

        action, _ = self._model.predict(obs, deterministic=self._deterministic)
        action_idx = int(action)

        # Remap simplified (13) action to raw (18) action if needed
        if self._action_map is not None:
            action_idx = self._action_map[action_idx]

        # Convert action index to UserInput
        keys = ACTION_TABLE[action_idx]
        user_input = UserInput()
        user_input.x_direction = -int(keys[0]) + int(keys[1])
        user_input.y_direction = -int(keys[2]) + int(keys[3])
        user_input.power_hit = int(keys[4])
        return user_input

    def reset(self, rng: Generator) -> None:
        self._last_obs = None
