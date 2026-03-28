"""
Adapter that wraps an SB3 model as an AIPolicy.

SB3 (stable-baselines3) is an optional dependency — only imported when used.
"""

from __future__ import annotations

from pathlib import Path

from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput
from pika_zoo.env.actions import ACTION_TABLE


class SB3ModelPolicy:
    """Wraps an SB3 model (PPO, etc.) as an AIPolicy.

    The model receives a simplified observation and returns a discrete action.
    This adapter converts between the engine's Player/Ball state and the
    model's expected observation format.

    Args:
        model_path: Path to saved SB3 model (.zip).
        deterministic: Use deterministic predictions (default True).
    """

    def __init__(self, model_path: str | Path, deterministic: bool = True) -> None:
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required to load SB3 models. Install with: pip install stable-baselines3"
            ) from None

        self._model = PPO.load(str(model_path), device="cpu")
        self._deterministic = deterministic
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

        action, _ = self._model.predict(self._last_obs, deterministic=self._deterministic)
        action_idx = int(action)

        # Convert action index to UserInput
        keys = ACTION_TABLE[action_idx]
        user_input = UserInput()
        user_input.x_direction = -int(keys[0]) + int(keys[1])
        user_input.y_direction = -int(keys[2]) + int(keys[3])
        user_input.power_hit = int(keys[4])
        return user_input

    def reset(self, rng: Generator) -> None:
        self._last_obs = None
