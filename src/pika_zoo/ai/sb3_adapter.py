"""
Adapter that wraps an SB3 model as an AIPolicy.

SB3 (stable-baselines3) is an optional dependency — only imported when used.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from numpy.random import Generator

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput
from pika_zoo.env.actions import ACTION_TABLE
from pika_zoo.env.observations import build_observation
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

    Works both as an ai_policy in PikachuVolleyballEnv and with the play script.

    Args:
        model_path: Path to saved SB3 model (.zip).
        deterministic: Use deterministic predictions (default True).
        action_simplified: If True, treat model output as simplified 13-action
            indices and remap through the per-player SimplifyAction table (default True).
        observation_simplified: If True, mirror player_2's x-axis observations
            using the same transform as SimplifyObservation (default False).
        observation_normalized: If True, normalize raw observations to [0, 1]
            using the same min-max scaling as NormalizeObservation (default True).
        frame_stack: Number of processed observations to stack for models trained
            with FrameStack. 1 disables stacking and preserves existing behavior.
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
        frame_stack: int = 1,
        agent: str | None = None,
    ) -> None:
        if (action_simplified or observation_simplified) and agent is None:
            raise ValueError("agent must be specified when action_simplified=True or observation_simplified=True")
        if action_simplified and agent not in _SIMPLIFIED_MAPS:
            raise ValueError(f"Unknown agent: {agent!r}. Must be 'player_1' or 'player_2'.")
        if not isinstance(frame_stack, int):
            raise TypeError("frame_stack must be an int")
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
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
        self._frame_stack = frame_stack
        self._frame_buffer: deque[np.ndarray] | None = deque(maxlen=frame_stack) if frame_stack > 1 else None
        self._sampling_rng: np.random.Generator | None = None
        self._prev_power_hit: int = 0
        self._opponent_prev_power_hit: int = 0

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        """Predict action from the model using current game state."""
        obs = build_observation(player, opponent, ball, self._prev_power_hit, self._opponent_prev_power_hit)

        if self._observation_simplified:
            obs = _mirror(obs)
        if self._observation_normalized:
            obs = np.clip((obs - OBS_LOW) / OBS_RANGE, 0.0, 1.0).astype(np.float32)

        model_obs = self._stack_observation(obs)
        if not self._deterministic and self._sampling_rng is None:
            self._reset_sampling_rng(rng)
        action, _ = self._predict_model(model_obs)
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

        # Track power_hit for next frame's observation
        self._prev_power_hit = int(keys[4])

        return user_input

    def reset(self, rng: Generator) -> None:
        self._prev_power_hit = 0
        self._opponent_prev_power_hit = 0
        if self._deterministic:
            self._sampling_rng = None
        else:
            self._reset_sampling_rng(rng)
        if self._frame_buffer is not None:
            self._frame_buffer.clear()

    def _reset_sampling_rng(self, rng: Generator) -> None:
        seed = int(rng.integers(0, 2**31))
        self._sampling_rng = np.random.default_rng(seed)

    def _predict_model(self, model_obs: np.ndarray) -> tuple[np.ndarray | int, object]:
        if self._deterministic:
            return self._model.predict(model_obs, deterministic=True)

        assert self._sampling_rng is not None
        seed = int(self._sampling_rng.integers(0, 2**31))

        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return self._model.predict(model_obs, deterministic=False)

    def _stack_observation(self, obs: np.ndarray) -> np.ndarray:
        if self._frame_buffer is None:
            return obs

        self._frame_buffer.append(obs.copy())
        frames = list(self._frame_buffer)
        if len(frames) < self._frame_stack:
            frames = [frames[0]] * (self._frame_stack - len(frames)) + frames
        return np.stack(frames, axis=0).astype(obs.dtype, copy=False)
