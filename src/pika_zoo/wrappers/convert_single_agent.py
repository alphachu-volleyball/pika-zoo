"""
Wrapper that converts PikachuVolleyballEnv (ParallelEnv) to a Gymnasium single-agent env.

The opponent is controlled by a fixed policy (AIPolicy, callable, or random).
The learning agent only sees its own observation and acts through the standard
Gymnasium step(action) → (obs, reward, terminated, truncated, info) interface.

This is required for SB3 compatibility (SubprocVecEnv, PPO, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv


class ConvertSingleAgent(gym.Env):
    """Convert PikachuVolleyballEnv to a single-agent Gymnasium env.

    Args:
        env: The PikachuVolleyballEnv instance to wrap.
        agent: Which agent the learner controls ("player_1" or "player_2").
        opponent_policy: How the opponent acts. Can be:
            - An AIPolicy instance (e.g. BuiltinAI) — injected into env.ai_policies
            - A callable (obs → action int)
            - None (random actions)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        env: PikachuVolleyballEnv,
        agent: str = "player_1",
        opponent_policy: AIPolicy | Callable[[np.ndarray], int] | None = None,
    ) -> None:
        super().__init__()

        assert agent in env.possible_agents, f"agent must be one of {env.possible_agents}"

        self._env = env
        self._agent = agent
        self._opponent = "player_2" if agent == "player_1" else "player_1"
        self._opponent_policy = opponent_policy

        # If AIPolicy, inject into env so physics-level AI works correctly
        if isinstance(opponent_policy, AIPolicy):
            self._env.ai_policies[self._opponent] = opponent_policy
            self._opponent_is_ai = True
        else:
            self._opponent_is_ai = False

        self.observation_space = env.observation_space(agent)
        self.action_space = env.action_space(agent)
        self.render_mode = env.render_mode
        self._last_observations: dict[str, np.ndarray] | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        observations, infos = self._env.reset(seed=seed, options=options)
        self._last_observations = observations
        return observations[self._agent], infos[self._agent]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        opponent_action = self._get_opponent_action()

        actions = {
            self._agent: action,
            self._opponent: opponent_action,
        }

        observations, rewards, terminations, truncations, infos = self._env.step(actions)
        self._last_observations = observations

        return (
            observations[self._agent],
            rewards[self._agent],
            terminations[self._agent],
            truncations[self._agent],
            infos[self._agent],
        )

    def render(self) -> np.ndarray | None:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def _get_opponent_action(self) -> int:
        # AIPolicy: handled by env.ai_policies, return dummy
        if self._opponent_is_ai:
            return 0

        # Callable: obs → action
        if self._opponent_policy is not None:
            assert self._last_observations is not None, "Call reset() before step()"
            obs = self._last_observations[self._opponent]
            return int(self._opponent_policy(obs))

        # None: random action
        return int(self._env.action_space(self._opponent).sample())
