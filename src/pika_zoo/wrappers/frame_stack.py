"""
Wrapper that stacks the last N observations along a time axis.

This preserves temporal structure as (n_frames, observation_size) instead of
flattening into a single vector.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseParallelWrapper


class FrameStack(BaseParallelWrapper):
    """Stack the last n_frames observations for each agent."""

    def __init__(self, env, n_frames: int = 4) -> None:
        super().__init__(env)
        if not isinstance(n_frames, int):
            raise TypeError("n_frames must be an int")
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        self.n_frames = n_frames
        self._buffers: dict[str, deque[np.ndarray]] = {}
        self._reset_next_step = False

    def observation_space(self, agent: str) -> spaces.Box:
        base_space = self.env.observation_space(agent)
        if not isinstance(base_space, spaces.Box):
            raise TypeError("FrameStack only supports Box observation spaces")
        low = np.stack([base_space.low] * self.n_frames, axis=0)
        high = np.stack([base_space.high] * self.n_frames, axis=0)
        return spaces.Box(low=low, high=high, dtype=base_space.dtype)

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self._buffers = {}
        self._reset_next_step = False
        for agent, obs in observations.items():
            self._buffers[agent] = deque([obs.copy() for _ in range(self.n_frames)], maxlen=self.n_frames)
        return self._stack_observations(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)
        if self._reset_next_step:
            self._reset_buffers(observations)
            self._reset_next_step = False
        else:
            for agent, obs in observations.items():
                if agent not in self._buffers:
                    self._buffers[agent] = deque([obs.copy() for _ in range(self.n_frames)], maxlen=self.n_frames)
                else:
                    self._buffers[agent].append(obs.copy())

        stacked = self._stack_observations(observations)
        if any(info.get("round_ended", False) for info in infos.values()) and not any(terminations.values()):
            self._reset_next_step = True
        return stacked, rewards, terminations, truncations, infos

    def _reset_buffers(self, observations: dict[str, np.ndarray]) -> None:
        self._buffers = {
            agent: deque([obs.copy() for _ in range(self.n_frames)], maxlen=self.n_frames)
            for agent, obs in observations.items()
        }

    def _stack_observations(self, observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {agent: np.stack(list(self._buffers[agent]), axis=0) for agent in observations}
