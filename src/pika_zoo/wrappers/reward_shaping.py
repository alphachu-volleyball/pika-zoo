"""
Wrapper that adds shaped rewards via pluggable reward channels.

Each channel is a (callable, coefficient) pair. The callable receives PikaPhysics
and returns (p1_reward, p2_reward), which is scaled by the coefficient.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.engine.physics import PikaPhysics
from pika_zoo.wrappers.reward_channels import linear_ball_position

RewardChannel = Callable[[PikaPhysics], tuple[float, float]]

PRESETS: dict[str, list[tuple[RewardChannel, float]]] = {
    "default": [(linear_ball_position, 0.01)],
}


class RewardShaping(BaseParallelWrapper):
    """Add shaped rewards via pluggable channels.

    Args:
        env: The base PikachuVolleyballEnv.
        channels: List of (channel_fn, coefficient) pairs.

    Example::

        RewardShaping(env, channels=[
            (linear_ball_position, 0.01),
            (normal_state_bonus, 0.005),
        ])
    """

    def __init__(
        self,
        env,
        channels: Sequence[tuple[RewardChannel, float]] = (),
    ) -> None:
        super().__init__(env)
        self._channels = list(channels)

    @classmethod
    def from_preset(cls, env, preset: str) -> RewardShaping:
        """Create RewardShaping from a named preset."""
        if preset not in PRESETS:
            available = ", ".join(sorted(PRESETS.keys()))
            raise KeyError(f"Unknown preset: {preset!r}. Available: {available}")
        return cls(env, channels=PRESETS[preset])

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        physics = self.env._physics
        if physics is None:
            return observations, rewards, terminations, truncations, infos

        for fn, coeff in self._channels:
            p1_r, p2_r = fn(physics)
            rewards["player_1"] += p1_r * coeff
            rewards["player_2"] += p2_r * coeff

        return observations, rewards, terminations, truncations, infos
