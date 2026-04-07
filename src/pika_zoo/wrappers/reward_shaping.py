"""
Wrapper that adds shaped rewards via pluggable reward channels.

Each channel is a (RewardChannel, coefficient) pair. The channel receives
PikaPhysics and returns (p1_reward, p2_reward), which is scaled by the coefficient.
"""

from __future__ import annotations

from collections.abc import Sequence

from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.wrappers.reward_channels import LinearBallPosition, RewardChannel

PRESETS: dict[str, list[tuple[RewardChannel, float]]] = {
    "default": [(LinearBallPosition(), 0.01)],
}


class RewardShaping(BaseParallelWrapper):
    """Add shaped rewards via pluggable channels.

    Args:
        env: The base PikachuVolleyballEnv.
        channels: List of (RewardChannel, coefficient) pairs.

    Example::

        RewardShaping(env, channels=[
            (LinearBallPosition(), 0.01),
            (QuadrantBallPosition(), 0.005),
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

        for channel, coeff in self._channels:
            p1_r, p2_r = channel(physics)
            rewards["player_1"] += p1_r * coeff
            rewards["player_2"] += p2_r * coeff

        return observations, rewards, terminations, truncations, infos
