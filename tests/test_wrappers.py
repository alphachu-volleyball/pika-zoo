"""Tests for the wrappers module."""

import pytest

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.env import env
from pika_zoo.env.observations import OBSERVATION_SIZE
from pika_zoo.wrappers import ConvertSingleAgent


class TestConvertSingleAgent:
    def test_gym_interface(self):
        """Should expose standard Gymnasium interface."""
        e = env()
        wrapped = ConvertSingleAgent(e)
        obs, info = wrapped.reset(seed=42)
        assert obs.shape == (OBSERVATION_SIZE,)
        assert isinstance(info, dict)

    def test_step_returns_single(self):
        """step() should return single obs/reward, not dicts."""
        e = env()
        wrapped = ConvertSingleAgent(e)
        wrapped.reset(seed=42)
        obs, reward, terminated, truncated, info = wrapped.step(0)
        assert obs.shape == (OBSERVATION_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_with_builtin_ai_opponent(self):
        """Should work with BuiltinAI as opponent."""
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=BuiltinAI())
        wrapped.reset(seed=42)

        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break

        assert game_ended

    def test_with_callable_opponent(self):
        """Should work with a callable as opponent."""
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=lambda obs: 0)
        wrapped.reset(seed=42)

        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break

        assert game_ended

    def test_with_random_opponent(self):
        """Should work with None (random) opponent."""
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=None)
        wrapped.reset(seed=42)

        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break

        assert game_ended

    def test_player2_as_agent(self):
        """Should work when learner is player_2."""
        e = env(winning_score=1)
        wrapped = ConvertSingleAgent(e, agent="player_2", opponent_policy=BuiltinAI())
        obs, _ = wrapped.reset(seed=42)
        # player_2 starts at x=396
        assert obs[0] == pytest.approx(396.0)

    def test_invalid_agent(self):
        """Should raise on invalid agent name."""
        e = env()
        with pytest.raises(AssertionError):
            ConvertSingleAgent(e, agent="player_3")

    def test_observation_action_spaces(self):
        """Should expose correct spaces."""
        e = env()
        wrapped = ConvertSingleAgent(e)
        assert wrapped.observation_space.shape == (OBSERVATION_SIZE,)
        assert wrapped.action_space.n == 18
