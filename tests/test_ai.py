"""Tests for the AI module."""

import numpy as np

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.ai.registry import get_ai, register_ai
from pika_zoo.engine.physics import PikaPhysics
from pika_zoo.engine.types import UserInput


class TestAIProtocol:
    def test_builtin_satisfies_protocol(self):
        assert isinstance(BuiltinAI(), AIPolicy)

    def test_custom_ai_satisfies_protocol(self):
        class MyAI:
            def compute_action(self, player, ball, opponent, rng):
                return UserInput()

            def reset(self, rng):
                pass

        assert isinstance(MyAI(), AIPolicy)


class TestBuiltinAI:
    def test_returns_user_input(self):
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        ai = BuiltinAI()
        result = ai.compute_action(physics.player1, physics.ball, physics.player2, rng)
        assert isinstance(result, UserInput)

    def test_deterministic(self):
        """Same state + seed should produce same AI decision."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        physics1 = PikaPhysics(rng1)
        physics2 = PikaPhysics(rng2)
        ai = BuiltinAI()

        result1 = ai.compute_action(physics1.player1, physics1.ball, physics1.player2, rng1)
        result2 = ai.compute_action(physics2.player1, physics2.ball, physics2.player2, rng2)

        assert result1.x_direction == result2.x_direction
        assert result1.y_direction == result2.y_direction
        assert result1.power_hit == result2.power_hit

    def test_ai_vs_ai_simulation(self):
        """Two AI players should be able to play a full round without errors."""
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        ai1 = BuiltinAI()
        ai2 = BuiltinAI()

        touched_ground = False
        for _ in range(1000):
            input1 = ai1.compute_action(physics.player1, physics.ball, physics.player2, rng)
            input2 = ai2.compute_action(physics.player2, physics.ball, physics.player1, rng)
            touched_ground = physics.run_engine_for_next_frame([input1, input2], rng)
            if touched_ground:
                break

        assert touched_ground, "Ball should eventually touch ground within 1000 frames"


class TestRegistry:
    def test_get_builtin(self):
        ai = get_ai("builtin")
        assert isinstance(ai, BuiltinAI)

    def test_register_custom(self):
        class DummyAI:
            def compute_action(self, player, ball, opponent, rng):
                return UserInput()

            def reset(self, rng):
                pass

        register_ai("dummy", DummyAI)
        ai = get_ai("dummy")
        assert isinstance(ai, DummyAI)

    def test_unknown_ai_raises(self):
        import pytest

        with pytest.raises(KeyError, match="Unknown AI"):
            get_ai("nonexistent")
