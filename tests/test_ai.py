"""Tests for the AI module."""

import sys
import types

import numpy as np
import pytest

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.ai.registry import get_ai, register_ai
from pika_zoo.engine.physics import PikaPhysics
from pika_zoo.engine.types import UserInput
from pika_zoo.env.observations import OBSERVATION_SIZE


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


class TestSB3ModelPolicy:
    def test_frame_stack_passes_stacked_observation(self, monkeypatch, tmp_path):
        fake_model = _install_fake_sb3(monkeypatch)

        from pika_zoo.ai.sb3_adapter import SB3ModelPolicy

        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"fake")
        policy = SB3ModelPolicy(model_path, agent="player_1", frame_stack=4, observation_normalized=False)

        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        policy.compute_action(physics.player1, physics.ball, physics.player2, rng)
        physics.ball.x += 1
        policy.compute_action(physics.player1, physics.ball, physics.player2, rng)

        assert fake_model.observations[0].shape == (4, OBSERVATION_SIZE)
        np.testing.assert_array_equal(fake_model.observations[0][0], fake_model.observations[0][-1])
        assert fake_model.observations[1].shape == (4, OBSERVATION_SIZE)
        np.testing.assert_array_equal(fake_model.observations[1][0], fake_model.observations[0][0])
        np.testing.assert_array_equal(fake_model.observations[1][-1], fake_model.observations[0][-1] + _ball_x_delta())

    def test_frame_stack_one_preserves_flat_observation(self, monkeypatch, tmp_path):
        fake_model = _install_fake_sb3(monkeypatch)

        from pika_zoo.ai.sb3_adapter import SB3ModelPolicy

        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"fake")
        policy = SB3ModelPolicy(model_path, agent="player_1", frame_stack=1)

        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        policy.compute_action(physics.player1, physics.ball, physics.player2, rng)

        assert fake_model.observations[0].shape == (OBSERVATION_SIZE,)

    def test_invalid_frame_stack(self, monkeypatch, tmp_path):
        _install_fake_sb3(monkeypatch)

        from pika_zoo.ai.sb3_adapter import SB3ModelPolicy

        model_path = tmp_path / "model.zip"
        model_path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="frame_stack"):
            SB3ModelPolicy(model_path, agent="player_1", frame_stack=0)


class TestModelConfig:
    def test_load_model_dir_accepts_frame_stack(self, tmp_path):
        from pika_zoo.scripts.play import _load_model_dir

        (tmp_path / "model.zip").write_bytes(b"fake")
        (tmp_path / "model.json").write_text(
            '{"side": "both", "frame_stack": 4, "name": "alphachu-v2", "unknown": true}'
        )

        model_path, config = _load_model_dir(tmp_path)

        assert model_path == tmp_path / "model.zip"
        assert config["side"] == "both"
        assert config["frame_stack"] == 4
        assert "name" not in config
        assert "unknown" not in config


class _FakeModel:
    def __init__(self) -> None:
        self.observations = []

    def predict(self, obs, deterministic=True):
        self.observations.append(obs.copy())
        return 0, None


def _install_fake_sb3(monkeypatch):
    fake_model = _FakeModel()
    module = types.ModuleType("stable_baselines3")

    class FakePPO:
        @staticmethod
        def load(path, device="cpu"):
            return fake_model

    module.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", module)
    return fake_model


def _ball_x_delta() -> np.ndarray:
    delta = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    delta[26] = 1.0
    return delta
