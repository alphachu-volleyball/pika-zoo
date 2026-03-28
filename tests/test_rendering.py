"""Tests for the rendering module."""

# ruff: noqa: E402
import os

import numpy as np
import pytest

# Skip all rendering tests if no display available (CI environment)
pytestmark = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None,
    reason="No display available for pygame",
)

from pika_zoo.env import env
from pika_zoo.rendering.renderer import SCREEN_HEIGHT, SCREEN_WIDTH
from pika_zoo.rendering.sprites import get_player_sprite_index


class TestSprites:
    def test_player_sprite_index(self):
        # State 0, frame 0 → index 0
        assert get_player_sprite_index(0, 0) == 0
        # State 0, frame 4 → index 4
        assert get_player_sprite_index(0, 4) == 4
        # State 1, frame 0 → index 5
        assert get_player_sprite_index(1, 0) == 5
        # State 3, frame 0 → index 15
        assert get_player_sprite_index(3, 0) == 15
        # State 4, frame 0 → index 17
        assert get_player_sprite_index(4, 0) == 17
        # State 5, frame 0 → index 18
        assert get_player_sprite_index(5, 0) == 18
        # State 6, frame 4 → index 27
        assert get_player_sprite_index(6, 4) == 27


class TestPygameRenderer:
    def test_rgb_array_output(self):
        e = env(render_mode="rgb_array")
        e.reset(seed=42)
        e.step({"player_1": 0, "player_2": 0})
        frame = e.render()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        assert frame.dtype == np.uint8
        e.close()

    def test_multiple_frames(self):
        e = env(render_mode="rgb_array")
        e.reset(seed=42)
        for _ in range(10):
            e.step({"player_1": 0, "player_2": 0})
            frame = e.render()
            assert frame is not None
            assert frame.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
        e.close()

    def test_no_render_without_mode(self):
        e = env(render_mode=None)
        e.reset(seed=42)
        e.step({"player_1": 0, "player_2": 0})
        assert e.render() is None

    def test_close_idempotent(self):
        e = env(render_mode="rgb_array")
        e.reset(seed=42)
        e.step({"player_1": 0, "player_2": 0})
        e.render()
        e.close()
        e.close()  # should not raise
