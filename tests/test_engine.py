"""Tests for the physics engine."""

import numpy as np

from pika_zoo.engine.constants import (
    BALL_TOUCHING_GROUND_Y_COORD,
    GROUND_HALF_WIDTH,
    GROUND_WIDTH,
    PLAYER_HALF_LENGTH,
    PLAYER_TOUCHING_GROUND_Y_COORD,
)
from pika_zoo.engine.physics import Ball, PikaPhysics, Player
from pika_zoo.engine.rand import rand
from pika_zoo.engine.types import PlayerState, UserInput


class TestConstants:
    def test_ground_half_width(self):
        assert GROUND_HALF_WIDTH == GROUND_WIDTH // 2 == 216


class TestRand:
    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            value = rand(rng)
            assert 0 <= value < 32768

    def test_deterministic(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        for _ in range(100):
            assert rand(rng1) == rand(rng2)


class TestUserInput:
    def test_default_values(self):
        ui = UserInput()
        assert ui.x_direction == 0
        assert ui.y_direction == 0
        assert ui.power_hit == 0

    def test_reset(self):
        ui = UserInput()
        ui.x_direction = 1
        ui.y_direction = -1
        ui.power_hit = 1
        ui.reset()
        assert ui.x_direction == 0
        assert ui.y_direction == 0
        assert ui.power_hit == 0


class TestPlayerState:
    def test_values(self):
        assert PlayerState.NORMAL == 0
        assert PlayerState.JUMPING == 1
        assert PlayerState.JUMPING_POWER_HIT == 2
        assert PlayerState.DIVING == 3
        assert PlayerState.LYING_DOWN == 4
        assert PlayerState.WIN == 5
        assert PlayerState.LOSE == 6


class TestPlayer:
    def test_initial_positions(self):
        rng = np.random.default_rng(42)
        p1 = Player(is_player2=False, rng=rng)
        p2 = Player(is_player2=True, rng=rng)
        assert p1.x == 36
        assert p2.x == GROUND_WIDTH - 36
        assert p1.y == PLAYER_TOUCHING_GROUND_Y_COORD
        assert p2.y == PLAYER_TOUCHING_GROUND_Y_COORD

    def test_initial_state(self):
        rng = np.random.default_rng(42)
        p = Player(is_player2=False, rng=rng)
        assert p.state == 0
        assert p.y_velocity == 0
        assert not p.is_collision_with_ball_happened
        assert 0 <= p.computer_boldness < 5

    def test_reinitialize(self):
        rng = np.random.default_rng(42)
        p = Player(is_player2=False, rng=rng)
        p.x = 999
        p.state = 3
        p.initialize_for_new_round(rng)
        assert p.x == 36
        assert p.state == 0


class TestBall:
    def test_initial_position_player1_serve(self):
        ball = Ball(is_player2_serve=False)
        assert ball.x == 56
        assert ball.y == 0
        assert ball.x_velocity == 0
        assert ball.y_velocity == 1

    def test_initial_position_player2_serve(self):
        ball = Ball(is_player2_serve=True)
        assert ball.x == GROUND_WIDTH - 56

    def test_reinitialize(self):
        ball = Ball(is_player2_serve=False)
        ball.x = 999
        ball.y = 999
        ball.initialize_for_new_round(is_player2_serve=True)
        assert ball.x == GROUND_WIDTH - 56
        assert ball.y == 0


class TestPikaPhysics:
    def test_creation(self):
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        assert physics.player1.x == 36
        assert physics.player2.x == GROUND_WIDTH - 36

    def test_ball_falls_with_no_input(self):
        """Ball should fall due to gravity when no player input."""
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        inputs = [UserInput(), UserInput()]

        touched_ground = False

        for _ in range(300):
            touched_ground = physics.run_engine_for_next_frame(inputs, rng)
            if touched_ground:
                break

        assert touched_ground
        assert physics.ball.y == BALL_TOUCHING_GROUND_Y_COORD

    def test_deterministic_simulation(self):
        """Same seed should produce identical simulation."""
        results1 = _run_simulation(seed=42, frames=100)
        results2 = _run_simulation(seed=42, frames=100)
        assert results1 == results2

    def test_different_seeds_differ(self):
        """Different seeds with ball-player collision should diverge due to rand()."""
        # With no input and no collision, rand() is only called during init (boldness).
        # Ball free-fall is deterministic regardless of seed.
        # So we compare boldness values which depend on the seed.
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        p1 = PikaPhysics(rng1)
        p2 = PikaPhysics(rng2)
        boldness1 = (p1.player1.computer_boldness, p1.player2.computer_boldness)
        boldness2 = (p2.player1.computer_boldness, p2.player2.computer_boldness)
        assert boldness1 != boldness2

    def test_player_jump(self):
        """Player should jump when y_direction is -1."""
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        jump_input = UserInput()
        jump_input.y_direction = -1
        no_input = UserInput()

        physics.run_engine_for_next_frame([jump_input, no_input], rng)
        # y_velocity is set to -16, then gravity adds +1 in the same frame → -15
        assert physics.player1.y_velocity == -15
        assert physics.player1.state == PlayerState.JUMPING

    def test_player_boundary(self):
        """Player 1 should not cross the net."""
        rng = np.random.default_rng(42)
        physics = PikaPhysics(rng)
        right_input = UserInput()
        right_input.x_direction = 1
        no_input = UserInput()

        for _ in range(100):
            physics.run_engine_for_next_frame([right_input, no_input], rng)

        assert physics.player1.x <= GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH


def _run_simulation(seed: int, frames: int) -> list[tuple[int, int, int, int]]:
    """Run a simulation and return ball positions per frame."""
    rng = np.random.default_rng(seed)
    physics = PikaPhysics(rng)
    inputs = [UserInput(), UserInput()]
    results = []
    for _ in range(frames):
        physics.run_engine_for_next_frame(inputs, rng)
        results.append((physics.ball.x, physics.ball.y, physics.ball.x_velocity, physics.ball.y_velocity))
    return results
