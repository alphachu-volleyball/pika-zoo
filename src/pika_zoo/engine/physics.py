"""
Pikachu Volleyball physics engine — 1:1 port of physics.js.

This module contains the pure physics simulation (no AI logic).
AI is separated into the pika_zoo.ai module.

Source: https://github.com/gorisanson/pikachu-volleyball/blob/main/src/resources/js/physics.js
"""

from __future__ import annotations

from numpy.random import Generator

from pika_zoo.engine.constants import (
    BALL_RADIUS,
    BALL_TOUCHING_GROUND_Y_COORD,
    GROUND_HALF_WIDTH,
    GROUND_WIDTH,
    INFINITE_LOOP_LIMIT,
    NET_PILLAR_HALF_WIDTH,
    NET_PILLAR_TOP_BOTTOM_Y_COORD,
    NET_PILLAR_TOP_TOP_Y_COORD,
    PLAYER_HALF_LENGTH,
    PLAYER_TOUCHING_GROUND_Y_COORD,
)
from pika_zoo.engine.rand import rand
from pika_zoo.engine.types import UserInput


def _trunc_div(a: int, b: int) -> int:
    """Integer division with truncation toward zero, matching JS ``(a / b) | 0``."""
    return int(a / b)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


class Player:
    """A player (Pikachu) in the game.

    Original: class Player in physics.js lines 124-215.
    """

    __slots__ = (
        "is_player2",
        "x",
        "y",
        "y_velocity",
        "is_collision_with_ball_happened",
        "state",
        "frame_number",
        "normal_status_arm_swing_direction",
        "delay_before_next_frame",
        "computer_boldness",
        "diving_direction",
        "lying_down_duration_left",
        "is_winner",
        "game_ended",
        "computer_where_to_stand_by",
        "sound",
    )

    def __init__(self, is_player2: bool, rng: Generator) -> None:
        self.is_player2: bool = is_player2

        # These persist across rounds
        self.diving_direction: int = 0
        self.lying_down_duration_left: int = -1
        self.is_winner: bool = False
        self.game_ended: bool = False
        self.computer_where_to_stand_by: int = 0
        self.sound: dict[str, bool] = {
            "pipikachu": False,
            "pika": False,
            "chu": False,
        }

        self.initialize_for_new_round(rng)

    def initialize_for_new_round(self, rng: Generator) -> None:
        """Reset player state for a new round."""
        self.x: int = 36 if not self.is_player2 else GROUND_WIDTH - 36
        self.y: int = PLAYER_TOUCHING_GROUND_Y_COORD
        self.y_velocity: int = 0
        self.is_collision_with_ball_happened: bool = False
        self.state: int = 0
        self.frame_number: int = 0
        self.normal_status_arm_swing_direction: int = 1
        self.delay_before_next_frame: int = 0
        self.computer_boldness: int = rand(rng) % 5


# ---------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------


class Ball:
    """The ball in the game.

    Original: class Ball in physics.js lines 226-290.
    """

    __slots__ = (
        "x",
        "y",
        "x_velocity",
        "y_velocity",
        "punch_effect_radius",
        "is_power_hit",
        "expected_landing_point_x",
        "rotation",
        "fine_rotation",
        "punch_effect_x",
        "punch_effect_y",
        "previous_x",
        "previous_previous_x",
        "previous_y",
        "previous_previous_y",
        "sound",
    )

    def __init__(self, is_player2_serve: bool) -> None:
        # These persist across rounds
        self.expected_landing_point_x: int = 0
        self.rotation: int = 0
        self.fine_rotation: int = 0
        self.punch_effect_x: int = 0
        self.punch_effect_y: int = 0
        self.previous_x: int = 0
        self.previous_previous_x: int = 0
        self.previous_y: int = 0
        self.previous_previous_y: int = 0
        self.sound: dict[str, bool] = {
            "power_hit": False,
            "ball_touches_ground": False,
        }

        self.initialize_for_new_round(is_player2_serve)

    def initialize_for_new_round(self, is_player2_serve: bool) -> None:
        """Reset ball state for a new round."""
        self.x: int = 56 if not is_player2_serve else GROUND_WIDTH - 56
        self.y: int = 0
        self.x_velocity: int = 0
        self.y_velocity: int = 1
        self.punch_effect_radius: int = 0
        self.is_power_hit: bool = False


# ---------------------------------------------------------------------------
# PikaPhysics — the main physics pack
# ---------------------------------------------------------------------------


class PikaPhysics:
    """Pack of physical objects (players + ball) with the physics engine.

    Original: class PikaPhysics in physics.js lines 70-97.
    """

    def __init__(self, rng: Generator) -> None:
        self.player1 = Player(is_player2=False, rng=rng)
        self.player2 = Player(is_player2=True, rng=rng)
        self.ball = Ball(is_player2_serve=False)

    def run_engine_for_next_frame(self, user_inputs: list[UserInput], rng: Generator) -> bool:
        """Run one frame of physics. Returns True if ball touches ground.

        Original: runEngineForNextFrame / physicsEngine in physics.js lines 303-371.
        """
        return _physics_engine(self.player1, self.player2, self.ball, user_inputs, rng)


# ---------------------------------------------------------------------------
# Physics engine functions
# ---------------------------------------------------------------------------


def _physics_engine(
    player1: Player,
    player2: Player,
    ball: Ball,
    user_inputs: list[UserInput],
    rng: Generator,
) -> bool:
    """The main physics engine — one frame of simulation.

    Original: physicsEngine in physics.js lines 303-371.
    NOTE: The ``if (player.isComputer)`` branch is removed.
    AI input is decided externally before calling this function.
    """
    is_ball_touching_ground = _process_collision_between_ball_and_world(ball)

    for i in range(2):
        if i == 0:
            player = player1
            the_other_player = player2
        else:
            player = player2
            the_other_player = player1

        _calculate_expected_landing_point_x(ball)

        _process_player_movement(player, user_inputs[i], the_other_player, ball)

    for i in range(2):
        player = player1 if i == 0 else player2

        is_happened = _is_collision_between_ball_and_player(ball, player.x, player.y)
        if is_happened:
            if not player.is_collision_with_ball_happened:
                _process_collision_between_ball_and_player(ball, player.x, user_inputs[i], player.state, rng)
                player.is_collision_with_ball_happened = True
        else:
            player.is_collision_with_ball_happened = False

    return is_ball_touching_ground


# ---------------------------------------------------------------------------
# Collision: ball vs world
# ---------------------------------------------------------------------------


def _process_collision_between_ball_and_world(ball: Ball) -> bool:
    """Process collision between ball and world boundaries.

    Original: processCollisionBetweenBallAndWorldAndSetBallPosition
    in physics.js lines 398-486.
    Returns True if ball touches ground.
    """
    # Track previous positions for trailing effect
    ball.previous_previous_x = ball.previous_x
    ball.previous_previous_y = ball.previous_y
    ball.previous_x = ball.x
    ball.previous_y = ball.y

    # Rotation: "(ball.xVelocity / 2) | 0" — use _trunc_div for toward-zero truncation
    future_fine_rotation = ball.fine_rotation + _trunc_div(ball.x_velocity, 2)
    if future_fine_rotation < 0:
        future_fine_rotation += 50
    elif future_fine_rotation > 50:
        future_fine_rotation += -50
    ball.fine_rotation = future_fine_rotation
    ball.rotation = _trunc_div(ball.fine_rotation, 10)

    # X boundary collision
    future_ball_x = ball.x + ball.x_velocity
    if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
        ball.x_velocity = -ball.x_velocity

    # Y upper boundary
    future_ball_y = ball.y + ball.y_velocity
    if future_ball_y < 0:
        ball.y_velocity = 1

    # Net collision
    if abs(ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
        if ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
            if ball.y_velocity > 0:
                ball.y_velocity = -ball.y_velocity
        else:
            if ball.x < GROUND_HALF_WIDTH:
                ball.x_velocity = -abs(ball.x_velocity)
            else:
                ball.x_velocity = abs(ball.x_velocity)

    # Ground collision
    future_ball_y = ball.y + ball.y_velocity
    if future_ball_y > BALL_TOUCHING_GROUND_Y_COORD:
        ball.sound["ball_touches_ground"] = True
        ball.y_velocity = -ball.y_velocity
        ball.punch_effect_x = ball.x
        ball.y = BALL_TOUCHING_GROUND_Y_COORD
        ball.punch_effect_radius = BALL_RADIUS
        ball.punch_effect_y = BALL_TOUCHING_GROUND_Y_COORD + BALL_RADIUS
        return True

    ball.y = future_ball_y
    ball.x = ball.x + ball.x_velocity
    ball.y_velocity += 1

    return False


# ---------------------------------------------------------------------------
# Collision detection: ball vs player
# ---------------------------------------------------------------------------


def _is_collision_between_ball_and_player(ball: Ball, player_x: int, player_y: int) -> bool:
    """Check if collision between ball and player happened.

    Original: isCollisionBetweenBallAndPlayerHappened in physics.js lines 381-390.
    """
    if abs(ball.x - player_x) <= PLAYER_HALF_LENGTH:
        if abs(ball.y - player_y) <= PLAYER_HALF_LENGTH:
            return True
    return False


# ---------------------------------------------------------------------------
# Collision response: ball vs player
# ---------------------------------------------------------------------------


def _process_collision_between_ball_and_player(
    ball: Ball,
    player_x: int,
    user_input: UserInput,
    player_state: int,
    rng: Generator,
) -> None:
    """Process collision between ball and player — set ball velocity.

    Original: processCollisionBetweenBallAndPlayer in physics.js lines 678-731.
    """
    # Ball x velocity based on distance from player center.
    # Math.abs ensures positive values, so _trunc_div == // here.
    if ball.x < player_x:
        ball.x_velocity = -(abs(ball.x - player_x) // 3)
    elif ball.x > player_x:
        ball.x_velocity = abs(ball.x - player_x) // 3

    # If ball velocity x is 0, randomly choose one of -1, 0, 1
    if ball.x_velocity == 0:
        ball.x_velocity = (rand(rng) % 3) - 1

    ball_abs_y_velocity = abs(ball.y_velocity)
    ball.y_velocity = -ball_abs_y_velocity

    if ball_abs_y_velocity < 15:
        ball.y_velocity = -15

    # Power hit (player state 2: jumping and power hitting)
    if player_state == 2:
        if ball.x < GROUND_HALF_WIDTH:
            ball.x_velocity = (abs(user_input.x_direction) + 1) * 10
        else:
            ball.x_velocity = -(abs(user_input.x_direction) + 1) * 10
        ball.punch_effect_x = ball.x
        ball.punch_effect_y = ball.y

        ball.y_velocity = abs(ball.y_velocity) * user_input.y_direction * 2
        ball.punch_effect_radius = BALL_RADIUS
        ball.sound["power_hit"] = True

        ball.is_power_hit = True
    else:
        ball.is_power_hit = False

    _calculate_expected_landing_point_x(ball)


# ---------------------------------------------------------------------------
# Expected landing point calculation
# ---------------------------------------------------------------------------


def _calculate_expected_landing_point_x(ball: Ball) -> None:
    """Calculate x coordinate of expected landing point of the ball.

    Uses a copy-ball lookahead simulation.
    Original: calculateExpectedLandingPointXFor in physics.js lines 738-788.
    """
    copy_x = ball.x
    copy_y = ball.y
    copy_x_velocity = ball.x_velocity
    copy_y_velocity = ball.y_velocity

    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_x = copy_x_velocity + copy_x
        if future_copy_x < BALL_RADIUS or future_copy_x > GROUND_WIDTH:
            copy_x_velocity = -copy_x_velocity
        if copy_y + copy_y_velocity < 0:
            copy_y_velocity = 1

        # Net collision for copy ball
        if abs(copy_x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_y > NET_PILLAR_TOP_TOP_Y_COORD:
            # NOTE: original uses < instead of <= for NET_PILLAR_TOP_BOTTOM_Y_COORD
            # (possible original author mistake, preserved for accuracy)
            if copy_y < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if copy_y_velocity > 0:
                    copy_y_velocity = -copy_y_velocity
            else:
                if copy_x < GROUND_HALF_WIDTH:
                    copy_x_velocity = -abs(copy_x_velocity)
                else:
                    copy_x_velocity = abs(copy_x_velocity)

        copy_y = copy_y + copy_y_velocity
        if copy_y > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= INFINITE_LOOP_LIMIT:
            break
        copy_x = copy_x + copy_x_velocity
        copy_y_velocity += 1

    ball.expected_landing_point_x = copy_x


# ---------------------------------------------------------------------------
# Player movement
# ---------------------------------------------------------------------------


def _process_player_movement(
    player: Player,
    user_input: UserInput,
    the_other_player: Player,
    ball: Ball,
) -> None:
    """Process player movement and set player position.

    Original: processPlayerMovementAndSetPlayerPosition
    in physics.js lines 496-648.
    NOTE: The ``if (player.isComputer)`` call to ``letComputerDecideUserInput``
    is removed. AI input is decided externally.
    """
    # If player is lying down, don't move
    if player.state == 4:
        player.lying_down_duration_left += -1
        if player.lying_down_duration_left < -1:
            player.state = 0
        return

    # X-direction movement
    player_velocity_x = 0
    if player.state < 5:
        if player.state < 3:
            player_velocity_x = user_input.x_direction * 6
        else:
            # player.state == 3: diving
            player_velocity_x = player.diving_direction * 8

    player.x = player.x + player_velocity_x

    # X-direction world boundary
    if not player.is_player2:
        if player.x < PLAYER_HALF_LENGTH:
            player.x = PLAYER_HALF_LENGTH
        elif player.x > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
    else:
        if player.x < GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH:
            player.x = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH
        elif player.x > GROUND_WIDTH - PLAYER_HALF_LENGTH:
            player.x = GROUND_WIDTH - PLAYER_HALF_LENGTH

    # Jump
    if player.state < 3 and user_input.y_direction == -1 and player.y == PLAYER_TOUCHING_GROUND_Y_COORD:
        player.y_velocity = -16
        player.state = 1
        player.frame_number = 0
        player.sound["chu"] = True

    # Gravity
    player.y = player.y + player.y_velocity
    if player.y < PLAYER_TOUCHING_GROUND_Y_COORD:
        player.y_velocity += 1
    elif player.y > PLAYER_TOUCHING_GROUND_Y_COORD:
        # Landing
        player.y_velocity = 0
        player.y = PLAYER_TOUCHING_GROUND_Y_COORD
        player.frame_number = 0
        if player.state == 3:
            # Diving → lying down
            player.state = 4
            player.frame_number = 0
            player.lying_down_duration_left = 3
        else:
            player.state = 0

    # Power hit / diving trigger
    if user_input.power_hit == 1:
        if player.state == 1:
            # Jumping → power hit
            player.delay_before_next_frame = 5
            player.frame_number = 0
            player.state = 2
            player.sound["pika"] = True
        elif player.state == 0 and user_input.x_direction != 0:
            # Standing → dive
            player.state = 3
            player.frame_number = 0
            player.diving_direction = user_input.x_direction
            player.y_velocity = -5
            player.sound["chu"] = True

    # Animation frame updates
    if player.state == 1:
        player.frame_number = (player.frame_number + 1) % 3
    elif player.state == 2:
        if player.delay_before_next_frame < 1:
            player.frame_number += 1
            if player.frame_number > 4:
                player.frame_number = 0
                player.state = 1
        else:
            player.delay_before_next_frame -= 1
    elif player.state == 0:
        player.delay_before_next_frame += 1
        if player.delay_before_next_frame > 3:
            player.delay_before_next_frame = 0
            future_frame_number = player.frame_number + player.normal_status_arm_swing_direction
            if future_frame_number < 0 or future_frame_number > 4:
                player.normal_status_arm_swing_direction = -player.normal_status_arm_swing_direction
            player.frame_number = player.frame_number + player.normal_status_arm_swing_direction

    # Game end animation
    if player.game_ended:
        if player.state == 0:
            if player.is_winner:
                player.state = 5
                player.sound["pipikachu"] = True
            else:
                player.state = 6
            player.delay_before_next_frame = 0
            player.frame_number = 0
        _process_game_end_frame(player)


def _process_game_end_frame(player: Player) -> None:
    """Process game end frame (winner/loser motions).

    Original: processGameEndFrameFor in physics.js lines 656-664.
    """
    if player.game_ended and player.frame_number < 4:
        player.delay_before_next_frame += 1
        if player.delay_before_next_frame > 4:
            player.delay_before_next_frame = 0
            player.frame_number += 1
