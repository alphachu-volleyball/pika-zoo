"""
Original built-in AI from gorisanson/pikachu-volleyball.

This AI has intentional bugs — it doesn't account for net side bounces
in ``_expected_landing_point_x_when_power_hit``, making it beatable.

Source: https://github.com/gorisanson/pikachu-volleyball/blob/main/src/resources/js/physics.js
Lines 803-1032 (letComputerDecideUserInput, decideWhetherInputPowerHit,
expectedLandingPointXWhenPowerHit)
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
    PLAYER_LENGTH,
)
from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.rand import rand
from pika_zoo.engine.types import UserInput


class BuiltinAI:
    """The original gorisanson AI with intentional bugs.

    Satisfies the AIPolicy protocol.
    """

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        """Decide user input for the computer-controlled player.

        Original: letComputerDecideUserInput in physics.js lines 803-895.
        """
        _calculate_expected_landing_point_x(ball)
        user_input = UserInput()
        _let_computer_decide_user_input(player, ball, opponent, user_input, rng)
        return user_input

    def reset(self, rng: Generator) -> None:
        """No per-round state needed for builtin AI."""


def _let_computer_decide_user_input(
    player: Player,
    ball: Ball,
    the_other_player: Player,
    user_input: UserInput,
    rng: Generator,
) -> None:
    """Computer controls its player.

    Original: letComputerDecideUserInput in physics.js lines 803-895.
    """
    user_input.x_direction = 0
    user_input.y_direction = 0
    user_input.power_hit = 0

    virtual_expected_landing_point_x = ball.expected_landing_point_x
    if abs(ball.x - player.x) > 100 and abs(ball.x_velocity) < player.computer_boldness + 5:
        left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
        if (
            ball.expected_landing_point_x <= left_boundary
            or ball.expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
        ) and player.computer_where_to_stand_by == 0:
            virtual_expected_landing_point_x = left_boundary + (GROUND_HALF_WIDTH // 2)

    if abs(virtual_expected_landing_point_x - player.x) > player.computer_boldness + 8:
        if player.x < virtual_expected_landing_point_x:
            user_input.x_direction = 1
        else:
            user_input.x_direction = -1
    elif rand(rng) % 20 == 0:
        player.computer_where_to_stand_by = rand(rng) % 2

    if player.state == 0:
        if (
            abs(ball.x_velocity) < player.computer_boldness + 3
            and abs(ball.x - player.x) < PLAYER_HALF_LENGTH
            and ball.y > -36
            and ball.y < 10 * player.computer_boldness + 84
            and ball.y_velocity > 0
        ):
            user_input.y_direction = -1

        left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
        right_boundary = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH
        if (
            ball.expected_landing_point_x > left_boundary
            and ball.expected_landing_point_x < right_boundary
            and abs(ball.x - player.x) > player.computer_boldness * 5 + PLAYER_LENGTH
            and ball.x > left_boundary
            and ball.x < right_boundary
            and ball.y > 174
        ):
            # Dive!
            user_input.power_hit = 1
            if player.x < ball.x:
                user_input.x_direction = 1
            else:
                user_input.x_direction = -1

    elif player.state == 1 or player.state == 2:
        if abs(ball.x - player.x) > 8:
            if player.x < ball.x:
                user_input.x_direction = 1
            else:
                user_input.x_direction = -1
        if abs(ball.x - player.x) < 48 and abs(ball.y - player.y) < 48:
            will_input_power_hit = _decide_whether_input_power_hit(player, ball, the_other_player, user_input, rng)
            if will_input_power_hit:
                user_input.power_hit = 1
                if abs(the_other_player.x - player.x) < 80 and user_input.y_direction != -1:
                    user_input.y_direction = -1


def _decide_whether_input_power_hit(
    player: Player,
    ball: Ball,
    the_other_player: Player,
    user_input: UserInput,
    rng: Generator,
) -> bool:
    """Decide whether to input power hit and set direction.

    Original: decideWhetherInputPowerHit in physics.js lines 908-953.
    """
    if rand(rng) % 2 == 0:
        for x_direction in range(1, -1, -1):
            for y_direction in range(-1, 2):
                expected_x = _expected_landing_point_x_when_power_hit(x_direction, y_direction, ball)
                if (
                    expected_x <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                ) and abs(expected_x - the_other_player.x) > PLAYER_LENGTH:
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    else:
        for x_direction in range(1, -1, -1):
            for y_direction in range(1, -2, -1):
                expected_x = _expected_landing_point_x_when_power_hit(x_direction, y_direction, ball)
                if (
                    expected_x <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                ) and abs(expected_x - the_other_player.x) > PLAYER_LENGTH:
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
    return False


def _calculate_expected_landing_point_x(ball: Ball) -> None:
    """Calculate x coordinate of expected landing point of the ball.

    Uses a copy-ball lookahead simulation. Moved from physics.py so that
    this expensive calculation only runs when BuiltinAI needs it.

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


def _expected_landing_point_x_when_power_hit(
    user_input_x_direction: int,
    user_input_y_direction: int,
    ball: Ball,
) -> int:
    """Calculate expected landing point x when power hit.

    Original: expectedLandingPointXWhenPowerHit in physics.js lines 965-1032.

    NOTE: This function has an intentional bug — it does not properly handle
    net side bounces (lines 998-1005 of original). This makes the AI beatable.
    """
    copy_x = ball.x
    copy_y = ball.y

    if copy_x < GROUND_HALF_WIDTH:
        copy_x_velocity = (abs(user_input_x_direction) + 1) * 10
    else:
        copy_x_velocity = -(abs(user_input_x_direction) + 1) * 10

    copy_y_velocity = abs(ball.y_velocity) * user_input_y_direction * 2

    loop_counter = 0
    while True:
        loop_counter += 1

        future_copy_x = copy_x + copy_x_velocity
        if future_copy_x < BALL_RADIUS or future_copy_x > GROUND_WIDTH:
            copy_x_velocity = -copy_x_velocity
        if copy_y + copy_y_velocity < 0:
            copy_y_velocity = 1

        # INTENTIONAL BUG: Only checks yVelocity, not net pillar sides.
        # This causes the AI to occasionally power-hit balls that bounce back
        # off the net pillar, because it doesn't anticipate the side bounce.
        if abs(copy_x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_y > NET_PILLAR_TOP_TOP_Y_COORD:
            if copy_y_velocity > 0:
                copy_y_velocity = -copy_y_velocity

        copy_y = copy_y + copy_y_velocity
        if copy_y > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= INFINITE_LOOP_LIMIT:
            return copy_x
        copy_x = copy_x + copy_x_velocity
        copy_y_velocity += 1
