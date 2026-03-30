"""
Ball path prediction and helper functions for duckll AI.

Ported from duckll/pikachu-volleyball, physics.js lines 782-1170.
Source: https://github.com/duckll/pikachu-volleyball
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from pika_zoo.engine.physics import Ball, Player


@dataclass
class BallPathEntry:
    """One frame in the ball's predicted path."""

    x: int
    y: int
    x_velocity: int
    y_velocity: int
    predict: list[list[PredictEntry]] = field(default_factory=list)


@dataclass
class PredictEntry:
    """One frame in a power-hit prediction sub-path."""

    x: int
    y: int
    x_velocity: int
    y_velocity: int


def compute_ball_path(ball: Ball) -> list[BallPathEntry]:
    """Compute ball path with per-frame 6-direction power-hit predictions.

    Original: calculateExpectedLandingPointXFor in physics.js lines 782-860.
    Also sets ball.expected_landing_point_x as a side effect.
    """
    copy_x = ball.x
    copy_y = ball.y
    copy_x_velocity = ball.x_velocity
    copy_y_velocity = ball.y_velocity

    path: list[BallPathEntry] = []

    # Initial predictions at current position
    initial_predicts = _compute_all_predicts(copy_x, copy_y, copy_x_velocity, copy_y_velocity)
    path.append(BallPathEntry(copy_x, copy_y, copy_x_velocity, copy_y_velocity, initial_predicts))

    loop_counter = 0
    while True:
        loop_counter += 1

        future_x = copy_x_velocity + copy_x
        if future_x < BALL_RADIUS or future_x > GROUND_WIDTH:
            copy_x_velocity = -copy_x_velocity
        if copy_y + copy_y_velocity < 0:
            copy_y_velocity = 1

        # Net collision
        if abs(copy_x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_y > NET_PILLAR_TOP_TOP_Y_COORD:
            if copy_y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
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

        if len(path) < 34:
            predicts = _compute_all_predicts(copy_x, copy_y, copy_x_velocity, copy_y_velocity)
        else:
            predicts = []
        path.append(BallPathEntry(copy_x, copy_y, copy_x_velocity, copy_y_velocity, predicts))

    ball.expected_landing_point_x = copy_x
    return path


def _compute_all_predicts(x: int, y: int, x_vel: int, y_vel: int) -> list[list[PredictEntry]]:
    """Compute 6 power-hit prediction paths for all direction combinations.

    Direction indices:
        0: y=-1, x=0 (up smash)
        1: y=-1, x=1 (up-forward smash)
        2: y=0,  x=0 (drop)
        3: y=0,  x=1 (forward)
        4: y=1,  x=0 (thunder)
        5: y=1,  x=1 (diagonal-down)
    """
    predicts: list[list[PredictEntry]] = []
    for y_direction in range(-1, 2):
        for x_direction in range(0, 2):
            predicts.append(_simulate_power_hit(x, y, x_vel, y_vel, x_direction, y_direction))
    return predicts


def _simulate_power_hit(
    x: int, y: int, x_vel: int, y_vel: int, x_direction: int, y_direction: int
) -> list[PredictEntry]:
    """Simulate a power hit from given position and return the resulting path.

    Original: calculateExpectedLandingPointXwithpredict in physics.js lines 862-934.
    """
    copy_x = x
    copy_y = y

    ball_abs_y_velocity = abs(y_vel)
    copy_y_velocity = -ball_abs_y_velocity
    if ball_abs_y_velocity < 15:
        copy_y_velocity = -15

    if copy_x < GROUND_HALF_WIDTH:
        copy_x_velocity = (abs(x_direction) + 1) * 10
    else:
        copy_x_velocity = -(abs(x_direction) + 1) * 10

    copy_y_velocity = abs(copy_y_velocity) * y_direction * 2

    loop_counter = 0
    tmp_path: list[PredictEntry] = []
    while True:
        loop_counter += 1

        future_x = copy_x_velocity + copy_x
        if future_x < BALL_RADIUS or future_x > GROUND_WIDTH:
            copy_x_velocity = -copy_x_velocity
        if copy_y + copy_y_velocity < 0:
            copy_y_velocity = 1

        # Net collision
        if abs(copy_x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_y > NET_PILLAR_TOP_TOP_Y_COORD:
            if copy_y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if copy_y_velocity > 0:
                    copy_y_velocity = -copy_y_velocity
            else:
                if copy_x < GROUND_HALF_WIDTH:
                    copy_x_velocity = -abs(copy_x_velocity)
                else:
                    copy_x_velocity = abs(copy_x_velocity)

        copy_y = copy_y + copy_y_velocity
        if copy_y > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= INFINITE_LOOP_LIMIT:
            tmp_path.append(PredictEntry(copy_x, copy_y, copy_x_velocity, copy_y_velocity))
            break
        copy_x = copy_x + copy_x_velocity
        copy_y_velocity += 1
        tmp_path.append(PredictEntry(copy_x, copy_y, copy_x_velocity, copy_y_velocity))

    return tmp_path


# ---------------------------------------------------------------------------
# Player Y prediction helpers
# ---------------------------------------------------------------------------


def player_y_predict(player: Player, frame: int) -> int:
    """Predict player Y position after 'frame' frames.

    Original: playerYpredict in physics.js lines 936-959.
    """
    if player.state == 0:
        total = -16
        speed = -15
        for _ in range(frame):
            total += speed
            speed += 1
        if total > 0:
            return PLAYER_TOUCHING_GROUND_Y_COORD
        return PLAYER_TOUCHING_GROUND_Y_COORD + total
    else:
        speed = player.y_velocity
        real_y = player.y + speed
        for _ in range(frame):
            speed += 1
            real_y += speed
        if real_y > PLAYER_TOUCHING_GROUND_Y_COORD:
            return PLAYER_TOUCHING_GROUND_Y_COORD
        return real_y


def player_y_predict_jump(player: Player, frame: int) -> int:
    """Predict player Y with double-jump (speed wraps at 17 → -16).

    Original: playerYpredictJump in physics.js lines 962-988.
    """
    if player.state == 0:
        total = -16
        speed = -15
        for _ in range(frame):
            total += speed
            speed += 1
            if speed == 17:
                speed = -16
        return PLAYER_TOUCHING_GROUND_Y_COORD + total
    else:
        speed = player.y_velocity
        real_y = player.y + speed
        for _ in range(frame):
            speed += 1
            if speed == 17:
                speed = -16
            real_y += speed
        if real_y > 244:
            real_y = 244
        return real_y


def other_player_y_predict(player: Player, frame: int) -> int:
    """Predict opponent's Y position (accounts for left-right asymmetry).

    Original: otherPlayerYpredict in physics.js lines 991-1018.
    """
    if player.is_player2:
        return player_y_predict_jump(player, frame)
    else:
        if player.state == 0:
            total = 0
            speed = -16
            for _ in range(frame):
                total += speed
                speed += 1
                if speed == 17:
                    speed = -16
            return PLAYER_TOUCHING_GROUND_Y_COORD + total
        else:
            speed = player.y_velocity
            real_y = player.y
            for _ in range(frame):
                real_y += speed
                speed += 1
                if speed == 17:
                    speed = -16
            return real_y


def sameside(player: Player, ball_x: int) -> bool:
    """Check if ball_x is on the same side as the player.

    Original: sameside in physics.js lines 1028-1035.
    """
    if ball_x == GROUND_HALF_WIDTH:
        return True
    if player.is_player2:
        return ball_x > GROUND_HALF_WIDTH
    return ball_x < GROUND_HALF_WIDTH


def samesideloss(player: Player, ball_x: int) -> bool:
    """Check if ball_x would cause a loss for the player (lands on their side).

    Original: samesideloss in physics.js lines 1044-1048.
    """
    if player.is_player2:
        return ball_x >= GROUND_HALF_WIDTH
    return ball_x < GROUND_HALF_WIDTH


def canblock(player: Player, predict: list[PredictEntry]) -> bool:
    """Check if player can block this predicted trajectory (jumping).

    Original: canblock in physics.js lines 1057-1079.
    """
    first = True
    for frame in range(len(predict)):
        if sameside(player, predict[frame].x):
            if first:
                if predict[frame].y > NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    return False
                first = False
            if abs(predict[frame].x - GROUND_HALF_WIDTH) > 60:
                return False
            if abs(other_player_y_predict(player, frame) - predict[frame].y) <= PLAYER_HALF_LENGTH:
                return True
    return False


def canblock_predict(player: Player, predict: list[PredictEntry]) -> bool:
    """Check if player can block (standing, predicts jump).

    Original: canblockPredict in physics.js lines 1088-1112.
    """
    first = True
    player_y = 244 - 32
    speed = -16
    for frame in range(len(predict)):
        player_y += speed
        if sameside(player, predict[frame].x):
            if first:
                if predict[frame].y > NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    return False
                first = False
                speed += 1
                continue
            if abs(predict[frame].x - GROUND_HALF_WIDTH) > 60:
                return False
            if predict[frame].y > player_y:
                return True
        speed += 1
    return False


def cantouch(player: Player, copyball: BallPathEntry, frame: int) -> bool:
    """Check if player can touch the ball at given frame.

    Original: cantouch in physics.js lines 1122-1152.
    """
    if copyball.y < 76:
        return False
    if sameside(player, copyball.x) and abs(copyball.x - player.x) <= 6 * frame + PLAYER_HALF_LENGTH + 6:
        needframe = -1
        if player.state > 0:
            needframe = 16 - player.y_velocity - (1 if player.is_player2 else 0)
            if frame < needframe:
                return abs(copyball.y - other_player_y_predict(player, frame)) <= PLAYER_HALF_LENGTH
        top = PLAYER_TOUCHING_GROUND_Y_COORD - PLAYER_HALF_LENGTH
        speed = -16
        for _ in range(frame - needframe):
            top += speed
            speed += 1
            if speed > 0:
                break
        return copyball.y >= top
    return False


def cancatch(player: Player, path: list[BallPathEntry]) -> bool:
    """Check if player can catch the ball along its path (ground level).

    Original: cancatch in physics.js lines 1159-1170.
    """
    for frame in range(len(path)):
        copyball = path[frame]
        if (
            copyball.y >= PLAYER_TOUCHING_GROUND_Y_COORD - PLAYER_HALF_LENGTH
            and abs(player.x - copyball.x) <= PLAYER_HALF_LENGTH + 6 * frame + 6
        ):
            return True
    return False
