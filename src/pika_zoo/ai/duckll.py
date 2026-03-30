"""
duckll's enhanced AI for Pikachu Volleyball.

A significantly stronger AI than the original builtin, featuring ball path
prediction with 6-direction lookahead, multi-touch "fancy" combos,
anti-block system, pre-programmed serve formulas, and configurable difficulty.

Source: https://github.com/duckll/pikachu-volleyball
Credits: duckll (AI), pxter7777 (serve machine), CBKM (testing)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

from numpy.random import Generator

from pika_zoo.ai.duckll_prediction import (
    BallPathEntry,
    canblock,
    canblock_predict,
    cancatch,
    cantouch,
    compute_ball_path,
    player_y_predict,
    player_y_predict_jump,
    sameside,
    samesideloss,
)
from pika_zoo.engine.constants import (
    GROUND_HALF_WIDTH,
    GROUND_WIDTH,
    NET_PILLAR_HALF_WIDTH,
    NET_PILLAR_TOP_BOTTOM_Y_COORD,
    PLAYER_HALF_LENGTH,
    PLAYER_LENGTH,
)
from pika_zoo.engine.physics import Ball, Player
from pika_zoo.engine.types import UserInput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DefenseMode(IntEnum):
    MID = 0
    MID_PLUS = 1
    MIRROR = 2
    PREDICT = 3
    CLOSE = 4


@dataclass(frozen=True)
class DuckllAIConfig:
    """Difficulty configuration for DuckllAI.

    Original: capability object + delay + defense in ui.js.
    """

    serve: bool = False
    fancy: bool = True
    block: bool = True
    diving: bool = True
    anti_block: bool = True
    early_ball: bool = False
    jump: bool = False
    delay: int = 0
    defense: DefenseMode = DefenseMode.CLOSE


# Preset difficulty levels from ui.js
# Parameter order: serve, fancy, block, diving, anti_block, early_ball, jump, delay, defense
DIFFICULTY_PRESETS: dict[int, DuckllAIConfig] = {
    0: DuckllAIConfig(False, False, False, False, False, False, True, 10, DefenseMode.MID),
    1: DuckllAIConfig(False, False, False, False, False, False, False, 10, DefenseMode.MID),
    2: DuckllAIConfig(False, False, True, True, False, False, False, 8, DefenseMode.MID_PLUS),
    3: DuckllAIConfig(False, False, True, True, True, False, False, 8, DefenseMode.PREDICT),
    4: DuckllAIConfig(False, False, True, True, True, False, False, 4, DefenseMode.MIRROR),
    5: DuckllAIConfig(False, False, True, True, True, False, False, 4, DefenseMode.MID_PLUS),
    6: DuckllAIConfig(False, False, True, True, True, False, True, 1, DefenseMode.CLOSE),
    7: DuckllAIConfig(False, False, True, True, False, True, False, 1, DefenseMode.CLOSE),
    8: DuckllAIConfig(False, False, True, True, True, False, False, 1, DefenseMode.PREDICT),
    9: DuckllAIConfig(False, True, True, True, True, False, False, 0, DefenseMode.MIRROR),
    10: DuckllAIConfig(False, True, True, True, True, False, False, 0, DefenseMode.CLOSE),
}


# ---------------------------------------------------------------------------
# Internal AI state
# ---------------------------------------------------------------------------


@dataclass
class _PlayerAIState:
    """Per-player state tracked across frames within a round."""

    tactics: int = 0  # 0=rally, 3=serve
    goodtime: int = -1
    attack_x: int = 0
    direction: int = 0
    fancy: bool = False
    second_attack: int = -1
    second_x: int = 0
    second_jump: bool = False
    juke: bool = False
    cooldown: int = 0
    front_frame: int = -1


# ---------------------------------------------------------------------------
# Serve Machine
# ---------------------------------------------------------------------------


class _ActionType(IntEnum):
    WAIT = 0
    FORWARD = 1
    FORWARD_UP = 2
    UP = 3
    BACKWARD_UP = 4
    BACKWARD = 5
    BACKWARD_DOWN = 6
    DOWN = 7
    FORWARD_DOWN = 8
    FORWARD_SMASH = 9
    FORWARD_UP_SMASH = 10
    UP_SMASH = 11
    BACKWARD_UP_SMASH = 12
    BACKWARD_SMASH = 13
    BACKWARD_DOWN_SMASH = 14
    DOWN_SMASH = 15
    FORWARD_DOWN_SMASH = 16


_A = _ActionType

# Serve formulas: list of (action, frames) tuples
_P1_FORMULAS: list[list[tuple[int, int]]] = [
    # 0. Break net
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 26), (_A.FORWARD_UP, 4), (_A.FORWARD_DOWN_SMASH, 1)],
    # 1. Break net (fake, flat)
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 30), (_A.FORWARD_UP, 1), (_A.FORWARD_SMASH, 2)],
    # 2. Head thunder
    [
        (_A.FORWARD, 1),
        (_A.WAIT, 20),
        (_A.FORWARD, 11),
        (_A.FORWARD_UP, 15),
        (_A.DOWN_SMASH, 1),
        (_A.FORWARD_DOWN_SMASH, 4),
    ],
    # 3. Head thunder (fake, flat)
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 11), (_A.FORWARD_UP, 15), (_A.FORWARD_SMASH, 1)],
    # 4. Net V smash
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 31), (_A.FORWARD_UP_SMASH, 3), (_A.WAIT, 16), (_A.DOWN_SMASH, 5)],
    # 5. Net R smash
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 31), (_A.FORWARD_UP_SMASH, 3), (_A.WAIT, 16), (_A.UP_SMASH, 1)],
    # 6. Net G smash
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 31), (_A.FORWARD_UP_SMASH, 3), (_A.WAIT, 16), (_A.FORWARD_SMASH, 1)],
    # 7. Net dodge
    [(_A.FORWARD, 1), (_A.WAIT, 20), (_A.FORWARD, 31), (_A.FORWARD_UP_SMASH, 3), (_A.WAIT, 16), (_A.BACKWARD, 1)],
    # 8. Tail thunder
    [(_A.FORWARD, 7), (_A.WAIT, 14), (_A.FORWARD, 11), (_A.FORWARD_UP, 15), (_A.DOWN_SMASH, 5)],
    # 9. Tail thunder (fake, flat)
    [(_A.FORWARD, 7), (_A.WAIT, 14), (_A.FORWARD, 11), (_A.FORWARD_UP, 15), (_A.FORWARD_SMASH, 1)],
]

_P2_FORMULAS: list[list[tuple[int, int]]] = [
    _P1_FORMULAS[0][:],  # 0. Break net
    _P1_FORMULAS[1][:],  # 1. Break net (fake, flat)
    _P1_FORMULAS[2][:],  # 2. Head thunder
    _P1_FORMULAS[3][:],  # 3. Head thunder (fake, flat)
    _P1_FORMULAS[4][:],  # 4. Net thunder
    _P1_FORMULAS[6][:],  # 5. Net thunder (fake, flat)
    # 6. Tail thunder
    [(_A.FORWARD, 7), (_A.WAIT, 14), (_A.FORWARD, 11), (_A.FORWARD_UP, 2), (_A.WAIT, 13), (_A.FORWARD_DOWN_SMASH, 5)],
    # 7. Tail thunder (fake, flat)
    [(_A.FORWARD, 7), (_A.WAIT, 14), (_A.FORWARD, 11), (_A.FORWARD_UP, 2), (_A.WAIT, 13), (_A.FORWARD_SMASH, 1)],
]

_NO_SERVE: list[tuple[int, int]] = [
    (_A.FORWARD_UP, 1),
    (_A.WAIT, 11),
    (_A.FORWARD_UP_SMASH, 1),
]


class _ServeMachine:
    """Pre-programmed serve formula system.

    Original: ServeMachine class in physics.js lines 3139-3260.
    """

    def __init__(self, is_player2: bool) -> None:
        self.is_player2 = is_player2
        self.skill_count = 8 if is_player2 else 10
        self.skill_list = list(range(self.skill_count))
        self.rand_serve_index = self.skill_count - 1
        self.using_full_skill = -1
        self.phase = 0
        self.frames_left = 0
        self.action = _ActionType.WAIT

    def shuffle(self) -> None:
        random.shuffle(self.skill_list)
        self.rand_serve_index = 0

    def choose_next_skill(self) -> None:
        while True:
            self.using_full_skill = self.skill_list[self.rand_serve_index]
            self.rand_serve_index += 1
            self.shuffle()
            # All skills are available (no filtering)
            return

    def initialize_for_new_round(self) -> None:
        self.choose_next_skill()
        self.frames_left = 0
        self.phase = 0

    def execute_move(
        self,
        is_player2: bool,
        user_input: UserInput,
        serve_enabled: bool,
    ) -> None:
        self._get_next_action(is_player2, serve_enabled)
        self.frames_left -= 1
        a = self.action
        if a == _A.FORWARD:
            user_input.x_direction = 1
        elif a == _A.FORWARD_UP:
            user_input.x_direction = 1
            user_input.y_direction = -1
        elif a == _A.FORWARD_DOWN_SMASH:
            user_input.x_direction = 1
            user_input.y_direction = 1
            user_input.power_hit = 1
        elif a == _A.FORWARD_SMASH:
            user_input.x_direction = 1
            user_input.power_hit = 1
        elif a == _A.DOWN_SMASH:
            user_input.y_direction = 1
            user_input.power_hit = 1
        elif a == _A.UP_SMASH:
            user_input.y_direction = -1
            user_input.power_hit = 1
        elif a == _A.FORWARD_UP_SMASH:
            user_input.x_direction = 1
            user_input.y_direction = -1
            user_input.power_hit = 1
        elif a == _A.BACKWARD:
            user_input.x_direction = -1

        if is_player2:
            user_input.x_direction = -user_input.x_direction

    def _get_next_action(self, is_player2: bool, serve_enabled: bool) -> None:
        if self.frames_left == 0:
            if not serve_enabled:
                if self.phase < len(_NO_SERVE):
                    self.action, self.frames_left = _NO_SERVE[self.phase]
                else:
                    self.action = _A.WAIT
                    self.frames_left = -1000
            elif not is_player2:
                formula = _P1_FORMULAS[self.using_full_skill]
                if self.phase < len(formula):
                    self.action, self.frames_left = formula[self.phase]
                else:
                    self.action = _A.WAIT
                    self.frames_left = -1000
            else:
                formula = _P2_FORMULAS[self.using_full_skill]
                if self.phase < len(formula):
                    self.action, self.frames_left = formula[self.phase]
                else:
                    self.action = _A.WAIT
                    self.frames_left = -1000
            self.phase += 1

        if self.frames_left == 0:
            self._get_next_action(is_player2, serve_enabled)


# ---------------------------------------------------------------------------
# DuckllAI
# ---------------------------------------------------------------------------


class DuckllAI:
    """duckll's enhanced AI — satisfies the AIPolicy protocol.

    Significantly stronger than BuiltinAI with full ball path prediction,
    fancy multi-touch combos, anti-block system, and serve formulas.
    """

    def __init__(self, config: DuckllAIConfig | None = None, preset: int | None = None) -> None:
        if preset is not None:
            if preset not in DIFFICULTY_PRESETS:
                available = ", ".join(str(k) for k in sorted(DIFFICULTY_PRESETS.keys()))
                raise ValueError(f"Unknown preset: {preset!r}. Available: {available}")
            self._config = DIFFICULTY_PRESETS[preset]
        elif config is not None:
            self._config = config
        else:
            self._config = DIFFICULTY_PRESETS[10]

        self._state = _PlayerAIState()
        self._serve_machine: _ServeMachine | None = None
        self._is_player2: bool | None = None
        self._frame_counter = 0
        self._is_player2_serve: bool | None = None
        self._true_rng = random.Random()

    def _true_rand(self) -> int:
        """Non-deterministic random, matching JS Math.random() usage."""
        return self._true_rng.randint(0, 32767)

    def compute_action(
        self,
        player: Player,
        ball: Ball,
        opponent: Player,
        rng: Generator,
    ) -> UserInput:
        """Decide user input for the computer-controlled player."""
        # Lazy initialization
        if self._is_player2 is None:
            self._is_player2 = player.is_player2
            self._serve_machine = _ServeMachine(player.is_player2)

        # Detect new round (ball at spawn position)
        if ball.y_velocity == 1 and ball.x_velocity == 0 and ball.y == 0:
            if self._frame_counter > 0:
                # New round started
                self._state = _PlayerAIState()
                self._serve_machine.initialize_for_new_round()
            self._frame_counter = 0
            self._is_player2_serve = ball.x > GROUND_HALF_WIDTH

        # Compute ball path prediction
        path = compute_ball_path(ball)

        user_input = UserInput()
        self._let_ai_decide(player, ball, opponent, user_input, path)

        self._frame_counter += 1
        return user_input

    def reset(self, rng: Generator) -> None:
        """Reset for a new round."""
        self._state = _PlayerAIState()
        self._frame_counter = 0
        if self._serve_machine is not None:
            self._serve_machine.initialize_for_new_round()

    def _let_ai_decide(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
    ) -> None:
        """Main AI decision function.

        Original: letAIDecideUserInput in physics.js lines 1187-3000.
        """
        cfg = self._config
        st = self._state

        # Serve detection
        if (
            self._frame_counter < 5
            and self._is_player2_serve is not None
            and (player.is_player2 == self._is_player2_serve)
        ):
            st.tactics = 3

        if st.tactics == 0:
            self._rally(player, ball, other, ui, path)

        if st.tactics == 3:
            assert self._serve_machine is not None
            self._serve_machine.execute_move(player.is_player2, ui, cfg.serve)
            if self._serve_machine.frames_left < -1000:
                st.tactics = 0

    # -------------------------------------------------------------------
    # Rally logic
    # -------------------------------------------------------------------

    def _rally(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
    ) -> None:
        cfg = self._config
        st = self._state

        virtual_x: int | None = None
        maychange = False
        diving = False

        # Check collision — cancel attack
        if player.is_collision_with_ball_happened and st.goodtime > -1 and st.second_attack > -1:
            st.goodtime = -1
            st.second_attack = -1
            st.fancy = False
            st.juke = False
            st.second_jump = False
            st.front_frame = -1

        # Detect maychange
        keep_touch = True
        for frame in range(len(path)):
            copyball = path[frame]
            if not sameside(other, copyball.x) and cantouch(player, copyball, frame):
                break
            if cantouch(other, copyball, frame):
                maychange = True
                if other.is_collision_with_ball_happened and other.state == 2 and keep_touch:
                    maychange = False
            else:
                keep_touch = False

        # Second jump
        if player.y_velocity == 16 and player.y == 244 and st.second_jump:
            ui.y_direction = -1
            st.second_jump = False

        if maychange and st.goodtime < 0 and st.second_attack < 0:
            # Predict defense
            virtual_x = self._predict_defense(player, ball, other, ui, path)
        else:
            st.cooldown += 1
            # Normal defense
            if st.cooldown > cfg.delay:
                virtual_x = ball.expected_landing_point_x

            if len(path) > 24:
                virtual_x = ball.expected_landing_point_x + (-6 if player.is_player2 else 6) * (len(path) - 24)

            # Jumping defense
            if player.state > 0 and player.y_velocity < 16 and not player.is_collision_with_ball_happened:
                for frame in range(len(path)):
                    copyball = path[frame]
                    if (
                        sameside(player, copyball.x)
                        and abs(player_y_predict(player, frame) - copyball.y) <= PLAYER_HALF_LENGTH
                        and abs(player.x - copyball.x) <= 6 * frame + PLAYER_HALF_LENGTH + 6
                    ):
                        virtual_x = copyball.x
                        break

            # Remove touch
            if (
                player.state > 0
                and player.y_velocity == ball.y_velocity - 1
                and player.is_collision_with_ball_happened
                and abs(player_y_predict(player, 0) - ball.y) <= PLAYER_HALF_LENGTH
                and abs(player.x - ball.x) <= PLAYER_HALF_LENGTH + 6
            ):
                cansmash = True
                if len(path) > 1:
                    predictball = path[1].predict[5]
                    if samesideloss(other, predictball[-1].x) and len(predictball) < 10:
                        cansmash = True
                    else:
                        cansmash = False

                if (
                    ball.x_velocity != 0
                    and (player.is_player2 == (ball.x_velocity < 0))
                    and ball.y_velocity < 13
                    and not cansmash
                ):
                    virtual_x = ball.x + (1 if player.is_player2 else -1) * (PLAYER_HALF_LENGTH - 3)
                else:
                    if abs(player.x - ball.x) > PLAYER_HALF_LENGTH:
                        virtual_x = player.x
                    else:
                        if abs(ball.x_velocity) < 2:
                            virtual_x = GROUND_WIDTH if player.is_player2 else 0
                        else:
                            virtual_x = GROUND_WIDTH if ball.x_velocity < 0 else 0

            # High ball emergency
            if ball.y_velocity > 60 and ball.y < 244 - 32:
                virtual_x = GROUND_HALF_WIDTH
                ui.y_direction = -1

            # Attack planning
            if (
                st.cooldown > cfg.delay
                and (st.second_attack < 0 or (st.second_attack > -1 and (player.state == 0 or player.y_velocity == 16)))
                and player.state < 3
            ):
                self._plan_attack(player, ball, other, ui, path)

            # Diving
            if (
                cfg.diving
                and player.state < 3
                and st.goodtime < 0
                and st.second_attack < 0
                and len(path) < 12
                and (ball.expected_landing_point_x > player.x) == (ball.x > player.x)
                and sameside(player, ball.expected_landing_point_x)
                and abs(player.x - ball.expected_landing_point_x) > len(path) * 6 + PLAYER_HALF_LENGTH
                and not cancatch(player, path)
            ):
                diving = True

        # Override position for planned attack
        if st.goodtime > -1:
            virtual_x = st.attack_x
        if st.front_frame > 0:
            virtual_x = GROUND_HALF_WIDTH
            st.front_frame -= 1
        if st.second_attack > -1 and st.goodtime < 0:
            virtual_x = st.second_x

        # Move toward target
        if virtual_x is not None and abs(player.x - virtual_x) > 3:
            if player.x < virtual_x:
                ui.x_direction = 1
            else:
                ui.x_direction = -1

        # Diving action
        if diving:
            ui.power_hit = 1

        # Super up smash
        if (
            st.second_attack > 0
            and player.state == 2
            and not player.is_collision_with_ball_happened
            and st.direction < 2
            and ball.y_velocity < -27
            and abs(player_y_predict(player, 0) - ball.y) <= PLAYER_HALF_LENGTH
        ):
            ui.x_direction = 0
            ui.y_direction = -1
            st.goodtime = -1
            st.second_attack = -1
            st.fancy = False
            st.juke = False
            st.second_jump = False
            st.front_frame = -1

        # Prevent self-kill
        if (
            st.second_attack > -1
            and player.state == 2
            and not player.is_collision_with_ball_happened
            and ui.x_direction == 0
            and ((player.is_player2 and ball.x_velocity == 20) or (not player.is_player2 and ball.x_velocity == -20))
            and abs(player.x - GROUND_HALF_WIDTH) < PLAYER_HALF_LENGTH + 25
            and abs(player.x - ball.x) <= PLAYER_HALF_LENGTH
            and abs(player_y_predict(player, 0) - ball.y) <= PLAYER_HALF_LENGTH
        ):
            ui.x_direction = 1 if player.is_player2 else -1

        # Attack execution
        if st.goodtime == 0 or (player.state > 0 and player.y_velocity < 16 and st.second_attack == 0):
            ui.power_hit = 1
            if not st.fancy:
                copyball = path[0]
                if st.second_attack == 0:
                    self._second_attack_decision(player, ball, other, ui, path, copyball)
                else:
                    self._first_attack_execution(player, ball, other, ui, path, copyball)
            else:
                # Cancel fancy if second X unreachable
                if st.direction == 0 and abs(player.x - st.second_x) > 6 * st.second_attack + PLAYER_LENGTH - 9:
                    ui.power_hit = 0
                    st.second_attack = -1
                st.fancy = False

            # Apply direction to input
            self._apply_direction(ui, st.direction, player.is_player2)

        st.goodtime -= 1
        st.second_attack -= 1

    # -------------------------------------------------------------------
    # Predict defense
    # -------------------------------------------------------------------

    def _predict_defense(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
    ) -> int:
        cfg = self._config
        st = self._state

        short_len = 1000
        short_x = ball.expected_landing_point_x
        short_direct = 4
        st.cooldown = 0

        keep_touch = True
        for frame in range(min(len(path), 32)):
            copyball = path[frame]
            if cantouch(other, copyball, frame):
                if other.is_collision_with_ball_happened and other.state == 2 and keep_touch:
                    continue
                for direct in range(6):
                    predict = copyball.predict[direct]
                    # Thunder detection
                    if direct > 3 and len(predict) > 3 and predict[3].y_velocity < 3 and sameside(other, predict[1].x):
                        short_len = 0
                        short_x = GROUND_HALF_WIDTH
                        break
                    else:
                        # Normal
                        if sameside(player, predict[-1].x):
                            if len(predict) < short_len:
                                short_len = len(predict)
                                short_x = predict[-1].x
                                short_direct = direct
                            if len(predict) == short_len:
                                short_x = int((short_x + predict[-1].x) / 2)
                                if direct == 0 or direct == 3:
                                    short_direct = direct

                if other.state > 0 and not other.is_collision_with_ball_happened:
                    break
            else:
                keep_touch = False
            if short_len == 0:
                break

        # Defense mode overrides
        defense = cfg.defense
        if defense == DefenseMode.MID:
            short_x = GROUND_HALF_WIDTH + (108 if player.is_player2 else -108)
        if defense == DefenseMode.MID_PLUS:
            sign = 1 if player.is_player2 else -1
            short_x = int(GROUND_HALF_WIDTH + sign * 54 + sign * abs(other.x - GROUND_HALF_WIDTH) / 2)
        if defense == DefenseMode.MIRROR:
            sign = 1 if player.is_player2 else -1
            short_x = GROUND_HALF_WIDTH + sign * abs(other.x - GROUND_HALF_WIDTH)
        # PREDICT (defense == 3) uses the calculated short_x as-is
        if (
            defense == DefenseMode.CLOSE
            and abs(short_x - GROUND_HALF_WIDTH) > 44
            and (
                player.state == 0
                or not (
                    (short_direct == 0 or short_direct == 3)
                    and player.y_velocity < 0
                    and abs(short_x - GROUND_HALF_WIDTH) > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
                )
            )
        ):
            short_x = GROUND_HALF_WIDTH + (44 if player.is_player2 else -44)

        # Jump when opponent jumps
        if cfg.jump and other.y_velocity == -10 and other.state == 1:
            ui.y_direction = -1
            short_x = GROUND_HALF_WIDTH

        return short_x

    # -------------------------------------------------------------------
    # Attack planning
    # -------------------------------------------------------------------

    def _plan_attack(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
    ) -> None:
        st = self._state
        short_path = 100

        # --- Jumping ball ---
        if (
            st.goodtime != 0
            and not st.fancy
            and player.state > 0
            and player.y_velocity < 16
            and abs(player_y_predict(player, 0) - ball.y) <= PLAYER_HALF_LENGTH
            and abs(player.x - ball.x) <= PLAYER_HALF_LENGTH + 6
            and sameside(player, ball.x)
            and not player.is_collision_with_ball_happened
        ):
            copyball = path[0]
            short_path = self._jumping_attack(player, ball, other, ui, path, copyball, short_path)

        if st.goodtime < 0:
            # --- 0-sec attack ---
            if (
                player.state == 0
                and abs(player.y - 16 - ball.y) <= PLAYER_HALF_LENGTH
                and abs(player.x - ball.x) <= PLAYER_HALF_LENGTH + 6
                and sameside(player, ball.x)
                and not player.is_collision_with_ball_happened
            ):
                copyball = path[0]
                self._zero_sec_attack(player, ball, other, ui, path, copyball, short_path)
            else:
                # --- Normal attack (back, front, normal) ---
                self._ground_attack(player, ball, other, ui, path, short_path)

    def _jumping_attack(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        copyball: BallPathEntry,
        short_path: int,
    ) -> int:
        """Handle attack while player is airborne near the ball."""
        cfg = self._config
        st = self._state

        # Fancy attacks from jumping
        flat_x = 0
        drop_x = 0
        for direct in range(6):
            if not cfg.fancy:
                break
            if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                continue
            predict = copyball.predict[direct]
            for pf in range(1, len(predict) - 1):
                pball = predict[pf]
                if not sameside(player, pball.x):
                    break
                if (
                    direct == 2
                    and abs(pball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
                    and pball.y > NET_PILLAR_TOP_BOTTOM_Y_COORD
                ):
                    drop_x = pball.x
                if direct == 3 and abs(pball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH:
                    flat_x = pball.x

                # Fancy flat jump
                if (
                    self._true_rand() % 10 < 7
                    and direct == 3
                    and player_y_predict_jump(player, pf + 1) == 228
                    and abs(pball.y - player_y_predict_jump(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict_jump(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 10000 <= short_path
                ):
                    short_path = pf - 10000
                    st.goodtime = 0
                    shift = -PLAYER_HALF_LENGTH + 9 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 9
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    ui.y_direction = -1
                    st.fancy = True
                    st.second_x = pball.x + (1 if player.is_player2 else -1) * int(
                        abs(pball.x - GROUND_HALF_WIDTH) / 10
                    )
                    st.second_attack = pf + 1
                    st.second_jump = True

                # Fancy jump
                if (
                    self._true_rand() % 10 < 7
                    and direct == 2
                    and player_y_predict_jump(player, pf + 1) != 244
                    and abs(pball.y - player_y_predict_jump(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict_jump(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                    and not (abs(pball.x - GROUND_HALF_WIDTH) > 20 or pball.y > 176)
                ):
                    short_path = pf - 100
                    st.goodtime = 0
                    shift = -PLAYER_HALF_LENGTH + 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 3
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.second_attack = pf + 1
                    st.second_jump = True

                # Fancy drop
                if (
                    self._true_rand() % 10 < 7
                    and direct == 2
                    and drop_x != 0
                    and abs(player.x - drop_x) > PLAYER_HALF_LENGTH
                    and abs(pball.y - player_y_predict(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                ):
                    short_path = pf - 100
                    st.goodtime = 0
                    shift = -PLAYER_HALF_LENGTH + 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 3
                    st.attack_x = copyball.x + shift
                    st.direction = direct if (player.is_player2 == (copyball.x > pball.x)) else -direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 3 if player.is_player2 else PLAYER_HALF_LENGTH - 3
                    st.second_x = pball.x + sign
                    if abs(st.second_x - drop_x) <= PLAYER_HALF_LENGTH:
                        st.second_x = drop_x + (
                            PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                        )
                    st.second_attack = pf + 1
                    st.second_jump = False
                    break

                # Fancy flat
                if (
                    self._true_rand() % 10 < 7
                    and direct == 3
                    and pf > 3
                    and abs(pball.y - player_y_predict(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 1000 <= short_path
                ):
                    short_path = pf - 1000
                    st.goodtime = 0
                    shift = -PLAYER_HALF_LENGTH + 9 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 9
                    st.attack_x = copyball.x + shift
                    st.direction = direct if ((pball.x < copyball.x) == player.is_player2) else -direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 3 if player.is_player2 else PLAYER_HALF_LENGTH - 3
                    st.second_x = pball.x + sign
                    if abs(st.second_x - flat_x) <= PLAYER_HALF_LENGTH:
                        st.second_x = flat_x + (
                            PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                        )
                    st.second_attack = pf + 1
                    st.second_jump = False
                    break

                # Generic fancy
                if (
                    self._true_rand() % 10 < 7
                    and abs(pball.y - player_y_predict(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                ):
                    short_path = pf - 100
                    st.goodtime = 0
                    shift_val = 9 if direct % 2 == 1 else 3
                    shift = -PLAYER_HALF_LENGTH + shift_val if pball.x < copyball.x else PLAYER_HALF_LENGTH - shift_val
                    st.attack_x = copyball.x + shift
                    st.direction = direct if ((pball.x < copyball.x) == player.is_player2) else -direct
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.second_attack = pf + 1
                    st.second_jump = False
                    break

        # Normal jump attack
        if short_path > -1:
            for direct in range(6):
                if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                    continue
                predict = copyball.predict[direct]
                if (samesideloss(other, predict[-1].x) or (player.state == 2 and len(predict) > 20)) and len(
                    predict
                ) <= short_path:
                    short_path = len(predict)
                    st.goodtime = 0
                    st.attack_x = copyball.x
                    st.direction = direct

        return short_path

    def _zero_sec_attack(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        copyball: BallPathEntry,
        short_path: int,
    ) -> None:
        """Handle 0-sec attack (ball within reach while standing)."""
        cfg = self._config
        st = self._state

        # Fancy
        for direct in range(2):
            if not cfg.fancy:
                break
            if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                continue
            if direct == 0 and ball.y_velocity > 60:
                continue
            predict = copyball.predict[direct]
            for pf in range(1, len(predict) - 1):
                pball = predict[pf]
                if not sameside(player, pball.x):
                    break
                if (
                    self._true_rand() % 10 < 7
                    and abs(pball.y - player_y_predict(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                ):
                    short_path = pf - 100
                    st.goodtime = 0
                    shift_val = 9 if direct % 2 == 1 else 3
                    shift = -PLAYER_HALF_LENGTH + shift_val if pball.x < copyball.x else PLAYER_HALF_LENGTH - shift_val
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.juke = False
                    if direct == 1 and abs(pball.x - GROUND_HALF_WIDTH) < 40:
                        st.second_x = pball.x + (PLAYER_HALF_LENGTH if player.is_player2 else -PLAYER_HALF_LENGTH)
                        st.juke = True
                    st.second_attack = pf + 1
                    break

        # Jump or miss
        if (
            st.second_attack < 0
            and short_path > -1
            and sameside(player, ball.expected_landing_point_x)
            and abs(player.x - ball.expected_landing_point_x) > len(path) * 6 + PLAYER_HALF_LENGTH
            and not cancatch(player, path)
        ):
            for direct in range(2):
                if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                    continue
                predict = copyball.predict[direct]
                if (samesideloss(other, predict[-1].x) or len(predict) > 20) and (
                    len(predict) < short_path or (len(predict) == short_path and self._true_rand() % 2 < 1)
                ):
                    short_path = len(predict)
                    st.goodtime = 0
                    st.attack_x = copyball.x
                    st.direction = direct
                    ui.y_direction = -1

    def _ground_attack(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        short_path: int,
    ) -> None:
        """Handle ground-based normal attack (back, front, normal patterns)."""
        cfg = self._config
        st = self._state

        for frame in range(1, min(len(path), 33)):
            copyball = path[frame]

            # Back attack
            if (
                abs(ball.x - GROUND_HALF_WIDTH) > abs(player.x - GROUND_HALF_WIDTH)
                and sameside(player, ball.x)
                and player_y_predict(player, frame) != 244
                and (copyball.x_velocity == 0 or (copyball.x_velocity < 0) == player.is_player2)
                and abs(player_y_predict(player, frame) - copyball.y) <= PLAYER_HALF_LENGTH
                and sameside(player, copyball.x)
                and abs(player.x - copyball.x) <= 6 * frame + PLAYER_HALF_LENGTH
            ):
                # Back fancy flat
                flat_x = 0
                for direct in range(6):
                    if not cfg.fancy:
                        break
                    if short_path < 0 and st.direction == 4:
                        break
                    predict = copyball.predict[direct]
                    for pf in range(1, len(predict) - 1):
                        pball = predict[pf]
                        if not sameside(player, pball.x):
                            break
                        if direct == 3 and abs(pball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH:
                            flat_x = pball.x
                        if (
                            self._true_rand() % 10 < 2
                            and direct == 3
                            and frame > 1
                            and pf > 3
                            and (pball.x < copyball.x) == player.is_player2
                            and abs(pball.y - player_y_predict(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                            and abs(copyball.x - GROUND_HALF_WIDTH) > PLAYER_LENGTH
                            and abs(pball.x - copyball.x) <= 6 * pf + PLAYER_LENGTH
                            and pf - 1000 <= short_path
                        ):
                            short_path = pf - 1000
                            st.goodtime = frame
                            shift = -PLAYER_HALF_LENGTH - 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH + 3
                            st.attack_x = copyball.x + shift
                            st.front_frame = int((frame - abs(player.x - st.attack_x) / 6) / 2)
                            st.direction = -3
                            ui.y_direction = -1
                            st.fancy = True
                            sign = -PLAYER_HALF_LENGTH + 3 if player.is_player2 else PLAYER_HALF_LENGTH - 3
                            st.second_x = pball.x + sign
                            if abs(st.second_x - flat_x) <= PLAYER_HALF_LENGTH:
                                st.second_x = flat_x + (
                                    PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                                )
                            st.second_attack = frame + pf + 1
                            st.second_jump = False
                            break

            # Front attack
            if (
                frame > 20
                and abs(copyball.x - GROUND_HALF_WIDTH) < 61
                and player_y_predict(player, frame) != 244
                and (copyball.x_velocity == 0 or (copyball.x_velocity > 0) == player.is_player2)
                and abs(player_y_predict(player, frame) - copyball.y) <= PLAYER_HALF_LENGTH
                and sameside(player, copyball.x)
                and abs(player.x - copyball.x) <= 6 * frame + PLAYER_HALF_LENGTH
            ):
                for direct in (3, 5):
                    predict = copyball.predict[direct]
                    if samesideloss(other, predict[-1].x) and (len(predict) < short_path or len(predict) == short_path):
                        short_path = len(predict) - 1000
                        st.goodtime = frame
                        shift = PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                        st.attack_x = copyball.x + shift
                        st.direction = direct
                        st.fancy = True
                        st.second_attack = -1
                        ui.y_direction = -1

            # Normal attack
            if (
                player_y_predict(player, frame) != 244
                and abs(player_y_predict(player, frame) - copyball.y) <= PLAYER_HALF_LENGTH
                and abs(player_y_predict(player, frame - 1) - path[frame - 1].y) > PLAYER_HALF_LENGTH
                and sameside(player, copyball.x)
                and abs(player.x - copyball.x) <= 6 * frame + PLAYER_HALF_LENGTH
            ):
                self._normal_attack(player, ball, other, ui, path, copyball, frame, short_path)
                short_path = self._last_short_path  # Updated by _normal_attack

    def _normal_attack(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        copyball: BallPathEntry,
        frame: int,
        short_path: int,
    ) -> None:
        """Normal attack from ground with fancy/normal sub-decisions."""
        cfg = self._config
        st = self._state

        # Fancy attacks
        flat_x = 0
        drop_x = 0
        for direct in range(6):
            if not cfg.fancy:
                break
            if short_path < 0 and st.direction == 4:
                break
            predict = copyball.predict[direct]
            for pf in range(1, len(predict) - 1):
                pball = predict[pf]
                if not sameside(player, pball.x):
                    break
                if (
                    direct == 2
                    and abs(pball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH
                    and pball.y > NET_PILLAR_TOP_BOTTOM_Y_COORD
                ):
                    drop_x = pball.x
                if direct == 3 and abs(pball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH:
                    flat_x = pball.x

                # Normal fancy flat jump
                if (
                    self._true_rand() % 10 < 7
                    and direct == 3
                    and player_y_predict_jump(player, frame + pf + 1) == 228
                    and abs(pball.y - player_y_predict_jump(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict_jump(player, frame + pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - copyball.x) <= 6 * pf + PLAYER_LENGTH + 6
                    and pf - 10000 <= short_path
                ):
                    short_path = pf - 10000
                    st.goodtime = frame
                    shift = -PLAYER_HALF_LENGTH + 9 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 9
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    ui.y_direction = -1
                    st.fancy = True
                    st.second_x = pball.x + (1 if player.is_player2 else -1) * int(
                        abs(pball.x - GROUND_HALF_WIDTH) / 10
                    )
                    st.second_attack = frame + pf + 1
                    st.second_jump = True

                # Normal fancy jump
                drop_second = False
                if (
                    self._true_rand() % 10 < 7
                    and direct == 2
                    and player_y_predict_jump(player, frame + pf + 1) != 244
                    and abs(pball.y - player_y_predict_jump(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict_jump(player, frame + pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - copyball.x) <= 6 * pf + PLAYER_LENGTH
                    and pf - 100 <= short_path
                    and not (abs(pball.x - GROUND_HALF_WIDTH) > 20 or pball.y > 176)
                ):
                    short_path = pf - 100
                    st.goodtime = frame
                    shift = -PLAYER_HALF_LENGTH + 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 3
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = PLAYER_HALF_LENGTH - 9 if player.is_player2 else -PLAYER_HALF_LENGTH + 9
                    st.second_x = pball.x + sign
                    st.second_attack = frame + pf + 1
                    st.second_jump = True
                    drop_second = True

                # Normal fancy drop
                if (
                    self._true_rand() % 10 < 7
                    and direct == 2
                    and drop_x != 0
                    and frame > 15
                    and abs(pball.y - player_y_predict(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(pball.x - copyball.x) <= 6 * pf + PLAYER_LENGTH
                    and pf - 100 <= short_path
                ):
                    short_path = pf - 100
                    st.goodtime = frame
                    shift = -PLAYER_HALF_LENGTH + 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 3
                    st.attack_x = copyball.x + shift
                    st.direction = direct if (player.is_player2 == (copyball.x > pball.x)) else -direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 3 if player.is_player2 else PLAYER_HALF_LENGTH - 3
                    st.second_x = pball.x + sign
                    if abs(st.second_x - drop_x) <= PLAYER_HALF_LENGTH:
                        st.second_x = drop_x + (
                            PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                        )
                    st.second_attack = frame + pf + 1
                    st.second_jump = False
                    break

                # Normal fancy flat
                if (
                    self._true_rand() % 10 < 7
                    and direct == 3
                    and frame > 1
                    and pf > 3
                    and abs(pball.y - player_y_predict(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(copyball.x - GROUND_HALF_WIDTH) < GROUND_HALF_WIDTH - 6
                    and abs(pball.x - copyball.x) <= 6 * pf + PLAYER_LENGTH
                    and pf - 1000 <= short_path
                ):
                    short_path = pf - 1000
                    st.goodtime = frame
                    shift = -PLAYER_HALF_LENGTH + 9 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 9
                    st.attack_x = copyball.x + shift
                    st.direction = direct if ((pball.x < copyball.x) == player.is_player2) else -direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 3 if player.is_player2 else PLAYER_HALF_LENGTH - 3
                    st.second_x = pball.x + sign
                    if abs(st.second_x - flat_x) <= PLAYER_HALF_LENGTH:
                        st.second_x = flat_x + (
                            PLAYER_HALF_LENGTH + 4 if player.is_player2 else -PLAYER_HALF_LENGTH - 4
                        )
                    st.second_attack = frame + pf + 1
                    st.second_jump = False
                    break

                # Generic normal fancy
                if (
                    (self._true_rand() % 10 < 6 or drop_second)
                    and abs(pball.y - player_y_predict(player, frame + pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict(player, frame + pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - copyball.x)
                    <= 6 * pf
                    + PLAYER_LENGTH
                    + (6 if (direct % 2 == 1 or (samesideloss(other, predict[-1].x) and direct > 0)) else 0)
                    and pf - 100 <= short_path
                ):
                    short_path = pf - 100
                    st.goodtime = frame
                    shift_val = 9 if direct % 2 == 1 else 3
                    shift = -PLAYER_HALF_LENGTH + shift_val if pball.x < copyball.x else PLAYER_HALF_LENGTH - shift_val
                    st.attack_x = copyball.x + shift
                    st.direction = direct if ((pball.x < copyball.x) == player.is_player2) else -direct
                    ui.y_direction = -1
                    st.fancy = True
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.juke = False
                    if direct == 1 and abs(pball.x - GROUND_HALF_WIDTH) < 40:
                        st.second_x = pball.x + (PLAYER_HALF_LENGTH if player.is_player2 else -PLAYER_HALF_LENGTH)
                        st.juke = True
                    st.second_attack = frame + pf + 1
                    st.second_jump = False
                    break

        # Normal (non-fancy) attack
        if short_path > -1:
            # Jump or miss
            if (
                st.second_attack < 0
                and samesideloss(player, ball.expected_landing_point_x)
                and abs(player.x - ball.expected_landing_point_x) > len(path) * 6 + PLAYER_HALF_LENGTH
                and not cancatch(player, path)
            ):
                for direct in range(6):
                    predict = copyball.predict[direct]
                    if (samesideloss(other, predict[-1].x) or len(predict) > 20) and (
                        len(predict) < short_path or (len(predict) == short_path and self._true_rand() % 2 < 1)
                    ):
                        short_path = len(predict)
                        st.goodtime = frame
                        st.attack_x = copyball.x
                        st.direction = direct
                        st.second_attack = -1
                        ui.y_direction = -1
            else:
                # Normal
                for direct in range(6):
                    # Skip up-forward
                    if direct == 1:
                        continue
                    # Early hit check
                    if (
                        not cfg.early_ball
                        and direct < 4
                        and player.state == 0
                        and frame < 10 + (self._true_rand() % 6)
                        and other.y_velocity > -1
                    ):
                        continue
                    predict = copyball.predict[direct]
                    # Up smash only front side
                    if direct == 0:
                        if abs(copyball.x - GROUND_HALF_WIDTH) > 50 and abs(copyball.x - GROUND_HALF_WIDTH) <= 144:
                            continue
                    # Long ball
                    if len(predict) > 20 + (self._true_rand() % 20) and (
                        other.y_velocity > -1 or abs(copyball.x - GROUND_HALF_WIDTH) > 72
                    ):
                        continue
                    # Far ball
                    if abs(copyball.x - GROUND_HALF_WIDTH) > 120 and direct > 0:
                        continue
                    if (
                        self._true_rand() % 10 < 6
                        and samesideloss(other, predict[-1].x)
                        and (len(predict) < short_path or (len(predict) == short_path and self._true_rand() % 2 < 1))
                    ):
                        short_path = len(predict)
                        st.goodtime = frame
                        shift_val = 15 if direct % 2 == 1 else 9
                        shift = -PLAYER_HALF_LENGTH + shift_val if player.is_player2 else PLAYER_HALF_LENGTH - shift_val
                        st.attack_x = copyball.x + shift
                        st.direction = direct
                        st.second_attack = -1
                        ui.y_direction = -1

        self._last_short_path = short_path

    # -------------------------------------------------------------------
    # Second attack decision (at execution time)
    # -------------------------------------------------------------------

    def _second_attack_decision(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        copyball: BallPathEntry,
    ) -> None:
        """Decide direction for second touch in a multi-hit combo."""
        cfg = self._config
        st = self._state

        short_path = 100
        new_direct = -1

        # Fancy (second touch fancy)
        for direct in range(6):
            if st.juke or not cfg.fancy:
                break
            if abs(ball.x - player.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                continue
            if abs(ball.x - player.x) > PLAYER_HALF_LENGTH + 6:
                continue
            if self._true_rand() % 10 < 1:
                continue
            predict = copyball.predict[direct]
            for pf in range(1, len(predict) - 1):
                pball = predict[pf]
                if not sameside(player, pball.x):
                    break
                # Second fancy jump
                if (
                    direct == 2
                    and player_y_predict_jump(player, pf + 1) != 244
                    and abs(pball.y - player_y_predict_jump(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict_jump(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                    and not (abs(pball.x - GROUND_HALF_WIDTH) > 20 or pball.y > 176)
                ):
                    short_path = pf - 100
                    st.goodtime = 0
                    shift = -PLAYER_HALF_LENGTH + 3 if pball.x < copyball.x else PLAYER_HALF_LENGTH - 3
                    st.attack_x = copyball.x + shift
                    st.direction = direct
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.second_attack = pf + 1
                    st.second_jump = True

                # Second fancy normal
                if (
                    abs(pball.y - player_y_predict(player, pf + 1)) <= PLAYER_HALF_LENGTH
                    and abs(predict[pf - 1].y - player_y_predict(player, pf)) > PLAYER_HALF_LENGTH
                    and abs(pball.x - player.x) <= 6 * pf + PLAYER_HALF_LENGTH + 6
                    and pf - 100 <= short_path
                ):
                    if short_path < 0 and (direct == 2 or direct == 3 or new_direct == 4):
                        break
                    short_path = pf - 100
                    st.goodtime = 0
                    shift_val = 9 if direct % 2 == 1 else 3
                    shift = -PLAYER_HALF_LENGTH + shift_val if pball.x < copyball.x else PLAYER_HALF_LENGTH - shift_val
                    st.attack_x = copyball.x + shift
                    st.direction = direct if ((pball.x < copyball.x) == player.is_player2) else -direct
                    new_direct = abs(direct)
                    sign = -PLAYER_HALF_LENGTH + 9 if player.is_player2 else PLAYER_HALF_LENGTH - 9
                    st.second_x = pball.x + sign
                    st.juke = False
                    if direct == 1 and abs(pball.x - GROUND_HALF_WIDTH) < 40:
                        st.second_x = pball.x + (PLAYER_HALF_LENGTH if player.is_player2 else -PLAYER_HALF_LENGTH)
                        st.juke = True
                    st.second_attack = pf + 1
                    st.second_jump = False
                    break

        # Normal second attack
        if short_path > -1:
            last_direction = st.direction
            st.direction = 1
            for direct in range(6):
                if abs(ball.x - player.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                    continue
                predict = copyball.predict[direct]
                if samesideloss(other, predict[-1].x):
                    if len(predict) < short_path:
                        st.direction = direct
                        short_path = len(predict)
                    elif len(predict) == short_path and self._true_rand() % 2 < 1:
                        st.direction = direct
                        short_path = len(predict)
                    if abs(ball.x - GROUND_HALF_WIDTH) > 120 and direct == 3:
                        st.direction = 1

            # Anti-predict
            if last_direction == 0 and other.state > 0 and st.direction > 3:
                if abs(ball.x - player.x) > PLAYER_HALF_LENGTH:
                    st.direction = 1
                else:
                    st.direction = 0

            # 2nd jump fail
            if player.y_velocity < -10 and st.direction < 4:
                if abs(ball.x - player.x) > PLAYER_HALF_LENGTH or abs(ball.x - GROUND_HALF_WIDTH) > 72:
                    st.direction = 1
                else:
                    st.direction = 0

            # Anti-block (second)
            if (
                st.direction == 3
                and abs(other.x - GROUND_HALF_WIDTH) < 72
                and (
                    abs(player.x - GROUND_HALF_WIDTH) > 108
                    or abs(other.y + other.y_velocity - ball.y) < PLAYER_HALF_LENGTH
                )
            ):
                if abs(ball.x - player.x) > PLAYER_HALF_LENGTH or abs(ball.x - GROUND_HALF_WIDTH) > 72:
                    st.direction = 1
                else:
                    st.direction = 0

        # Juke
        if st.juke and st.second_attack == 0:
            if (self._true_rand() % 10 < 5 or ball.x == GROUND_HALF_WIDTH) and not (
                other.state > 0 and abs(other.x - GROUND_HALF_WIDTH) < 72 and other.y_velocity < 10
            ):
                st.direction = -3
            st.juke = False

    # -------------------------------------------------------------------
    # First attack execution
    # -------------------------------------------------------------------

    def _first_attack_execution(
        self,
        player: Player,
        ball: Ball,
        other: Player,
        ui: UserInput,
        path: list[BallPathEntry],
        copyball: BallPathEntry,
    ) -> None:
        """Execute first hit — attack farthest, anti-block, block check."""
        cfg = self._config
        st = self._state

        org = copyball.predict[st.direction]
        # Attack farthest
        if (
            player.y <= other.y
            and abs(other.x - org[-1].x) <= len(org) * 6 + PLAYER_HALF_LENGTH
            and self._true_rand() % 10 < 2
        ):
            farthest = 0
            for direct in range(6):
                if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                    continue
                predict = copyball.predict[direct]
                far = abs(other.x - predict[-1].x)
                if len(predict) < 50 and samesideloss(other, predict[-1].x) and far >= farthest:
                    farthest = far
                    st.direction = direct

        # Anti-block
        if (
            cfg.anti_block
            and other.state < 3
            and (player.y_velocity < 0 or abs(ball.x - GROUND_HALF_WIDTH) > 144)
            and abs(other.x - GROUND_HALF_WIDTH) < 144
        ):
            canbypass = False
            short_path_ab = 1000

            if other.state == 0 or other.y_velocity > 12:
                if canblock_predict(other, copyball.predict[st.direction]):
                    for direct in range(6):
                        if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                            continue
                        predict = copyball.predict[direct]
                        if (
                            samesideloss(other, predict[-1].x)
                            and len(predict) <= short_path_ab
                            and (direct > 3 or not canblock_predict(other, predict))
                        ):
                            short_path_ab = len(predict)
                            st.direction = direct
                            canbypass = True
                else:
                    canbypass = True
            else:
                if canblock(other, copyball.predict[st.direction]):
                    for direct in range(6):
                        if abs(player.x - ball.x) > PLAYER_HALF_LENGTH and direct % 2 == 0:
                            continue
                        predict = copyball.predict[direct]
                        if (
                            samesideloss(other, predict[-1].x)
                            and len(predict) <= short_path_ab
                            and ((direct > 3 and self._true_rand() % 10 < 2) or not canblock(other, predict))
                        ):
                            short_path_ab = len(predict)
                            st.direction = direct
                            canbypass = True
                else:
                    canbypass = True

            if canbypass and st.direction == 0 and player.y_velocity < -10 and abs(ball.x - GROUND_HALF_WIDTH) > 72:
                st.direction = 1

            if not canbypass:
                st.direction = 1

        # Block check
        if (
            not cfg.block
            and ball.is_power_hit
            and (player.is_player2 == (ball.x_velocity > 0))
            and abs(ball.x - GROUND_HALF_WIDTH) < 72
        ):
            ui.power_hit = 0

    # -------------------------------------------------------------------
    # Direction to input mapping
    # -------------------------------------------------------------------

    @staticmethod
    def _apply_direction(ui: UserInput, direction: int, is_player2: bool) -> None:
        """Convert direction code to UserInput fields.

        Original: direction mapping in physics.js lines 2924-2990.
        """
        forward = -1 if is_player2 else 1

        abs_dir = abs(direction)
        sign = 1 if direction >= 0 else -1

        if abs_dir == 0:
            ui.y_direction = -1
            ui.x_direction = 0
        elif abs_dir == 1:
            ui.y_direction = -1
            if ui.x_direction == 0:
                ui.x_direction = forward * sign
        elif abs_dir == 2:
            ui.y_direction = 0
            ui.x_direction = 0
        elif abs_dir == 3:
            ui.y_direction = 0
            if ui.x_direction == 0:
                ui.x_direction = forward * sign
        elif abs_dir == 4:
            ui.y_direction = 1
            ui.x_direction = 0
        elif abs_dir == 5:
            ui.y_direction = 1
            if ui.x_direction == 0:
                ui.x_direction = forward * sign
