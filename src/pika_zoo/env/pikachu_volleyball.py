"""
PettingZoo ParallelEnv for Pikachu Volleyball.

Two agents ("player_1", "player_2") act simultaneously each frame.
The physics engine runs at 25 FPS equivalent (one step = one frame).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.engine.constants import GROUND_HALF_WIDTH
from pika_zoo.engine.physics import PikaPhysics
from pika_zoo.engine.types import NoiseConfig
from pika_zoo.env.actions import NUM_ACTIONS, ActionConverter
from pika_zoo.env.observations import build_observation, build_observation_space


class PikachuVolleyballEnv(ParallelEnv):
    """Pikachu Volleyball as a PettingZoo ParallelEnv.

    Args:
        winning_score: Score needed to win the game (default 15).
        serve: Serve rule — "winner" (scorer serves), "alternate", or "random".
        ai_policies: Dict mapping agent name to AIPolicy. When set, the env
            calls policy.compute_action() before physics step, overriding
            the action for that agent.
        render_mode: "human" or "rgb_array" (not yet implemented).
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pikachu_volleyball_v0",
        "is_parallelizable": True,
        "render_fps": 25,
    }

    def __init__(
        self,
        winning_score: int = 15,
        serve: str = "winner",
        ai_policies: dict[str, AIPolicy] | None = None,
        render_mode: str | None = None,
        noise: NoiseConfig | None = None,
        p1_skin: str = "yellow",
        p2_skin: str = "yellow",
        p1_label: str = "",
        p2_label: str = "",
    ) -> None:
        super().__init__()

        self.winning_score = winning_score
        self.serve = serve
        self.ai_policies = ai_policies or {}
        self.render_mode = render_mode
        self.noise = noise
        self._p1_skin = p1_skin
        self._p2_skin = p2_skin
        self._p1_label = p1_label
        self._p2_label = p2_label

        self.possible_agents = ["player_1", "player_2"]

        self._action_converters: dict[str, ActionConverter] = {}
        self._physics: PikaPhysics | None = None
        self._renderer = None  # Lazy-initialized PygameRenderer
        self._scores: list[int] = [0, 0]
        self._is_player2_serve: bool = False
        self._round_ended: bool = False
        self._game_ended: bool = False
        self._np_random: np.random.Generator | None = None
        self._last_user_inputs: list | None = None

    @functools.cache
    def observation_space(self, agent: str) -> spaces.Box:
        return build_observation_space()

    @functools.cache
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(NUM_ACTIONS)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        self.agents = list(self.possible_agents)

        self._physics = PikaPhysics(self._np_random)
        self._physics.ball.initialize_for_new_round(self._is_player2_serve, noise=self.noise, rng=self._np_random)
        self._scores = [0, 0]
        self._round_ended = False
        self._game_ended = False

        self._action_converters = {agent: ActionConverter() for agent in self.agents}

        # Reset AI policies
        for agent_name, policy in self.ai_policies.items():
            policy.reset(self._np_random)

        observations = self._get_observations()
        infos = self._get_infos()
        return observations, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        assert self._physics is not None, "Call reset() before step()"
        assert self._np_random is not None

        # Start new round if previous ended but game continues
        if self._round_ended and not self._game_ended:
            self._physics.player1.initialize_for_new_round(self._np_random)
            self._physics.player2.initialize_for_new_round(self._np_random)
            self._physics.ball.initialize_for_new_round(self._is_player2_serve, noise=self.noise, rng=self._np_random)
            self._round_ended = False
            for converter in self._action_converters.values():
                converter.reset()

        # Override actions for AI-controlled agents
        players = {"player_1": self._physics.player1, "player_2": self._physics.player2}
        opponents = {"player_1": self._physics.player2, "player_2": self._physics.player1}

        user_inputs = []
        for agent in self.possible_agents:
            if agent in self.ai_policies:
                ai_input = self.ai_policies[agent].compute_action(
                    players[agent], self._physics.ball, opponents[agent], self._np_random
                )
                user_inputs.append(ai_input)
            else:
                action = actions.get(agent, 0)
                user_inputs.append(self._action_converters[agent].convert(action))
        self._last_user_inputs = user_inputs

        # Clear sound flags
        for sound_dict in [
            self._physics.player1.sound,
            self._physics.player2.sound,
            self._physics.ball.sound,
        ]:
            for key in sound_dict:
                sound_dict[key] = False

        # Run physics
        is_ball_touching_ground = self._physics.run_engine_for_next_frame(user_inputs, self._np_random)

        # Process scoring
        rewards: dict[str, float] = {"player_1": 0.0, "player_2": 0.0}

        if is_ball_touching_ground and not self._round_ended and not self._game_ended:
            if self._physics.ball.punch_effect_x < GROUND_HALF_WIDTH:
                # Ball landed on player_1's side → player_2 scores
                self._is_player2_serve = True
                self._scores[1] += 1
                rewards = {"player_1": -1.0, "player_2": 1.0}
                if self._scores[1] >= self.winning_score:
                    self._game_ended = True
                    self._physics.player2.is_winner = True
                    self._physics.player1.game_ended = True
                    self._physics.player2.game_ended = True
            else:
                # Ball landed on player_2's side → player_1 scores
                self._is_player2_serve = False
                self._scores[0] += 1
                rewards = {"player_1": 1.0, "player_2": -1.0}
                if self._scores[0] >= self.winning_score:
                    self._game_ended = True
                    self._physics.player1.is_winner = True
                    self._physics.player1.game_ended = True
                    self._physics.player2.game_ended = True
            self._round_ended = True

        terminations = {agent: self._game_ended for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        if self._game_ended:
            self.agents = []

        observations = self._get_observations()
        infos = self._get_infos()

        return observations, rewards, terminations, truncations, infos

    def render(self) -> np.ndarray | None:
        if self.render_mode is None or self._physics is None:
            return None

        if self._renderer is None:
            from pika_zoo.rendering.renderer import PygameRenderer

            self._renderer = PygameRenderer(render_mode=self.render_mode, p1_skin=self._p1_skin, p2_skin=self._p2_skin)

        return self._renderer.render(
            self._physics.player1,
            self._physics.player2,
            self._physics.ball,
            self._scores,
            metadata={
                "noise": self.noise,
                "p1_label": self._p1_label,
                "p2_label": self._p2_label,
            },
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _get_observations(self) -> dict[str, np.ndarray]:
        assert self._physics is not None
        p1_prev = self._action_converters.get("player_1")
        p2_prev = self._action_converters.get("player_2")
        p1_prev_power = p1_prev._prev_power_hit if p1_prev else 0
        p2_prev_power = p2_prev._prev_power_hit if p2_prev else 0

        return {
            "player_1": build_observation(
                self._physics.player1, self._physics.player2, self._physics.ball, p1_prev_power, p2_prev_power
            ),
            "player_2": build_observation(
                self._physics.player2, self._physics.player1, self._physics.ball, p2_prev_power, p1_prev_power
            ),
        }

    def _get_infos(self) -> dict[str, dict]:
        inputs = self._last_user_inputs if self._last_user_inputs else None
        user_inputs = {}
        if inputs:
            for i, agent in enumerate(self.possible_agents):
                u = inputs[i]
                user_inputs[agent] = {
                    "x_direction": u.x_direction,
                    "y_direction": u.y_direction,
                    "power_hit": u.power_hit,
                }
        base = {"scores": list(self._scores), "round_ended": self._round_ended, "user_inputs": user_inputs}
        if self._physics is not None:
            p1 = self._physics.player1.events
            p2 = self._physics.player2.events
            ball = self._physics.ball.events
            events = {
                "p1_touch_ball": p1["touch_ball"],
                "p1_power_hit": p1["power_hit"],
                "p1_diving": p1["diving"],
                "p2_touch_ball": p2["touch_ball"],
                "p2_power_hit": p2["power_hit"],
                "p2_diving": p2["diving"],
                "ball_wall_bounce": ball["wall_bounce"],
                "ball_net_collision": ball["net_collision"],
            }
        else:
            events = {}
        return {
            "player_1": {**base, "events": events},
            "player_2": {**base, "events": events},
        }

    def _get_serve(self) -> bool:
        """Determine who serves. Returns True if player 2 serves."""
        if self.serve == "winner":
            return self._is_player2_serve
        elif self.serve == "random":
            assert self._np_random is not None
            return bool(self._np_random.integers(0, 2))
        else:  # alternate
            return (self._scores[0] + self._scores[1]) % 2 == 1
