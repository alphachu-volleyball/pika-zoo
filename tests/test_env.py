"""Tests for the PettingZoo environment."""

import pytest

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.env import env
from pika_zoo.env.actions import NUM_ACTIONS, ActionConverter
from pika_zoo.env.observations import OBSERVATION_SIZE


class TestActionConverter:
    def test_noop(self):
        converter = ActionConverter()
        ui = converter.convert(0)
        assert ui.x_direction == 0
        assert ui.y_direction == 0
        assert ui.power_hit == 0

    def test_directions(self):
        converter = ActionConverter()
        # Action 3: RIGHT
        ui = converter.convert(3)
        assert ui.x_direction == 1
        assert ui.y_direction == 0
        # Action 4: LEFT
        ui = converter.convert(4)
        assert ui.x_direction == -1
        # Action 2: UP
        ui = converter.convert(2)
        assert ui.y_direction == -1
        # Action 5: DOWN
        ui = converter.convert(5)
        assert ui.y_direction == 1

    def test_power_hit_debounce(self):
        converter = ActionConverter()
        # First press: should trigger
        ui = converter.convert(1)  # FIRE
        assert ui.power_hit == 1
        # Held down: should NOT trigger
        ui = converter.convert(1)  # FIRE again
        assert ui.power_hit == 0
        # Released then pressed: should trigger
        ui = converter.convert(0)  # NOOP
        ui = converter.convert(1)  # FIRE
        assert ui.power_hit == 1

    def test_reset(self):
        converter = ActionConverter()
        converter.convert(1)  # FIRE
        converter.reset()
        ui = converter.convert(1)  # FIRE again after reset
        assert ui.power_hit == 1


class TestPikachuVolleyballEnv:
    def test_create_env(self):
        e = env()
        assert e.possible_agents == ["player_1", "player_2"]

    def test_reset(self):
        e = env()
        observations, infos = e.reset(seed=42)
        assert set(observations.keys()) == {"player_1", "player_2"}
        assert observations["player_1"].shape == (OBSERVATION_SIZE,)
        assert observations["player_2"].shape == (OBSERVATION_SIZE,)
        assert infos["player_1"]["scores"] == [0, 0]

    def test_action_space(self):
        e = env()
        e.reset(seed=42)
        assert e.action_space("player_1").n == NUM_ACTIONS
        assert e.action_space("player_2").n == NUM_ACTIONS

    def test_observation_space(self):
        e = env()
        e.reset(seed=42)
        obs_space = e.observation_space("player_1")
        assert obs_space.shape == (OBSERVATION_SIZE,)

    def test_step_noop(self):
        e = env()
        e.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}
        obs, rewards, terms, truncs, infos = e.step(actions)
        assert set(obs.keys()) == {"player_1", "player_2"}
        assert obs["player_1"].shape == (OBSERVATION_SIZE,)

    def test_round_scoring(self):
        """Ball should eventually touch ground and score a point."""
        e = env()
        e.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}

        scored = False
        for _ in range(300):
            obs, rewards, terms, truncs, infos = e.step(actions)
            if infos["player_1"]["round_ended"]:
                scored = True
                total = infos["player_1"]["scores"]
                assert total[0] + total[1] == 1
                break

        assert scored, "A point should be scored within 300 frames"

    def test_full_game(self):
        """A full game should end with one player reaching winning_score."""
        e = env(winning_score=3)
        e.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}

        game_ended = False
        for _ in range(5000):
            obs, rewards, terms, truncs, infos = e.step(actions)
            if any(terms.values()):
                game_ended = True
                scores = infos["player_1"]["scores"]
                assert max(scores) >= 3
                break

        assert game_ended, "Game should end within 5000 frames with winning_score=3"

    def test_agents_cleared_on_game_end(self):
        """agents list should be empty after game ends."""
        e = env(winning_score=1)
        e.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}

        for _ in range(300):
            obs, rewards, terms, truncs, infos = e.step(actions)
            if any(terms.values()):
                assert e.agents == []
                return

        pytest.fail("Game should end within 300 frames with winning_score=1")

    def test_ai_policy_override(self):
        """AI policies should override actions for controlled agents."""
        ai = BuiltinAI()
        e = env(ai_policies={"player_2": ai}, winning_score=3)
        e.reset(seed=42)

        game_ended = False
        for _ in range(5000):
            actions = {"player_1": 0, "player_2": 0}  # player_2 action ignored
            obs, rewards, terms, truncs, infos = e.step(actions)
            if any(terms.values()):
                game_ended = True
                break

        assert game_ended

    def test_agent_centric_observations(self):
        """Each agent should see itself first in the observation."""
        e = env()
        obs, _ = e.reset(seed=42)

        # Player 1's obs[0] should be player1.x, player 2's obs[0] should be player2.x
        p1_x = obs["player_1"][0]
        p2_x = obs["player_2"][0]

        # Player 1 starts at x=36, player 2 at x=396
        assert p1_x == pytest.approx(36.0)
        assert p2_x == pytest.approx(396.0)

    def test_zero_sum_rewards(self):
        """Rewards should be zero-sum."""
        e = env()
        e.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}

        for _ in range(300):
            obs, rewards, terms, truncs, infos = e.step(actions)
            assert rewards["player_1"] + rewards["player_2"] == pytest.approx(0.0)

    def test_deterministic(self):
        """Same seed should produce identical trajectories."""

        def run_episode(seed):
            e = env(winning_score=2)
            e.reset(seed=seed)
            actions = {"player_1": 0, "player_2": 0}
            total_rewards = {"player_1": 0.0, "player_2": 0.0}
            steps = 0
            for _ in range(3000):
                obs, rewards, terms, truncs, infos = e.step(actions)
                total_rewards["player_1"] += rewards["player_1"]
                total_rewards["player_2"] += rewards["player_2"]
                steps += 1
                if any(terms.values()):
                    break
            return total_rewards, steps

        r1, s1 = run_episode(42)
        r2, s2 = run_episode(42)
        assert r1 == r2
        assert s1 == s2
