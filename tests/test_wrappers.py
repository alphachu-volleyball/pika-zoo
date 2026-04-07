"""Tests for the wrappers module."""

import json

import numpy as np
import pytest

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.env import env
from pika_zoo.env.observations import OBSERVATION_SIZE
from pika_zoo.wrappers import (
    ConvertSingleAgent,
    LinearBallPosition,
    NormalizeObservation,
    QuadrantBallPosition,
    RecordGame,
    RewardShaping,
    SimplifyAction,
    SimplifyObservation,
)
from pika_zoo.wrappers.simplify_action import NUM_SIMPLIFIED_ACTIONS


class TestConvertSingleAgent:
    def test_gym_interface(self):
        e = env()
        wrapped = ConvertSingleAgent(e)
        obs, info = wrapped.reset(seed=42)
        assert obs.shape == (OBSERVATION_SIZE,)
        assert isinstance(info, dict)

    def test_step_returns_single(self):
        e = env()
        wrapped = ConvertSingleAgent(e)
        wrapped.reset(seed=42)
        obs, reward, terminated, truncated, info = wrapped.step(0)
        assert obs.shape == (OBSERVATION_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_with_builtin_ai_opponent(self):
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=BuiltinAI())
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break
        assert game_ended

    def test_with_callable_opponent(self):
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=lambda obs: 0)
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break
        assert game_ended

    def test_with_random_opponent(self):
        e = env(winning_score=2)
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=None)
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break
        assert game_ended

    def test_callable_opponent_with_wrapper(self):
        """Callable opponent should work even when env is wrapped (no private access)."""
        from pika_zoo.wrappers import NormalizeObservation

        e = NormalizeObservation(env(winning_score=2))
        wrapped = ConvertSingleAgent(e, agent="player_1", opponent_policy=lambda obs: 0)
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated:
                game_ended = True
                break
        assert game_ended

    def test_player2_as_agent(self):
        e = env(winning_score=1)
        wrapped = ConvertSingleAgent(e, agent="player_2", opponent_policy=BuiltinAI())
        obs, _ = wrapped.reset(seed=42)
        assert obs[0] == pytest.approx(396.0)

    def test_invalid_agent(self):
        e = env()
        with pytest.raises(AssertionError):
            ConvertSingleAgent(e, agent="player_3")

    def test_observation_action_spaces(self):
        e = env()
        wrapped = ConvertSingleAgent(e)
        assert wrapped.observation_space.shape == (OBSERVATION_SIZE,)
        assert wrapped.action_space.n == 18


class TestSimplifyAction:
    def test_action_space_size(self):
        e = env()
        wrapped = SimplifyAction(e)
        assert wrapped.action_space("player_1").n == NUM_SIMPLIFIED_ACTIONS

    def test_step_with_simplified_actions(self):
        e = env()
        wrapped = SimplifyAction(e)
        wrapped.reset(seed=42)
        actions = {"player_1": 0, "player_2": 0}
        obs, rewards, terms, truncs, infos = wrapped.step(actions)
        assert obs["player_1"].shape == (OBSERVATION_SIZE,)

    def test_toward_net_maps_correctly(self):
        """Player 1 toward_net (3) should map to RIGHT (3), player 2 to LEFT (4)."""
        e = env()
        wrapped = SimplifyAction(e)
        assert wrapped._maps["player_1"][3] == 3  # RIGHT
        assert wrapped._maps["player_2"][3] == 4  # LEFT

    def test_full_game(self):
        e = env(winning_score=2)
        wrapped = SimplifyAction(e)
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            actions = {"player_1": 0, "player_2": 0}
            obs, rewards, terms, truncs, infos = wrapped.step(actions)
            if any(terms.values()):
                game_ended = True
                break
        assert game_ended


class TestSimplifyObservation:
    def test_player1_unchanged(self):
        """Player 1 observations should pass through unmodified."""
        raw_env = env()
        raw_obs, _ = raw_env.reset(seed=42)

        wrapped_env = env()
        wrapped = SimplifyObservation(wrapped_env)
        wrapped_obs, _ = wrapped.reset(seed=42)

        np.testing.assert_array_equal(wrapped_obs["player_1"], raw_obs["player_1"])

    def test_player2_x_mirrored(self):
        """Player 2 x-positions should be mirrored (432 - x)."""
        raw_env = env()
        raw_obs, _ = raw_env.reset(seed=42)

        wrapped_env = env()
        wrapped = SimplifyObservation(wrapped_env)
        wrapped_obs, _ = wrapped.reset(seed=42)

        assert wrapped_obs["player_2"][0] == pytest.approx(432.0 - raw_obs["player_2"][0])
        assert wrapped_obs["player_2"][13] == pytest.approx(432.0 - raw_obs["player_2"][13])
        assert wrapped_obs["player_2"][26] == pytest.approx(432.0 - raw_obs["player_2"][26])

    def test_player2_direction_negated(self):
        """Player 2 x-direction/velocity should be negated."""
        raw_env = env()
        raw_obs, _ = raw_env.reset(seed=42)

        wrapped_env = env()
        wrapped = SimplifyObservation(wrapped_env)
        wrapped_obs, _ = wrapped.reset(seed=42)

        assert wrapped_obs["player_2"][3] == pytest.approx(-raw_obs["player_2"][3])
        assert wrapped_obs["player_2"][32] == pytest.approx(-raw_obs["player_2"][32])

    def test_step_mirrors_player2(self):
        """Mirroring should also apply to step() observations."""
        raw_env = env()
        raw_env.reset(seed=42)
        raw_obs, _, _, _, _ = raw_env.step({"player_1": 0, "player_2": 0})

        wrapped_env = env()
        wrapped = SimplifyObservation(wrapped_env)
        wrapped.reset(seed=42)
        wrapped_obs, _, _, _, _ = wrapped.step({"player_1": 0, "player_2": 0})

        assert wrapped_obs["player_2"][0] == pytest.approx(432.0 - raw_obs["player_2"][0])
        assert wrapped_obs["player_2"][26] == pytest.approx(432.0 - raw_obs["player_2"][26])

    def test_obs_shape_preserved(self):
        e = env()
        wrapped = SimplifyObservation(e)
        obs, _ = wrapped.reset(seed=42)
        for agent_obs in obs.values():
            assert agent_obs.shape == (OBSERVATION_SIZE,)
            assert agent_obs.dtype == np.float32

    def test_symmetric_initial_positions(self):
        """After mirroring, both players should see similar self.x at start."""
        e = env()
        wrapped = SimplifyObservation(e)
        obs, _ = wrapped.reset(seed=42)
        assert obs["player_1"][0] == pytest.approx(obs["player_2"][0])

    def test_full_game(self):
        e = env(winning_score=2)
        wrapped = SimplifyObservation(e)
        wrapped.reset(seed=42)
        game_ended = False
        for _ in range(3000):
            actions = {"player_1": 0, "player_2": 0}
            obs, rewards, terms, truncs, infos = wrapped.step(actions)
            if any(terms.values()):
                game_ended = True
                break
        assert game_ended


class TestNormalizeObservation:
    def test_obs_range(self):
        e = env()
        wrapped = NormalizeObservation(e)
        obs, _ = wrapped.reset(seed=42)
        for agent_obs in obs.values():
            assert np.all(agent_obs >= 0.0)
            assert np.all(agent_obs <= 1.0)

    def test_obs_space(self):
        e = env()
        wrapped = NormalizeObservation(e)
        space = wrapped.observation_space("player_1")
        assert space.low.min() == pytest.approx(0.0)
        assert space.high.max() == pytest.approx(1.0)

    def test_step_normalized(self):
        e = env()
        wrapped = NormalizeObservation(e)
        wrapped.reset(seed=42)
        obs, _, _, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
        for agent_obs in obs.values():
            assert np.all(agent_obs >= 0.0)
            assert np.all(agent_obs <= 1.0)


class TestRewardShaping:
    def test_ball_position_reward(self):
        """Shaped rewards should be non-zero even when no scoring."""
        e = env()
        wrapped = RewardShaping(e, channels=[(LinearBallPosition(), 0.01)])
        wrapped.reset(seed=42)
        obs, rewards, _, _, infos = wrapped.step({"player_1": 0, "player_2": 0})
        # Ball starts on player_1's side (x=56 < 216), so player_1 gets negative bonus
        if not infos["player_1"]["round_ended"]:
            assert rewards["player_1"] != 0.0

    def test_zero_sum_with_shaping(self):
        """Ball position rewards should remain zero-sum."""
        e = env()
        wrapped = RewardShaping(e, channels=[(LinearBallPosition(), 0.05)])
        wrapped.reset(seed=42)
        for _ in range(50):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            assert rewards["player_1"] + rewards["player_2"] == pytest.approx(0.0, abs=1e-6)
            if any(terms.values()):
                break

    def test_empty_channels(self):
        """No channels should produce no shaping."""
        e = env()
        wrapped = RewardShaping(e, channels=[])
        wrapped.reset(seed=42)
        obs, rewards, _, _, infos = wrapped.step({"player_1": 0, "player_2": 0})
        if not infos["player_1"]["round_ended"]:
            assert rewards["player_1"] == 0.0
            assert rewards["player_2"] == 0.0

    def test_from_preset(self):
        """Preset should create a valid RewardShaping."""
        e = env()
        wrapped = RewardShaping.from_preset(e, "default")
        wrapped.reset(seed=42)
        obs, rewards, _, _, infos = wrapped.step({"player_1": 0, "player_2": 0})
        if not infos["player_1"]["round_ended"]:
            assert rewards["player_1"] != 0.0

    def test_multiple_channels(self):
        """Multiple channels should combine additively."""
        e = env()
        wrapped = RewardShaping(
            e,
            channels=[
                (LinearBallPosition(), 0.01),
                (QuadrantBallPosition(), 0.005),
            ],
        )
        wrapped.reset(seed=42)
        obs, rewards, _, _, infos = wrapped.step({"player_1": 0, "player_2": 0})
        if not infos["player_1"]["round_ended"]:
            # Both channels contribute, so reward should be non-zero
            assert rewards["player_1"] != 0.0


class TestRecordGame:
    def test_records_frames(self):
        e = env(winning_score=1)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(300):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        assert record is not None
        assert record.num_frames > 0
        assert len(record.frames) == record.num_frames

    def test_episode_stats_in_info(self):
        e = env(winning_score=1)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(300):
            obs, rewards, terms, _, infos = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                assert "episode" in infos["player_1"]
                assert "length" in infos["player_1"]["episode"]
                assert "scores" in infos["player_1"]["episode"]
                return
        pytest.fail("Game should end within 300 frames")

    def test_to_dict_serializable(self):
        e = env(winning_score=1)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(300):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        d = record.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_round_records(self):
        e = env(winning_score=3)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(5000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        assert record is not None
        # Total rounds should match total score
        assert len(record.rounds) == sum(record.scores)
        # Each round should have a valid scorer and server
        for r in record.rounds:
            assert r.scorer in ("player_1", "player_2")
            assert r.server in ("player_1", "player_2")
            assert r.start_frame <= r.end_frame
        # Round numbers should be sequential
        for i, r in enumerate(record.rounds):
            assert r.round_number == i + 1

    def test_frames_grouped_by_round(self):
        e = env(winning_score=3)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(5000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        # Each round should have non-empty frames
        for r in record.rounds:
            assert len(r.frames) > 0
        # Total frames across rounds should equal total_frames
        assert sum(len(r.frames) for r in record.rounds) == record.num_frames
        # Frame numbers within each round should be contiguous and ascending
        for r in record.rounds:
            frame_nums = [f.frame for f in r.frames]
            assert frame_nums == list(range(frame_nums[0], frame_nums[-1] + 1))

    def test_round_reward(self):
        e = env(winning_score=2)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(3000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        for r in record.rounds:
            # Reward should be zero-sum
            assert r.reward["player_1"] + r.reward["player_2"] == pytest.approx(0.0)
            # Scorer should get +1
            assert r.reward[r.scorer] == pytest.approx(1.0)

    def test_winner(self):
        e = env(winning_score=2)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(3000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        assert record.winner is not None
        # Winner should be the player with higher score
        if record.scores[0] > record.scores[1]:
            assert record.winner == "player_1"
        else:
            assert record.winner == "player_2"

    def test_server_tracking(self):
        """First round server should be player_1, then scorer serves next."""
        e = env(winning_score=3)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(5000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        # First round: player_1 serves (default)
        assert record.rounds[0].server == "player_1"
        # Subsequent rounds: scorer of previous round serves
        for i in range(1, len(record.rounds)):
            assert record.rounds[i].server == record.rounds[i - 1].scorer

    def test_round_num_frames(self):
        e = env(winning_score=2)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(3000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        for r in record.rounds:
            assert r.num_frames == r.end_frame - r.start_frame + 1
            assert r.num_frames == len(r.frames)
            # backward compat
            assert r.duration == r.num_frames

    def test_event_counts(self):
        """event_counts should aggregate frame-level events."""
        e = env(winning_score=3)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(5000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        # Game-level event_counts
        ec = record.event_counts
        assert isinstance(ec, dict)
        assert "p1_touch_ball" in ec
        assert ec["p1_touch_ball"] > 0 or ec["p2_touch_ball"] > 0
        # Round-level event_counts should sum to game-level
        for key in ec:
            assert ec[key] == sum(r.event_counts[key] for r in record.rounds)

    def test_scores_computed(self):
        """scores should be computed from rounds, not stored."""
        e = env(winning_score=2)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(3000):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        scores = record.scores
        p1_scored = sum(1 for r in record.rounds if r.scorer == "player_1")
        p2_scored = sum(1 for r in record.rounds if r.scorer == "player_2")
        assert scores == [p1_scored, p2_scored]

    def test_frame_record_has_events(self):
        """FrameRecord should include event flags."""
        e = env(winning_score=1)
        wrapped = RecordGame(e)
        wrapped.reset(seed=42)
        for _ in range(300):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        f = record.frames[0]
        assert hasattr(f, "p1_touch_ball")
        assert hasattr(f, "ball_wall_bounce")
        assert hasattr(f, "round_number")

    def test_no_frames_when_disabled(self):
        e = env(winning_score=1)
        wrapped = RecordGame(e, record_frames=False)
        wrapped.reset(seed=42)
        for _ in range(300):
            obs, rewards, terms, _, _ = wrapped.step({"player_1": 0, "player_2": 0})
            if any(terms.values()):
                break
        record = wrapped.get_game_record()
        assert record.num_frames > 0
        assert len(record.frames) == 0
