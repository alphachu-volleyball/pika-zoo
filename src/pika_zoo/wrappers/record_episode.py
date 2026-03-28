"""
Wrapper that records episode statistics and per-frame game state for replay.

Records:
- Episode length and cumulative rewards (in infos)
- Per-round scoring records with frames grouped by round
- Per-frame game state snapshots for later JSON export
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pettingzoo.utils import BaseParallelWrapper


@dataclass
class FrameSnapshot:
    """A single frame of game state."""

    frame: int
    player1_action: int
    player1_x: int
    player1_y: int
    player1_state: int
    player2_action: int
    player2_x: int
    player2_y: int
    player2_state: int
    ball_x: int
    ball_y: int
    ball_x_velocity: int
    ball_y_velocity: int
    ball_is_power_hit: bool


@dataclass
class RoundRecord:
    """Record of a single round (one point scored)."""

    round_number: int
    server: str  # "player_1" or "player_2" — who served this round
    scorer: str  # "player_1" or "player_2" — who scored
    reward: dict[str, float]
    start_frame: int
    end_frame: int
    frames: list[FrameSnapshot] = field(default_factory=list)

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class EpisodeRecord:
    """Complete record of an episode (full game)."""

    scores: list[int] = field(default_factory=lambda: [0, 0])
    total_frames: int = 0
    cumulative_rewards: dict[str, float] = field(default_factory=lambda: {"player_1": 0.0, "player_2": 0.0})
    rounds: list[RoundRecord] = field(default_factory=list)

    @property
    def winner(self) -> str | None:
        """Return the winner, or None if game is incomplete / draw."""
        if self.scores[0] > self.scores[1]:
            return "player_1"
        elif self.scores[1] > self.scores[0]:
            return "player_2"
        return None

    @property
    def frames(self) -> list[FrameSnapshot]:
        """Flat list of all frames across all rounds (backward compatible)."""
        return [f for r in self.rounds for f in r.frames]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "total_frames": self.total_frames,
            "cumulative_rewards": self.cumulative_rewards,
            "winner": self.winner,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "server": r.server,
                    "scorer": r.scorer,
                    "reward": r.reward,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "frames": [asdict(f) for f in r.frames],
                }
                for r in self.rounds
            ],
        }


class RecordEpisode(BaseParallelWrapper):
    """Record episode statistics and game state snapshots.

    Args:
        env: The base PikachuVolleyballEnv.
        record_frames: Whether to record per-frame snapshots (default True).
            Set to False for lightweight stats-only recording.
    """

    def __init__(self, env, record_frames: bool = True) -> None:
        super().__init__(env)
        self._record_frames = record_frames
        self._current_episode: EpisodeRecord | None = None
        self._frame_count: int = 0
        self._round_start_frame: int = 1
        self._prev_scores: list[int] = [0, 0]
        self._current_round_frames: list[FrameSnapshot] = []
        self._current_server: str = "player_1"

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self._current_episode = EpisodeRecord()
        self._frame_count = 0
        self._round_start_frame = 1
        self._prev_scores = [0, 0]
        self._current_round_frames = []
        # First round server: check env's _is_player2_serve
        self._current_server = "player_2" if self.env._is_player2_serve else "player_1"
        return observations, infos

    def step(self, actions):
        p1_action = actions.get("player_1", 0)
        p2_action = actions.get("player_2", 0)
        observations, rewards, terminations, truncations, infos = super().step(actions)

        if self._current_episode is None:
            return observations, rewards, terminations, truncations, infos

        # 1. Increment frame count
        self._frame_count += 1
        self._current_episode.total_frames = self._frame_count

        # 2. Update cumulative rewards
        self._current_episode.cumulative_rewards["player_1"] += rewards.get("player_1", 0.0)
        self._current_episode.cumulative_rewards["player_2"] += rewards.get("player_2", 0.0)

        # 3. Record frame snapshot into current round buffer (BEFORE round detection)
        physics = self.env._physics
        if physics is not None and self._record_frames:
            self._current_round_frames.append(
                FrameSnapshot(
                    frame=self._frame_count,
                    player1_action=int(p1_action),
                    player1_x=int(physics.player1.x),
                    player1_y=int(physics.player1.y),
                    player1_state=int(physics.player1.state),
                    player2_action=int(p2_action),
                    player2_x=int(physics.player2.x),
                    player2_y=int(physics.player2.y),
                    player2_state=int(physics.player2.state),
                    ball_x=int(physics.ball.x),
                    ball_y=int(physics.ball.y),
                    ball_x_velocity=int(physics.ball.x_velocity),
                    ball_y_velocity=int(physics.ball.y_velocity),
                    ball_is_power_hit=bool(physics.ball.is_power_hit),
                )
            )

        # 4. Detect round end by score change
        current_scores = list(self.env._scores)
        if current_scores != self._prev_scores:
            round_number = len(self._current_episode.rounds) + 1
            if current_scores[0] > self._prev_scores[0]:
                scorer = "player_1"
                reward = {"player_1": 1.0, "player_2": -1.0}
            else:
                scorer = "player_2"
                reward = {"player_1": -1.0, "player_2": 1.0}

            self._current_episode.rounds.append(
                RoundRecord(
                    round_number=round_number,
                    server=self._current_server,
                    scorer=scorer,
                    reward=reward,
                    start_frame=self._round_start_frame,
                    end_frame=self._frame_count,
                    frames=self._current_round_frames,
                )
            )
            self._current_round_frames = []
            self._round_start_frame = self._frame_count + 1
            self._prev_scores = current_scores
            # Next round server: the scorer serves (env uses "winner" serve by default)
            self._current_server = "player_2" if self.env._is_player2_serve else "player_1"

        # 5. Add episode stats to infos on game end
        if any(terminations.values()):
            self._current_episode.scores = current_scores
            for agent in infos:
                infos[agent]["episode"] = {
                    "length": self._current_episode.total_frames,
                    "rewards": dict(self._current_episode.cumulative_rewards),
                    "scores": list(self._current_episode.scores),
                    "winner": self._current_episode.winner,
                }

        return observations, rewards, terminations, truncations, infos

    def get_episode_record(self) -> EpisodeRecord | None:
        """Return the current episode record (call after episode ends)."""
        return self._current_episode
