"""
Unified game recording with hierarchical frame → round → game → games structure.

Each level provides a consistent interface:
- num_frames: frame count
- event_counts: dict[str, int] of aggregated event flags
- to_frames_df(): flat DataFrame (pandas lazy import)

Records per-frame physics state + event flags, grouped by round.
Export to CSV via to_frames_df().to_csv() or JSON via to_dict().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.env.actions import user_input_to_action

_EVENT_KEYS: list[str] = [
    "p1_touch_ball",
    "p1_power_hit",
    "p1_diving",
    "p2_touch_ball",
    "p2_power_hit",
    "p2_diving",
    "ball_wall_bounce",
    "ball_net_collision",
]


@dataclass
class FrameRecord:
    """A single frame of game state + events."""

    frame: int
    round_number: int
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
    p1_touch_ball: bool = False
    p1_power_hit: bool = False
    p1_diving: bool = False
    p2_touch_ball: bool = False
    p2_power_hit: bool = False
    p2_diving: bool = False
    ball_wall_bounce: bool = False
    ball_net_collision: bool = False


# Backward compatibility alias
FrameSnapshot = FrameRecord


@dataclass
class RoundRecord:
    """Record of a single round (one point scored)."""

    round_number: int
    server: str  # "player_1" or "player_2"
    scorer: str  # "player_1" or "player_2"
    reward: dict[str, float]
    start_frame: int
    end_frame: int
    frames: list[FrameRecord] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    # Backward compatibility
    @property
    def duration(self) -> int:
        return self.num_frames

    @property
    def event_counts(self) -> dict[str, int]:
        counts = {k: 0 for k in _EVENT_KEYS}
        for f in self.frames:
            for k in _EVENT_KEYS:
                if getattr(f, k):
                    counts[k] += 1
        return counts

    def to_frames_df(self):
        """DataFrame of all frames in this round."""
        import pandas as pd

        return pd.DataFrame([asdict(f) for f in self.frames])


@dataclass
class GameRecord:
    """Complete record of a game (all rounds up to winning_score)."""

    num_frames: int = 0
    cumulative_rewards: dict[str, float] = field(default_factory=lambda: {"player_1": 0.0, "player_2": 0.0})
    rounds: list[RoundRecord] = field(default_factory=list)

    @property
    def scores(self) -> list[int]:
        p1 = sum(1 for r in self.rounds if r.scorer == "player_1")
        p2 = sum(1 for r in self.rounds if r.scorer == "player_2")
        return [p1, p2]

    @property
    def winner(self) -> str | None:
        s = self.scores
        if s[0] > s[1]:
            return "player_1"
        elif s[1] > s[0]:
            return "player_2"
        return None

    @property
    def frames(self) -> list[FrameRecord]:
        return [f for r in self.rounds for f in r.frames]

    @property
    def event_counts(self) -> dict[str, int]:
        totals: dict[str, int] = {k: 0 for k in _EVENT_KEYS}
        for r in self.rounds:
            for k, v in r.event_counts.items():
                totals[k] += v
        return totals

    def to_frames_df(self):
        """Flat DataFrame of all frames across all rounds."""
        import pandas as pd

        return pd.DataFrame([asdict(f) for f in self.frames])

    def to_rounds_df(self):
        """One row per round with aggregated stats."""
        import pandas as pd

        rows = []
        for r in self.rounds:
            rows.append(
                {
                    "round_number": r.round_number,
                    "server": r.server,
                    "scorer": r.scorer,
                    "num_frames": r.num_frames,
                    **r.event_counts,
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": self.scores,
            "num_frames": self.num_frames,
            "cumulative_rewards": self.cumulative_rewards,
            "winner": self.winner,
            "event_counts": self.event_counts,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "server": r.server,
                    "scorer": r.scorer,
                    "reward": r.reward,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "num_frames": r.num_frames,
                    "event_counts": r.event_counts,
                    "frames": [asdict(f) for f in r.frames],
                }
                for r in self.rounds
            ],
        }


@dataclass
class GamesRecord:
    """Collection of multiple games for aggregate analysis."""

    games: list[GameRecord] = field(default_factory=list)

    @property
    def num_games(self) -> int:
        return len(self.games)

    @property
    def num_frames(self) -> int:
        return sum(g.num_frames for g in self.games)

    @property
    def scores(self) -> list[int]:
        p1 = sum(g.scores[0] for g in self.games)
        p2 = sum(g.scores[1] for g in self.games)
        return [p1, p2]

    @property
    def win_counts(self) -> dict[str, int]:
        counts = {"player_1": 0, "player_2": 0}
        for g in self.games:
            if g.winner:
                counts[g.winner] += 1
        return counts

    @property
    def win_rate(self) -> dict[str, float]:
        n = self.num_games
        if n == 0:
            return {"player_1": 0.0, "player_2": 0.0}
        wc = self.win_counts
        return {"player_1": wc["player_1"] / n, "player_2": wc["player_2"] / n}

    @property
    def event_counts(self) -> dict[str, int]:
        totals: dict[str, int] = {k: 0 for k in _EVENT_KEYS}
        for g in self.games:
            for k, v in g.event_counts.items():
                totals[k] += v
        return totals

    def to_frames_df(self):
        """Flat DataFrame of all frames across all games."""
        import pandas as pd

        rows = []
        for i, g in enumerate(self.games):
            for f in g.frames:
                d = asdict(f)
                d["game_number"] = i + 1
                rows.append(d)
        return pd.DataFrame(rows)

    def to_rounds_df(self):
        """All rounds across all games."""
        import pandas as pd

        rows = []
        for i, g in enumerate(self.games):
            for r in g.rounds:
                rows.append(
                    {
                        "game_number": i + 1,
                        "round_number": r.round_number,
                        "server": r.server,
                        "scorer": r.scorer,
                        "num_frames": r.num_frames,
                        **r.event_counts,
                    }
                )
        return pd.DataFrame(rows)

    def to_games_df(self):
        """One row per game with summary stats."""
        import pandas as pd

        rows = []
        for i, g in enumerate(self.games):
            s = g.scores
            rows.append(
                {
                    "game_number": i + 1,
                    "p1_score": s[0],
                    "p2_score": s[1],
                    "winner": g.winner,
                    "num_frames": g.num_frames,
                    **g.event_counts,
                }
            )
        return pd.DataFrame(rows)


class RecordGame(BaseParallelWrapper):
    """Record game statistics and state snapshots.

    Args:
        env: The base PikachuVolleyballEnv.
        record_frames: Whether to record per-frame snapshots (default True).
            Set to False for lightweight stats-only recording.
    """

    def __init__(self, env, record_frames: bool = True) -> None:
        super().__init__(env)
        self._record_frames = record_frames
        self._current_game: GameRecord | None = None
        self._frame_count: int = 0
        self._round_start_frame: int = 1
        self._prev_scores: list[int] = [0, 0]
        self._current_round_frames: list[FrameRecord] = []
        self._current_server: str = "player_1"

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed=seed, options=options)
        self._current_game = GameRecord()
        self._frame_count = 0
        self._round_start_frame = 1
        self._prev_scores = [0, 0]
        self._current_round_frames = []
        self._current_server = "player_2" if self.env._is_player2_serve else "player_1"
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = super().step(actions)

        if self._current_game is None:
            return observations, rewards, terminations, truncations, infos

        self._frame_count += 1
        self._current_game.num_frames = self._frame_count

        self._current_game.cumulative_rewards["player_1"] += rewards.get("player_1", 0.0)
        self._current_game.cumulative_rewards["player_2"] += rewards.get("player_2", 0.0)

        # Record frame with physics state + events + user inputs
        physics = self.env._physics
        p1_info = infos.get("player_1", {})
        events = p1_info.get("events", {})
        ui = p1_info.get("user_inputs", {})
        p1_ui = ui.get("player_1", {})
        p2_ui = ui.get("player_2", {})
        if physics is not None and self._record_frames:
            def _ui_tuple(ui: dict) -> tuple[int, int, int]:
                return int(ui.get("x_direction", 0)), int(ui.get("y_direction", 0)), int(ui.get("power_hit", 0))

            p1_ui_vals = _ui_tuple(p1_ui)
            p2_ui_vals = _ui_tuple(p2_ui)
            self._current_round_frames.append(
                FrameRecord(
                    frame=self._frame_count,
                    round_number=len(self._current_game.rounds) + 1,
                    player1_action=user_input_to_action(*p1_ui_vals),
                    player1_x=int(physics.player1.x),
                    player1_y=int(physics.player1.y),
                    player1_state=int(physics.player1.state),
                    player2_action=user_input_to_action(*p2_ui_vals),
                    player2_x=int(physics.player2.x),
                    player2_y=int(physics.player2.y),
                    player2_state=int(physics.player2.state),
                    ball_x=int(physics.ball.x),
                    ball_y=int(physics.ball.y),
                    ball_x_velocity=int(physics.ball.x_velocity),
                    ball_y_velocity=int(physics.ball.y_velocity),
                    ball_is_power_hit=bool(physics.ball.is_power_hit),
                    p1_touch_ball=events.get("p1_touch_ball", False),
                    p1_power_hit=events.get("p1_power_hit", False),
                    p1_diving=events.get("p1_diving", False),
                    p2_touch_ball=events.get("p2_touch_ball", False),
                    p2_power_hit=events.get("p2_power_hit", False),
                    p2_diving=events.get("p2_diving", False),
                    ball_wall_bounce=events.get("ball_wall_bounce", False),
                    ball_net_collision=events.get("ball_net_collision", False),
                )
            )

        # Detect round end by score change
        current_scores = list(self.env._scores)
        if current_scores != self._prev_scores:
            round_number = len(self._current_game.rounds) + 1
            if current_scores[0] > self._prev_scores[0]:
                scorer = "player_1"
                reward = {"player_1": 1.0, "player_2": -1.0}
            else:
                scorer = "player_2"
                reward = {"player_1": -1.0, "player_2": 1.0}

            self._current_game.rounds.append(
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
            self._current_server = "player_2" if self.env._is_player2_serve else "player_1"

        # Add game stats to infos on game end (uses "episode" key for SB3 compatibility)
        if any(terminations.values()):
            for agent in infos:
                infos[agent]["episode"] = {
                    "length": self._current_game.num_frames,
                    "rewards": dict(self._current_game.cumulative_rewards),
                    "scores": self._current_game.scores,
                    "winner": self._current_game.winner,
                }

        return observations, rewards, terminations, truncations, infos

    def get_game_record(self) -> GameRecord | None:
        """Return the current game record (call after game ends)."""
        return self._current_game
