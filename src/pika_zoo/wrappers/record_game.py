"""
PettingZoo wrapper that records game state into the records hierarchy.

Data classes (FrameRecord, RoundRecord, GameRecord, GamesRecord) live in
pika_zoo.records — this module only contains the wrapper that populates them.
"""

from __future__ import annotations

from pettingzoo.utils import BaseParallelWrapper

from pika_zoo.env.actions import user_input_to_action
from pika_zoo.records import FrameRecord, FrameSnapshot, GameRecord, GamesRecord, RoundRecord

# Re-export for backward compatibility
__all__ = ["FrameRecord", "FrameSnapshot", "GameRecord", "GamesRecord", "RecordGame", "RoundRecord"]


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

            def _ui_tuple(u: dict) -> tuple[int, int, int]:
                return int(u.get("x_direction", 0)), int(u.get("y_direction", 0)), int(u.get("power_hit", 0))

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
