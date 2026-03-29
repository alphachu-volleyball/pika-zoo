# records — Game Record Data Structures

Pure data containers for game recording and analysis. No dependency on PettingZoo or the environment — usable by any consumer (training-center, vs-recorder, etc.).

## Hierarchy

```
GamesRecord
└── games: list[GameRecord]
    └── rounds: list[RoundRecord]
        └── frames: list[FrameRecord]
```

## FrameRecord (23 columns)

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame number (1-based) |
| `round_number` | int | Current round (1-based) |
| `player1_action` | int | Player 1 action index (0-17) |
| `player1_x` | int | Player 1 x position |
| `player1_y` | int | Player 1 y position |
| `player1_state` | int | Player 1 state (0-6, see below) |
| `player2_action` | int | Player 2 action index (0-17) |
| `player2_x` | int | Player 2 x position |
| `player2_y` | int | Player 2 y position |
| `player2_state` | int | Player 2 state |
| `ball_x` | int | Ball x position |
| `ball_y` | int | Ball y position |
| `ball_x_velocity` | int | Ball x velocity |
| `ball_y_velocity` | int | Ball y velocity |
| `ball_is_power_hit` | bool | Ball has power hit status |
| `p1_touch_ball` | bool | Player 1 touched the ball (rising edge) |
| `p1_power_hit` | bool | Player 1 power-hit the ball |
| `p1_diving` | bool | Player 1 initiated a dive |
| `p2_touch_ball` | bool | Player 2 touched the ball |
| `p2_power_hit` | bool | Player 2 power-hit the ball |
| `p2_diving` | bool | Player 2 initiated a dive |
| `ball_wall_bounce` | bool | Ball bounced off a wall |
| `ball_net_collision` | bool | Ball hit the net pillar |

### Player States

| Value | State | Description |
|-------|-------|-------------|
| 0 | NORMAL | Idle / moving on ground |
| 1 | JUMPING | In the air |
| 2 | JUMPING_POWER_HIT | Jump + power hit stance |
| 3 | DIVING | Diving |
| 4 | LYING_DOWN | Recovering after dive |
| 5 | WIN | Victory pose (game end) |
| 6 | LOSE | Defeat pose (game end) |

### Action vs Event

`player1_action` is the input (what the player chose to do). `p1_power_hit` is the event (the ball was actually hit with power). A player can be in state 2 (power hit stance) for multiple frames, but `p1_power_hit` is True only on the single frame where ball contact occurs.

## RoundRecord

| Field | Type | Description |
|-------|------|-------------|
| `round_number` | int | Sequential round number |
| `server` | str | Who served ("player_1" or "player_2") |
| `scorer` | str | Who scored |
| `reward` | dict | {"player_1": ±1.0, "player_2": ∓1.0} |
| `start_frame` | int | First frame of round |
| `end_frame` | int | Last frame of round |
| `frames` | list | FrameRecords in this round |

Computed: `num_frames`, `event_counts`, `to_frames_df()`

## GameRecord

| Field | Type | Description |
|-------|------|-------------|
| `num_frames` | int | Total frame count |
| `cumulative_rewards` | dict | Accumulated rewards per player |
| `rounds` | list | RoundRecords |

Computed: `scores`, `winner`, `event_counts`, `to_frames_df()`, `to_rounds_df()`, `to_dict()`

## GamesRecord

| Field | Type | Description |
|-------|------|-------------|
| `games` | list | GameRecords |

Computed: `num_games`, `num_frames`, `scores`, `win_counts`, `win_rate`, `event_counts`, `to_frames_df()`, `to_rounds_df()`, `to_games_df()`

## Consistent Interface

| | `num_frames` | `event_counts` | `scores` | `to_frames_df()` |
|---|---|---|---|---|
| RoundRecord | O | O | - | O |
| GameRecord | O | O | O (computed) | O |
| GamesRecord | O | O | O (sum) | O (+game_number) |

## CSV Format

`--stats` exports `to_frames_df()` as CSV. One row per frame, 23 columns, no index. Booleans as `True`/`False`.

## Files

| File | Description |
|------|-------------|
| `types.py` | FrameRecord, RoundRecord, GameRecord, GamesRecord dataclasses |
