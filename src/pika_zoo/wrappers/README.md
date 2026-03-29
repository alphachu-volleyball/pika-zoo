# wrappers — PettingZoo / Gymnasium Wrappers

Opt-in, composable wrappers that transform the environment interface. Each wrapper addresses a specific concern.

## Recommended Stacking Order

```python
e = env(winning_score=15)
e = SimplifyAction(e)              # 1. action space reduction
e = SimplifyObservation(e)         # 2. x-axis mirroring (optional)
e = NormalizeObservation(e)        # 3. observation scaling
e = RewardShaping(e)               # 4. shaped rewards (optional)
e = ConvertSingleAgent(e)          # 5. multi-agent → single-agent
```

`SimplifyObservation` must come before `NormalizeObservation` — mirroring operates on raw coordinates.

## SimplifyAction

Reduces the action space from 18 absolute to 13 relative actions. Maps LEFT/RIGHT to TOWARD_NET/AWAY_FROM_NET per player.

| Index | Action |
|-------|--------|
| 0 | NOOP |
| 1 | FIRE |
| 2 | UP |
| 3 | TOWARD_NET |
| 4 | AWAY_FROM_NET |
| 5 | DOWN |
| 6 | UP + TOWARD_NET |
| 7 | UP + AWAY_FROM_NET |
| 8 | DOWN + TOWARD_NET |
| 9 | DOWN + AWAY_FROM_NET |
| 10 | UP + FIRE |
| 11 | TOWARD_NET + FIRE |
| 12 | AWAY_FROM_NET + FIRE |

Player 1 (left): TOWARD_NET = RIGHT, AWAY_FROM_NET = LEFT.
Player 2 (right): TOWARD_NET = LEFT, AWAY_FROM_NET = RIGHT.

## SimplifyObservation

Mirrors player_2's x-axis observations so both players see themselves on the left side of the court. Player_1 observations pass through unchanged.

### Mirrored Indices (player_2 only)

| Indices | Features | Transform |
|---------|----------|-----------|
| 0, 13 | self.x, opponent.x | 432 - x |
| 26, 28, 30 | ball.x, ball.prev_x, ball.prev_prev_x | 432 - x |
| 3, 16 | self.diving_direction, opponent.diving_direction | negate |
| 32 | ball.x_velocity | negate |

All other features (y positions, y velocities, states, etc.) are left unchanged.

> **Note**: This hides the physical left-right asymmetries of the original game engine. See the [Asymmetry section in README.md](../../../README.md#physics-engine-left-right-asymmetry) for details.

## NormalizeObservation

Min-max scales all observation features to [0, 1] using known physical ranges. Uses fixed bounds — no running statistics.

## RewardShaping

Adds dense rewards on top of the sparse scoring signal (+1/-1):

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ball_position_coeff` | 0.01 | Bonus when ball is on opponent's side, penalty on own side |
| `normal_state_coeff` | 0.0 | Small reward for staying in normal (ready) state |

Shaped rewards are zero-sum across agents.

## ConvertSingleAgent

Converts the two-player `ParallelEnv` to a single-agent `gymnasium.Env` for SB3 compatibility.

```python
ConvertSingleAgent(e, agent="player_1", opponent_policy=BuiltinAI())
```

Opponent can be:
- `AIPolicy` instance (e.g. `BuiltinAI`) — injected into `env.ai_policies`
- Callable `(obs → action)` — called each step
- `None` — random actions

## RecordGame

Unified game recording with hierarchical `FrameRecord → RoundRecord → GameRecord → GamesRecord` structure.

### FrameRecord (23 columns)

Each frame records player actions, positions/states, ball state, and event flags:

| Column | Type | Description |
|--------|------|-------------|
| `frame` | int | Frame number (1-based) |
| `round_number` | int | Current round (1-based) |
| `player1_action` | int | Player 1 action index (0-17) |
| `player1_x` | int | Player 1 x position |
| `player1_y` | int | Player 1 y position |
| `player1_state` | int | Player 1 state (0=normal, 1=jumping, 2=power_hitting, 3=diving, 4=lying_down) |
| `player2_action` | int | Player 2 action index (0-17) |
| `player2_x` | int | Player 2 x position |
| `player2_y` | int | Player 2 y position |
| `player2_state` | int | Player 2 state |
| `ball_x` | int | Ball x position |
| `ball_y` | int | Ball y position |
| `ball_x_velocity` | int | Ball x velocity |
| `ball_y_velocity` | int | Ball y velocity |
| `ball_is_power_hit` | bool | Ball has power hit status |
| `p1_touch_ball` | bool | Player 1 touched the ball |
| `p1_power_hit` | bool | Player 1 power-hit the ball |
| `p1_diving` | bool | Player 1 initiated a dive |
| `p2_touch_ball` | bool | Player 2 touched the ball |
| `p2_power_hit` | bool | Player 2 power-hit the ball |
| `p2_diving` | bool | Player 2 initiated a dive |
| `ball_wall_bounce` | bool | Ball bounced off a wall |
| `ball_net_collision` | bool | Ball hit the net pillar |

Action indices work correctly for both AI and human players (AI actions are reverse-mapped from UserInput via `user_input_to_action()`).

### CSV format (`--stats`)

`uv run play --stats game.csv` exports `to_frames_df()` as CSV. One row per frame, 23 columns, no index. Booleans as `True`/`False`.

### Consistent interface at each level

| | `num_frames` | `event_counts` | `scores` | `to_frames_df()` | `to_rounds_df()` | `to_games_df()` |
|---|---|---|---|---|---|---|
| RoundRecord | O | O | - | O | - | - |
| GameRecord | O | O | O (computed) | O | O | - |
| GamesRecord | O | O | O (sum) | O (+game_number) | O (+game_number) | O |

### Export

```python
record = e.get_game_record()

# Frame-level analysis (positions + events in one DataFrame)
df = record.to_frames_df()
df[df.p1_power_hit].ball_x.hist()              # ball position on power hits

# Round-level aggregation
record.to_rounds_df()

# Multi-game analysis
games = GamesRecord()
games.games.append(record)
games.win_rate                                  # {"player_1": 0.0, "player_2": 1.0}
games.to_games_df()
```

## Files

| File | Description |
|------|-------------|
| `simplify_action.py` | 18 → 13 relative-direction actions |
| `simplify_observation.py` | Mirror player_2 x-axis observations |
| `normalize_observation.py` | Min-max normalization to [0, 1] |
| `reward_shaping.py` | Ball position + normal state rewards |
| `convert_single_agent.py` | ParallelEnv → Gymnasium for SB3 |
| `record_game.py` | Per-round frame recording + JSON export |
