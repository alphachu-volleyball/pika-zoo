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

Adds dense rewards via pluggable reward channels. Each channel is a `(RewardChannel, coefficient)` pair.

```python
from pika_zoo.wrappers import RewardShaping, LinearBallPosition, QuadrantBallPosition

# Channel-based
RewardShaping(e, channels=[
    (LinearBallPosition(), 0.01),
    (QuadrantBallPosition(), 0.005),
])

# Preset
RewardShaping.from_preset(e, "default")
```

### Built-in Channels

| Channel | Description | Zero-sum |
|---------|-------------|----------|
| `LinearBallPosition()` | Continuous reward based on ball x position | Yes |
| `QuadrantBallPosition()` | Zone-based reward (4 quadrants, configurable) | Configurable |
| `OpponentDistance()` | Reward when ball is far from opponent on their side | Yes |
| `BallDownwardVelocity()` | Reward for fast downward velocity (spike) | Yes |

### Presets

| Preset | Channels |
|--------|----------|
| `"default"` | `LinearBallPosition` (0.01) |

### Custom Channels

Extend `RewardChannel` and implement `__call__` and `__repr__`:

```python
from pika_zoo.wrappers import RewardChannel

class MyChannel(RewardChannel):
    def __call__(self, physics):
        # Access physics.player1, physics.player2, physics.ball
        return (p1_reward, p2_reward)

    def __repr__(self):
        return "MyChannel()"
```

`__repr__` is used for W&B config logging (`str(channel)`).

## ConvertSingleAgent

Converts the two-player `ParallelEnv` to a single-agent `gymnasium.Env` for SB3 compatibility.

```python
ConvertSingleAgent(e, agent="player_1", opponent_policy=BuiltinAI())
```

Opponent can be:
- `AIPolicy` instance (e.g. `BuiltinAI`) — injected into `env.ai_policies`
- Callable `(obs → action)` — called each step with the previous step's observation
- `None` — random actions

> **Note**: Callable opponents receive the observation from the previous `step()`/`reset()`, not a fresh observation. This avoids accessing private env internals and works correctly with wrapped environments (e.g. `NormalizeObservation`).

## RecordGame

Wraps the env to record per-frame state into the hierarchical record structure. Data classes live in [`pika_zoo.records`](../records/README.md).

```python
from pika_zoo.wrappers import RecordGame
from pika_zoo.records import GameRecord, GamesRecord

e = RecordGame(env(winning_score=15))
e.reset()
# ... game loop ...
record = e.get_game_record()    # → GameRecord

df = record.to_frames_df()      # 23-column DataFrame
record.to_rounds_df()           # per-round aggregation
record.to_dict()                # JSON export
```

## Files

| File | Description |
|------|-------------|
| `simplify_action.py` | 18 → 13 relative-direction actions |
| `simplify_observation.py` | Mirror player_2 x-axis observations |
| `normalize_observation.py` | Min-max normalization to [0, 1] |
| `reward_shaping.py` | Pluggable reward channels wrapper |
| `reward_channels.py` | Built-in reward channel functions |
| `convert_single_agent.py` | ParallelEnv → Gymnasium for SB3 |
| `record_game.py` | Per-round frame recording + JSON export |
