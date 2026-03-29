# env — PettingZoo Environment

PettingZoo `ParallelEnv` for Pikachu Volleyball. Two agents (`player_1`, `player_2`) act simultaneously each frame.

## Action Space

18 discrete actions covering all combinations of 3 x-directions, 3 y-directions, and 2 power_hit states.

Each action maps to a 5-element binary vector `[left, right, up, down, power_hit]`:

| Index | Action | Keys |
|-------|--------|------|
| 0 | NOOP | — |
| 1 | FIRE | power_hit |
| 2 | UP | up |
| 3 | RIGHT | right |
| 4 | LEFT | left |
| 5 | DOWN | down |
| 6 | UP+RIGHT | up, right |
| 7 | UP+LEFT | up, left |
| 8 | DOWN+RIGHT | down, right |
| 9 | DOWN+LEFT | down, left |
| 10 | UP+FIRE | up, power_hit |
| 11 | RIGHT+FIRE | right, power_hit |
| 12 | LEFT+FIRE | left, power_hit |
| 13 | DOWN+FIRE | down, power_hit |
| 14 | UP+RIGHT+FIRE | up, right, power_hit |
| 15 | UP+LEFT+FIRE | up, left, power_hit |
| 16 | DOWN+RIGHT+FIRE | down, right, power_hit |
| 17 | DOWN+LEFT+FIRE | down, left, power_hit |

`ActionConverter` handles power_hit debouncing — `power_hit=1` only triggers on the rising edge (key was not pressed → now pressed), matching the original game behavior.

## Observation Space

35-element agent-centric vector: `[self(13), opponent(13), ball(9)]`.

Each agent always sees itself at indices 0–12 and its opponent at 13–25, regardless of which side it is on. Coordinates are **absolute** (x=0 is left wall, x=432 is right wall).

### Player Features (13 per player)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | x | [0, 432] | Horizontal position |
| 1 | y | [0, 244] | Vertical position (0=top, 244=ground) |
| 2 | y_velocity | [-16, 16] | Vertical velocity |
| 3 | diving_direction | {-1, 0, 1} | Dive horizontal direction |
| 4 | lying_down_duration_left | [-1, 3] | Frames until recovery |
| 5 | frame_number | [0, 4] | Current animation frame |
| 6 | delay_before_next_frame | [0, 5] | Animation delay counter |
| 7 | state: normal | {0, 1} | One-hot |
| 8 | state: jumping | {0, 1} | One-hot |
| 9 | state: power_hitting | {0, 1} | One-hot |
| 10 | state: diving | {0, 1} | One-hot |
| 11 | state: lying_down | {0, 1} | One-hot |
| 12 | prev_power_hit | {0, 1} | Previous frame's power_hit key state |

### Ball Features (9)

| Index | Feature | Range |
|-------|---------|-------|
| 26 | x | [0, 432] |
| 27 | y | [0, 304] |
| 28 | previous_x | [0, 432] |
| 29 | previous_y | [0, 304] |
| 30 | previous_previous_x | [0, 432] |
| 31 | previous_previous_y | [0, 304] |
| 32 | x_velocity | [-20, 20] |
| 33 | y_velocity | [-30, 30] |
| 34 | is_power_hit | {0, 1} |

## Files

| File | Description |
|------|-------------|
| `pikachu_volleyball.py` | `PikachuVolleyballEnv(ParallelEnv)` — main environment class |
| `actions.py` | `ACTION_TABLE` (18 actions) + `ActionConverter` (debouncing) |
| `observations.py` | `build_observation()` — 35-element vector builder |
