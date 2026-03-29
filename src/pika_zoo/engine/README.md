# engine — Physics Engine

Pure Python port of the Pikachu Volleyball (1997) physics engine. No AI logic — input is provided externally.

Source: [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) (reverse-engineered JS)

## Coordinate System

- **Ground**: 432 x 304 pixels (x: [0, 432], y: [0, 304])
- **Origin**: top-left corner
- **X**: right-increasing
- **Y**: down-increasing (0 = top, 304 = bottom)
- **Net**: x = 216 (center), top at y = 176, bottom of pillar top at y = 192

## Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `GROUND_WIDTH` | 432 | Court width |
| `GROUND_HALF_WIDTH` | 216 | Net x position |
| `PLAYER_HALF_LENGTH` | 32 | Player collision radius |
| `PLAYER_TOUCHING_GROUND_Y_COORD` | 244 | Player ground level |
| `BALL_RADIUS` | 20 | Ball collision radius |
| `BALL_TOUCHING_GROUND_Y_COORD` | 252 | Ball ground level |
| `NET_PILLAR_HALF_WIDTH` | 25 | Net collision half-width |
| `NET_PILLAR_TOP_TOP_Y_COORD` | 176 | Top of net pillar |
| `NET_PILLAR_TOP_BOTTOM_Y_COORD` | 192 | Bottom of net pillar top |

## Objects

### Player

Two players: player 1 (left, starting x=36) and player 2 (right, starting x=396).

States: `NORMAL`(0), `JUMPING`(1), `JUMPING_POWER_HIT`(2), `DIVING`(3), `LYING_DOWN`(4), `WIN`(5), `LOSE`(6).

### Ball

Starts at x=56 (player 1 serve) or x=376 (player 2 serve), y=0, y_velocity=1 (falling).

Collision handling:
- **Walls**: left at x < 20 (surface), right at x > 432 (center — see asymmetry note)
- **Ceiling**: y < 0 → y_velocity = 1
- **Net**: pillar half-width = 25, top bounce vs side bounce depending on y position
- **Ground**: y ≥ 252 → round ends, scorer determined by which side the ball lands on

### PikaPhysics

Main physics container. Holds player1, player2, and ball. `run_engine_for_next_frame(user_inputs, rng)` advances one frame.

## Left-Right Asymmetry

The original game has several asymmetries that are intentionally preserved. See [README.md](../../../README.md#physics-engine-left-right-asymmetry) for the complete list.

## rand()

`rand.py` wraps Python's RNG to match the original game's `rand()` behavior: returns integers in [0, 32767].

## Files

| File | Description |
|------|-------------|
| `physics.py` | `PikaPhysics`, `Player`, `Ball` + collision/movement logic |
| `constants.py` | Game constants (court dimensions, net, ball radius, etc.) |
| `types.py` | `UserInput`, `PlayerState(IntEnum)`, `NoiseConfig` |
| `rand.py` | `rand()` wrapper matching original [0, 32767] range |
