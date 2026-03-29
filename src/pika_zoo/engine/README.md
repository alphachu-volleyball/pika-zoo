# engine — Physics Engine

Pure Python port of the Pikachu Volleyball (1997) physics engine. No AI logic — input is provided externally.

Source: [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) (reverse-engineered JS)

## Coordinate System

- **Origin**: top-left corner
- **X**: right-increasing, range [0, 432] (`GROUND_WIDTH`)
- **Y**: down-increasing, range [0, 304] (0 = top, 304 = bottom)
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

The original game has several left-right asymmetries that are **intentionally preserved** so that RL agents train under the same conditions as the original game.

### 1. Net collision boundary

When the ball hits the side of the net pillar, its horizontal direction is determined by which side of center (`x = 216`) it is on.
The condition uses strict less-than (`<`), so a ball at exactly `x = 216` is treated as being on the **right side** and pushed rightward.
The net's effective center is biased 1px to the right.

### 2. Net collision: actual vs predicted

The physics engine uses two separate collision checks for the ball against the top of the net:

- **Actual ball**: uses `<=` (ball at the boundary counts as top collision, vertical bounce)
- **Predicted ball** (for AI landing-point calculation): uses `<` (same position counts as side collision, horizontal bounce)

At the exact boundary value, the real ball bounces vertically but the built-in AI predicts a horizontal bounce — causing occasional mispredictions.

### 3. Power hit direction

A power hit's horizontal direction is determined by **which side of the court the ball is on**, not by which player hit it or their input direction.
The player's directional input affects only the **speed** (10 if no direction key, 20 if any direction key is held), while the `abs()` strips the actual direction.

If a player jumps near the net and power-hits while the ball is on the opponent's side, the ball flies back into their own court.
At `x = 216`, the ball is treated as right-side (same `<` boundary as net collision).

### 4. Wall bounce asymmetry

- **Left wall**: ball bounces when its center < `BALL_RADIUS` (20) — effectively when the ball's **surface** touches `x = 0`
- **Right wall**: ball bounces when its center > `GROUND_WIDTH` (432) — the ball's **center** must pass the wall, ignoring the radius

This means the ball can travel 20px further to the right before bouncing. Measured from center court (216): 196px to the left wall, 216px to the right wall.

## rand()

`rand.py` wraps Python's RNG to match the original game's `rand()` behavior: returns integers in [0, 32767].

## Files

| File | Description |
|------|-------------|
| `physics.py` | `PikaPhysics`, `Player`, `Ball` + collision/movement logic |
| `constants.py` | Game constants (court dimensions, net, ball radius, etc.) |
| `types.py` | `UserInput`, `PlayerState(IntEnum)`, `NoiseConfig` |
| `rand.py` | `rand()` wrapper matching original [0, 32767] range |
