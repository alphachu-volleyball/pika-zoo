# rendering — Pygame Renderer

Pygame-based renderer for visualizing game state. Supports live display and headless frame capture.

## Render Modes

| Mode | Description |
|------|-------------|
| `"human"` | Opens a pygame window (432 x 304) |
| `"rgb_array"` | Returns frames as numpy arrays (for recording) |

## Overlays

The renderer draws metadata overlays on top of the game:

- **Score**: current score at top of screen
- **Player labels**: names below each player
- **Mode label**: "normal" or `noise(x=5, xv=3, yv=0)` at top center

## Skins

Player sprites are loaded from `assets/pikachu_sprites/{skin}/`. Available skins are determined by subdirectory names (e.g. `yellow`, `orange`, `white`, `lime`).

Skin conventions:

| Skin | Used for |
|------|----------|
| yellow | Human player |
| orange | Builtin AI |
| lime | Random AI |
| white | RL model (SB3) |

Each skin contains sprite sheets for the 6 player states + ball and background sprites.

## Files

| File | Description |
|------|-------------|
| `renderer.py` | `PygameRenderer` — main rendering class |
| `sprites.py` | Sprite loading + per-player skin management |
| `overlays.py` | `TextOverlay`, `MetadataOverlay` |
| `assets/` | PNG sprites (pikachu_sprites, ball, background) |
