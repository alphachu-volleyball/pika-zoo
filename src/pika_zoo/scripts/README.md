# scripts — CLI Commands

Command-line tools for playing, watching, and recording Pikachu Volleyball matches.

## play

Main entry point. Registered as `uv run play` in pyproject.toml.

### Usage

```bash
uv run play                                        # builtin vs builtin, windowed
uv run play --p1 human                             # human vs builtin AI
uv run play --p1 human --p2 human                  # human vs human
uv run play --p1 model.zip --p2 builtin            # SB3 model vs builtin
uv run play --no-render --record match.mp4         # headless recording
uv run play --record match.mp4                     # windowed + recording
uv run play --no-render                            # headless (stats only)
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--p1`, `--p2` | `builtin` | Player spec: `"builtin"`, `"duckll"`, `"duckll:N"`, `"random"`, `"stone"`, `"human"`, or model path |
| `--winning-score` | 15 | Score to win |
| `--seed` | None | Random seed |
| `--fps` | 25 | Frame rate |
| `--no-render` | off | Disable pygame window |
| `--record FILE` | None | Record to MP4 (requires ffmpeg) |
| `--stats FILE` | None | Save per-frame stats to CSV (positions + events) |
| `--noise-x N` | None | Ball x position noise ±N pixels |
| `--noise-x-vel N` | None | Ball x velocity noise ±N |
| `--noise-y-vel N` | None | Ball y velocity noise ±N |
| `--p1-skin`, `--p2-skin` | auto | Override pikachu skin (azure, gray, lime, orange, white, yellow) |
| `--p1-label`, `--p2-label` | auto | Override display label |
| `--p1-keymap`, `--p2-keymap` | P1=original, P2=arrows | Keyboard layout preset |

### Keyboard Controls

Three keymap presets are available, selectable via `--p1-keymap` / `--p2-keymap`:

| Action | `original` (P1 default) | `wasd` | `arrows` (P2 default) |
|--------|------------------------|--------|----------------------|
| Move left | D | A | Left arrow |
| Move right | G | D | Right arrow |
| Jump | R | W | Up arrow |
| Dive | V | S | Down arrow |
| Power hit | Z | Enter | Space |

```bash
uv run play --p1 human --p2 duckll --p1-keymap arrows    # P1 uses arrow keys
uv run play --p1 human --p2 human --p1-keymap wasd --p2-keymap arrows
```

## Files

| File | Description |
|------|-------------|
| `play.py` | `uv run play` — watch, play, or record matches |
| `keyboard.py` | Keyboard input handler (pygame key mapping) |
| `video.py` | `FFmpegWriter` for MP4 recording |
