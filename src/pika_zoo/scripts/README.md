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
| `--p1`, `--p2` | `builtin` | Player spec: `"builtin"`, `"random"`, `"human"`, or path to `.zip` model |
| `--winning-score` | 15 | Score to win |
| `--seed` | None | Random seed |
| `--fps` | 25 | Frame rate |
| `--no-render` | off | Disable pygame window |
| `--record FILE` | None | Record to MP4 (requires ffmpeg) |
| `--stats FILE` | None | Save per-frame stats to CSV (positions + events) |
| `--noise-x N` | None | Ball x position noise ±N pixels |
| `--noise-x-vel N` | None | Ball x velocity noise ±N |
| `--noise-y-vel N` | None | Ball y velocity noise ±N |
| `--p1-skin`, `--p2-skin` | auto | Override pikachu skin |
| `--p1-label`, `--p2-label` | auto | Override display label |

### Keyboard Controls

| Player 1 | Player 2 | Action |
|----------|----------|--------|
| D | Left arrow | Move left |
| G | Right arrow | Move right |
| R | Up arrow | Jump |
| V | Down arrow | Dive |
| Z | Enter | Power hit |

## Files

| File | Description |
|------|-------------|
| `play.py` | `uv run play` — watch, play, or record matches |
| `keyboard.py` | Keyboard input handler (pygame key mapping) |
| `video.py` | `FFmpegWriter` for MP4 recording |
