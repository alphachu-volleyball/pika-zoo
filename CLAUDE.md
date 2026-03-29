# pika-zoo - Claude Development Guide

## Project Overview

Python port of the Pikachu Volleyball physics engine + PettingZoo/Gymnasium reinforcement learning environment, based on the reverse-engineered JS source code ([gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball)).

### Goals

- Accurately port the game physics from the reverse-engineered JS to Python
- Provide standard RL interfaces via PettingZoo `ParallelEnv` (two-player) + Gymnasium wrapper
- Referenced by training-center as a Git tag dependency for RL training

### Observation / Action Space

- **Observation**: 35-element agent-centric vector — see [env/README.md](src/pika_zoo/env/README.md#observation-space)
- **Action**: 18 discrete actions (13 with SimplifyAction) — see [env/README.md](src/pika_zoo/env/README.md#action-space)
- CPU parallelization is the bottleneck, not GPU

## Architecture

```
alphachu-volleyball/
├── pika-zoo (this repo)      ← RL environment + physics engine
├── training-center           ← PPO, Self-play, PFSP training
├── world-tournament          ← Web demo (GitHub Pages)
└── vs-recorder               ← Replay analysis (future)
```

training-center → pika-zoo: Git tag pinning (`pika-zoo @ git+...@v0.1.0`)
training-center → world-tournament: ONNX models (GitHub Releases)

### Package Structure

Each package has its own README with detailed documentation:

| Package | Description | README |
|---------|-------------|--------|
| `engine/` | Pure physics engine (no AI) | [engine/README.md](src/pika_zoo/engine/README.md) |
| `ai/` | Pluggable AI system | [ai/README.md](src/pika_zoo/ai/README.md) |
| `env/` | PettingZoo ParallelEnv | [env/README.md](src/pika_zoo/env/README.md) |
| `wrappers/` | PettingZoo / Gymnasium wrappers | [wrappers/README.md](src/pika_zoo/wrappers/README.md) |
| `records/` | Game record data structures (FrameRecord, GameRecord, etc.) | [records/README.md](src/pika_zoo/records/README.md) |
| `rendering/` | Pygame renderer + overlays | [rendering/README.md](src/pika_zoo/rendering/README.md) |
| `scripts/` | CLI commands | [scripts/README.md](src/pika_zoo/scripts/README.md) |

### Key Design: AI Separation

The original JS embeds AI inside the physics engine (`processPlayerMovementAndSetPlayerPosition` calls `letComputerDecideUserInput`). In pika-zoo, AI is fully separated:

- `engine/physics.py`: pure function `(state, inputs) → next_state`
- `ai/protocol.py`: `AIPolicy` Protocol — any object with `compute_action()` works
- Environment layer calls `AIPolicy.compute_action()` before stepping physics

## Physics Engine: Left-Right Asymmetry

The original game has several left-right asymmetries in its integer physics. These are **intentionally preserved** — do not "fix" them. See [engine/README.md](src/pika_zoo/engine/README.md#left-right-asymmetry) for the full list.

Key rules when working on the physics engine:

- **Do not symmetrize** collision boundaries, wall ranges, or power hit logic — they must match the original JS
- Observation mirroring is provided as an opt-in wrapper (`SimplifyObservation`), not baked into the engine or env
- By default, player 1 and player 2 models are trained and evaluated separately

## Development Environment

- **Python**: 3.10+
- **Package manager**: uv (`pyproject.toml` + `uv.lock`)
- **Linter/Formatter**: ruff
- **Testing**: pytest

### Commands

```bash
uv sync                  # Install dependencies
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pytest            # Test
```

## Code Quality

### ruff Configuration

```toml
[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

## Version Control & Git

### Semantic Versioning

`MAJOR.MINOR.PATCH` — Versions are tagged with Git tags (e.g. `v0.1.0`)

### Branch Workflow

```
feat/* ──(squash merge)──► release/{version} ──(merge commit)──► main ──► tag
fix/*  ──(squash merge)──►
```

- feat/fix → release: squash merge (PR required)
- release → main: merge commit (PR required)

### Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <subject>

feat(env): add random ball position mode
fix(physics): correct ball-net collision
docs(readme): update architecture diagram
chore(ci): add ruff lint workflow
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`

## CI/CD (GitHub Actions)

| Trigger | Action |
|---------|--------|
| PR, push to main | ruff lint, pytest |
| tag push (`v*`) | Create release |

Training is not run in CI (requires GPU, long-running).

## Code Copy Policy

No submodules — significant customization needed. When copying external code, always include:

- Original source URL
- License file (LICENSE)
- Change log (ATTRIBUTION.md)

### Reference Sources

| Source | License | Notes |
|--------|---------|-------|
| [helpingstar/pika-zoo](https://github.com/helpingstar/pika-zoo) | MIT | PettingZoo environment reference |
| [hankluo6/gym-pikachu-volleyball](https://github.com/hankluo6/gym-pikachu-volleyball) | TBD | Gymnasium environment reference |
| [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) | UNLICENSED (TBD) | Original reverse-engineered JS |

## Hardware Notes

- AMD Ryzen 7 3700X (8C/16T), NVIDIA RTX 2080 Super (8GB)
- Low-dimensional vector obs + MLP policy → CPU (env parallelization) is the bottleneck
- SB3 `SubprocVecEnv` with 8–16 parallel environments
