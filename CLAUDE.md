# pika-zoo - Claude Development Guide

## Project Overview

Python port of the Pikachu Volleyball (1997) physics engine + PettingZoo/Gymnasium reinforcement learning environment, based on reverse-engineered source code.

### Goals

- Accurately port the game physics from the original JS implementation ([gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball)) to Python
- Provide standard RL interfaces via PettingZoo `ParallelEnv` (two-player) + Gymnasium wrapper
- Referenced by training-center as a Git tag dependency for RL training

### Observation / Action Space

- **Observation**: Low-dimensional vector (positions, velocities, etc.) — CPU parallelization is the bottleneck, not GPU
- **Action**: Discrete action space (directional keys + jump combinations)

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

```
src/pika_zoo/
├── engine/              # Pure physics engine (no AI)
│   ├── constants.py     # Game constants (432×304, net, ball radius, etc.)
│   ├── types.py         # UserInput, PlayerState(IntEnum)
│   ├── rand.py          # rand() wrapper matching original [0, 32767]
│   └── physics.py       # PikaPhysics, Player, Ball + collision/movement
├── ai/                  # Pluggable AI system
│   ├── protocol.py      # AIPolicy (typing.Protocol)
│   ├── builtin.py       # Original gorisanson AI (with intentional bugs)
│   └── registry.py      # Name-based AI lookup
├── env/                 # PettingZoo ParallelEnv
│   ├── actions.py       # 18 discrete actions + ActionConverter (debouncing)
│   ├── observations.py  # 35-element agent-centric observation builder
│   └── pikachu_volleyball.py  # PikachuVolleyballEnv(ParallelEnv)
├── wrappers/            # PettingZoo wrappers
│   ├── convert_single_agent.py  # ParallelEnv → Gymnasium (for SB3)
│   ├── simplify_action.py       # 18 → 13 relative-direction actions
│   ├── normalize_observation.py # Min-max normalization to [0, 1]
│   ├── reward_shaping.py        # Ball position + normal state rewards
│   └── record_episode.py        # Per-round frame recording + JSON export
├── rendering/           # Pygame renderer + overlays (planned)
└── utils/               # Replay, random mode (planned)
```

### Key Design: AI Separation

The original JS embeds AI inside the physics engine (`processPlayerMovementAndSetPlayerPosition` calls `letComputerDecideUserInput`). In pika-zoo, AI is fully separated:

- `engine/physics.py`: pure function `(state, inputs) → next_state`
- `ai/protocol.py`: `AIPolicy` Protocol — any object with `compute_action()` works
- Environment layer calls `AIPolicy.compute_action()` before stepping physics

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
