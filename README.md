# pika-zoo

[![Release](https://img.shields.io/github/v/release/alphachu-volleyball/pika-zoo?label=release&logo=github)](https://github.com/alphachu-volleyball/pika-zoo/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Python port of [Pikachu Volleyball Game](https://gorisanson.github.io/pikachu-volleyball/en/) as a [PettingZoo](https://pettingzoo.farama.org/) / [Gymnasium](https://gymnasium.farama.org/) reinforcement learning environment.

> **Original game**: Pikachu Volleyball (対戦ぴかちゅ～　ﾋﾞｰﾁﾊﾞﾚｰ編)
> — 1997 (C) SACHI SOFT / SAWAYAKAN Programmers, Satoshi Takenouchi
>
> **Reverse-engineered JS**: [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball)
> — The source code this project is based on

## Overview

Python port of the reverse-engineered JS source code, wrapped with standard RL interfaces.

- **Physics Engine**: Accurately reproduces the original ball trajectory, character movement, net collision, and scoring logic
- **PettingZoo**: Two-player multi-agent environment (`ParallelEnv`)
- **Gymnasium**: Single-agent wrapper (opponent fixed with a built-in policy)

### RL Pipeline

```mermaid
graph TD
    subgraph pika-zoo
        ENGINE["🎮 Physics Engine<br>Ball · Character · Collision · Scoring"]
        PZ["🔵 PettingZoo ParallelEnv<br>Player 1 ↔ Player 2"]
        GYM["🟢 Gymnasium Wrapper<br>Opponent fixed as built-in policy<br>Single-agent interface"]

        ENGINE -->|wrap| PZ
        PZ -->|fix opponent| GYM
    end

    subgraph training-center
        SB3["🟡 SB3 PPO Training<br>MLP policy network"]
    end

    GYM -->|train| SB3
```

> [!NOTE]
> **Why so complex?** — Pikachu Volleyball is a two-player game, but major RL libraries like SB3 only support single-agent training. We first create a multi-agent environment with PettingZoo, then use a Gymnasium wrapper that fixes the opponent inside the environment to make it look like a single-player game. During self-play, the opponent policy inside the wrapper is periodically swapped with past model versions.

## Quick Start

```bash
# Install
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Environment

- **Observation**: 35-element agent-centric vector `[self(13), opponent(13), ball(9)]` — [details](src/pika_zoo/env/README.md)
- **Action**: 18 discrete actions (3 x-dir x 3 y-dir x 2 power_hit) — [details](src/pika_zoo/env/README.md#action-space)
- **Wrappers**: opt-in, composable — [details](src/pika_zoo/wrappers/README.md)

```python
from pika_zoo.env import env
from pika_zoo.wrappers import SimplifyAction, SimplifyObservation, NormalizeObservation, ConvertSingleAgent

e = env(winning_score=15)
e = SimplifyAction(e)              # 18 → 13 relative actions
e = SimplifyObservation(e)         # mirror player_2 x-axis (optional)
e = NormalizeObservation(e)        # scale observations to [0, 1]
e = ConvertSingleAgent(e)          # PettingZoo → Gymnasium for SB3
```

## Physics Engine: Left-Right Asymmetry

The original Pikachu Volleyball uses integer-based physics with several left-right asymmetries (net collision boundary, power hit direction, wall bounce ranges, etc.). pika-zoo **intentionally preserves** these asymmetries so that RL agents train under the same conditions as the original game.

> [!IMPORTANT]
> Due to these asymmetries, **a single model cannot play both sides equally** unless `SimplifyObservation` is applied to mirror player_2's x-axis. By default, this project trains separate models for player 1 (left) and player 2 (right).

See [engine/README.md](src/pika_zoo/engine/README.md#left-right-asymmetry) for the full technical breakdown.

## Development

See [CLAUDE.md](CLAUDE.md) for the full development guide.

### Branch Workflow

```
feat/* ──(squash)──► release/{version} ──(merge)──► main ──► tag
```

## Related Projects

- [gorisanson/pikachu-volleyball](https://github.com/gorisanson/pikachu-volleyball) — Reverse-engineered JS reimplementation of the original game
- [helpingstar/pika-zoo](https://github.com/helpingstar/pika-zoo) — Pikachu Volleyball PettingZoo environment
- [hankluo6/Pikachu-VolleyBall-RL](https://github.com/hankluo6/Pikachu-VolleyBall-RL) — Prior work with PPO/ES
