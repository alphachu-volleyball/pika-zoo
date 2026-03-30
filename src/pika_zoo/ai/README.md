# ai — Pluggable AI System

AI is fully separated from the physics engine. Any object satisfying the `AIPolicy` protocol can control a player.

## AIPolicy Protocol

```python
class AIPolicy(Protocol):
    def compute_action(self, player: Player, ball: Ball, opponent: Player, rng: Generator) -> UserInput: ...
    def reset(self, rng: Generator) -> None: ...
```

The environment calls `compute_action()` each frame for agents registered in `ai_policies`. The returned `UserInput` overrides the discrete action for that agent.

## Built-in Implementations

### BuiltinAI

Port of the original gorisanson AI. Includes intentional bugs from the original code (e.g., the net collision prediction mismatch). Registered as `"builtin"` with orange skin.

Exposes `BuiltinAI.calculate_expected_landing_point_x(ball)` as a static method — a lookahead simulation that predicts where the ball will land. Other AI implementations can reuse this without instantiating BuiltinAI.

### DuckllAI

Port of duckll's enhanced AI from [duckll/pikachu-volleyball](https://github.com/duckll/pikachu-volleyball). Significantly stronger than BuiltinAI, featuring:

- **Ball path prediction** — full trajectory simulation with 6-direction power-hit lookahead per frame
- **Fancy combos** — multi-touch aerial attacks (hit → reposition → hit again)
- **Anti-block** — predicts opponent blocking and switches direction
- **Serve machine** — 10 (P1) / 8 (P2) pre-programmed frame-perfect serve formulas
- **5 defense modes** — mid, mid_plus, mirror, predict, close

Registered as `"duckll"` with azure skin. 11 difficulty presets (0–10):

```python
DuckllAI()            # default: preset 10 (invincible)
DuckllAI(preset=0)    # beginner
DuckllAI(preset=5)    # mid-level
```

The `label` property returns `"duckll lv.N"` for display.

Attack pattern randomness is derived from the env seed via `reset(rng)`, ensuring reproducibility while maintaining per-episode variety.

### RandomAI

Selects a random action each frame. Registered as `"random"` with lime skin.

### SB3ModelPolicy

Adapter that wraps an SB3 model (PPO, etc.) as an `AIPolicy`. Used by the `play` script to load trained models.

```python
SB3ModelPolicy(
    model_path="model.zip",
    agent="player_1",              # required: determines action/observation mapping
    action_simplified=True,        # remap 13 simplified actions → 18 raw actions
    observation_simplified=False,  # mirror player_2 x-axis (SimplifyObservation)
    observation_normalized=True,   # normalize observations to [0, 1]
)
```

| Parameter | Default | Corresponding Wrapper |
|-----------|---------|----------------------|
| `action_simplified` | `True` | `SimplifyAction` — remap model output (0–12) to raw actions (0–17) |
| `observation_simplified` | `False` | `SimplifyObservation` — mirror player_2 x-axis (only applied for player_2) |
| `observation_normalized` | `True` | `NormalizeObservation` — min-max scale to [0, 1] |

Processing order: simplify → normalize (same as wrapper stacking order).

## Registry

Name-based lookup with associated skins:

```python
from pika_zoo.ai.registry import get_ai, get_skin

ai = get_ai("builtin")     # → BuiltinAI()
skin = get_skin("builtin")  # → "orange"
```

| Name | Class | Skin |
|------|-------|------|
| `"builtin"` | `BuiltinAI` | orange |
| `"duckll"` | `DuckllAI` | azure |
| `"random"` | `RandomAI` | lime |

## Files

| File | Description |
|------|-------------|
| `protocol.py` | `AIPolicy` typing.Protocol |
| `builtin.py` | Original gorisanson AI (with intentional bugs) |
| `duckll.py` | duckll's enhanced AI (prediction, decision, serve machine, config) |
| `random.py` | Random action baseline |
| `sb3_adapter.py` | SB3 model → AIPolicy adapter (optional dep) |
| `registry.py` | Name-based AI lookup + skin mapping |
