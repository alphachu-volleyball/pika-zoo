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

### RandomAI

Selects a random action each frame. Registered as `"random"` with lime skin.

### SB3ModelPolicy

Adapter that wraps an SB3 model (PPO, etc.) as an `AIPolicy`. Used by the `play` script to load trained models.

```python
SB3ModelPolicy(
    model_path="model.zip",
    agent="player_1",       # required: determines action/observation mapping
    simplified=True,        # remap 13 simplified actions → 18 raw actions
    normalized=True,        # normalize observations before prediction
)
```

- `simplified=True`: remaps model output (0–12) through the per-player `SimplifyAction` mapping table to raw actions (0–17)
- `normalized=True`: applies the same min-max scaling as `NormalizeObservation` before passing observations to the model

Both default to `True` since the standard training pipeline uses `SimplifyAction` + `NormalizeObservation` wrappers.

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
| `"random"` | `RandomAI` | lime |

## Files

| File | Description |
|------|-------------|
| `protocol.py` | `AIPolicy` typing.Protocol |
| `builtin.py` | Original gorisanson AI (with intentional bugs) |
| `random.py` | Random action baseline |
| `sb3_adapter.py` | SB3 model → AIPolicy adapter (optional dep) |
| `registry.py` | Name-based AI lookup + skin mapping |
