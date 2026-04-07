"""
AI registry — name-based lookup for AI implementations with skin mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.ai.duckll import DuckllAI
from pika_zoo.ai.random import RandomAI
from pika_zoo.ai.stone import StoneAI

DEFAULT_SKIN = "lime"


@dataclass
class AIEntry:
    cls: type
    skin: str


_REGISTRY: dict[str, AIEntry] = {}


def register_ai(name: str, cls: type, skin: str = DEFAULT_SKIN) -> None:
    """Register an AI class by name with an associated skin.

    Args:
        name: Unique name for this AI.
        cls: AI class (must satisfy AIPolicy protocol).
        skin: Pikachu skin folder name (default: "lime").
    """
    _REGISTRY[name] = AIEntry(cls=cls, skin=skin)


def _parse_name(spec: str) -> tuple[str, dict[str, Any]]:
    """Parse AI spec like ``"duckll"`` or ``"duckll:3"`` into (name, kwargs).

    The ``:N`` suffix is passed as ``preset=N`` to the constructor.
    """
    if ":" in spec:
        name, arg = spec.split(":", 1)
        return name, {"preset": int(arg)}
    return spec, {}


def get_ai(spec: str) -> Any:
    """Create an AI instance by spec.

    Supports ``"name"`` or ``"name:arg"`` format (e.g. ``"duckll:3"``).
    """
    name, kwargs = _parse_name(spec)
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown AI: {name!r}. Available: {available}")
    return _REGISTRY[name].cls(**kwargs)


def get_skin(spec: str) -> str:
    """Get the skin associated with an AI spec.

    Returns DEFAULT_SKIN for unknown names (e.g. "human").
    """
    name, _ = _parse_name(spec)
    if name in _REGISTRY:
        return _REGISTRY[name].skin
    return DEFAULT_SKIN


# Auto-register builtins
register_ai("builtin", BuiltinAI, skin="orange")
register_ai("random", RandomAI, skin="lime")
register_ai("duckll", DuckllAI, skin="azure")
register_ai("stone", StoneAI, skin="gray")
register_ai("stone_random", lambda: StoneAI(random_position=True), skin="gray")
