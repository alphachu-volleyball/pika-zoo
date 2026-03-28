"""
AI registry — name-based lookup for AI implementations.
"""

from __future__ import annotations

from typing import Any

from pika_zoo.ai.builtin import BuiltinAI

_REGISTRY: dict[str, type] = {}


def register_ai(name: str, cls: type) -> None:
    """Register an AI class by name."""
    _REGISTRY[name] = cls


def get_ai(name: str) -> Any:
    """Create an AI instance by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown AI: {name!r}. Available: {available}")
    return _REGISTRY[name]()


# Auto-register builtins
register_ai("builtin", BuiltinAI)
