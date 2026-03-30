"""
Keyboard input handler for human players.

Supports multiple keymap presets:
  original:   D(left) G(right) R(up) V(down) Z(power hit)
  wasd:       A(left) D(right) W(up) S(down) Enter(power hit)
  arrows:     Arrow keys + Space(power hit)

Returns a discrete action index (0-17) compatible with the env's action space.
"""

from __future__ import annotations

import numpy as np
import pygame

from pika_zoo.env.actions import ACTION_TABLE

# Keymap presets: each maps logical actions to pygame keys
KEYMAPS: dict[str, dict[str, int]] = {
    "original": {
        "left": pygame.K_d,
        "right": pygame.K_g,
        "up": pygame.K_r,
        "down": pygame.K_v,
        "power": pygame.K_z,
    },
    "wasd": {
        "left": pygame.K_a,
        "right": pygame.K_d,
        "up": pygame.K_w,
        "down": pygame.K_s,
        "power": pygame.K_RETURN,
    },
    "arrows": {
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
        "power": pygame.K_SPACE,
    },
}

# Default keymaps per player
DEFAULT_KEYMAPS = {"player_1": "original", "player_2": "arrows"}


def get_keymap(name: str) -> dict[str, int]:
    """Get a keymap by name."""
    if name not in KEYMAPS:
        available = ", ".join(sorted(KEYMAPS.keys()))
        raise KeyError(f"Unknown keymap: {name!r}. Available: {available}")
    return KEYMAPS[name]


def keymap_help(name: str) -> str:
    """Return a human-readable description of a keymap."""
    km = get_keymap(name)
    key_name = pygame.key.name
    return (
        f"{key_name(km['left'])}(left) {key_name(km['right'])}(right) "
        f"{key_name(km['up'])}(up) {key_name(km['down'])}(down) "
        f"{key_name(km['power'])}(power hit)"
    )


def get_action_from_keys(
    keys: pygame.key.ScancodeWrapper,
    player: str = "player_1",
    keymap: str | None = None,
) -> int:
    """Read currently pressed keys and return the matching action index.

    Args:
        keys: Result of pygame.key.get_pressed()
        player: "player_1" or "player_2"
        keymap: Keymap preset name. If None, uses the default for the player.

    Returns:
        Discrete action index (0-17)
    """
    keymap_name = keymap or DEFAULT_KEYMAPS[player]
    bindings = get_keymap(keymap_name)

    key_array = np.array(
        [
            int(keys[bindings["left"]]),
            int(keys[bindings["right"]]),
            int(keys[bindings["up"]]),
            int(keys[bindings["down"]]),
            int(keys[bindings["power"]]),
        ],
        dtype=np.int32,
    )

    # Find matching action in ACTION_TABLE
    for i, row in enumerate(ACTION_TABLE):
        if np.array_equal(key_array, row):
            return i

    # No exact match (e.g. left+right pressed simultaneously) → NOOP
    return 0
