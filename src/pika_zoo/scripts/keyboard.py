"""
Keyboard input handler for human players.

Key bindings (matching original game):
  Player 1: D(left) G(right) R(up) V(down) Z(power hit)
  Player 2: Arrow keys + Enter (power hit)

Returns a discrete action index (0-17) compatible with the env's action space.
"""

from __future__ import annotations

import numpy as np
import pygame

from pika_zoo.env.actions import ACTION_TABLE

# Player 1: D/G/R/V + Z (original game layout)
P1_KEYS = {
    "left": pygame.K_d,
    "right": pygame.K_g,
    "up": pygame.K_r,
    "down": pygame.K_v,
    "power": pygame.K_z,
}

# Player 2: Arrow keys + Enter
P2_KEYS = {
    "left": pygame.K_LEFT,
    "right": pygame.K_RIGHT,
    "up": pygame.K_UP,
    "down": pygame.K_DOWN,
    "power": pygame.K_RETURN,
}


def get_action_from_keys(keys: pygame.key.ScancodeWrapper, player: str = "player_1") -> int:
    """Read currently pressed keys and return the matching action index.

    Args:
        keys: Result of pygame.key.get_pressed()
        player: "player_1" or "player_2"

    Returns:
        Discrete action index (0-17)
    """
    bindings = P1_KEYS if player == "player_1" else P2_KEYS

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
