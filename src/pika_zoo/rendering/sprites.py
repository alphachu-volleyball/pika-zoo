"""
Sprite loading and color tinting for the Pikachu Volleyball renderer.

Loads PNG sprite assets and provides color tinting for visual distinction
between different models/algorithms.
"""

from __future__ import annotations

from pathlib import Path

import pygame

ASSETS_DIR = Path(__file__).parent / "assets"


def _load(name: str) -> pygame.Surface:
    """Load a sprite image from the assets directory."""
    return pygame.image.load(str(ASSETS_DIR / name)).convert_alpha()


def load_all_sprites() -> dict[str, pygame.Surface | tuple[pygame.Surface, ...]]:
    """Load all game sprites and return as a dict.

    Returns:
        Dict with keys: ball, ball_hyper, ball_punch, ball_trail,
        pikachu (28 frames), number (10 digits), shadow, cloud, wave,
        sky_blue, mountain, ground_red, ground_yellow, ground_line,
        ground_line_leftmost, ground_line_rightmost, net_pillar, net_pillar_top
    """
    sprites: dict[str, pygame.Surface | tuple[pygame.Surface, ...]] = {}

    # Ball sprites (6: rotation 0-4 + hyper)
    sprites["ball"] = tuple(_load(f"ball_{i}.png") for i in range(5)) + (_load("ball_hyper.png"),)
    sprites["ball_punch"] = _load("ball_punch.png")
    sprites["ball_trail"] = _load("ball_trail.png")

    # Player sprites (28 total across 7 states)
    pikachu_frames: list[pygame.Surface] = []
    for state, frame_count in [(0, 5), (1, 5), (2, 5), (3, 2), (4, 1), (5, 5), (6, 5)]:
        for frame in range(frame_count):
            pikachu_frames.append(_load(f"pikachu_{state}_{frame}.png"))
    sprites["pikachu"] = tuple(pikachu_frames)

    # Number sprites (0-9)
    sprites["number"] = tuple(_load(f"number_{i}.png") for i in range(10))

    # Environment
    sprites["shadow"] = _load("shadow.png")
    sprites["cloud"] = _load("cloud.png")
    sprites["wave"] = _load("wave.png")
    sprites["sky_blue"] = _load("sky_blue.png")
    sprites["mountain"] = _load("mountain.png")
    sprites["ground_red"] = _load("ground_red.png")
    sprites["ground_yellow"] = _load("ground_yellow.png")
    sprites["ground_line"] = _load("ground_line.png")
    sprites["ground_line_leftmost"] = _load("ground_line_leftmost.png")
    sprites["ground_line_rightmost"] = _load("ground_line_rightmost.png")
    sprites["net_pillar"] = _load("net_pillar.png")
    sprites["net_pillar_top"] = _load("net_pillar_top.png")

    return sprites


def get_player_sprite_index(state: int, frame_number: int) -> int:
    """Map player state + frame_number to sprite index in the pikachu tuple.

    State layout: 0(5) + 1(5) + 2(5) + 3(2) + 4(1) + 5(5) + 6(5) = 28 total
    """
    offsets = [0, 5, 10, 15, 17, 18, 23]
    return offsets[state] + frame_number


def tint_surface(surface: pygame.Surface, color: tuple[int, int, int]) -> pygame.Surface:
    """Apply a color tint to a surface for visual distinction.

    Args:
        surface: The original surface.
        color: RGB color to tint with (e.g. (255, 100, 100) for reddish).

    Returns:
        A new tinted surface.
    """
    tinted = surface.copy()
    tint_overlay = pygame.Surface(tinted.get_size(), flags=pygame.SRCALPHA)
    tint_overlay.fill((*color, 64))  # Semi-transparent tint
    tinted.blit(tint_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return tinted
