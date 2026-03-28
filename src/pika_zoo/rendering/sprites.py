"""
Sprite loading and color tinting for the Pikachu Volleyball renderer.

Loads PNG sprite assets and provides color tinting for visual distinction
between different models/algorithms.
"""

from __future__ import annotations

from pathlib import Path

import pygame

ASSETS_DIR = Path(__file__).parent / "assets"
PIKACHU_SPRITES_DIR = ASSETS_DIR / "pikachu_sprites"
DEFAULT_PIKACHU_SKIN = "yellow"


def _load(path: str | Path, alpha: bool = True) -> pygame.Surface:
    """Load a sprite image."""
    surface = pygame.image.load(str(path))
    return surface.convert_alpha() if alpha else surface.convert()


def load_pikachu_sprites(skin: str = DEFAULT_PIKACHU_SKIN) -> tuple[pygame.Surface, ...]:
    """Load 28 pikachu sprite frames for a given skin.

    Args:
        skin: Skin folder name under pikachu_sprites/ (e.g. "yellow").
    """
    skin_dir = PIKACHU_SPRITES_DIR / skin
    frames: list[pygame.Surface] = []
    for state, frame_count in [(0, 5), (1, 5), (2, 5), (3, 2), (4, 1), (5, 5), (6, 5)]:
        for frame in range(frame_count):
            frames.append(_load(skin_dir / f"pikachu_{state}_{frame}.png"))
    return tuple(frames)


def load_all_sprites(
    p1_skin: str = DEFAULT_PIKACHU_SKIN,
    p2_skin: str = DEFAULT_PIKACHU_SKIN,
) -> dict[str, pygame.Surface | tuple[pygame.Surface, ...]]:
    """Load all game sprites and return as a dict.

    Args:
        p1_skin: Pikachu skin for player 1.
        p2_skin: Pikachu skin for player 2.
    """
    sprites: dict[str, pygame.Surface | tuple[pygame.Surface, ...]] = {}

    # Ball sprites (6: rotation 0-4 + hyper)
    sprites["ball"] = tuple(_load(ASSETS_DIR / f"ball_{i}.png") for i in range(5)) + (
        _load(ASSETS_DIR / "ball_hyper.png"),
    )
    sprites["ball_punch"] = _load(ASSETS_DIR / "ball_punch.png")
    sprites["ball_trail"] = _load(ASSETS_DIR / "ball_trail.png")

    # Player sprites (28 total across 7 states)
    sprites["pikachu_p1"] = load_pikachu_sprites(p1_skin)
    sprites["pikachu_p2"] = load_pikachu_sprites(p2_skin)

    # Number sprites (0-9)
    sprites["number"] = tuple(_load(ASSETS_DIR / f"number_{i}.png") for i in range(10))

    # Environment
    sprites["shadow"] = _load(ASSETS_DIR / "shadow.png")
    sprites["cloud"] = _load(ASSETS_DIR / "cloud.png")
    sprites["wave"] = _load(ASSETS_DIR / "wave.png")
    sprites["sky_blue"] = _load(ASSETS_DIR / "sky_blue.png", alpha=False)
    sprites["mountain"] = _load(ASSETS_DIR / "mountain.png", alpha=False)
    sprites["ground_red"] = _load(ASSETS_DIR / "ground_red.png", alpha=False)
    sprites["ground_yellow"] = _load(ASSETS_DIR / "ground_yellow.png", alpha=False)
    sprites["ground_line"] = _load(ASSETS_DIR / "ground_line.png", alpha=False)
    sprites["ground_line_leftmost"] = _load(ASSETS_DIR / "ground_line_leftmost.png", alpha=False)
    sprites["ground_line_rightmost"] = _load(ASSETS_DIR / "ground_line_rightmost.png", alpha=False)
    sprites["net_pillar"] = _load(ASSETS_DIR / "net_pillar.png", alpha=False)
    sprites["net_pillar_top"] = _load(ASSETS_DIR / "net_pillar_top.png", alpha=False)

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
