"""
Overlay system for rendering text and debug information on the game surface.

Overlays are drawn on top of the game after all game elements are rendered.
"""

from __future__ import annotations

from typing import Any, Protocol

import pygame


class Overlay(Protocol):
    """Protocol for overlay renderers."""

    def draw(self, surface: pygame.Surface, metadata: dict[str, Any]) -> None:
        """Draw overlay content onto the surface."""
        ...


class TextOverlay:
    """Renders text labels on the game surface.

    Useful for displaying model names, debug info, or match metadata.

    Args:
        texts: Dict mapping label text to (x, y) position.
        font_size: Font size in pixels.
        color: RGB text color.
    """

    def __init__(
        self,
        texts: dict[str, tuple[int, int]] | None = None,
        font_size: int = 12,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self._texts = texts or {}
        self._font_size = font_size
        self._color = color
        self._font: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface, metadata: dict[str, Any]) -> None:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", self._font_size)

        for text, (x, y) in self._texts.items():
            rendered = self._font.render(text, True, self._color)
            surface.blit(rendered, (x, y))


class MetadataOverlay:
    """Renders metadata values (from env or wrapper infos) as text.

    Reads keys from the metadata dict passed at render time.

    Args:
        keys: List of metadata keys to display.
        position: (x, y) starting position for the first line.
        font_size: Font size in pixels.
        color: RGB text color.
        line_spacing: Pixels between lines.
    """

    def __init__(
        self,
        keys: list[str] | None = None,
        position: tuple[int, int] = (5, 5),
        font_size: int = 10,
        color: tuple[int, int, int] = (255, 255, 255),
        line_spacing: int = 14,
    ) -> None:
        self._keys = keys or []
        self._position = position
        self._font_size = font_size
        self._color = color
        self._line_spacing = line_spacing
        self._font: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface, metadata: dict[str, Any]) -> None:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", self._font_size)

        x, y = self._position
        for key in self._keys:
            if key in metadata:
                text = f"{key}: {metadata[key]}"
                rendered = self._font.render(text, True, self._color)
                surface.blit(rendered, (x, y))
                y += self._line_spacing
