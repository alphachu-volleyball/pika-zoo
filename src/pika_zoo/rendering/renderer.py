"""
Pygame-based renderer for Pikachu Volleyball.

Renders the game state (players, ball, background, scores) to a pygame surface.
Supports "human" (display window) and "rgb_array" (headless) render modes.

Reference: helpingstar/pika-zoo pikazoo_env.py rendering methods
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pygame

from pika_zoo.engine.physics import Ball, Player
from pika_zoo.rendering.overlays import Overlay
from pika_zoo.rendering.sprites import get_player_sprite_index, load_all_sprites

SCREEN_WIDTH = 432
SCREEN_HEIGHT = 304
SHADOW_Y = 273
FPS = 25


def _blit_center(screen: pygame.Surface, source: pygame.Surface, center: tuple[int, int]) -> None:
    """Blit a surface centered at the given position."""
    rect = source.get_rect(center=center)
    screen.blit(source, rect)


class PygameRenderer:
    """Renders the Pikachu Volleyball game state using pygame.

    Args:
        render_mode: "human" (display window) or "rgb_array" (headless).
        overlays: Optional list of overlay objects to draw on top.
    """

    def __init__(
        self,
        render_mode: str = "rgb_array",
        overlays: list[Overlay] | None = None,
        p1_skin: str = "yellow",
        p2_skin: str = "yellow",
    ) -> None:
        self._render_mode = render_mode
        self._overlays = overlays or []
        self._p1_skin = p1_skin
        self._p2_skin = p2_skin
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._sprites: dict[str, Any] | None = None
        self._mode_font: pygame.font.Font | None = None
        self._initialized = False

    def _init_pygame(self) -> None:
        """Lazy-initialize pygame and load sprites."""
        if self._initialized:
            return

        pygame.init()
        if self._render_mode == "human":
            self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Pikachu Volleyball")
            self._clock = pygame.time.Clock()
        else:
            # Headless: need a dummy display for convert_alpha() to work
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.display.set_mode((1, 1))
            self._screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self._sprites = load_all_sprites(p1_skin=self._p1_skin, p2_skin=self._p2_skin)
        self._initialized = True

    def render(
        self,
        player1: Player,
        player2: Player,
        ball: Ball,
        scores: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Render one frame.

        Returns:
            RGB array if render_mode is "rgb_array", None if "human".
        """
        self._init_pygame()
        assert self._screen is not None
        assert self._sprites is not None

        self._draw_background()
        self._draw_player(player1)
        self._draw_player(player2)
        self._draw_ball(ball)
        self._draw_scores(scores)
        self._draw_player_labels(metadata or {})
        self._draw_mode_label(metadata or {})

        # Draw overlays
        for overlay in self._overlays:
            overlay.draw(self._screen, metadata or {})

        frame = np.transpose(pygame.surfarray.pixels3d(self._screen), axes=(1, 0, 2)).copy()

        if self._render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            assert self._clock is not None
            self._clock.tick(FPS)

        return frame

    def capture_frame(self) -> np.ndarray | None:
        """Capture the current screen as an RGB array, regardless of render mode.

        Works in both 'human' and 'rgb_array' modes.
        Returns None if pygame is not initialized.
        """
        if not self._initialized or self._screen is None:
            return None
        return np.transpose(pygame.surfarray.pixels3d(self._screen), axes=(1, 0, 2)).copy()

    def close(self) -> None:
        """Clean up pygame resources."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            self._screen = None
            self._sprites = None

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _draw_background(self) -> None:
        assert self._screen is not None
        assert self._sprites is not None
        s = self._sprites

        # Sky (12 rows × 27 cols of 16×16 tiles)
        for y in range(12):
            for x in range(27):
                self._screen.blit(s["sky_blue"], (x * 16, y * 16))

        # Mountain
        self._screen.blit(s["mountain"], (0, 188))

        # Ground red (1 row)
        for x in range(27):
            self._screen.blit(s["ground_red"], (x * 16, 248))

        # Ground line
        self._screen.blit(s["ground_line_leftmost"], (0, 264))
        for x in range(1, 26):
            self._screen.blit(s["ground_line"], (x * 16, 264))
        self._screen.blit(s["ground_line_rightmost"], (416, 264))

        # Ground yellow (2 rows)
        for row in range(2):
            for x in range(27):
                self._screen.blit(s["ground_yellow"], (x * 16, 280 + row * 16))

        # Net pillar
        self._screen.blit(s["net_pillar_top"], (213, 176))
        for i in range(12):
            self._screen.blit(s["net_pillar"], (213, 184 + i * 8))

    # ------------------------------------------------------------------
    # Players
    # ------------------------------------------------------------------

    def _draw_player(self, player: Player) -> None:
        assert self._screen is not None
        assert self._sprites is not None

        sprite_idx = get_player_sprite_index(player.state, player.frame_number)
        sprite_key = "pikachu_p2" if player.is_player2 else "pikachu_p1"
        sprite = self._sprites[sprite_key][sprite_idx]

        # Determine if sprite should be flipped
        if not player.is_player2:
            # Player 1: flip when diving/lying left
            x_flip = (player.state == 3 or player.state == 4) and player.diving_direction == -1
        else:
            # Player 2: flip when NOT diving/lying right
            x_flip = not ((player.state == 3 or player.state == 4) and player.diving_direction == 1)

        if x_flip:
            sprite = pygame.transform.flip(sprite, True, False)

        _blit_center(self._screen, sprite, (player.x, player.y))

        # Shadow
        _blit_center(self._screen, self._sprites["shadow"], (player.x, SHADOW_Y))

    # ------------------------------------------------------------------
    # Ball
    # ------------------------------------------------------------------

    def _draw_ball(self, ball: Ball) -> None:
        assert self._screen is not None
        assert self._sprites is not None
        s = self._sprites

        # Power hit trail effects
        if ball.is_power_hit:
            _blit_center(self._screen, s["ball_trail"], (ball.previous_previous_x, ball.previous_previous_y))
            hyper_sprite = s["ball"][5]  # ball_hyper
            _blit_center(self._screen, hyper_sprite, (ball.previous_x, ball.previous_y))

        # Main ball
        ball_sprite = s["ball"][ball.rotation]
        _blit_center(self._screen, ball_sprite, (ball.x, ball.y))

        # Punch effect (shrinks by 2 pixels per frame, same as original)
        if ball.punch_effect_radius > 0:
            punch = s["ball_punch"]
            size = ball.punch_effect_radius * 2
            if size > 0:
                scaled = pygame.transform.scale(punch, (size, size))
                _blit_center(self._screen, scaled, (ball.punch_effect_x, ball.punch_effect_y))
            ball.punch_effect_radius -= 2

        # Shadow
        _blit_center(self._screen, s["shadow"], (ball.x, SHADOW_Y))

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _draw_scores(self, scores: list[int]) -> None:
        assert self._screen is not None
        assert self._sprites is not None
        numbers = self._sprites["number"]

        # Player 1 score (left)
        if scores[0] >= 10:
            self._screen.blit(numbers[1], (14, 10))
        self._screen.blit(numbers[scores[0] % 10], (46, 10))

        # Player 2 score (right)
        if scores[1] >= 10:
            self._screen.blit(numbers[1], (356, 10))
        self._screen.blit(numbers[scores[1] % 10], (388, 10))

    # ------------------------------------------------------------------
    # Player labels
    # ------------------------------------------------------------------

    def _draw_player_labels(self, metadata: dict[str, Any]) -> None:
        """Draw player labels (model names) below scores."""
        assert self._screen is not None
        if self._mode_font is None:
            pygame.font.init()
            self._mode_font = pygame.font.SysFont("monospace", 14, bold=True)

        p1_label = metadata.get("p1_label", "")
        p2_label = metadata.get("p2_label", "")

        if p1_label:
            rendered = self._mode_font.render(p1_label, True, (0, 0, 0))
            self._screen.blit(rendered, (14, 42))

        if p2_label:
            rendered = self._mode_font.render(p2_label, True, (0, 0, 0))
            x = SCREEN_WIDTH - 14 - rendered.get_width()
            self._screen.blit(rendered, (x, 42))

    # ------------------------------------------------------------------
    # Mode label
    # ------------------------------------------------------------------

    def _draw_mode_label(self, metadata: dict[str, Any]) -> None:
        """Draw mode label at top center — 'normal' or noise config details."""
        noise = metadata.get("noise")
        assert self._screen is not None
        if self._mode_font is None:
            pygame.font.init()
            self._mode_font = pygame.font.SysFont("monospace", 14, bold=True)
        if noise is not None:
            label = f"noise(x={noise.x_range}, xv={noise.x_velocity_range}, yv={noise.y_velocity_range})"
            color = (0, 0, 255)
        else:
            label = "normal"
            color = (0, 0, 0)
        rendered = self._mode_font.render(label, True, color)
        x = SCREEN_WIDTH // 2 - rendered.get_width() // 2
        self._screen.blit(rendered, (x, 10))
