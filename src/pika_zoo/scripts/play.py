"""
Unified play/demo/record script for Pikachu Volleyball.

Usage:
    uv run play                                          # AI vs AI, render only
    uv run play --p1 human                               # Human vs AI
    uv run play --no-render --record match.mp4           # Headless record
    uv run play --record match.mp4                       # Render + record
    uv run play --no-render                              # Headless (stats only)

Controls (when render is on and player is human):
    Player 1: D(left) G(right) R(up) V(down) Z(power hit)
    Player 2: Arrow keys + Enter (power hit)
"""

from __future__ import annotations

import argparse
import warnings

from pika_zoo.ai.registry import get_ai, get_skin
from pika_zoo.env import env
from pika_zoo.scripts.video import FFmpegWriter


def play(
    p1: str = "builtin",
    p2: str = "builtin",
    winning_score: int = 15,
    seed: int | None = None,
    fps: int = 25,
    render: bool = True,
    record: str | None = None,
    random_mode: bool = False,
    p1_skin: str | None = None,
    p2_skin: str | None = None,
) -> None:
    """Run a Pikachu Volleyball match.

    Args:
        p1: Player 1 — AI name or "human".
        p2: Player 2 — AI name or "human".
        winning_score: Score to win.
        seed: Random seed.
        fps: Frame rate (for render and/or recording).
        render: Show pygame window.
        record: Output MP4 path, or None to skip recording.
        random_mode: Randomize ball starting position/velocity each round.
        p1_skin: Pikachu skin for P1 (default: auto from AI registry, "yellow" for human).
        p2_skin: Pikachu skin for P2 (default: auto from AI registry, "yellow" for human).
    """
    p1_human = p1 == "human"
    p2_human = p2 == "human"

    # Warn if human without render
    if (p1_human or p2_human) and not render:
        warnings.warn("Human player requires render=True. Falling back to AI.", stacklevel=2)
        if p1_human:
            p1 = "builtin"
            p1_human = False
        if p2_human:
            p2 = "builtin"
            p2_human = False

    ai_policies = {}
    if not p1_human:
        ai_policies["player_1"] = get_ai(p1)
    if not p2_human:
        ai_policies["player_2"] = get_ai(p2)

    # Resolve skins: explicit > registry > "yellow" (human)
    resolved_p1_skin = p1_skin or ("yellow" if p1_human else get_skin(p1))
    resolved_p2_skin = p2_skin or ("yellow" if p2_human else get_skin(p2))

    if render:
        render_mode = "human"
    elif record:
        render_mode = "rgb_array"
    else:
        render_mode = None

    e = env(
        render_mode=render_mode,
        ai_policies=ai_policies,
        winning_score=winning_score,
        random_mode=random_mode,
        p1_skin=resolved_p1_skin,
        p2_skin=resolved_p2_skin,
    )
    e.reset(seed=seed)

    # Print match info
    p1_label = "Human" if p1_human else p1
    p2_label = "Human" if p2_human else p2
    print(f"Pikachu Volleyball — {p1_label} vs {p2_label} (first to {winning_score})")
    if p1_human:
        print("  P1: D(left) G(right) R(up) V(down) Z(power hit)")
    if p2_human:
        print("  P2: Arrow keys + Enter (power hit)")

    # Init render if needed
    if render_mode is not None:
        e.render()

    # Set up recording
    writer = None
    if record:
        if render and e._renderer is not None:
            frame = e._renderer.capture_frame()
        elif render_mode == "rgb_array":
            frame = e.render()
        else:
            frame = None

        if frame is not None:
            h, w = frame.shape[:2]
            writer = FFmpegWriter(record, w, h, fps)
            writer.write_frame(frame)
            print(f"  Recording to {record}...")

    # Import pygame only when rendering
    pygame = None
    get_action_from_keys = None
    if render:
        import pygame as _pygame

        pygame = _pygame
        from pika_zoo.scripts.keyboard import get_action_from_keys

    frame_count = 0
    running = True
    while running:
        # Handle events (only when rendering)
        if pygame is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
            if not running:
                break

        # Get actions
        actions = {"player_1": 0, "player_2": 0}
        if pygame is not None and (p1_human or p2_human):
            keys = pygame.key.get_pressed()
            if p1_human:
                actions["player_1"] = get_action_from_keys(keys, "player_1")
            if p2_human:
                actions["player_2"] = get_action_from_keys(keys, "player_2")

        obs, rewards, terms, truncs, infos = e.step(actions)

        if render_mode is not None:
            e.render()

        frame_count += 1

        # Capture for recording
        if writer is not None:
            if render and e._renderer is not None:
                captured = e._renderer.capture_frame()
            elif render_mode == "rgb_array":
                captured = e.render()
            else:
                captured = None
            if captured is not None:
                writer.write_frame(captured)

        if any(terms.values()):
            scores = infos["player_1"]["scores"]
            winner = "Player 1" if scores[0] > scores[1] else "Player 2"
            print(f"Game over! {winner} wins {scores[0]}-{scores[1]} ({frame_count} frames)")
            break

    if writer is not None:
        writer.close()
        print(f"Saved to {record}")

    e.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play, watch, or record Pikachu Volleyball")
    parser.add_argument("--winning-score", type=int, default=15, help="Score to win (default: 15)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--p1", type=str, default="builtin", help="Player 1: AI name or 'human' (default: builtin)")
    parser.add_argument("--p2", type=str, default="builtin", help="Player 2: AI name or 'human' (default: builtin)")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window (headless)")
    parser.add_argument("--record", type=str, default=None, metavar="FILE", help="Record to MP4 (requires ffmpeg)")
    parser.add_argument("--random", action="store_true", help="Random ball start position/velocity each round")
    parser.add_argument("--p1-skin", type=str, default=None, help="P1 pikachu skin (default: auto from AI)")
    parser.add_argument("--p2-skin", type=str, default=None, help="P2 pikachu skin (default: auto from AI)")
    args = parser.parse_args(argv)

    play(
        p1=args.p1,
        p2=args.p2,
        winning_score=args.winning_score,
        seed=args.seed,
        fps=args.fps,
        render=not args.no_render,
        record=args.record,
        random_mode=args.random,
        p1_skin=args.p1_skin,
        p2_skin=args.p2_skin,
    )


if __name__ == "__main__":
    main()
