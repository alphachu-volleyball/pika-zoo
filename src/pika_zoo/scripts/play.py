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
from pathlib import Path

from pika_zoo.ai.registry import get_ai, get_skin
from pika_zoo.engine.types import NoiseConfig
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
    stats: str | None = None,
    noise: NoiseConfig | None = None,
    p1_skin: str | None = None,
    p2_skin: str | None = None,
    p1_label: str | None = None,
    p2_label: str | None = None,
) -> None:
    """Run a Pikachu Volleyball match.

    Args:
        p1: Player 1 — AI name, "human", or model path.
        p2: Player 2 — AI name, "human", or model path.
        winning_score: Score to win.
        seed: Random seed.
        fps: Frame rate (for render and/or recording).
        render: Show pygame window.
        record: Output MP4 path, or None to skip recording.
        stats: Output CSV path for per-frame event stats, or None to skip.
        noise: Noise configuration for ball initialization. None disables noise.
        p1_skin: Pikachu skin for P1 (default: auto).
        p2_skin: Pikachu skin for P2 (default: auto).
        p1_label: Display label for P1 (default: auto from spec).
        p2_label: Display label for P2 (default: auto from spec).
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
    sb3_policies: dict[str, object] = {}  # SB3 models need obs updates

    for agent, spec, is_human in [("player_1", p1, p1_human), ("player_2", p2, p2_human)]:
        if is_human:
            continue
        if Path(spec).exists():
            from pika_zoo.ai.sb3_adapter import SB3ModelPolicy

            policy = SB3ModelPolicy(spec, agent=agent)
            ai_policies[agent] = policy
            sb3_policies[agent] = policy
        else:
            ai_policies[agent] = get_ai(spec)

    def _resolve_skin(spec: str, is_human: bool, explicit: str | None) -> str:
        if explicit:
            return explicit
        if is_human:
            return "yellow"
        if Path(spec).exists():
            return "white"  # SB3 model default skin
        return get_skin(spec)

    resolved_p1_skin = _resolve_skin(p1, p1_human, p1_skin)
    resolved_p2_skin = _resolve_skin(p2, p2_human, p2_skin)

    if render:
        render_mode = "human"
    elif record:
        render_mode = "rgb_array"
    else:
        render_mode = None

    def _make_label(spec: str, is_human: bool) -> str:
        if is_human:
            return "human"
        if Path(spec).exists():
            p = Path(spec)
            return p.parent.name if p.stem == "model" else p.stem
        return spec

    p1_label = p1_label or _make_label(p1, p1_human)
    p2_label = p2_label or _make_label(p2, p2_human)

    e = env(
        render_mode=render_mode,
        ai_policies=ai_policies,
        winning_score=winning_score,
        noise=noise,
        p1_skin=resolved_p1_skin,
        p2_skin=resolved_p2_skin,
        p1_label=p1_label,
        p2_label=p2_label,
    )
    e.reset(seed=seed)

    # Print match info
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

    event_rows: list[dict] = []
    current_round = 1

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

        # Feed observations to SB3 models before step
        if sb3_policies:
            current_obs = e._get_observations()
            for agent, policy in sb3_policies.items():
                policy.set_observation(current_obs[agent])

        obs, rewards, terms, truncs, infos = e.step(actions)

        if render_mode is not None:
            e.render()

        frame_count += 1

        # Collect events
        if stats and "events" in infos.get("player_1", {}):
            event_rows.append({"frame": frame_count, "round": current_round, **infos["player_1"]["events"]})
            if infos["player_1"].get("round_ended"):
                current_round += 1

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

    if stats and event_rows:
        import csv

        with open(stats, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=event_rows[0].keys())
            w.writeheader()
            w.writerows(event_rows)
        print(f"Stats saved to {stats}")

    e.close()


def _build_noise(args: argparse.Namespace) -> NoiseConfig | None:
    if args.noise_x is None and args.noise_x_vel is None and args.noise_y_vel is None:
        return None
    return NoiseConfig(
        x_range=args.noise_x or 0,
        x_velocity_range=args.noise_x_vel or 0,
        y_velocity_range=args.noise_y_vel or 0,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play, watch, or record Pikachu Volleyball")
    parser.add_argument("--winning-score", type=int, default=15, help="Score to win (default: 15)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--p1", type=str, default="builtin", help="Player 1: AI name or 'human' (default: builtin)")
    parser.add_argument("--p2", type=str, default="builtin", help="Player 2: AI name or 'human' (default: builtin)")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (default: 25)")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame window (headless)")
    parser.add_argument("--record", type=str, default=None, metavar="FILE", help="Record to MP4 (requires ffmpeg)")
    parser.add_argument("--stats", type=str, default=None, metavar="FILE", help="Save per-frame event stats to CSV")
    parser.add_argument("--noise-x", type=int, default=None, metavar="N", help="Ball x position noise ±N pixels")
    parser.add_argument("--noise-x-vel", type=int, default=None, metavar="N", help="Ball x velocity noise ±N")
    parser.add_argument("--noise-y-vel", type=int, default=None, metavar="N", help="Ball y velocity noise ±N")
    parser.add_argument("--p1-skin", type=str, default=None, help="P1 pikachu skin (default: auto from AI)")
    parser.add_argument("--p2-skin", type=str, default=None, help="P2 pikachu skin (default: auto from AI)")
    parser.add_argument("--p1-label", type=str, default=None, help="P1 display label (default: auto)")
    parser.add_argument("--p2-label", type=str, default=None, help="P2 display label (default: auto)")
    args = parser.parse_args(argv)

    play(
        p1=args.p1,
        p2=args.p2,
        winning_score=args.winning_score,
        seed=args.seed,
        fps=args.fps,
        render=not args.no_render,
        record=args.record,
        stats=args.stats,
        noise=_build_noise(args),
        p1_skin=args.p1_skin,
        p2_skin=args.p2_skin,
        p1_label=args.p1_label,
        p2_label=args.p2_label,
    )


if __name__ == "__main__":
    main()
