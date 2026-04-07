"""
Headless benchmark for measuring environment throughput (FPS).

Usage:
    uv run benchmark                                    # builtin vs builtin, 10000 frames
    uv run benchmark --p1 duckll --p2 duckll            # AI matchup
    uv run benchmark --frames 50000                     # more frames
    uv run benchmark --warmup 500                       # adjust warmup
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pika_zoo.ai.registry import get_ai
from pika_zoo.env import env


def benchmark(
    p1: str = "builtin",
    p2: str = "builtin",
    frames: int = 10_000,
    warmup: int = 1_000,
    seed: int | None = None,
) -> None:
    """Run a headless benchmark and report FPS."""
    ai_policies = {}
    for agent, spec in [("player_1", p1), ("player_2", p2)]:
        if Path(spec).exists():
            from pika_zoo.ai.sb3_adapter import SB3ModelPolicy

            ai_policies[agent] = SB3ModelPolicy(spec, agent=agent)
        else:
            ai_policies[agent] = get_ai(spec)

    e = env(render_mode=None, ai_policies=ai_policies)
    e.reset(seed=seed)

    actions = {"player_1": 0, "player_2": 0}

    # Warmup
    for _ in range(warmup):
        _, _, terms, _, _ = e.step(actions)
        if any(terms.values()):
            e.reset()

    # Benchmark
    start = time.perf_counter()
    for _ in range(frames):
        _, _, terms, _, _ = e.step(actions)
        if any(terms.values()):
            e.reset()
    elapsed = time.perf_counter() - start

    e.close()

    fps = frames / elapsed
    print(f"Benchmark: {p1} vs {p2}")
    print(f"  Frames: {frames:,} | Time: {elapsed:.2f}s | FPS: {fps:,.0f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Headless benchmark for environment throughput")
    parser.add_argument("--p1", type=str, default="builtin", help="Player 1 AI spec (default: builtin)")
    parser.add_argument("--p2", type=str, default="builtin", help="Player 2 AI spec (default: builtin)")
    parser.add_argument("--frames", type=int, default=10_000, help="Frames to measure (default: 10000)")
    parser.add_argument("--warmup", type=int, default=1_000, help="Warmup frames, excluded from measurement (default: 1000)")  # noqa: E501
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)

    benchmark(p1=args.p1, p2=args.p2, frames=args.frames, warmup=args.warmup, seed=args.seed)


if __name__ == "__main__":
    main()
