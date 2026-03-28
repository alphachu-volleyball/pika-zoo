"""
Record a Pikachu Volleyball match to MP4 video using ffmpeg.

Streams raw frames to ffmpeg via stdin — no moviepy dependency, low memory usage.

Usage:
    uv run record                                  # AI vs AI, default output
    uv run record -o match.mp4 --winning-score 5
    uv run record --p1 builtin --p2 builtin --seed 42

Requires: ffmpeg installed and available on PATH.
"""

from __future__ import annotations

import argparse
import subprocess

import numpy as np

from pika_zoo.ai.registry import get_ai
from pika_zoo.env import env


def record_match(
    output: str = "episode.mp4",
    winning_score: int = 15,
    seed: int | None = None,
    fps: int = 25,
    p1: str = "builtin",
    p2: str = "builtin",
) -> None:
    """Run a match in rgb_array mode and save as MP4 via ffmpeg streaming.

    Args:
        output: Output video file path.
        winning_score: Score to win.
        seed: Random seed.
        fps: Video frame rate.
        p1: Player 1 AI name.
        p2: Player 2 AI name.
    """
    ai_policies = {
        "player_1": get_ai(p1),
        "player_2": get_ai(p2),
    }

    e = env(
        render_mode="rgb_array",
        ai_policies=ai_policies,
        winning_score=winning_score,
    )
    e.reset(seed=seed)

    # Get first frame to determine dimensions
    first_frame = e.render()
    h, w = first_frame.shape[:2]

    # Start ffmpeg process
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-loglevel", "warning",
            output,
        ],
        stdin=subprocess.PIPE,
    )

    frame_count = 0

    def write_frame(frame: np.ndarray) -> None:
        nonlocal frame_count
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        frame_count += 1

    write_frame(first_frame)
    print(f"Recording: {p1} vs {p2} (first to {winning_score})...")

    while True:
        actions = {"player_1": 0, "player_2": 0}
        obs, rewards, terms, truncs, infos = e.step(actions)
        frame = e.render()
        if frame is not None:
            write_frame(frame)

        if any(terms.values()):
            scores = infos["player_1"]["scores"]
            winner = "Player 1" if scores[0] > scores[1] else "Player 2"
            print(f"Game over! {winner} wins {scores[0]}-{scores[1]} ({frame_count} frames)")
            break

    proc.stdin.close()
    proc.wait()
    e.close()

    print(f"Saved to {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record a Pikachu Volleyball match to MP4")
    parser.add_argument("-o", "--output", default="episode.mp4", help="Output video path (default: episode.mp4)")
    parser.add_argument("--winning-score", type=int, default=15, help="Score to win (default: 15)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fps", type=int, default=25, help="Video frame rate (default: 25)")
    parser.add_argument("--p1", type=str, default="builtin", help="Player 1 AI (default: builtin)")
    parser.add_argument("--p2", type=str, default="builtin", help="Player 2 AI (default: builtin)")
    args = parser.parse_args(argv)

    record_match(
        output=args.output,
        winning_score=args.winning_score,
        seed=args.seed,
        fps=args.fps,
        p1=args.p1,
        p2=args.p2,
    )


if __name__ == "__main__":
    main()
