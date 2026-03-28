"""
FFmpeg video writer for streaming raw frames to MP4.
"""

from __future__ import annotations

import subprocess

import numpy as np


class FFmpegWriter:
    """Stream RGB frames to ffmpeg and produce an MP4 file.

    Args:
        output: Output video file path.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Video frame rate.
    """

    def __init__(self, output: str, width: int, height: int, fps: int = 25) -> None:
        self._output = output
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-loglevel",
                "warning",
                output,
            ],
            stdin=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single RGB frame."""
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        """Finalize the video file."""
        self._proc.stdin.close()
        self._proc.wait()
