"""
Random number generator matching the original game's rand() behavior.

The original game uses _rand() from Visual Studio 1988 Library,
which generates random integers in [0, 32767].

Source: https://github.com/gorisanson/pikachu-volleyball/blob/main/src/resources/js/rand.js
"""

from __future__ import annotations

from numpy.random import Generator


def rand(rng: Generator) -> int:
    """Return random integer in [0, 32767], matching the original rand() behavior."""
    return int(rng.integers(0, 32768))
