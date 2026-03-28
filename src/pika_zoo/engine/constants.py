"""
Game constants from the original Pikachu Volleyball physics engine.

Source: https://github.com/gorisanson/pikachu-volleyball/blob/main/src/resources/js/physics.js

Ground width: 432 (0x1B0)
Ground height: 304 (0x130)
X position coordinate: [0, 432], right-direction increasing
Y position coordinate: [0, 304], down-direction increasing
"""

# Ground
GROUND_WIDTH: int = 432
GROUND_HALF_WIDTH: int = 216  # also the net pillar x coordinate

# Player (Pikachu)
PLAYER_LENGTH: int = 64
PLAYER_HALF_LENGTH: int = 32
PLAYER_TOUCHING_GROUND_Y_COORD: int = 244

# Ball
BALL_RADIUS: int = 20
BALL_TOUCHING_GROUND_Y_COORD: int = 252

# Net pillar
NET_PILLAR_HALF_WIDTH: int = 25
NET_PILLAR_TOP_TOP_Y_COORD: int = 176
NET_PILLAR_TOP_BOTTOM_Y_COORD: int = 192

# Safety limit for trajectory prediction loops.
# Not in the original machine code; added to prevent infinite loops
# when ball x coord range is modified for left-right symmetry.
INFINITE_LOOP_LIMIT: int = 1000
