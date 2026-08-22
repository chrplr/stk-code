# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Turning the server's facts into an observation array.

The server reports raw quantities in game units - metres, metres per second,
radians. The scaling, the choice of which facts to include and the layout are
all decided here, so trying a different encoding costs an edit rather than a
rebuild of the game.

Every shape is derived from the handshake (``lookahead_k``, ``max_karts``,
``track_length``), never from a constant repeated here, so the two sides cannot
drift apart.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

__all__ = ["OBS_MODES", "space_for", "encode", "progress_of"]

OBS_MODES = ("vector", "vector_karts")

# Scales that turn game units into roughly [-1, 1]. They are constants of the
# encoding, not of the game: a kart tops out near 25 m/s with a zipper, and the
# lookahead reaches about ten metres per driveline node.
_SPEED_SCALE = 30.0
_LOOKAHEAD_SCALE = 50.0
_CENTRE_SCALE = 5.0

# Facts about the kart itself, before the lookahead and the opponents.
_N_SCALARS = 10


def progress_of(state: dict[str, Any], track_length: float) -> float:
    """Distance travelled along the track, in metres, counting laps.

    This is the quantity a progress reward wants, and building it correctly is
    the one piece of arithmetic this package has to get right.

    ``distance_down_track`` is measured from the start line and wraps to zero
    there, while ``finished_laps`` counts -1 before the line is first crossed.
    The two therefore compose: a kart sitting on the grid is a metre or so
    *before* the line, which comes out as a small negative number, and the sum
    runs smoothly through every crossing.

    Do not use ``overall_distance`` for this. It is the game's own measure and
    it is built on the checkline-validated distance, which only moves when a
    checkline is crossed - it exists to defeat shortcuts, not to reward
    progress, and it is flat in between.
    """
    return state.get("finished_laps", 0) * track_length + state.get(
        "distance_down_track", 0.0
    )


def space_for(mode: str, meta: dict[str, Any]) -> spaces.Space:
    """Return the observation space implied by \\p mode and the handshake."""
    if mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {mode!r}; choose from {OBS_MODES}")
    size = _N_SCALARS + 2 * int(meta.get("lookahead_k", 0))
    if mode == "vector_karts":
        size += 4 * int(meta.get("max_karts", 0))
    # Bounds are advisory: a kart launched off a ramp can leave them briefly,
    # and clipping the observation would hide that from the agent.
    return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)


def encode(state: dict[str, Any], mode: str, meta: dict[str, Any]) -> np.ndarray:
    """Encode one state as a float32 array matching :func:`space_for`."""
    if mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {mode!r}; choose from {OBS_MODES}")

    lookahead_k = int(meta.get("lookahead_k", 0))
    track_length = float(meta.get("track_length", 0.0)) or 1.0
    num_karts = max(int(meta.get("num_karts", 1)), 1)

    speed = float(state.get("speed", 0.0))
    max_speed = float(state.get("max_speed", 0.0)) or 1.0
    values = [
        speed / _SPEED_SCALE,
        speed / max_speed,
        float(state.get("steer", 0.0)),
        1.0 if state.get("on_road", True) else 0.0,
        1.0 if state.get("on_ground", True) else 0.0,
        1.0 if state.get("wrong_way", False) else 0.0,
        np.tanh(float(state.get("distance_to_center", 0.0)) / _CENTRE_SCALE),
        float(state.get("nitro_energy", 0.0)) / 100.0,
        (float(state.get("rank", 1)) - 1.0) / max(num_karts - 1, 1),
        float(state.get("distance_down_track", 0.0)) / track_length,
    ]
    assert len(values) == _N_SCALARS

    # The lookahead is already in kart-local coordinates: x to the right, z
    # ahead. Only those two matter for steering; the height is dropped.
    lookahead = state.get("lookahead") or []
    last = (0.0, 0.0)
    for i in range(lookahead_k):
        if i < len(lookahead):
            point = lookahead[i]
            last = (float(point.get("x", 0.0)), float(point.get("z", 0.0)))
        # A track with no driveline, or one whose graph runs out, repeats the
        # last point rather than padding with zeros, which would read as "the
        # racing line is exactly where I am".
        values.append(last[0] / _LOOKAHEAD_SCALE)
        values.append(last[1] / _LOOKAHEAD_SCALE)

    if mode == "vector_karts":
        max_karts = int(meta.get("max_karts", 0))
        own = progress_of(state, track_length)
        karts = state.get("karts") or []
        # Nearest first, so the slot an opponent occupies means something even
        # though the game lists them by id.
        karts = sorted(
            karts[:max_karts],
            key=lambda k: float(k.get("x", 0.0)) ** 2 + float(k.get("z", 0.0)) ** 2,
        )
        for i in range(max_karts):
            if i < len(karts):
                other = karts[i]
                values += [
                    float(other.get("x", 0.0)) / _LOOKAHEAD_SCALE,
                    float(other.get("z", 0.0)) / _LOOKAHEAD_SCALE,
                    float(other.get("speed", 0.0)) / _SPEED_SCALE,
                    (progress_of(other, track_length) - own) / track_length,
                ]
            else:
                # An eliminated or absent kart is reported as all zeros, which
                # is distinguishable from a real one because a real opponent is
                # never at the agent's exact position.
                values += [0.0, 0.0, 0.0, 0.0]

    return np.asarray(values, dtype=np.float32)
