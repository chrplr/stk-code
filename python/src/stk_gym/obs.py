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

__all__ = ["OBS_MODES", "space_for", "encode", "progress_of", "SAMPLE_FIELDS", "flatten_state"]

OBS_MODES = ("vector", "vector_karts", "pixels")

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
    if mode == "pixels":
        # The size is the game window's, as the handshake reports it: the
        # requested --screensize, unless the display scales it.
        shape = (int(meta.get("frame_height", 0)), int(meta.get("frame_width", 0)), 3)
        return spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    size = _N_SCALARS + 2 * int(meta.get("lookahead_k", 0))
    if mode == "vector_karts":
        size += 4 * int(meta.get("max_karts", 0))
    # Bounds are advisory: a kart launched off a ramp can leave them briefly,
    # and clipping the observation would hide that from the agent.
    return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)


def encode(state: dict[str, Any], mode: str, meta: dict[str, Any]) -> np.ndarray:
    """Encode one state as a float32 array matching :func:`space_for`.

    ``pixels`` is not encoded from the state at all - the frame comes with the
    answer - so it is refused here; :class:`StkEnv` never asks.
    """
    if mode not in OBS_MODES or mode == "pixels":
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


# -- The state as one row of floats -------------------------------------------

# The scalars of a state that an analysis wants per sample, in the order they
# are listed; "controls" is flattened to ctrl_*, and xyz to x, y, z.
_STATE_SCALARS = (
    "tick", "time", "phase", "speed", "max_speed", "heading", "pitch", "roll",
    "distance_down_track", "distance_to_center", "overall_distance",
    "on_road", "on_ground", "wrong_way", "rank", "finished_laps", "finished",
    "finish_time", "nitro_energy", "powerup", "num_powerup", "eliminated",
)
_CONTROLS = ("steer", "accel", "brake", "nitro", "skid", "fire", "rescue", "look_back")
#: The keys :func:`flatten_state` returns, always all of them.
SAMPLE_FIELDS: tuple[str, ...] = (
    _STATE_SCALARS + ("x", "y", "z") + tuple("ctrl_" + c for c in _CONTROLS) + ("progress_m",)
)


def _number(value: Any) -> float:
    """A state field as a float (bools become 0/1); NaN when it is absent."""
    if isinstance(value, (bool, int, float)):
        return float(value)
    return float("nan")


def flatten_state(state: dict[str, Any], track_length: float) -> dict[str, float]:
    """One state as a row of floats, :data:`SAMPLE_FIELDS` every time.

    This is the shape a logger wants: booleans are 0/1, and a field the game
    did not report - the race gone, as after the pause menu's quit - is NaN
    rather than missing, so rows logged side by side stay the same shape. The
    ``ctrl_*`` fields are the controls the kart applied, which for a person at
    the wheel (or a ``keys`` agent) is the record of what they did.
    """
    row = {key: _number(state.get(key)) for key in _STATE_SCALARS}
    xyz = state.get("xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
        row["x"], row["y"], row["z"] = (float(c) for c in xyz)
    else:
        row["x"] = row["y"] = row["z"] = float("nan")
    controls = state.get("controls") or {}
    for key in _CONTROLS:
        row["ctrl_" + key] = _number(controls.get(key))
    row["progress_m"] = (
        float(progress_of(state, track_length)) if "tick" in state else float("nan")
    )
    return row
