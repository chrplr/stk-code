# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Turning a gym action into the control values the game accepts.

The server always accepts the whole control surface - steer, accel, brake,
nitro, skid, fire, rescue - and this module decides which subset an agent gets
to use. Adding an action space is therefore a Python edit, symmetric with
:mod:`stk_gym.obs` deciding the observation encoding.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

__all__ = ["ACTION_MODES", "space_for", "to_control", "DISCRETE_ACTIONS", "KEYS"]

ACTION_MODES = ("discrete", "continuous", "continuous_full", "keys")

# The keys mode: one flag per key a keyboard player has, in this order. The
# game's own player controller turns presses and releases into controls, so
# the steering ramp, the skid direction and nitro-only-while-accelerating are
# STK's, not this module's. The server needs --gym-keys for it.
KEYS = ("left", "right", "up", "down", "nitro", "skid", "fire", "rescue")

# The discrete table: five steering positions crossed with accelerate, coast and
# brake. Small on purpose - it is the easy first target, and every entry is a
# control a human could hold down.
_STEERS = (-1.0, -0.5, 0.0, 0.5, 1.0)
_THROTTLES = (("accel", 1.0, False), ("coast", 0.0, False), ("brake", 0.0, True))
DISCRETE_ACTIONS = tuple(
    {"steer": steer, "accel": accel, "brake": brake}
    for steer in _STEERS
    for _name, accel, brake in _THROTTLES
)


def space_for(mode: str) -> spaces.Space:
    """Return the action space for \\p mode."""
    if mode == "discrete":
        return spaces.Discrete(len(DISCRETE_ACTIONS))
    if mode == "continuous":
        # steer, accel, brake
        return spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
    if mode == "continuous_full":
        # steer, accel, brake, nitro, skid
        return spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
    if mode == "keys":
        return spaces.MultiBinary(len(KEYS))
    raise ValueError(f"unknown action_mode {mode!r}; choose from {ACTION_MODES}")


def to_control(action: Any, mode: str) -> dict[str, Any]:
    """Convert a gym action into the control dict sent over the wire.

    Values are plain Python floats and bools: Discrete.sample() returns
    np.int64 and Box.sample() returns np.float32, neither of which json will
    encode.
    """
    if mode == "discrete":
        index = int(action)
        if not 0 <= index < len(DISCRETE_ACTIONS):
            raise ValueError(
                f"discrete action {index} is outside 0..{len(DISCRETE_ACTIONS) - 1}"
            )
        return dict(DISCRETE_ACTIONS[index])

    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if mode == "keys":
        if values.size != len(KEYS):
            raise ValueError(f"keys action must have {len(KEYS)} values, got {values.size}")
        return {"keys": {name: bool(v > 0.5) for name, v in zip(KEYS, values)}}
    if mode == "continuous":
        if values.size != 3:
            raise ValueError(f"continuous action must have 3 values, got {values.size}")
        return {
            "steer": float(np.clip(values[0], -1.0, 1.0)),
            "accel": float(np.clip(values[1], 0.0, 1.0)),
            "brake": bool(values[2] > 0.5),
        }
    if mode == "continuous_full":
        if values.size != 5:
            raise ValueError(f"continuous_full action must have 5 values, got {values.size}")
        steer = float(np.clip(values[0], -1.0, 1.0))
        # Skidding needs a direction, and taking it from the steering is what
        # the game's own player controller does: a skid with no steering does
        # nothing useful.
        if values[4] <= 0.5:
            skid = 0                      # SC_NONE
        elif steer < -0.05:
            skid = 2                      # SC_LEFT
        elif steer > 0.05:
            skid = 3                      # SC_RIGHT
        else:
            skid = 1                      # SC_NO_DIRECTION
        return {
            "steer": steer,
            "accel": float(np.clip(values[1], 0.0, 1.0)),
            "brake": bool(values[2] > 0.5),
            "nitro": bool(values[3] > 0.5),
            "skid": skid,
        }
    raise ValueError(f"unknown action_mode {mode!r}; choose from {ACTION_MODES}")
