# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""SuperTuxKart as a Gymnasium environment.

    import gymnasium, stk_gym
    env = gymnasium.make("SuperTuxKart-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The physics live in the game, which is run as a child process and driven over a
JSON-lines protocol. This package is the client: it owns the observation
encoding, the reward and the episode budget, none of which need a rebuild of the
game to change.
"""

from __future__ import annotations

import gymnasium

# The release tag this package downloads its engine and asset pack from:
# the gym-release workflow refuses a tag that does not match it, so a wheel
# and the engine it fetches cannot drift apart.
__version__ = "0.1.1"

from .actions import ACTION_MODES
from .baselines import run_expert, run_random, run_straight
from .binary import BinaryNotFound, default_cwd, find_binary
from .engine import (
    CommandFailed,
    Engine,
    EngineDied,
    EngineError,
    ProtocolError,
    server_args,
)
from .env import REWARD_SCHEMES, StkEnv, compute_reward, state_info
from .human import HumanSession
from .obs import OBS_MODES, progress_of
from .vector_env import StkVectorEnv

__all__ = [
    "ACTION_MODES",
    "BinaryNotFound",
    "CommandFailed",
    "Engine",
    "HumanSession",
    "EngineDied",
    "EngineError",
    "OBS_MODES",
    "ProtocolError",
    "REWARD_SCHEMES",
    "StkEnv",
    "StkVectorEnv",
    "compute_reward",
    "default_cwd",
    "find_binary",
    "progress_of",
    "register",
    "run_expert",
    "run_random",
    "run_straight",
    "server_args",
    "state_info",
]

# A step is 6 physics ticks, or 0.05 s of race time, so 3000 steps is two and a
# half minutes - comfortably more than a lap for a policy that can drive, and a
# firm stop for one that cannot.
_SPECS = (
    dict(id="SuperTuxKart-v0", max_episode_steps=3000, kwargs={}),
    dict(
        id="SuperTuxKart-Easy-v0",
        max_episode_steps=3000,
        # Alone on the track, so the only thing to learn is how to drive it.
        kwargs={"num_karts": 1, "laps": 1, "difficulty": 0},
    ),
    dict(
        id="SuperTuxKart-Race-v0",
        max_episode_steps=6000,
        kwargs={"num_karts": 4, "laps": 3, "obs_mode": "vector_karts"},
    ),
)


def register() -> None:
    """Register the environment ids. Called on import; safe to call again."""
    for spec in _SPECS:
        if spec["id"] in gymnasium.registry:
            continue
        gymnasium.register(
            entry_point="stk_gym.env:StkEnv",
            vector_entry_point="stk_gym.vector_env:StkVectorEnv",
            **spec,
        )


register()
