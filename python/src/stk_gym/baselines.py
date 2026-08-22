# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Policies that do not learn, to measure the ones that do against.

Three of them, and the gap between them is the whole point:

- :func:`run_straight` holds the throttle down and never steers. It is the floor
  - what an agent that has learned nothing about the track achieves.
- :func:`run_random` samples the action space. Usually worse than straight.
- :func:`run_expert` is SuperTuxKart's own racing AI in the agent's seat. It is
  the ceiling. A learned policy that does not approach it has not learned
  anything worth having.

The expert needs an environment built with ``expert=True``: the game puts
SkiddingAI in the player's seat and ignores the actions sent to it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

__all__ = ["EpisodeResult", "run_episode", "run_random", "run_straight", "run_expert"]


class EpisodeResult(dict):
    """One episode's outcome, readable as attributes or as keys."""

    __getattr__ = dict.__getitem__

    def __repr__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"<{self['steps']} steps, reward {self['reward']:.1f}, "
            f"{self['progress_m']:.0f} m, lap {self['laps']}, "
            f"{'finished' if self['finished'] else 'unfinished'}>"
        )


def run_episode(
    env,
    policy: Callable[[Any, dict], Any],
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> EpisodeResult:
    """Run one episode under \\p policy and report what happened."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    steps = 0
    off_road = 0
    while True:
        action = policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        off_road += 0 if info["on_road"] else 1
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break
    return EpisodeResult(
        steps=steps,
        reward=total,
        progress_m=float(info["progress_m"]),
        laps=int(info["lap"]),
        rank=int(info["rank"]),
        race_time=float(info["race_time"]),
        finished=bool(info["is_success"]),
        off_road_fraction=off_road / max(steps, 1),
    )


def _straight_action(env):
    """Full throttle, no steering, in whichever action space env uses."""
    mode = getattr(env.unwrapped, "action_mode", "continuous")
    if mode == "discrete":
        from .actions import DISCRETE_ACTIONS

        # The entry that steers straight and accelerates.
        for index, control in enumerate(DISCRETE_ACTIONS):
            if control["steer"] == 0.0 and control["accel"] == 1.0:
                return index
        return 0
    if mode == "continuous":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def run_straight(env, *, seed: int | None = None, max_steps: int | None = None):
    """The floor: accelerate, never steer."""
    action = _straight_action(env)
    return run_episode(env, lambda _o, _i: action, seed=seed, max_steps=max_steps)


def run_random(env, *, seed: int | None = None, max_steps: int | None = None):
    """Sample the action space every step."""
    return run_episode(
        env, lambda _o, _i: env.action_space.sample(), seed=seed, max_steps=max_steps
    )


def run_expert(env, *, seed: int | None = None, max_steps: int | None = None):
    """The ceiling: SuperTuxKart's own AI drives, and the actions are ignored."""
    if not getattr(env.unwrapped, "expert", False):
        raise ValueError(
            "run_expert needs an environment built with expert=True, which puts "
            "SkiddingAI in the agent's seat; this one is driven by its actions."
        )
    action = _straight_action(env)
    return run_episode(env, lambda _o, _i: action, seed=seed, max_steps=max_steps)
