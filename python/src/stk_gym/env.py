# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""SuperTuxKart as a single Gymnasium environment."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from . import actions as _actions
from . import obs as _obs
from .binary import default_cwd, find_binary
from .engine import Engine, server_args

__all__ = ["StkEnv", "REWARD_SCHEMES", "compute_reward", "state_info"]

REWARD_SCHEMES = ("progress", "progress_only", "sparse")

# Paid once when the kart crosses the finish line.
_FINISH_BONUS = 50.0
# Paid every step, so that a policy which drives well but slowly is worse than
# one which drives well and quickly.
_TIME_PENALTY = 0.05
# Paid every step spent off the driveline. Leaving the road already costs speed,
# so this is a nudge and not the main signal.
_OFF_ROAD_PENALTY = 0.5


def compute_reward(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    scheme: str,
    track_length: float,
) -> float:
    """Reward for one step, from the states either side of it.

    A free function rather than a method because :class:`StkVectorEnv` needs
    exactly the same arithmetic, and two copies of it would be two things to
    keep in step.
    """
    if scheme not in REWARD_SCHEMES:
        raise ValueError(f"unknown reward_scheme {scheme!r}; choose from {REWARD_SCHEMES}")

    # The bonus is paid on the crossing, not for having crossed. An agent that
    # honours `terminated` never sees the difference, but one that keeps
    # stepping - a wrapper that ignores it, a hand-written loop - would
    # otherwise sit past the line collecting the bonus once per step.
    just_finished = bool(after.get("finished", False)) and not bool(
        before.get("finished", False)
    )
    if scheme == "sparse":
        return _FINISH_BONUS if just_finished else 0.0

    delta = _obs.progress_of(after, track_length) - _obs.progress_of(
        before, track_length
    )
    # A rescue teleports the kart back to the driveline and a lap counter change
    # can land either side of the wrap, so a single step can appear to jump most
    # of a lap. Anything that large is bookkeeping rather than driving, and
    # paying for it would teach the agent to trigger it.
    limit = track_length / 2.0
    if not -limit < delta < limit:
        delta = 0.0

    if scheme == "progress_only":
        return float(delta)

    reward = delta - _TIME_PENALTY
    if not after.get("on_road", True):
        reward -= _OFF_ROAD_PENALTY
    if just_finished:
        reward += _FINISH_BONUS
    return float(reward)


def state_info(state: dict[str, Any], track_length: float) -> dict[str, Any]:
    """The per-step info dict, shared with the vector environment."""
    return {
        # Monitor turns is_success into a success rate.
        "is_success": bool(state.get("finished", False)),
        "progress_m": _obs.progress_of(state, track_length),
        "lap": state.get("finished_laps", 0),
        "rank": state.get("rank", 0),
        "speed": state.get("speed", 0.0),
        "on_road": bool(state.get("on_road", True)),
        "wrong_way": bool(state.get("wrong_way", False)),
        "race_time": state.get("time", 0.0),
    }


class StkEnv(gymnasium.Env):
    """One kart in one race, driven a few physics ticks at a time.

    The track is fixed for the life of the environment: loading one costs about
    half a second, so ``reset`` restarts the race in place instead, which costs
    well under a millisecond. Racing a different track means a different
    environment.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}

    def __init__(
        self,
        track: str = "hacienda",
        *,
        laps: int | None = None,
        num_karts: int | None = None,
        difficulty: int | None = None,
        obs_mode: str = "vector",
        action_mode: str = "continuous",
        reward_scheme: str = "progress",
        frame_skip: int | None = None,
        lookahead: int | None = None,
        expert: bool = False,
        render_mode: str | None = None,
        binary: str | None = None,
        cwd: str | None = None,
        capture_stderr: bool = False,
    ):
        if obs_mode not in _obs.OBS_MODES:
            raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {_obs.OBS_MODES}")
        if action_mode not in _actions.ACTION_MODES:
            raise ValueError(
                f"unknown action_mode {action_mode!r}; choose from {_actions.ACTION_MODES}"
            )
        if reward_scheme not in REWARD_SCHEMES:
            raise ValueError(
                f"unknown reward_scheme {reward_scheme!r}; choose from {REWARD_SCHEMES}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")

        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.reward_scheme = reward_scheme
        self.render_mode = render_mode
        self.expert = expert

        path = find_binary(binary)
        self.engine = Engine(
            path,
            server_args(
                track=track,
                laps=laps,
                num_karts=num_karts,
                difficulty=difficulty,
                frame_skip=frame_skip,
                lookahead=lookahead,
                expert=expert,
                include_karts=obs_mode == "vector_karts",
                render=render_mode == "human",
            ),
            cwd=cwd if cwd is not None else default_cwd(path),
            capture_stderr=capture_stderr,
        )

        meta = self.engine.meta
        # The spaces come from the handshake rather than from constants
        # duplicated here, so the two sides cannot drift apart.
        self.observation_space = _obs.space_for(obs_mode, meta)
        self.action_space = _actions.space_for(action_mode)
        self.track_length = float(meta.get("track_length", 0.0)) or 1.0
        self.metadata = dict(
            self.metadata,
            render_fps=int(round(1.0 / float(meta.get("step_dt", 0.05) or 0.05))),
        )
        self._state: dict[str, Any] = {}

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        message: dict[str, Any] = {"cmd": "reset"}
        if seed is not None:
            message["seed"] = int(seed)
        if options:
            if "seed" in options:
                message["seed"] = int(options["seed"])
            # Anything else is forwarded so the server can refuse it by name;
            # asking for a different track answers not_supported rather than
            # being quietly ignored.
            for key in options:
                if key != "seed":
                    message.setdefault("options", {})[key] = options[key]
        self._state = self.engine.state(message)
        return self._observation(), self._info()

    def step(self, action):
        previous = self._state
        control = _actions.to_control(action, self.action_mode)
        self._state = self.engine.state({"cmd": "step", "action": control})
        reward = compute_reward(
            previous,
            self._state,
            scheme=self.reward_scheme,
            track_length=self.track_length,
        )
        terminated = bool(self._state.get("finished", False))
        # Truncation is a TimeLimit decision; this class keeps no step budget.
        return self._observation(), reward, terminated, False, self._info()

    def render(self):
        if self.render_mode == "ansi":
            s = self._state
            return (
                f"t={s.get('time', 0.0):6.2f}s  lap {s.get('finished_laps', 0)}"
                f"/{self.engine.meta.get('laps', '?')}  rank {s.get('rank', 0)}  "
                f"{s.get('speed', 0.0):5.1f} m/s  "
                f"{_obs.progress_of(s, self.track_length):8.1f} m  "
                f"{'on' if s.get('on_road', True) else 'OFF'}-road"
            )
        # "human" is drawn by the game's own window; nothing to do here.
        return None

    def close(self):
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):
        self.close()

    # -- Internals -----------------------------------------------------------

    def _observation(self) -> np.ndarray:
        return _obs.encode(self._state, self.obs_mode, self.engine.meta)

    def _info(self) -> dict[str, Any]:
        return state_info(self._state, self.track_length)
