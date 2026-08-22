# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Several races at once.

Every other game in this family keeps N sessions inside one server process and
steps them with one round trip. SuperTuxKart cannot: World, RaceManager, Track
and the physics world are all process globals, so one race per process is the
hard limit. This class therefore owns N children.

What it does buy back is overlap. A step posts its request to every child before
reading any reply, so the races simulate at the same time rather than one after
another; the cost of a vector step is the slowest child, not their sum.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

try:                                        # gymnasium >= 1.0
    from gymnasium.vector import AutoresetMode
    _AUTORESET = {"autoreset_mode": AutoresetMode.NEXT_STEP}
except ImportError:                         # pragma: no cover
    _AUTORESET = {}

from . import actions as _actions
from . import obs as _obs
from .binary import default_cwd, find_binary
from .engine import Engine, server_args
from .env import REWARD_SCHEMES, compute_reward, state_info

__all__ = ["StkVectorEnv"]


class StkVectorEnv(VectorEnv):
    """A batch of independent races, one child process each."""

    metadata = {"render_modes": [], **_AUTORESET}

    def __init__(
        self,
        num_envs: int = 4,
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
        binary: str | None = None,
        cwd: str | None = None,
        capture_stderr: bool = False,
        **_ignored: Any,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if reward_scheme not in REWARD_SCHEMES:
            raise ValueError(
                f"unknown reward_scheme {reward_scheme!r}; choose from {REWARD_SCHEMES}"
            )
        self.num_envs = num_envs
        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.reward_scheme = reward_scheme

        path = find_binary(binary)
        args = server_args(
            track=track,
            laps=laps,
            num_karts=num_karts,
            difficulty=difficulty,
            frame_skip=frame_skip,
            lookahead=lookahead,
            expert=expert,
            include_karts=obs_mode == "vector_karts",
            render=False,
        )
        run_in = cwd if cwd is not None else default_cwd(path)
        self.engines: list[Engine] = []
        try:
            for _ in range(num_envs):
                self.engines.append(
                    Engine(path, args, cwd=run_in, capture_stderr=capture_stderr)
                )
        except BaseException:
            # A partially built vector would leave orphan children behind.
            self.close()
            raise

        meta = self.engines[0].meta
        self.single_observation_space = _obs.space_for(obs_mode, meta)
        self.single_action_space = _actions.space_for(action_mode)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.track_length = float(meta.get("track_length", 0.0)) or 1.0

        self._states: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        # NEXT_STEP autoreset: an environment that ended on the previous step is
        # reset by this one, and reports the reset observation with no reward.
        self._needs_reset = np.zeros(num_envs, dtype=bool)

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | list[int] | None = None, options: dict | None = None):
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            # Seeded the way SyncVectorEnv seeds its sub-environments, so the
            # two agree episode for episode.
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(
                    f"expected {self.num_envs} seeds, got {len(seeds)}"
                )

        messages = []
        for s in seeds:
            message: dict[str, Any] = {"cmd": "reset"}
            if s is not None:
                message["seed"] = int(s)
            messages.append(message)
        self._states = self._round_trip(messages)
        self._needs_reset[:] = False
        return self._observations(), self._infos()

    def step(self, actions):
        previous = list(self._states)
        messages: list[dict[str, Any]] = []
        for i in range(self.num_envs):
            if self._needs_reset[i]:
                messages.append({"cmd": "reset"})
            else:
                control = _actions.to_control(actions[i], self.action_mode)
                messages.append({"cmd": "step", "action": control})
        self._states = self._round_trip(messages)

        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminations = np.zeros(self.num_envs, dtype=bool)
        for i in range(self.num_envs):
            if self._needs_reset[i]:
                self._needs_reset[i] = False
                continue
            rewards[i] = compute_reward(
                previous[i],
                self._states[i],
                scheme=self.reward_scheme,
                track_length=self.track_length,
            )
            terminations[i] = bool(self._states[i].get("finished", False))
        self._needs_reset |= terminations
        truncations = np.zeros(self.num_envs, dtype=bool)
        return (
            self._observations(),
            rewards,
            terminations,
            truncations,
            self._infos(),
        )

    def flush_autoreset(self):
        """Perform any pending autoresets now, without stepping the others.

        NEXT_STEP autoreset hands the terminal observation back on the step that
        ended the episode and only resets on the following one. Stable-Baselines3
        expects the reset observation immediately, so its adapter calls this to
        bring the two conventions into line. Environments with nothing to reset
        are asked for their current state, which does not advance them.
        """
        if not self._needs_reset.any():
            return self._observations(), self._infos()
        messages = [
            {"cmd": "reset"} if need else {"cmd": "state"}
            for need in self._needs_reset
        ]
        self._states = self._round_trip(messages)
        self._needs_reset[:] = False
        return self._observations(), self._infos()

    def close(self, **_kwargs):
        for engine in getattr(self, "engines", []):
            engine.close()
        self.engines = []

    def __del__(self):
        self.close()

    # -- Internals -----------------------------------------------------------

    def _round_trip(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Post every request, then collect every answer.

        The whole point of this class: while the last child is still being
        written to, the first is already simulating.
        """
        pending = [engine.send(message)
                   for engine, message in zip(self.engines, messages)]
        return [engine.recv(*token)["state"]
                for engine, token in zip(self.engines, pending)]

    def _observations(self) -> np.ndarray:
        meta = self.engines[0].meta
        return np.stack(
            [_obs.encode(state, self.obs_mode, meta) for state in self._states]
        )

    def _infos(self) -> dict[str, Any]:
        """Infos as a dict of arrays, which is the shape gymnasium expects."""
        per_env = [state_info(state, self.track_length) for state in self._states]
        out: dict[str, Any] = {}
        for key in per_env[0]:
            out[key] = np.array([info[key] for info in per_env])
            # gymnasium pairs every info key with a boolean mask saying which
            # sub-environments supplied it; here that is always all of them.
            out[f"_{key}"] = np.ones(self.num_envs, dtype=bool)
        return out
