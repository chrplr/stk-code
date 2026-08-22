# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Stable-Baselines3 VecEnv on top of :class:`stk_gym.StkVectorEnv`.

SB3 predates gymnasium's vector API and differs from it in exactly two ways
worth naming, because both are silent when got wrong:

- **Autoreset timing.** Gymnasium's NEXT_STEP mode returns the terminal
  observation on the step that ended the episode and resets on the next one.
  SB3 expects the reset observation on that same step, with the terminal one
  tucked into ``info["terminal_observation"]``.
- **terminated vs truncated.** SB3 has a single ``done`` and marks a time limit
  with ``info["TimeLimit.truncated"]``. Collapsing the two without that flag
  makes the value function treat a time-out as a real terminal state, which
  quietly biases every long episode.

SB3 is imported lazily so it stays an optional dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .vector_env import StkVectorEnv

__all__ = ["make_sb3_vec_env"]


def make_sb3_vec_env(num_envs: int = 4, max_episode_steps: int = 3000, **kwargs: Any):
    """Return an SB3 ``VecEnv`` backed by ``num_envs`` racing processes."""
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class _Adapter(VecEnv):
        def __init__(self):
            self.venv = StkVectorEnv(num_envs=num_envs, **kwargs)
            super().__init__(
                self.venv.num_envs,
                self.venv.single_observation_space,
                self.venv.single_action_space,
            )
            self._actions: np.ndarray | None = None
            self._steps = np.zeros(self.venv.num_envs, dtype=np.int64)

        # -- VecEnv API ------------------------------------------------------

        def reset(self):
            obs, _infos = self.venv.reset(seed=self._seed)
            self._seed = None
            self._steps[:] = 0
            return obs

        def step_async(self, actions):
            self._actions = actions

        def step_wait(self):
            obs, rewards, terminations, truncations, infos = self.venv.step(
                self._actions
            )
            self._steps += 1
            # The step budget lives here rather than in the vector environment,
            # which keeps none, the same way TimeLimit sits outside StkEnv.
            truncations = truncations | (self._steps >= max_episode_steps)
            dones = terminations | truncations

            per_env = [
                {key: infos[key][i] for key in infos if not key.startswith("_")}
                for i in range(self.num_envs)
            ]
            if dones.any():
                terminal_obs = obs.copy()
                for i in np.flatnonzero(dones):
                    per_env[i]["terminal_observation"] = terminal_obs[i]
                    per_env[i]["TimeLimit.truncated"] = bool(
                        truncations[i] and not terminations[i]
                    )
                # Truncation ends the episode here but not in the game, which
                # only autoresets on termination; asking for the reset makes the
                # two agree.
                for i in np.flatnonzero(truncations & ~terminations):
                    self.venv._needs_reset[i] = True
                obs, _ = self.venv.flush_autoreset()
                self._steps[dones] = 0
            return obs, rewards, dones, per_env

        def close(self):
            self.venv.close()

        # -- The rest of the interface, which SB3 requires but barely uses ---

        _seed = None

        def seed(self, seed=None):
            self._seed = seed
            return [seed] * self.num_envs

        def get_attr(self, attr_name, indices=None):
            value = getattr(self.venv, attr_name, None)
            return [value] * len(self._indices(indices))

        def set_attr(self, attr_name, value, indices=None):
            setattr(self.venv, attr_name, value)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            raise NotImplementedError(
                "the racing environments live in child processes; there is no "
                "Python object to call methods on"
            )

        def env_is_wrapped(self, wrapper_class, indices=None):
            return [False] * len(self._indices(indices))

        def _indices(self, indices):
            if indices is None:
                return range(self.num_envs)
            if isinstance(indices, int):
                return [indices]
            return indices

    return _Adapter()
