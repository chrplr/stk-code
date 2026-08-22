# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Stable-Baselines3 bridge.

What is worth testing here is only the two places the conventions differ:
autoreset timing, and terminated versus truncated.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stk_gym.sb3 import make_sb3_vec_env  # noqa: E402

from conftest import FAST  # noqa: E402

N = 2
LIMIT = 12
STRAIGHT = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (N, 1))


@pytest.fixture
def venv(binary):
    v = make_sb3_vec_env(
        num_envs=N, max_episode_steps=LIMIT, binary=binary, **FAST
    )
    yield v
    v.close()


def test_it_looks_like_a_vec_env(venv):
    obs = venv.reset()
    assert obs.shape == (N, venv.observation_space.shape[0])
    obs, rewards, dones, infos = venv.step(STRAIGHT)
    assert rewards.shape == (N,)
    assert dones.shape == (N,)
    assert len(infos) == N


def test_the_step_budget_is_reported_as_a_time_limit(venv):
    venv.reset()
    for step in range(LIMIT - 1):
        _obs, _r, dones, _infos = venv.step(STRAIGHT)
        assert not dones.any(), f"ended early at step {step}"
    obs, _r, dones, infos = venv.step(STRAIGHT)
    assert dones.all()
    for info in infos:
        # Without this flag the value function treats a time-out as a real
        # terminal state, which quietly biases every long episode.
        assert info["TimeLimit.truncated"] is True
        assert "terminal_observation" in info


def test_the_observation_returned_on_a_done_step_is_the_reset_one(venv):
    venv.reset()
    for _ in range(LIMIT - 1):
        venv.step(STRAIGHT)
    obs, _r, dones, infos = venv.step(STRAIGHT)
    assert dones.all()
    final = np.stack([info["terminal_observation"] for info in infos])
    # SB3 wants the fresh episode's observation in the return value and the
    # last one of the old episode in the info.
    assert not np.allclose(obs, final)
    # A kart on the grid is stopped; one that has been driving is not.
    assert obs[0][0] == pytest.approx(0.0)
    assert final[0][0] > 0.0


def test_a_short_training_run_completes(venv):
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", venv, n_steps=16, batch_size=16, verbose=0)
    model.learn(total_timesteps=64)
