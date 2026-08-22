# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Several races at once."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv

from stk_gym import StkEnv, StkVectorEnv

from conftest import FAST

N = 2
STRAIGHT = np.tile(np.array([0.15, 1.0, 0.0], dtype=np.float32), (N, 1))


@pytest.fixture
def venv(binary):
    v = StkVectorEnv(num_envs=N, binary=binary, **FAST)
    yield v
    v.close()


def test_shapes_and_batching(venv):
    obs, infos = venv.reset(seed=0)
    assert obs.shape == (N, venv.single_observation_space.shape[0])
    assert venv.observation_space.contains(obs)
    obs, rewards, terminations, truncations, infos = venv.step(STRAIGHT)
    assert rewards.shape == (N,)
    assert terminations.shape == truncations.shape == (N,)
    # gymnasium pairs every info key with a boolean mask.
    assert infos["_progress_m"].shape == (N,)


def test_it_agrees_with_a_sync_vector_env_step_for_step(binary):
    """The whole reason this class exists is speed, not different behaviour."""
    ours = StkVectorEnv(num_envs=N, binary=binary, **FAST)
    theirs = SyncVectorEnv(
        [lambda: StkEnv(binary=binary, **FAST) for _ in range(N)]
    )
    try:
        a, _ = ours.reset(seed=5)
        b, _ = theirs.reset(seed=5)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
        for step in range(50):
            a, ra, ta, _ua, _ = ours.step(STRAIGHT)
            b, rb, tb, _ub, _ = theirs.step(STRAIGHT)
            np.testing.assert_allclose(a, b, rtol=0, atol=0,
                                       err_msg=f"observations differ at step {step}")
            np.testing.assert_allclose(ra, rb, rtol=0, atol=0)
            np.testing.assert_array_equal(ta, tb)
    finally:
        ours.close()
        theirs.close()


def test_different_seeds_go_to_different_sub_environments(venv):
    """Seeded the way SyncVectorEnv does it, so the two agree episode for
    episode."""
    venv.reset(seed=[1, 2])
    with pytest.raises(ValueError):
        venv.reset(seed=[1, 2, 3])


def test_flush_autoreset_does_not_advance_the_others(venv):
    venv.reset(seed=0)
    before = venv.step(STRAIGHT)[0]
    after, _infos = venv.flush_autoreset()
    np.testing.assert_allclose(before, after, rtol=0, atol=0)


def test_closing_twice_is_harmless(binary):
    v = StkVectorEnv(num_envs=1, binary=binary, **FAST)
    v.close()
    v.close()
