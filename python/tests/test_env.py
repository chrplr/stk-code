# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Gymnasium environment, against a real race."""

from __future__ import annotations

import os

import gymnasium
import numpy as np
import pytest

import stk_gym
from stk_gym import StkEnv

from conftest import FAST

STRAIGHT = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def test_it_passes_the_gymnasium_checker(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env.unwrapped, skip_render_check=True)


def test_reset_returns_an_observation_in_the_space(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["is_success"] is False
    assert info["race_time"] == 0


def test_a_step_moves_the_kart_and_pays_for_it(env):
    env.reset(seed=0)
    rewards = []
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(STRAIGHT)
        rewards.append(reward)
    assert env.observation_space.contains(obs)
    assert not terminated and not truncated
    assert info["speed"] > 1.0
    # Accelerating in a straight line from the grid earns its keep for a while.
    assert sum(rewards) > 0.0


def test_reset_restarts_the_race(env):
    env.reset(seed=0)
    for _ in range(100):
        env.step(STRAIGHT)
    _obs, info = env.reset(seed=0)
    assert info["race_time"] == 0
    assert info["lap"] == -1
    assert info["speed"] == 0


def test_progress_is_not_stale_at_the_start_of_an_episode(env):
    """The trap this guards: track sectors are only recomputed by the world's
    own update, so without an explicit refresh the first observation of an
    episode still holds the previous episode's distance."""
    env.reset(seed=0)
    for _ in range(200):
        env.step(STRAIGHT)
    _obs, info = env.reset(seed=0)
    assert abs(info["progress_m"]) < 20.0


def test_registered_ids_make_working_environments(binary):
    env = gymnasium.make("SuperTuxKart-Easy-v0", binary=binary)
    try:
        obs, _info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        env.step(env.action_space.sample())
    finally:
        env.close()


def test_the_step_budget_truncates_rather_than_terminates(binary):
    env = gymnasium.wrappers.TimeLimit(
        StkEnv(binary=binary, **FAST), max_episode_steps=10
    )
    try:
        env.reset(seed=0)
        for _ in range(9):
            _o, _r, terminated, truncated, _i = env.step(STRAIGHT)
            assert not terminated and not truncated
        _o, _r, terminated, truncated, _i = env.step(STRAIGHT)
        assert truncated and not terminated
    finally:
        env.close()


@pytest.mark.parametrize("action_mode", stk_gym.ACTION_MODES)
def test_every_action_mode_drives(binary, action_mode):
    env = StkEnv(binary=binary, action_mode=action_mode, **FAST)
    try:
        env.reset(seed=0)
        for _ in range(20):
            env.step(env.action_space.sample())
    finally:
        env.close()


def test_opponents_appear_in_the_wider_observation(binary):
    env = StkEnv(binary=binary, obs_mode="vector_karts", num_karts=4, laps=1)
    try:
        obs, _info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.engine.meta["max_karts"] == 3
        narrow = stk_gym.obs.space_for("vector", env.engine.meta).shape[0]
        assert obs.shape[0] == narrow + 4 * 3
    finally:
        env.close()


def test_the_expert_finishes_the_race_and_a_straight_line_does_not(binary):
    """Floor and ceiling. If this gap ever closes, something is wrong with the
    action mapping or the observation, and no amount of training would show it."""
    with StkEnv(binary=binary, **FAST) as env:
        floor = stk_gym.run_straight(env, seed=0, max_steps=1500)
    with StkEnv(binary=binary, expert=True, **FAST) as env:
        ceiling = stk_gym.run_expert(env, seed=0, max_steps=4000)
    assert not floor.finished
    assert ceiling.finished
    assert ceiling.progress_m > 5 * max(floor.progress_m, 1.0)


def test_run_expert_refuses_an_environment_that_is_not_an_expert(env):
    with pytest.raises(ValueError):
        stk_gym.run_expert(env, max_steps=1)


def test_ansi_render_says_where_the_kart_is(binary):
    env = StkEnv(binary=binary, render_mode="ansi", **FAST)
    try:
        env.reset(seed=0)
        env.step(STRAIGHT)
        text = env.render()
        assert "lap" in text and "m/s" in text
    finally:
        env.close()


def test_bad_construction_arguments_are_refused_by_name(binary):
    for kwargs in (
        {"obs_mode": "pixel_soup"},
        {"action_mode": "wiggle"},
        {"reward_scheme": "vibes"},
        {"render_mode": "opengl"},
    ):
        with pytest.raises(ValueError):
            StkEnv(binary=binary, **kwargs, **FAST)


# -- Frames and keys: a window is needed ---------------------------------------

_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
needs_display = pytest.mark.skipif(not _HAS_DISPLAY, reason="frames need a display")


@pytest.fixture(scope="module")
def frame_env(binary):
    """An environment that draws, small and hidden; shared, so reset it."""
    if not _HAS_DISPLAY:
        pytest.skip("frames need a display")
    e = StkEnv(
        binary=binary, render_mode="rgb_array", action_mode="keys",
        frame_skip=2, screensize=(320, 180), **FAST,
    )
    yield e
    e.close()


@needs_display
def test_rgb_array_frames_have_the_advertised_shape(frame_env):
    env = frame_env
    assert env.engine.meta["frame_supported"] and env.engine.meta["hidden"]
    env.reset(seed=0)
    first = env.render()
    assert first.shape == (180, 320, 3) and first.dtype == np.uint8
    assert first.std() > 1.0, "a frame that is one colour was not rendered"
    for _ in range(2):
        env.step(np.zeros(8, dtype=np.int8))
    assert env.render().shape == first.shape
    assert env.metadata["render_fps"] == 60


@needs_display
def test_pixels_observations_live_in_their_space(binary):
    env = StkEnv(binary=binary, obs_mode="pixels", screensize=(320, 180), **FAST)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(STRAIGHT)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@needs_display
def test_held_keys_reach_the_player_controller(frame_env):
    """Holding left and up must show up as the game's own controls: a negative
    steer that ramps rather than jumps, and full acceleration."""
    env = frame_env
    env.reset(seed=0)
    left_up = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
    steers = []
    for _ in range(12):
        env.step(left_up)
        steers.append(env.sample()["ctrl_steer"])
    assert env.sample()["ctrl_accel"] == 1.0
    assert steers[-1] < 0.0 and steers[0] > steers[-1], steers
    assert steers[0] > -1.0, "the steering ramps over several ticks, not at once"
    # Released: the controller returns to straight.
    for _ in range(30):
        env.step(np.zeros(8, dtype=np.int8))
    assert env.sample()["ctrl_steer"] == 0.0
