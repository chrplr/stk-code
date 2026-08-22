# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""The encoding and the reward, which are pure functions and need no race."""

from __future__ import annotations

import numpy as np
import pytest

from stk_gym import compute_reward, progress_of
from stk_gym.actions import ACTION_MODES, DISCRETE_ACTIONS, to_control
from stk_gym.actions import space_for as action_space_for
from stk_gym.obs import OBS_MODES, encode
from stk_gym.obs import space_for as obs_space_for

META = {"lookahead_k": 3, "max_karts": 2, "track_length": 1000.0, "num_karts": 3}
LAP = META["track_length"]


def a_state(**overrides):
    state = {
        "speed": 10.0, "max_speed": 20.0, "steer": 0.5, "on_road": True,
        "on_ground": True, "wrong_way": False, "distance_to_center": 1.0,
        "nitro_energy": 20.0, "rank": 2, "distance_down_track": 100.0,
        "finished_laps": 0, "finished": False,
        "lookahead": [{"x": 1.0, "y": 0.0, "z": 10.0},
                      {"x": 2.0, "y": 0.0, "z": 20.0},
                      {"x": 3.0, "y": 0.0, "z": 30.0}],
        "karts": [{"x": 5.0, "z": 5.0, "speed": 9.0, "distance_down_track": 90.0,
                   "finished_laps": 0},
                  {"x": 1.0, "z": 1.0, "speed": 8.0, "distance_down_track": 80.0,
                   "finished_laps": 0}],
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize("mode", OBS_MODES)
def test_encoding_matches_the_declared_space(mode):
    space = obs_space_for(mode, META)
    obs = encode(a_state(), mode, META)
    assert obs.shape == space.shape
    assert obs.dtype == np.float32
    assert space.contains(obs)


def test_a_short_lookahead_is_padded_to_the_declared_length():
    """A track whose driveline runs out must not change the observation shape."""
    obs = encode(a_state(lookahead=[{"x": 1.0, "y": 0.0, "z": 10.0}]), "vector", META)
    assert obs.shape == obs_space_for("vector", META).shape
    # The missing points repeat the last real one rather than reading as
    # "the racing line is exactly where I am".
    assert obs[-1] == pytest.approx(obs[-3]) == pytest.approx(10.0 / 50.0)


def test_missing_facts_do_not_crash_the_encoder():
    """An arena has no driveline, so most of these keys are simply absent."""
    obs = encode({"speed": 1.0}, "vector", META)
    assert obs.shape == obs_space_for("vector", META).shape
    assert np.all(np.isfinite(obs))


def test_opponents_are_ordered_nearest_first():
    obs = encode(a_state(), "vector_karts", META)
    first_x = obs[-8] * 50.0
    assert first_x == pytest.approx(1.0)  # the kart at (1, 1), not the one at (5, 5)


def test_progress_runs_smoothly_through_the_start_line():
    before = progress_of({"finished_laps": -1, "distance_down_track": 999.0}, LAP)
    after = progress_of({"finished_laps": 0, "distance_down_track": 1.0}, LAP)
    assert before == pytest.approx(-1.0)
    assert after == pytest.approx(1.0)
    assert after - before == pytest.approx(2.0)


def test_reward_pays_for_distance_covered():
    before = a_state(distance_down_track=100.0)
    after = a_state(distance_down_track=110.0)
    assert compute_reward(before, after, scheme="progress_only",
                          track_length=LAP) == pytest.approx(10.0)


def test_a_rescue_sized_jump_is_not_paid_for():
    """A teleport is bookkeeping, not driving; paying for it teaches the wrong
    thing."""
    before = a_state(distance_down_track=900.0)
    after = a_state(distance_down_track=10.0, finished_laps=0)
    assert compute_reward(before, after, scheme="progress_only",
                          track_length=LAP) == 0.0


def test_finishing_is_worth_more_than_a_step_of_driving():
    before = a_state()
    after = a_state(finished=True)
    plain = compute_reward(before, a_state(), scheme="progress", track_length=LAP)
    done = compute_reward(before, after, scheme="progress", track_length=LAP)
    assert done > plain + 10.0


def test_sparse_pays_only_at_the_finish():
    assert compute_reward(a_state(), a_state(distance_down_track=200.0),
                          scheme="sparse", track_length=LAP) == 0.0
    assert compute_reward(a_state(), a_state(finished=True),
                          scheme="sparse", track_length=LAP) > 0.0


def test_unknown_schemes_and_modes_are_refused():
    with pytest.raises(ValueError):
        compute_reward(a_state(), a_state(), scheme="vibes", track_length=LAP)
    with pytest.raises(ValueError):
        obs_space_for("pixels", META)
    with pytest.raises(ValueError):
        action_space_for("wiggle")


@pytest.mark.parametrize("mode", ACTION_MODES)
def test_every_sampled_action_converts_to_json_safe_controls(mode):
    space = action_space_for(mode)
    for _ in range(20):
        control = to_control(space.sample(), mode)
        for key, value in control.items():
            # Discrete.sample() gives np.int64 and Box.sample() np.float32,
            # neither of which json will encode.
            assert type(value) in (float, bool, int), (key, type(value))
        assert -1.0 <= control["steer"] <= 1.0
        assert 0.0 <= control["accel"] <= 1.0


def test_skidding_takes_its_direction_from_the_steering():
    assert to_control([-1.0, 1.0, 0.0, 0.0, 1.0], "continuous_full")["skid"] == 2
    assert to_control([1.0, 1.0, 0.0, 0.0, 1.0], "continuous_full")["skid"] == 3
    assert to_control([0.0, 1.0, 0.0, 0.0, 0.0], "continuous_full")["skid"] == 0


def test_the_discrete_table_covers_steering_and_throttle():
    assert len(DISCRETE_ACTIONS) == 15
    assert {c["steer"] for c in DISCRETE_ACTIONS} == {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_out_of_range_discrete_actions_are_refused():
    with pytest.raises(ValueError):
        to_control(len(DISCRETE_ACTIONS), "discrete")
