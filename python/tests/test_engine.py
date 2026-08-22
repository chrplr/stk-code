# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transport, tested against the real game binary."""

from __future__ import annotations

import pytest

from stk_gym import CommandFailed, Engine, EngineDied, default_cwd, server_args
from stk_gym.engine import PROTOCOL

from conftest import FAST


@pytest.fixture
def engine(binary):
    e = Engine(binary, server_args(**FAST), cwd=default_cwd(binary))
    yield e
    e.close()


def test_handshake_reports_everything_the_client_needs(engine):
    meta = engine.meta
    assert meta["protocol"] == PROTOCOL
    assert meta["game"] == "supertuxkart"
    assert meta["track"] == "hacienda"
    # The spaces are built from these, so their absence must fail loudly here
    # rather than as a shape mismatch much later.
    for key in ("lookahead_k", "max_karts", "track_length", "frames", "step_dt"):
        assert key in meta, key
    assert meta["track_length"] > 0


def test_step_before_reset_is_refused_by_name(engine):
    with pytest.raises(CommandFailed) as excinfo:
        engine.request({"cmd": "step", "action": {"accel": 1.0}})
    assert excinfo.value.kind == "not_reset"


def test_unknown_command_is_refused_by_name(engine):
    with pytest.raises(CommandFailed) as excinfo:
        engine.request({"cmd": "fly"})
    assert excinfo.value.kind == "unknown_cmd"


def test_a_second_environment_id_is_refused(engine):
    engine.request({"cmd": "reset"})
    with pytest.raises(CommandFailed) as excinfo:
        engine.request({"cmd": "step", "env_id": 1, "action": {}})
    assert excinfo.value.kind == "no_such_env"


def test_changing_track_is_refused_rather_than_ignored(engine):
    with pytest.raises(CommandFailed) as excinfo:
        engine.request({"cmd": "reset", "options": {"track": "cornfield_crossing"}})
    assert excinfo.value.kind == "not_supported"


def test_a_bad_action_type_is_refused(engine):
    engine.request({"cmd": "reset"})
    with pytest.raises(CommandFailed) as excinfo:
        engine.request({"cmd": "step", "action": 3})
    assert excinfo.value.kind == "bad_action"


def test_the_server_survives_a_bad_request(engine):
    """A client bug must not take the server down."""
    engine.request({"cmd": "reset"})
    with pytest.raises(CommandFailed):
        engine.request({"cmd": "nonsense"})
    state = engine.state({"cmd": "step", "action": {"accel": 1.0}})
    assert state["tick"] > 0


def test_every_step_advances_the_clock_by_the_frame_skip(engine):
    frames = engine.meta["frames"]
    state = engine.state({"cmd": "reset"})
    previous = state["tick"]
    for _ in range(20):
        state = engine.state({"cmd": "step", "action": {"accel": 1.0}})
        assert state["tick"] - previous == frames
        previous = state["tick"]


def test_state_does_not_advance_the_race(engine):
    engine.request({"cmd": "reset"})
    first = engine.state({"cmd": "step", "action": {"accel": 1.0}})
    again = engine.state({"cmd": "state"})
    assert again["tick"] == first["tick"]


def test_close_is_idempotent_and_using_a_closed_engine_says_so(engine):
    engine.close()
    engine.close()
    with pytest.raises(EngineDied):
        engine.request({"cmd": "hello"})


def test_two_processes_with_the_same_seed_agree(binary):
    """The guarantee this package offers: a whole run reproduces exactly."""
    def rollout():
        with Engine(binary, server_args(**FAST), cwd=default_cwd(binary)) as e:
            e.request({"cmd": "reset", "seed": 3})
            return [
                tuple(e.state({"cmd": "step", "action": {"steer": 0.3, "accel": 1.0}})["xyz"])
                for _ in range(40)
            ]

    assert rollout() == rollout()
