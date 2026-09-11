# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""--gym-human: the game keeps its own clock; the protocol only reports."""

from __future__ import annotations

import os
import time

import pytest

from stk_gym import CommandFailed, HumanSession, server_args

from conftest import FAST


def test_server_args_human_mode():
    args = server_args(human=True, **FAST, fullscreen=True, screensize="1024x768",
                       race_now=True, extra=["--no-console"])
    assert args[0] == "--gym-human"
    assert "--no-graphics" not in args, "a person needs the window"
    assert "--fullscreen" in args and "--screensize=1024x768" in args
    assert "--race-now" in args and args[-1] == "--no-console"
    # The agent mode is unchanged by the new keywords' defaults.
    assert server_args(**FAST)[:2] == ["--gym", "--no-graphics"]


# Human mode opens a real window: skipped where there is no display to open it
# on, which is what a CI runner is.
needs_display = pytest.mark.skipif(
    not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="human mode opens a window",
)


@needs_display
def test_human_session_runs_on_its_own_clock(binary):
    with HumanSession(binary=binary, race_now=True, screensize="640x480",
                      extra_args=["--no-console"], **FAST) as session:
        assert session.meta["human"] is True
        first = session.reset(seed=1)
        assert "controls" in first and "tick" in first

        # Nothing is stepped from here, yet the race clock advances at wall
        # rate: the game is running by itself.
        t0 = time.perf_counter()
        a = session.state()
        time.sleep(0.5)
        b = session.state()
        wall = time.perf_counter() - t0
        race = b["time"] - a["time"]
        assert 0.7 * wall <= race <= 1.3 * wall, (race, wall)
        assert b["tick"] > a["tick"]

        # step is refused by name, not silently answered.
        with pytest.raises(CommandFailed) as err:
            session.engine.state({"cmd": "step", "action": {"steer": 0}})
        assert err.value.kind == "not_supported"

        # info is the same flat summary StkEnv gives.
        info = session.info()
        assert {"race_time", "speed", "lap", "progress_m"} <= info.keys()
