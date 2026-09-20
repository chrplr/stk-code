# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fixtures.

Every test here starts a real race. Loading a track costs about half a second,
so the module-scoped fixtures below are shared rather than rebuilt per test; the
ones that need a pristine environment make their own.
"""

from __future__ import annotations

import os

import pytest

# Never let a test run fetch the release pack. Without a local binary
# these tests should skip in a second, not download the game; find_binary's last
# resort is exactly the kind of thing that should not fire unasked in CI. An
# explicit setting in the environment still wins, so a deliberate test of the
# download path can have it.
os.environ.setdefault("STK_ENV_OFFLINE", "1")

from stk_gym import BinaryNotFound, StkEnv, find_binary

# A short, cheap race: one kart, one lap, no opponents to simulate.
FAST = dict(num_karts=1, laps=1, track="hacienda")


@pytest.fixture(scope="session")
def binary() -> str:
    try:
        return find_binary()
    except BinaryNotFound as exc:
        pytest.skip(f"no supertuxkart binary: {exc}")


@pytest.fixture(scope="module")
def env(binary):
    """A shared environment. Tests must leave it usable, i.e. reset it."""
    e = StkEnv(binary=binary, **FAST)
    yield e
    e.close()
