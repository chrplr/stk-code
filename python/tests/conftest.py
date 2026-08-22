# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fixtures.

Every test here starts a real race. Loading a track costs about half a second,
so the module-scoped fixtures below are shared rather than rebuilt per test; the
ones that need a pristine environment make their own.
"""

from __future__ import annotations

import pytest

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
