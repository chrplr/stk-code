# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Finding the ``supertuxkart`` binary that serves the race.

The game is the single source of truth for the physics, so this package is
useless without it. Unlike a Go server, it cannot be built on demand: a
SuperTuxKart build takes minutes, not seconds, and silently starting one from
inside ``gymnasium.make`` would be a very unpleasant surprise. The search
therefore ends with a message saying exactly what to run.

The binary also needs its assets. It resolves them relative to the working
directory, so :func:`default_cwd` reports the checkout to run it in.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = ["find_binary", "default_cwd", "BinaryNotFound"]

_EXE = ".exe" if sys.platform == "win32" else ""
_BINARY_NAME = "supertuxkart" + _EXE
_ENV_VAR = "STK_ENV_BIN"
# The file that identifies a stk-code checkout carrying this feature. Checking
# for the gym server rather than for the repository root means an old checkout
# reports "no gym support" instead of failing later with unknown_cmd.
_MARKER = Path("src") / "gym" / "gym_server.hpp"
_BUILD_HINT = (
    "    cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\n"
)


class BinaryNotFound(RuntimeError):
    """The game binary could not be located."""


def find_binary(explicit: str | os.PathLike[str] | None = None) -> str:
    """Return the path to the SuperTuxKart binary.

    Tried in order: an explicit path, ``$STK_ENV_BIN``, the build directory of
    the checkout this package lives in, and finally the ``PATH``.

    The checkout is preferred over the ``PATH`` here, the opposite of the usual
    order, because a system-wide SuperTuxKart is almost certainly a release
    without ``--gym``, while the checkout next to this file is the one being
    worked on.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise BinaryNotFound(f"no such binary: {path}")
        return str(path)

    env = os.environ.get(_ENV_VAR)
    if env:
        if not Path(env).exists():
            raise BinaryNotFound(f"${_ENV_VAR} points at a missing file: {env}")
        return env

    repo = find_repo()
    if repo is not None:
        for candidate in (
            repo / "build" / "bin" / _BINARY_NAME,
            repo / "build" / _BINARY_NAME,
            repo / "cmake_build" / "bin" / _BINARY_NAME,
        ):
            if candidate.exists():
                return str(candidate)
        raise BinaryNotFound(
            f"found the stk-code checkout at {repo} but no built binary.\n"
            "Build it with:\n" + _BUILD_HINT +
            f"or point ${_ENV_VAR} at an existing one."
        )

    on_path = shutil.which(_BINARY_NAME)
    if on_path:
        return on_path

    raise BinaryNotFound(
        "cannot find the supertuxkart binary.\n"
        "Build it from a stk-code checkout:\n" + _BUILD_HINT +
        f"then point ${_ENV_VAR} at build/bin/{_BINARY_NAME}."
    )


def find_repo() -> Path | None:
    """Walk up from this file looking for the stk-code checkout it lives in."""
    for parent in Path(__file__).resolve().parents:
        if (parent / _MARKER).is_file():
            return parent
    return None


def default_cwd(binary: str | os.PathLike[str]) -> str | None:
    """Return the directory to run \\p binary from, or None to inherit.

    SuperTuxKart looks for its assets relative to the working directory - a
    checkout keeps them in a sibling ``stk-assets`` - so a child started
    anywhere else dies with "Set $SUPERTUXKART_DATADIR". An installed binary
    knows its own prefix and needs nothing.
    """
    if os.environ.get("SUPERTUXKART_DATADIR"):
        return None
    for parent in Path(binary).resolve().parents:
        if (parent / _MARKER).is_file():
            return str(parent)
    return str(find_repo()) if find_repo() is not None else None
