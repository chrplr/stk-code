# Copyright (C) 2026 SuperTuxKart-Team
# Distributed under the GNU General Public License v3 or later.
"""A person at the wheel, another process watching.

:class:`HumanSession` starts the game with ``--gym-human``: it opens its usual
window, reads the keyboard or gamepad itself, and runs on its own clock. The
protocol then only *reports* - :meth:`HumanSession.state` returns the race as
it is at the next frame, including the controls the kart applied - and
:meth:`HumanSession.reset` restarts the race. There is no ``step``: the game
refuses it in this mode, so a caller cannot mistake a race that runs by itself
for one it is driving.

This is the shape an experiment harness wants: it presents the game to a
participant and samples the trajectory at its own rate, rather than pacing the
physics from Python.
"""

from __future__ import annotations

from typing import Any, Iterable

from .binary import default_cwd, find_binary
from .engine import Engine, server_args
from .env import state_info
from .obs import SAMPLE_FIELDS, flatten_state

__all__ = ["HumanSession", "SAMPLE_FIELDS"]


class HumanSession:
    """One race in a window, driven by a person; polled, never stepped."""

    def __init__(
        self,
        track: str = "hacienda",
        *,
        laps: int | None = None,
        num_karts: int | None = None,
        difficulty: int | None = None,
        lookahead: int | None = None,
        include_karts: bool = True,
        fullscreen: bool = False,
        screensize: str | None = None,
        race_now: bool = False,
        extra_args: Iterable[str] = (),
        binary: str | None = None,
        cwd: str | None = None,
        capture_stderr: bool = False,
    ):
        path = find_binary(binary)
        self.engine = Engine(
            path,
            server_args(
                track=track,
                laps=laps,
                num_karts=num_karts,
                difficulty=difficulty,
                lookahead=lookahead,
                include_karts=include_karts,
                human=True,
                fullscreen=fullscreen,
                screensize=screensize,
                race_now=race_now,
                extra=extra_args,
            ),
            cwd=cwd if cwd is not None else default_cwd(path),
            capture_stderr=capture_stderr,
        )
        meta = self.engine.meta
        if not meta.get("human"):
            self.close()
            raise RuntimeError(
                f"{path} did not start in --gym-human mode (handshake: {meta}); "
                "the binary predates human mode"
            )
        self.track_length = float(meta.get("track_length", 0.0)) or 1.0
        self._state: dict[str, Any] = {}

    @property
    def meta(self) -> dict[str, Any]:
        """The handshake: track, laps, num_karts, physics_fps, track_length..."""
        return self.engine.meta

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Restart the race in place and return its first state."""
        request: dict[str, Any] = {"cmd": "reset"}
        if seed is not None:
            request["seed"] = int(seed)
        self._state = self.engine.state(request)
        return self._state

    def state(self) -> dict[str, Any]:
        """The race as of the game's next frame (one frame of latency)."""
        self._state = self.engine.state({"cmd": "state"})
        return self._state

    def info(self) -> dict[str, Any]:
        """The flat per-sample summary :class:`StkEnv` puts in ``info``."""
        return state_info(self._state, self.track_length)

    def sample(self) -> dict[str, float]:
        """The last state as one row of floats, :data:`SAMPLE_FIELDS` every time."""
        return flatten_state(self._state, self.track_length)

    def close(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

    def __enter__(self) -> "HumanSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
