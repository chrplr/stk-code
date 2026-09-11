# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""The client half of the JSON-lines protocol.

One :class:`Engine` owns one ``supertuxkart --gym`` child process and speaks to
it in strict alternation: write one line, read one line. That is the whole
transport. Everything about the race itself is decided on the other side.

There is exactly one race per process - World, RaceManager and the physics world
are all globals in the game - so a vector environment runs several children
rather than several sessions in one child.
"""

from __future__ import annotations

import atexit
import itertools
import json
import subprocess
import weakref
from typing import Any, Iterable

__all__ = ["Engine", "EngineError", "EngineDied", "ProtocolError", "CommandFailed",
           "server_args", "PROTOCOL"]

# Bumped in lockstep with PROTOCOL_VERSION in src/gym/gym_server.cpp.
PROTOCOL = 1


class EngineError(RuntimeError):
    """Base class for transport failures."""


class EngineDied(EngineError):
    """The child process went away."""


class ProtocolError(EngineError):
    """The child said something the client cannot make sense of.

    This is unrecoverable by design: an out-of-step reply means the request and
    response streams no longer line up, and every later answer would be
    attributed to the wrong request. The engine is closed rather than resynced.
    """


class CommandFailed(EngineError):
    """The server answered ``ok: false``."""

    def __init__(self, kind: str, message: str, request: dict[str, Any]):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.request = request


def _reap(proc: subprocess.Popen) -> None:
    """Best-effort termination, safe to call from a finalizer or at exit."""
    if proc.poll() is not None:
        return
    try:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class Engine:
    """A live ``supertuxkart --gym`` process.

    The handshake runs in the constructor, so :attr:`meta` - the track, the lap
    count, the frame skip and the observation dimensions - is available
    immediately and the caller never has to guess the spaces.

    Starting one is not cheap: the game loads a track, which takes about half a
    second. Keep an environment rather than making a new one per episode; that
    is what ``reset`` is for, and it costs well under a millisecond.
    """

    def __init__(
        self,
        binary: str,
        args: Iterable[str] = (),
        *,
        cwd: str | None = None,
        capture_stderr: bool = False,
    ):
        argv = [binary, *args]
        self._closed = False
        # stderr is inherited, not piped: a pipe nobody drains fills up and the
        # child blocks writing to it while we wait for stdout - a deadlock with
        # no symptom other than a hang. It matters more here than in a quieter
        # game, because the child logs a screenful while loading a track.
        stderr = subprocess.PIPE if capture_stderr else subprocess.DEVNULL
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=cwd,
                # Binary pipes: text mode would translate newlines on Windows
                # and corrupt the framing.
                text=False,
                close_fds=True,
            )
        except OSError as exc:
            raise EngineDied(f"cannot start {argv[0]}: {exc}") from exc

        # A finalizer rather than __del__, which is unreliable during
        # interpreter shutdown; the atexit sweep covers processes still alive
        # when the interpreter goes down in an orderly way. The backstop for
        # everything else is on the game side: it exits when its stdin closes.
        self._finalizer = weakref.finalize(self, _reap, self._proc)
        _LIVE.add(self)

        self._ids = itertools.count(1)
        answer = self.request({"cmd": "hello"})
        self.meta: dict[str, Any] = answer.get("meta") or {}
        if self.meta.get("protocol") != PROTOCOL:
            self.close()
            raise ProtocolError(
                f"{binary} speaks protocol {self.meta.get('protocol')}, "
                f"this client speaks {PROTOCOL}"
            )

    # -- The one operation ---------------------------------------------------

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its answer."""
        return self.recv(*self.send(message))

    def send(self, message: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Write one request without waiting for its answer.

        Split from :meth:`request` so that a vector environment can post a step
        to every child before reading any reply, letting the races run at the
        same time instead of one after another. Every send must be matched by
        exactly one :meth:`recv`, in order.
        """
        if self._closed:
            raise EngineDied("the engine is closed")

        request_id = next(self._ids)
        payload = dict(message, id=request_id)
        line = json.dumps(payload, separators=(",", ":")).encode() + b"\n"

        try:
            self._proc.stdin.write(line)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._died(exc) from exc
        return request_id, payload

    def recv(self, request_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        """Read the answer to the request :meth:`send` returned."""
        if self._closed:
            raise EngineDied("the engine is closed")

        raw = self._proc.stdout.readline()
        if not raw:
            raise self._died("no answer")

        try:
            answer = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.close()
            raise ProtocolError(f"unparseable answer: {raw!r}") from exc

        if answer.get("id") != request_id:
            self.close()
            raise ProtocolError(
                f"answer carries id {answer.get('id')}, expected {request_id} - "
                "the request and response streams are out of step"
            )
        if not answer.get("ok"):
            raise CommandFailed(
                answer.get("kind", "unknown"), answer.get("error", ""), payload
            )
        return answer

    def state(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a request that answers with one state, and return that state."""
        return self.request(message)["state"]

    def _died(self, reason: object) -> EngineDied:
        code = self._proc.poll()
        self.close()
        detail = f" (exit status {code})" if code is not None else ""
        return EngineDied(
            f"the environment server stopped responding{detail}: {reason}. "
            "Pass capture_stderr=True, or run the binary by hand, to see why."
        )

    # -- Lifecycle -----------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Shut the child down. Idempotent, and safe to call after a failure."""
        if self._closed:
            return
        self._closed = True
        _LIVE.discard(self)

        if self._proc.poll() is None:
            try:
                self._proc.stdin.write(b'{"id":0,"cmd":"close"}\n')
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
        self._finalizer()  # runs _reap once, whatever happened above

        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


# Engines still running when the interpreter exits. A WeakSet so an engine that
# is simply garbage collected is finalized the usual way instead.
_LIVE: "weakref.WeakSet[Engine]" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for engine in list(_LIVE):
        engine.close()


def server_args(
    *,
    track: str | None = None,
    laps: int | None = None,
    num_karts: int | None = None,
    difficulty: int | None = None,
    frame_skip: int | None = None,
    lookahead: int | None = None,
    expert: bool = False,
    include_karts: bool = True,
    render: bool = False,
    human: bool = False,
    fullscreen: bool = False,
    screensize: str | None = None,
    race_now: bool = False,
    extra: Iterable[str] = (),
) -> list[str]:
    """Build the command line for :class:`Engine` from environment options.

    Anything left as None keeps the game's own default, which the handshake then
    reports back, so there is exactly one source of truth for each knob.

    Rendering is a flag rather than a separate binary: the game already decides
    at run time whether it has a window, so watching a policy play means leaving
    ``--no-graphics`` off. ``human`` puts a person at the wheel (``--gym-human``:
    the game keeps its own clock and input, the protocol only reports) and
    implies a window; ``fullscreen``, ``screensize`` and ``race_now`` (skip the
    countdown) are the game's own flags, and ``extra`` passes any others.
    """
    args: list[str] = ["--gym-human" if human else "--gym"]
    if not render and not human:
        args.append("--no-graphics")
    for flag, value in (
        ("--track", track),
        ("--laps", laps),
        ("--numkarts", num_karts),
        ("--difficulty", difficulty),
        ("--gym-frame-skip", frame_skip),
        ("--gym-lookahead", lookahead),
    ):
        if value is not None:
            args.append(f"{flag}={value}")
    if expert:
        args.append("--gym-expert")
    if not include_karts:
        args.append("--gym-no-karts")
    if fullscreen:
        args.append("--fullscreen")
    if screensize:
        args.append(f"--screensize={screensize}")
    if race_now:
        args.append("--race-now")
    args.extend(extra)
    return args
