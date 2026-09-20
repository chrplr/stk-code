# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Finding the ``supertuxkart`` binary that serves the race.

The game is the single source of truth for the physics, so this package is
useless without it. Unlike a Go server, it cannot be built on demand: a
SuperTuxKart build takes minutes, not seconds, and silently starting one from
inside ``gymnasium.make`` would be a very unpleasant surprise. So a checkout is
used when there is one, and otherwise the prebuilt pack for this package
version is fetched once from its GitHub release.

The binary also needs its assets. It resolves them relative to the working
directory, so :func:`default_cwd` reports the directory to run it in -- the
checkout, or the pack, which is laid out to look like one.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
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

# Where a release lives, and what the pack is called on each platform the
# gym-release workflow builds for. The package version is the tag it fetches
# from, so a wheel and its engine cannot drift apart.
#
# Linux x86_64 only for now. The game is C++ and every other target needs a
# native runner rather than a cross-compile, so the table grows a row at a time
# as the workflow grows a job; a machine with no row gets told so.
_RELEASES = "https://github.com/chrplr/stk-code/releases/download"
_PACKS: dict[tuple[str, str], str] = {
    ("linux", "x86_64"): "SuperTuxKart-gym-linux-x86_64.tar.xz",
    ("linux", "amd64"): "SuperTuxKart-gym-linux-x86_64.tar.xz",
}
# Set to a non-empty value to never touch the network (an offline machine, or
# a test that must fail loudly rather than fetch 160 MB).
_OFFLINE_VAR = "STK_ENV_OFFLINE"
# What a downloaded pack contains, relative to its root. See make_asset_pack.sh:
# the nesting is what lets the game resolve its assets with no environment
# variables, and checking for it is how default_cwd recognises a pack.
_PACK_DATA = Path("stk") / "data"
_PACK_ASSETS = Path("stk-assets")


class BinaryNotFound(RuntimeError):
    """The game binary could not be located."""


def find_binary(explicit: str | os.PathLike[str] | None = None) -> str:
    """Return the path to the SuperTuxKart binary.

    Tried in order: an explicit path, ``$STK_ENV_BIN``, the build directory of
    the checkout this package lives in, the ``PATH``, and finally the prebuilt
    pack for this package's version, fetched once into the user cache.
    ``$STK_ENV_OFFLINE`` forbids that last step.

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

    return _download()


def find_repo() -> Path | None:
    """Walk up from this file looking for the stk-code checkout it lives in."""
    for parent in Path(__file__).resolve().parents:
        if (parent / _MARKER).is_file():
            return parent
    return None


def find_pack(binary: str | os.PathLike[str]) -> Path | None:
    """The downloaded pack \\p binary belongs to, if it belongs to one."""
    for parent in Path(binary).resolve().parents:
        if (parent / _PACK_DATA).is_dir() and (parent / _PACK_ASSETS).is_dir():
            return parent
    return None


def default_cwd(binary: str | os.PathLike[str]) -> str | None:
    """Return the directory to run \\p binary from, or None to inherit.

    SuperTuxKart looks for its assets relative to the working directory - a
    checkout keeps them in a sibling ``stk-assets`` - so a child started
    anywhere else dies with "Set $SUPERTUXKART_DATADIR". An installed binary
    knows its own prefix and needs nothing.

    A downloaded pack is deliberately shaped like a checkout, so it is served
    the same way: its ``stk`` directory holds ``data`` and sits beside
    ``stk-assets``, and the game resolves both from there unaided.

    A pack the binary sits in wins over any checkout that happens to enclose
    it. Unpacking a pack inside a checkout is an ordinary thing to do -- it is
    what the release workflow does -- and serving it the checkout's working
    directory sends the game looking for assets beside the checkout, where
    there are none. Whichever is nearer the binary is the one that owns it,
    and that is always the pack.
    """
    if os.environ.get("SUPERTUXKART_DATADIR"):
        return None
    pack = find_pack(binary)
    if pack is not None:
        return str(pack / "stk")
    for parent in Path(binary).resolve().parents:
        if (parent / _MARKER).is_file():
            return str(parent)
    return str(find_repo()) if find_repo() is not None else None


# ── Release download ─────────────────────────────────────────────────────────


def _release_tag() -> str:
    from . import __version__

    return "gym-v" + __version__


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "supertuxkart-gym"


def _pack_name() -> str | None:
    """The release pack for this machine, or None when none is built."""
    return _PACKS.get((sys.platform, platform.machine().lower()))


def _download(tag: str | None = None, cache: Path | None = None) -> str:
    """Fetch this version's engine and assets from its GitHub release, once.

    The pack is 160 MB, so this is not something to do by accident: it happens
    only when every local source has been ruled out, it says what it is doing,
    and it never happens twice. The archive is checked against the release's
    SHA256SUMS before anything is unpacked, and the result lands in the user
    cache under the tag, so two package versions never share an engine.
    """
    tag = tag or _release_tag()
    cache = cache or _cache_dir()
    dest = cache / tag
    binary = dest / _BINARY_NAME
    if binary.exists():
        return str(binary)

    manual = (
        f"Alternatively, download the pack from "
        f"https://github.com/chrplr/stk-code/releases/tag/{tag}, unpack it, and "
        f"point ${_ENV_VAR} at its {_BINARY_NAME}; or build the game from a "
        "stk-code checkout with:\n" + _BUILD_HINT
    )
    if os.environ.get(_OFFLINE_VAR):
        raise BinaryNotFound(
            f"cannot find the supertuxkart binary, and ${_OFFLINE_VAR} is set, "
            f"so it was not fetched from the {tag} release.\n{manual}"
        )
    pack = _pack_name()
    if pack is None:
        raise BinaryNotFound(
            f"cannot find the supertuxkart binary, and the {tag} release has no "
            f"prebuilt pack for {sys.platform}/{platform.machine()}.\n{manual}"
        )

    base = f"{_RELEASES}/{tag}"
    print(
        f"supertuxkart-gym: fetching {pack} (about 160 MB) from the {tag} "
        "release. This happens once.",
        file=sys.stderr,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / pack
        try:
            sums = _fetch(f"{base}/SHA256SUMS").decode("utf-8", "replace")
            got = _fetch_to(f"{base}/{pack}", archive)
        except urllib.error.URLError as exc:
            raise BinaryNotFound(
                f"could not fetch {pack} from {base}: {exc}\n{manual}"
            ) from exc

        want = _expected_sum(sums, pack)
        if want is None or got != want:
            raise BinaryNotFound(
                f"{pack} from {base} does not match the release's SHA256SUMS "
                f"(got {got}, want {want}); not installing it.\n{manual}"
            )

        staged = Path(tmp) / "pack"
        _extract(archive, staged)
        root = _pack_root(staged)
        (root / _BINARY_NAME).chmod(0o755)
        # A rename within the same directory is atomic: a second process racing
        # for the same pack sees either no directory or a complete one. It can
        # lose the race, which is why an existing destination is not an error.
        try:
            os.replace(root, dest)
        except OSError:
            if not binary.exists():
                raise
    return str(binary)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _fetch_to(url: str, dest: Path) -> str:
    """Stream \\p url into \\p dest and return its SHA256.

    Streamed rather than read into memory: the pack is 160 MB, and hashing it
    on the way past means it is never held twice.
    """
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        tty = sys.stderr.isatty()
        with open(dest, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if tty and total:
                    print(f"\r  {100 * done // total:3d}%  "
                          f"{done >> 20} / {total >> 20} MiB",
                          end="", file=sys.stderr)
        if tty and total:
            print(file=sys.stderr)
    return digest.hexdigest()


def _expected_sum(sums: str, asset: str) -> str | None:
    """The digest SHA256SUMS lists for asset (``<hex>  <name>`` per line)."""
    for line in sums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == asset:
            return parts[0].lower()
    return None


def _extract(archive: Path, into: Path) -> None:
    """Unpack a release pack, refusing members that would escape \\p into."""
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tf:
        # "data" rejects absolute paths, parent traversal, links out of the
        # tree and device nodes. It is the default from Python 3.14; passing it
        # explicitly is what makes the older versions this package supports
        # behave the same way.
        try:
            tf.extractall(into, filter="data")
        except TypeError:  # Python < 3.11.4 has no filter argument
            _extract_checked(tf, into)


def _extract_checked(tf: tarfile.TarFile, into: Path) -> None:
    """extractall for interpreters without the "data" filter."""
    root = into.resolve()
    for member in tf.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise BinaryNotFound(f"refusing link or device in pack: {member.name}")
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise BinaryNotFound(f"refusing path outside pack: {member.name}")
    tf.extractall(into)


def _pack_root(staged: Path) -> Path:
    """The pack directory inside what was just unpacked.

    The tarball holds one top-level directory, but a pack rolled by hand may
    not, so the marker decides rather than the shape.
    """
    if (staged / _PACK_DATA).is_dir():
        return staged
    for child in sorted(staged.iterdir()):
        if child.is_dir() and (child / _PACK_DATA).is_dir():
            return child
    raise BinaryNotFound(
        f"the downloaded pack has no {_PACK_DATA} in it; it is not a gym pack"
    )
