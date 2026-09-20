#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check an asset pack is as complete as the full stk-assets.

A pack missing a texture does not crash: the material falls back and the race
runs to completion with an exit status of 0. The first pack built for this tool
passed its test suite while drawing an untextured white landscape. So "it ran"
proves nothing, and two independent checks are run instead.

**What the game says it could not find.** Every texture it fails to resolve is
named on its log, and a complete pack produces not one such line -- the full
asset set scores exactly zero, which makes this a check with no noise floor to
argue about. It is the sharper of the two: the second pack built here rendered
close enough to pass the frame comparison below while quietly missing 284
textures, and only the log said so.

**What it actually drew.** The log cannot catch a file that resolves to the
wrong content, or an asset that is found but never reached, so frames are
compared too.

Two details make the frame comparison mean something:

* Compare the frame at reset, not during a race. Two runs of the same race with
  the same seed and the same actions diverge -- 98% of pixels differ by the
  300th step, with the full asset set on both sides -- so a mid-race diff
  measures the physics, not the assets.

* Run the control. Even at reset the renderer is not bit-exact between runs
  (about 1% of pixels, from kart selection and particle timing), so the full
  set is rendered twice and the pack is judged against that noise floor rather
  than against zero.

Both sides run with a cold texture cache, since a warm one would serve textures
the pack does not contain.

Usage:

    tools/gym/verify_pack.py --pack /tmp/pack --assets ../stk-assets
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# This tool checks a pack that is already on disk; it must never fetch one.
# find_binary's last resort is a 255 MB download, and a verification step that
# quietly installs the artefact it is verifying is worse than no check at all.
os.environ.setdefault("STK_ENV_OFFLINE", "1")

import numpy as np

FRAME = """
import sys, numpy as np
from stk_gym import StkEnv
env = StkEnv(num_karts=4, laps=1, track=sys.argv[1], obs_mode="pixels")
obs, info = env.reset(seed=0)
env.close()
np.save(sys.argv[2], np.asarray(obs))
"""

TRACKS = ["hacienda", "cornfield_crossing", "snowmountain", "lighthouse", "scotland"]

# How much worse than the run-to-run noise a pack may be before it is called
# broken. The gap between "fine" and "untextured" is not subtle -- the bad pack
# scored 90x its control -- so this does not need to be tight.
TOLERANCE = 4.0

# A control this bad is not a noise floor, it is a broken measurement: the
# reference rendered two different things. Observed once in a hundred-odd runs,
# a lighthouse control of 161 against a usual 0.3-1.4, and it silently made
# that track's verdict meaningless -- anything passes when the bar is that low.
# Such a row is retried, then reported as inconclusive rather than as a pass.
MAX_CONTROL_MEAN = 10.0


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "supertuxkart"


class RaceFailed(RuntimeError):
    """The game could not be started or could not reach the first frame."""


def frame(script: Path, track: str, out: Path, env: dict[str, str]) -> np.ndarray:
    """Render the reset frame, cold, or say why it could not be rendered.

    A pack can be broken in two ways, and both have to be reported rather than
    raised through: missing a file the game merely draws with (which renders
    wrongly and is caught by the diff) or missing one it refuses to start
    without (which lands here).
    """
    shutil.rmtree(cache_dir(), ignore_errors=True)
    # An empty value means "unset": a stale SUPERTUXKART_DATADIR inherited from
    # the caller's shell would silently redirect the run being measured.
    child = {**os.environ, **env}
    child = {k: v for k, v in child.items() if v != ""}
    # check=False: a failed race is a result to report, not an exception.
    done = subprocess.run([sys.executable, str(script), track, str(out)],
                          env=child, check=False,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if done.returncode != 0:
        tail = done.stderr.decode(errors="replace").strip().splitlines()
        raise RaceFailed(tail[-1] if tail else f"exit status {done.returncode}")
    return np.load(out).astype(np.int16)


# What the game prints when it cannot resolve an asset. A complete pack prints
# none of these, so any is a defect; there is no noise floor here.
_MISSING = ("Cannot determine texture full path", "Failed to load")


def missing_assets(binary: Path, cwd: Path, track: str,
                   env: dict[str, str]) -> list[str]:
    """Names the game says it could not find, racing `track` from `cwd`.

    Run by hand rather than through the gym, because the engine sends the
    child's stderr to /dev/null -- a pipe nobody drains deadlocks while a track
    loads -- and this is precisely the output that matters here.
    """
    shutil.rmtree(cache_dir(), ignore_errors=True)
    child = {k: v for k, v in {**os.environ, **env}.items() if v != ""}
    done = subprocess.run(
        [str(binary), "--gym", "--no-graphics", f"--track={track}"],
        input=b'{"cmd":"hello"}\n', cwd=str(cwd), env=child, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
    )
    log = done.stdout.decode(errors="replace")
    names = set()
    for line in log.splitlines():
        for marker in _MISSING:
            if marker in line:
                names.add(line.split(marker, 1)[1].strip(" :.").split()[0])
    return sorted(names)


def reference_game(explicit: Path | None = None) -> tuple[Path, Path]:
    """The checkout's own build, and the directory to run it from.

    Deliberately not find_binary(): its last resort is to download the release
    pack, which is the very thing being verified, and in CI -- package
    installed into site-packages, no checkout above it -- that is exactly the
    branch it takes. It then fails on a release that this run is meant to
    produce. The reference is always the build in the checkout this script
    lives in.
    """
    if explicit is not None:
        return explicit, explicit.parent
    repo = Path(__file__).resolve().parents[2]
    for candidate in (repo / "build" / "bin" / "supertuxkart",
                      repo / "build" / "supertuxkart",
                      repo / "cmake_build" / "bin" / "supertuxkart"):
        if candidate.exists():
            # Run from the checkout root, where ./data/ and ../../stk-assets
            # resolve the way the game expects.
            return candidate, repo
    raise SystemExit(
        f"no built supertuxkart in {repo}; build it, or pass "
        "--reference-binary"
    )


def diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a - b)
    px = 100.0 * (d.max(axis=2) > 0).sum() / (d.shape[0] * d.shape[1])
    return px, float(d.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--assets", required=True, type=Path,
                    help="the full stk-assets checkout, as the reference")
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    ap.add_argument("--reference-binary", type=Path, default=None,
                    help="the binary to render the reference with "
                         "(default: this checkout's build)")
    ap.add_argument("--no-frames", action="store_true",
                    help="skip the frame comparison, which needs a GL context. "
                         "Leaves the completeness check, which does not, and is "
                         "what CI runs.")
    args = ap.parse_args()

    # The reference: the checkout's own binary against the full asset tree.
    full = {
        "SUPERTUXKART_ASSETS_DIR": str(args.assets.resolve()),
        "SUPERTUXKART_DATADIR": "",
        "STK_ENV_BIN": "",
    }
    # The pack: nothing but a path to its binary, which is how a user gets it.
    # Leaving SUPERTUXKART_* unset on purpose -- the pack is meant to resolve
    # its own data and assets from its layout, and setting them here would test
    # a configuration nobody runs.
    pack = {
        "STK_ENV_BIN": str((args.pack / "supertuxkart").resolve()),
        "SUPERTUXKART_DATADIR": "",
        "SUPERTUXKART_ASSETS_DIR": "",
    }

    # The completeness check first: it is sharper, needs no GL context, and a
    # pack that fails it is not worth rendering.
    print("Assets the game cannot find (a complete pack: none)\n")
    print(f"{'track':<22}{'full':>8}{'pack':>8}   verdict")
    incomplete = {}
    ref_binary, ref_cwd = reference_game(args.reference_binary)
    for track in args.tracks:
        # The control is measured, not assumed: if the reference tree is itself
        # incomplete, the pack should not be blamed for matching it.
        ref = missing_assets(ref_binary, ref_cwd, track, full)
        got = missing_assets((args.pack / "supertuxkart").resolve(),
                             (args.pack / "stk").resolve(), track, pack)
        if got:
            incomplete[track] = got
        print(f"{track:<22}{len(ref):>8}{len(got):>8}   "
              f"{'OK' if not got else 'INCOMPLETE'}")
    if incomplete:
        print(file=sys.stderr)
        for track, names in incomplete.items():
            shown = ", ".join(names[:8]) + ("..." if len(names) > 8 else "")
            print(f"  {track}: {len(names)} missing -- {shown}", file=sys.stderr)
        print("\nregenerate the manifest with tools/gym/trace_assets.py",
              file=sys.stderr)
        return 1

    if args.no_frames:
        print("\nframes not compared (--no-frames)")
        return 0

    print("\nFrames against the full asset set\n")
    print(f"{'track':<22}{'control':>18}{'pack':>18}   verdict")
    bad = []
    inconclusive = []
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "frame.py"
        script.write_text(FRAME)
        for track in args.tracks:
            try:
                f1 = frame(script, track, Path(td) / "f1.npy", full)
                f2 = frame(script, track, Path(td) / "f2.npy", full)
            except RaceFailed as exc:
                print(f"{track:<22}{'reference race failed':>37}   SKIPPED")
                print(f"  the full asset set could not race it: {exc}",
                      file=sys.stderr)
                continue
            try:
                p = frame(script, track, Path(td) / "p.npy", pack)
            except RaceFailed as exc:
                bad.append(track)
                print(f"{track:<22}{'pack race failed':>37}   BROKEN")
                print(f"  {exc}", file=sys.stderr)
                continue
            c_px, c_mean = diff(f1, f2)
            if c_mean > MAX_CONTROL_MEAN:
                # Once, before calling the measurement unusable.
                f2 = frame(script, track, Path(td) / "f2.npy", full)
                c_px, c_mean = diff(f1, f2)
            p_px, p_mean = diff(f1, p)
            if c_mean > MAX_CONTROL_MEAN:
                inconclusive.append(track)
                print(f"{track:<22}{c_px:7.2f}% {c_mean:8.3f}"
                      f"{p_px:7.2f}% {p_mean:8.3f}   INCONCLUSIVE")
                continue
            # A pack cannot be better than the noise floor, so allow an
            # absolute slack as well for the tracks whose control is near zero.
            ok = p_mean <= max(c_mean * TOLERANCE, c_mean + 2.0)
            if not ok:
                bad.append(track)
            print(f"{track:<22}{c_px:7.2f}% {c_mean:8.3f}"
                  f"{p_px:7.2f}% {p_mean:8.3f}   {'OK' if ok else 'DIFFERS'}")

    if bad:
        print(f"\n{len(bad)} track(s) render differently: {', '.join(bad)}",
              file=sys.stderr)
        print("the pack is missing assets those tracks load; re-run "
              "tools/gym/trace_assets.py", file=sys.stderr)
        return 1
    if inconclusive:
        print(f"\n{len(inconclusive)} track(s) could not be judged: "
              f"{', '.join(inconclusive)}", file=sys.stderr)
        print("the reference disagreed with itself, so the pack was not "
              "compared against anything; run it again", file=sys.stderr)
        return 1
    print("\npack renders within run-to-run noise of the full asset set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
