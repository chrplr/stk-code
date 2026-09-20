#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check an asset pack renders what the full stk-assets renders.

A pack missing a texture does not crash and does not warn: the material falls
back to a flat colour and the race runs to completion with an exit status of 0.
The first pack built for this tool passed its test suite while drawing an
untextured white landscape. So "it ran" proves nothing, and this compares
pixels instead.

Two details make the comparison mean something:

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

    print(f"{'track':<22}{'control':>18}{'pack':>18}   verdict")
    bad = []
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
            p_px, p_mean = diff(f1, p)
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
    print("\npack renders within run-to-run noise of the full asset set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
