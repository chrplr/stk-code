#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regenerate tools/gym/asset-manifest.txt by watching the game load.

The gym needs a fraction of stk-assets: 1.5 GB of tracks and karts, of which a
race touches under a tenth. Rather than guess which files that is -- a guess
fails silently, because a missing texture falls back to a flat colour instead of
raising -- this runs a real race per track under strace and keeps every asset
the game actually opened.

Two things make a naive trace wrong, and both are handled here:

* The texture cache. SuperTuxKart caches decoded textures in
  ~/.cache/supertuxkart, and a warm cache serves them without ever opening the
  originals: tracing warm found 518 files where cold found 855. A pack built
  from a warm trace works on the machine that built it and breaks on every
  machine that starts cold. Each race below therefore runs with the cache
  cleared.

* Coverage. A trace only records what that particular race exercised. One kart
  for sixty steps misses the other karts' models and most of the track, so the
  race is deliberately busy: four karts, three hundred steps, pixels on, which
  forces the render path to load the textures the RL path never asks for.

Verify the result with tools/gym/verify_pack.py rather than by reading the log:
the failure this guards against does not log anything.

Usage:

    tools/gym/trace_assets.py --assets ../stk-assets [--out tools/gym/asset-manifest.txt]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# The tracks the pack ships. Varied on purpose: the texture sets barely overlap,
# so each one costs about 18 MB compressed and a set of look-alike tracks would
# buy nothing.
TRACKS = ["hacienda", "cornfield_crossing", "snowmountain", "lighthouse", "scotland"]

# A race busy enough to pull in what a quiet one misses. See the docstring.
RACE = """
import sys, numpy as np
from stk_gym import StkEnv
env = StkEnv(num_karts=4, laps=1, track=sys.argv[1], obs_mode="pixels")
obs, info = env.reset(seed=0)
rng = np.random.default_rng(0)
n = getattr(env.action_space, "n", None)
for _ in range(300):
    a = int(rng.integers(n)) if n else env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    if term or trunc:
        break
env.close()
"""


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "supertuxkart"


def trace_one(track: str, race_py: Path, assets: Path, tmp: Path) -> set[str]:
    """Race `track` under strace and return the stk-assets paths it opened."""
    out = tmp / f"trace_{track}.txt"
    # Cold cache, or the trace silently under-reports by about 40%.
    shutil.rmtree(cache_dir(), ignore_errors=True)
    subprocess.run(
        ["strace", "-f", "-qq", "-e", "trace=openat", "-o", str(out),
         sys.executable, str(race_py), track],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    root = str(assets.resolve())
    used: set[str] = set()
    for line in out.read_text(errors="replace").splitlines():
        # A failed open says nothing about what the pack needs.
        if "ENOENT" in line or "EACCES" in line:
            continue
        for field in line.split('"')[1::2]:
            if field.startswith(root + "/"):
                rel = field[len(root) + 1:]
                if (assets / rel).is_file():
                    used.add(rel)
    return used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assets", required=True, type=Path,
                    help="the stk-assets checkout to trace against")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("asset-manifest.txt"))
    ap.add_argument("--tracks", nargs="+", default=TRACKS)
    args = ap.parse_args()

    if shutil.which("strace") is None:
        print("strace is required: apt install strace", file=sys.stderr)
        return 1
    if not args.assets.is_dir():
        print(f"no such assets directory: {args.assets}", file=sys.stderr)
        return 1

    # The game resolves assets relative to this, so the trace sees the paths of
    # the checkout being packed rather than whatever the package would find.
    os.environ["SUPERTUXKART_ASSETS_DIR"] = str(args.assets.resolve())

    revision = "unknown"
    # check=False: a checkout without svn, or no svn installed, just means
    # the revision goes unrecorded -- not a reason to refuse to write it.
    info = subprocess.run(["svn", "info", str(args.assets)],
                          capture_output=True, text=True, check=False)
    for line in info.stdout.splitlines():
        if line.startswith("Revision:"):
            revision = line.split(":", 1)[1].strip()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        race_py = tmp / "race.py"
        race_py.write_text(RACE)
        used: set[str] = set()
        for track in args.tracks:
            found = trace_one(track, race_py, args.assets, tmp)
            print(f"{track:<22} {len(found):>5} files", file=sys.stderr)
            used |= found

    header = [
        "# Assets the gym needs, one path relative to the stk-assets root.",
        "#",
        "# Generated by tools/gym/trace_assets.py: every file the game actually",
        "# opened while racing each track below, with a cold texture cache. A warm",
        "# ~/.cache/supertuxkart hides ~40% of these reads, so the trace that",
        "# produced this list cleared it first. Do not edit by hand.",
        "#",
        f"# stk-assets revision: {revision}",
        f"# tracks: {' '.join(args.tracks)}",
    ]
    args.out.write_text("\n".join(header + sorted(used)) + "\n")
    print(f"{args.out}: {len(used)} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
