#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Replay a logged block and say how far the race departs from the log.

fmri-gym logs one .npz per game block - the 8-vector of keys held at every
frame, the seed of every episode, and the race state sampled every frame -
next to a manifest.json that holds the phase's settings. This starts a fresh
game with those settings, plays the keys back through the same environment,
and prints the largest difference in position along the track and in space.
Millimetres are a replay; metres are a bug (or a different binary).

    python examples/replay_log.py <session dir>/block-01_stk_gym_supertuxkart.npz

Episodes are replayed in the order they were played, from process start,
which is what makes a run reproduce (see README, "Reproducibility").
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from stk_gym import StkEnv


def load_spec(npz_path: str) -> dict:
    """The curriculum entry that produced this block, from the manifest beside it."""
    manifest = json.load(open(os.path.join(os.path.dirname(npz_path), "manifest.json")))
    name = os.path.basename(npz_path)
    for phase in manifest["phases"]:
        if phase.get("data_file") == name:
            return manifest["curriculum"][phase["index"]]
    raise SystemExit(f"{name} is not a block of {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz")
    parser.add_argument("--binary", default=None, help="the game binary (default: found)")
    args = parser.parse_args()

    log = np.load(args.npz, allow_pickle=True)
    spec = load_spec(args.npz)
    actions, episode_id = log["actions"], log["episode_id"]
    logged = np.stack([log[k] for k in ("distance_down_track", "x", "y", "z")], axis=1)
    replayed = np.full_like(logged, np.nan)

    env = StkEnv(
        track=spec.get("track", "hacienda"), laps=spec.get("laps"),
        num_karts=spec.get("num_karts"), difficulty=spec.get("difficulty"),
        action_mode="keys", render_mode="rgb_array",
        frame_skip=int(spec.get("frame_skip", 2)), screensize=spec.get("screensize", "640x360"),
        seed=spec.get("seed"), binary=args.binary,
    )
    with env:
        for episode, seed in enumerate(log["episode_seeds"]):
            env.reset(seed=int(seed))
            for i in np.flatnonzero(episode_id == episode):
                env.step(actions[i])
                s = env.sample()
                replayed[i] = (s["distance_down_track"], s["x"], s["y"], s["z"])

    diff = np.abs(replayed - logged)
    along = np.nanmax(diff[:, 0])
    space = np.nanmax(np.linalg.norm(replayed[:, 1:] - logged[:, 1:], axis=1))
    print(
        f"{len(actions)} frames, {len(log['episode_seeds'])} episode(s): largest departure "
        f"{along * 1000:.1f} mm along the track, {space * 1000:.1f} mm in space; "
        f"the log ends at {logged[-1, 0]:.1f} m along the track, the replay at "
        f"{replayed[-1, 0]:.1f} m"
    )


if __name__ == "__main__":
    main()
