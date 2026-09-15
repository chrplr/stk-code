#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""How long a step with a frame takes, and whether it fits a screen refresh.

A harness that shows the frames to a person - fmri-gym - steps once per screen
refresh, so a step (physics + render + read-back + transfer) has to fit in one
refresh period with room to spare. This prints the distribution, first for a
step that brings the frame back and then for the same step without it, so
that the cost of the picture is visible on its own. Run it on the machine the
experiment will run on; the numbers are the machine's, not the code's.

    python examples/measure_frames.py --screensize 640x360 --fps 60 --steps 600
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from stk_gym import StkEnv

# Hold up and a little left: a corner of the track gets rendered, not one wall.
KEYS_UP_LEFT = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.int8)


def measure(env: StkEnv, steps: int) -> np.ndarray:
    env.reset(seed=0)
    costs = np.empty(steps)
    for i in range(steps):
        t0 = time.perf_counter()
        env.step(KEYS_UP_LEFT)
        costs[i] = time.perf_counter() - t0
    return costs * 1000.0


def report(label: str, ms: np.ndarray, budget_ms: float) -> None:
    over = float(np.mean(ms > budget_ms) * 100.0)
    print(
        f"{label:<14} n={ms.size}  mean {ms.mean():5.2f}  p50 {np.median(ms):5.2f}  "
        f"p95 {np.percentile(ms, 95):5.2f}  max {ms.max():5.2f} ms   "
        f"over {budget_ms:.1f} ms: {over:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="hacienda")
    parser.add_argument("--num-karts", type=int, default=4)
    parser.add_argument("--screensize", default="640x360")
    parser.add_argument("--fps", type=int, default=60, help="steps per second the harness wants")
    parser.add_argument("--physics-fps", type=int, default=120, help="the game's tick rate")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    budget_ms = 1000.0 / args.fps
    common = dict(
        track=args.track, num_karts=args.num_karts, laps=3, action_mode="keys",
        screensize=args.screensize, frame_skip=max(1, round(args.physics_fps / args.fps)),
    )
    with StkEnv(render_mode="rgb_array", **common) as env:
        meta = env.engine.meta
        print(
            f"frame {meta['frame_width']}x{meta['frame_height']}, {args.num_karts} karts on "
            f"{args.track}, physics {meta['physics_fps']} Hz, {meta['frames']} ticks per "
            f"step ({1.0 / meta['step_dt']:.0f} steps/s), budget {budget_ms:.1f} ms"
        )
        report("with frame", measure(env, args.steps), budget_ms)
    with StkEnv(render_mode=None, hidden=True, **common) as env:
        report("no frame", measure(env, args.steps), budget_ms)


if __name__ == "__main__":
    main()
