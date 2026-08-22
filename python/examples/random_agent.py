#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Floor and ceiling: what to beat, and what to aim at.

Run this before training anything. It prints three numbers and a step rate, and
it doubles as a wiring check: if the expert stops finishing, or a straight line
stops crashing, the observation or the action mapping has broken in a way the
unit tests cannot see.

    python examples/random_agent.py --track hacienda --laps 1
"""

from __future__ import annotations

import argparse
import time

from stk_gym import StkEnv, run_expert, run_random, run_straight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="hacienda")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--karts", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=None,
                        help="0 Beginner to 3 SuperTux; also sets how hard "
                             "the reference AI drives, so quote it with any "
                             "lap time")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    common = dict(track=args.track, laps=args.laps, num_karts=args.karts,
                  difficulty=args.difficulty)
    runs = (
        ("random", run_random, {}),
        ("straight", run_straight, {}),
        ("expert (SkiddingAI)", run_expert, {"expert": True}),
    )

    print(f"{args.track}, {args.laps} lap(s), {args.karts} kart(s), "
          f"{args.episodes} episodes each\n")
    print(f"{'policy':<22}{'progress m':>12}{'lap':>5}{'finished':>10}"
          f"{'race s':>9}{'off-road':>10}{'reward':>10}")
    print("-" * 78)

    total_steps = 0
    started = time.time()
    for name, run, extra in runs:
        with StkEnv(**common, **extra) as env:
            results = [
                run(env, seed=i, max_steps=args.max_steps)
                for i in range(args.episodes)
            ]
        total_steps += sum(r.steps for r in results)
        mean = lambda f: sum(f(r) for r in results) / len(results)  # noqa: E731
        print(
            f"{name:<22}{mean(lambda r: r.progress_m):>12.1f}"
            f"{mean(lambda r: r.laps):>5.1f}"
            f"{mean(lambda r: float(r.finished)):>10.0%}"
            f"{mean(lambda r: r.race_time):>9.1f}"
            f"{mean(lambda r: r.off_road_fraction):>10.0%}"
            f"{mean(lambda r: r.reward):>10.1f}"
        )

    elapsed = time.time() - started
    print(f"\n{total_steps} steps in {elapsed:.1f}s "
          f"({total_steps / elapsed:.0f} steps/s, one environment, "
          "including the time to load a track three times)")
    print("\nA learned policy that does not beat 'straight' has learned nothing;")
    print("one that approaches 'expert' has learned to drive.")


if __name__ == "__main__":
    main()
