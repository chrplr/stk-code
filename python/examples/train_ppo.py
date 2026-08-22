#!/usr/bin/env python3
# Copyright (c) 2026 SuperTuxKart-Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Train a kart to drive with PPO, then measure it against the baselines.

    pip install -e "python[rl]"
    python examples/train_ppo.py --steps 300000 --save kart.zip

Training uses several racing processes at once; evaluation uses one, so the
numbers line up with examples/random_agent.py.
"""

from __future__ import annotations

import argparse
import time

from stk_gym import StkEnv, run_expert, run_straight
from stk_gym.sb3 import make_sb3_vec_env


def evaluate(model, episodes: int, max_steps: int, **env_kwargs):
    from stk_gym.baselines import run_episode

    with StkEnv(**env_kwargs) as env:
        def policy(obs, _info):
            action, _state = model.predict(obs, deterministic=True)
            return action

        return [
            run_episode(env, policy, seed=i, max_steps=max_steps)
            for i in range(episodes)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--track", default="hacienda")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument("--karts", type=int, default=1)
    parser.add_argument("--difficulty", type=int, default=None,
                        help="0 Beginner to 3 SuperTux; also sets how hard "
                             "the reference AI drives, so quote it with any "
                             "lap time")
    parser.add_argument("--max-episode-steps", type=int, default=3000)
    parser.add_argument("--save", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    env_kwargs = dict(track=args.track, laps=args.laps, num_karts=args.karts,
                      difficulty=args.difficulty)
    venv = VecMonitor(
        make_sb3_vec_env(
            num_envs=args.envs,
            max_episode_steps=args.max_episode_steps,
            **env_kwargs,
        )
    )
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        # A racing kart is driven on the CPU: the network is tiny and the
        # simulation is the expensive part.
        device="cpu",
        verbose=1,
    )
    started = time.time()
    model.learn(total_timesteps=args.steps)
    elapsed = time.time() - started
    print(f"\n{args.steps} steps in {elapsed / 60:.1f} min "
          f"({args.steps / elapsed:.0f} steps/s, {args.envs} environments)")
    venv.close()

    if args.save:
        model.save(args.save)
        print(f"saved to {args.save}")

    print("\nEvaluating against the baselines")
    print(f"{'policy':<22}{'progress m':>12}{'finished':>10}{'reward':>10}")
    print("-" * 54)

    def report(name, results):
        mean = lambda f: sum(f(r) for r in results) / len(results)  # noqa: E731
        print(f"{name:<22}{mean(lambda r: r.progress_m):>12.1f}"
              f"{mean(lambda r: float(r.finished)):>10.0%}"
              f"{mean(lambda r: r.reward):>10.1f}")

    with StkEnv(**env_kwargs) as env:
        report("straight", [run_straight(env, seed=i, max_steps=args.max_episode_steps)
                            for i in range(args.episodes)])
    report("ppo", evaluate(model, args.episodes, args.max_episode_steps, **env_kwargs))
    with StkEnv(expert=True, **env_kwargs) as env:
        report("expert (SkiddingAI)",
               [run_expert(env, seed=i, max_steps=args.max_episode_steps)
                for i in range(args.episodes)])


if __name__ == "__main__":
    main()
