# Letting an AI drive: SuperTuxKart as a reinforcement learning environment

SuperTuxKart can be driven by a program instead of a person. The game runs as a
child process, an agent sends it steering and throttle a few physics ticks at a
time, and the game sends back what the kart can see. On the Python side that is
a [Gymnasium](https://gymnasium.farama.org/) environment, so any standard RL
library can train on it.

This page is a walk-through for someone who has not done this before.
[`python/README.md`](python/README.md) is the reference for the details.

---

## 1. Install

Build the game, then install the Python package:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
pip install -e python
```

Check it works:

```sh
python -c "
import gymnasium, stk_gym
env = gymnasium.make('SuperTuxKart-Easy-v0')
obs, info = env.reset(seed=0)
print('observation', obs.shape, 'action space', env.action_space)
env.close()"
```

Expected output:

```
observation (20,) action space Box([-1.  0.  0.], 1.0, (3,), float32)
```

The first call takes about half a second: the game is loading a track.

## 2. The two words you need

An **episode** is one attempt at the track: the kart starts on the grid and the
episode ends when it crosses the finish line, or when it runs out of its step
budget. A **step** is one decision. Here a step is 6 physics ticks, or 0.05
seconds of race time, so the agent decides 20 times a second.

## 3. What the agent does

The default action is three numbers:

```python
action = [steer, accelerate, brake]   # steer in [-1, 1], the others in [0, 1]
```

`brake` is thresholded at 0.5. There are two other action spaces —
`action_mode="discrete"` for a 15-entry menu, and `action_mode="continuous_full"`
which adds nitro and skidding — and switching between them costs no rebuild,
because the game always accepts the whole control surface and Python decides
which part of it the agent gets.

## 4. What the agent sees

Twenty numbers. Ten describe the kart: how fast it is going, which way its
wheels are turned, whether it is on the road, how far it is from the centre
line, whether it is facing the wrong way, and so on. The other ten are **the
next five points of the racing line, in the kart's own frame of reference** —
x to the right, z ahead. This is roughly what SuperTuxKart's own AI steers by.

That last part is what makes this learnable in minutes rather than days. An
agent that can see where the track goes next only has to learn to point at it.

For racing against opponents rather than against the clock, `obs_mode
="vector_karts"` adds four numbers per rival: where they are relative to you,
how fast, and how far ahead or behind.

## 5. Your first agent

```python
import gymnasium, numpy as np, stk_gym

env = gymnasium.make("SuperTuxKart-Easy-v0")
obs, info = env.reset(seed=0)
for _ in range(600):
    obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0]))
    if terminated or truncated:
        break
print(f"travelled {info['progress_m']:.0f} m of 1167, on the road: {info['on_road']}")
env.close()
```

```
travelled 139 m of 1167, on the road: False
```

Full throttle and no steering gets 139 metres into a 1167 metre lap and then
sits in the scenery. **That is the expected result, not a bug.** It is the floor
everything else is measured against.

## 6. Baselines before training

Run this before training anything:

```sh
python python/examples/random_agent.py
```

Measured on an Intel Core Ultra 7 165H, `hacienda`, one kart, one lap, three
episodes each:

```
policy                  progress m  lap  finished   race s  off-road    reward
------------------------------------------------------------------------------
random                       121.8  0.0        0%    150.0       92%   -1399.3
straight                     138.5  0.0        0%    150.0       96%   -1450.6
expert (SkiddingAI)         1166.8  1.0      100%     95.5        1%    1111.0
```

The **expert** row is SuperTuxKart's own racing AI sitting in the agent's seat,
driven by the same environment. It is the ceiling. A learned policy that does
not beat `straight` has learned nothing; one that approaches `expert` has
learned to drive.

The expert's lap time moves by a few seconds from run to run — it uses items and
nitro, and those draws depend on the seed. The other two rows do not move.

This script is also a wiring check. If the expert ever stops finishing, or a
straight line stops crashing, the observation or the action mapping has broken in
a way the unit tests cannot see.

## 7. The reward

By default a step pays the metres of track covered since the last step, minus
0.05 for the time it took, minus 0.5 if the kart is off the road, plus 50 for
crossing the finish line. So the return is roughly "metres of progress, less a
penalty for dawdling and for driving on the grass".

Two things about it are worth knowing.

**The off-road penalty is deliberately large enough to notice.** It is why
`straight` scores −1450 rather than +139: the kart spends 96% of the episode in
the scenery. Leaving the road already costs speed, so this is a nudge on top of
a real cost, but it does mean early training reward is strongly negative.

**A step covering more than half a lap is paid nothing.** That is a rescue or a
lap-counter crossing, not driving, and paying for it would teach the agent to
trigger it.

Both live in `python/src/stk_gym/env.py`, in a free function, and changing them
needs no rebuild of the game. `reward_scheme="progress_only"` and
`reward_scheme="sparse"` are there to compare against.

## 8. Train

```sh
pip install -e "python[rl]"
python python/examples/train_ppo.py --steps 500000 --envs 8 --save kart.zip
```

Measured on the same machine, `hacienda`, one kart, one lap, Beginner, an 8-way
vector environment: 500,000 steps in **2.0 minutes** (4,100 steps/s end to end).
Evaluating the result over six episodes, each in a fresh process:

| policy | finishes | lap time | 
|---|---|---|
| straight | 0 / 6 | — (139 m of 1167) |
| ppo, 500k steps | 6 / 6 | 80.3 s (identical every episode) |
| expert (SkiddingAI, Beginner) | 6 / 6 | 92.6 s mean, 89.7–95.5 s |

So half a lap of untrained flailing becomes a policy that completes every lap
and, on this track and at this difficulty, laps about twelve seconds faster than
the game's own AI.

The trained policy is exactly repeatable because it is deterministic and never
picks anything up. The reference AI is not: it uses items and nitro, and those
draws depend on the seed, which is why its lap time is quoted as a range.

Training runs eight races at once, one process each. The game is not the
bottleneck — a single environment does about 8,500 steps per second, and an
8-way vector environment about 14,000 — so most of the wall-clock time goes into
the policy network.

### Read that result carefully

Beating `expert` here is less impressive than it sounds. The reference AI's
strength is set by `difficulty`, which defaults to 0, Beginner. On the same
track and the same lap, SkiddingAI laps in:

| difficulty | 0 Beginner | 1 Intermediate | 2 Expert | 3 SuperTux |
|---|---|---|---|---|
| lap time (mean of 6) | 92.6 s | 58.5 s | 50.3 s | 44.2 s |
| range | 89.7–95.5 | 58.5–58.5 | 49.2–51.3 | 43.8–45.8 |

The trained policy's 80.3 s beats Beginner and is nowhere near SuperTux. Train
with `--difficulty` set to the level you want to be compared against, and quote
the level alongside any lap time — otherwise two runs are not comparable.

### One thing that cost me time

The game reports two different distances along the track, and picking the wrong
one produces a reward that looks reasonable and teaches nothing.
`overall_distance` is SuperTuxKart's own progress measure, and it is built on
the *checkline-validated* distance: it only moves when the kart crosses a
checkline, because its job is to defeat shortcuts. Between checklines it is
flat, so a reward built on it pays nothing for most steps and then pays a large
lump. Use `info["progress_m"]`, which is built from the per-tick distance and
the lap counter, and runs smoothly through the start line.

## 9. Watch it drive

```python
import stk_gym
from stable_baselines3 import PPO

model = PPO.load("kart.zip")
env = stk_gym.StkEnv(track="hacienda", laps=1, num_karts=1, render_mode="human")
obs, info = env.reset(seed=0)
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

There is no separate build for this: the game already decides at run time
whether it opens a window, so `render_mode="human"` just leaves `--no-graphics`
off. To watch the reference AI instead, pass `expert=True` and send any action —
the game ignores it and SkiddingAI drives.

Rendering runs at the display's refresh rate, so a race plays at roughly three
times real time rather than at 8,500 steps per second.

## 9b. Let a person drive, and watch from Python

`--gym-human` (or `stk_gym.HumanSession`) is the reverse of everything above:
the game runs as usual — its window, its keyboard and gamepad, its own clock —
and the protocol only reports, once per frame: `state` (which now includes a
`controls` object, the record of what the person did) and `reset`. `step` is
refused. An experiment harness that presents the game to a participant and
samples the trajectory at its own rate needs exactly this and nothing more;
see `python/README.md`, "A person at the wheel".

## 9c. Get the picture, and send the keys

The other way to put a person in front of the game is to make the *environment*
show them the game: `render_mode="rgb_array"` brings the frame the game drew
back with every `reset` and `step`, and `action_mode="keys"` lets the client
send the keys being held — `{"keys": {"left": true, "up": true}}` — to
SuperTuxKart's own player controller, so that steering ramps and skids latch
exactly as they do for a keyboard. A harness then draws the frames in its own
window, reads its own keyboard, and the person and an agent are in front of
the same environment object, replayable from the seeds and the key vectors.
The game renders in a window that is created hidden (`--gym-hidden`), at the
`--screensize` asked for; the frame follows the JSON line as raw bytes. See
`python/README.md`, "Frames", for the details and the measured cost.

## 10. Talk to the game yourself

The protocol is plain text and meant to be typed:

```sh
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"reset","seed":1}\n{"id":3,"cmd":"step","action":{"steer":0,"accel":1}}\n{"id":4,"cmd":"close"}\n' \
  | ./build/bin/supertuxkart --gym --no-graphics --track=hacienda 2>/dev/null
```

Note the `=`. SuperTuxKart's own command line requires it: `--track hacienda`
with a space is parsed as two unrecognised arguments and the game quietly starts
its default track instead. (`stk_gym` always passes `--track=...`, so this only
bites when driving the binary by hand.)

One JSON object per line in each direction, paired by `id`. `hello` describes
the race and every dimension the client needs, so Python never hard-codes a
size. `reset` restarts, `step` drives, `state` looks without moving, `close`
quits — and so does closing stdin, which is how the game exits when its parent
dies.

**The server reports facts and never a reward.** The reward scheme, the
termination rule and the observation encoding all live in Python. That is the
one design rule worth remembering: to change how the agent is rewarded or what
it sees, you edit Python, and the game is untouched.

One exception to one-object-per-line: a request with `"frame": true` (needs a
window, so `--gym-hidden` rather than `--no-graphics`) is answered by a line
that announces `"frame": {"width", "height", "format": "rgb8"}` and is followed
by that many raw bytes. Pixels do not fit a text protocol, and encoding them as
text would cost more than drawing them.

Everything the game prints — its own logs, irrlicht's, the sound system's — goes
to stderr, because `--gym` hands stdout to the protocol and points file
descriptor 1 at stderr. That is why `2>/dev/null` above is safe.

## 11. Things that are not bugs

- **A straight line crashes after 139 metres.** Yes. See §5.
- **Different seeds give nearly identical episodes.** The grid, the track and
  the driveline are fixed, so the seed only changes the game's random draws,
  such as what an item box contains. A policy that just steers sees no variation
  at all; the built-in AI, which uses items, varies by a few seconds a lap.
  Either way there is much less variety here than in a procedurally generated
  environment.
- **Two episodes in the same process are not bit-identical.** Restarting a race
  does not restore the physics world completely. Measured on `hacienda`, two
  episodes with the same seed and the same actions start about 0.7 mm apart and
  drift to about 2 cm over 300 steps. A whole *run* from process start does
  reproduce exactly.
- **Gymnasium warns that the observation bounds are infinite.** Deliberate: the
  values are scaled to roughly [−1, 1] but a kart off a ramp can exceed it, and
  clipping would hide that from the agent.

## 12. When something goes wrong

| symptom | cause |
|---|---|
| `BinaryNotFound` | the game is not built, or `$STK_ENV_BIN` points at nothing. See §1. |
| `EngineDied` immediately | the game died while starting. Pass `capture_stderr=True`, or run the command from §10 by hand and read stderr. |
| `Set $SUPERTUXKART_DATADIR` on stderr | the game cannot find its assets. It resolves them relative to the working directory, and the package normally runs it from the checkout; pass `cwd=` if yours is elsewhere. |
| `CommandFailed: not_reset` | `step` before `reset`. |
| `CommandFailed: not_supported` | asking `reset` for a different track. The track is fixed per environment; make another one. |
| `CommandFailed: no_such_env` | `env_id` other than 0. There is one race per process; a vector environment uses several processes. |
| episodes never end | only crossing the finish line terminates. The step budget comes from the registered id's `max_episode_steps`, or from wrapping in `TimeLimit` yourself. |
| observation shape mismatch | the game and the package disagree. Rebuild the game; the shapes come from its handshake. |

## Where the code is

| | |
|---|---|
| `src/gym/` | the server: protocol, state extraction, the controller, the loop |
| `src/gym/gym_state.cpp` | the only file that reaches into the rest of the game for observations |
| `python/src/stk_gym/obs.py` | what the agent sees |
| `python/src/stk_gym/actions.py` | what the agent can do |
| `python/src/stk_gym/env.py` | the reward and the episode |
