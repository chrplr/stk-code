# supertuxkart-gym

SuperTuxKart as a [Gymnasium](https://gymnasium.farama.org/) environment. The
physics are the game's own: `supertuxkart --gym` runs as a child process and is
driven one step at a time over a line-based JSON protocol.

This page is the reference. For a walk-through aimed at someone who has not used
Gymnasium before, read [`../README-AI.md`](../README-AI.md).

```python
import gymnasium, stk_gym

env = gymnasium.make("SuperTuxKart-Easy-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Install

The game must be built first; this package will not build it for you, because a
SuperTuxKart build takes minutes.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
pip install -e python
```

The binary is looked for in this order: an explicit `binary=` argument,
`$STK_ENV_BIN`, `build/bin/supertuxkart` in the checkout this package lives in,
and finally the `PATH`. The checkout comes before the `PATH` on purpose: a
system-wide SuperTuxKart is almost certainly a release without `--gym`.

## Registered ids

| id | karts | laps | observation | step budget |
|---|---|---|---|---|
| `SuperTuxKart-v0` | game default | game default | `vector` | 3000 |
| `SuperTuxKart-Easy-v0` | 1 | 1 | `vector` | 3000 |
| `SuperTuxKart-Race-v0` | 4 | 3 | `vector_karts` | 6000 |

`StkEnv` takes the same arguments directly: `track`, `laps`, `num_karts`,
`difficulty`, `obs_mode`, `action_mode`, `reward_scheme`, `frame_skip`,
`lookahead`, `expert`, `render_mode`, `binary`, `cwd`, `capture_stderr`.

## Observations — `obs_mode`

The vector modes are a flat `Box(-inf, inf, (n,), float32)` whose length comes
from the handshake, never from a constant repeated in Python.

| mode | contents | length with the defaults |
|---|---|---|
| `vector` | 10 scalars about the kart, then `lookahead_k` driveline points as (x, z) in kart-local coordinates | 20 |
| `vector_karts` | the above plus 4 numbers per opponent, nearest first | 20 + 4·(karts−1) |
| `pixels` | the frame the game drew, `Box(0, 255, (H, W, 3), uint8)` — see *Frames* below | 360 × 640 × 3 |

The ten scalars are: speed ÷ 30, speed ÷ current max speed, steering, on-road,
on-ground, wrong-way, `tanh(distance to centre ÷ 5)`, nitro ÷ 100, rank
normalised to [0, 1], and distance along the lap ÷ track length.

The bounds are infinite deliberately. Every value is scaled to roughly [−1, 1],
but a kart launched off a ramp can leave that range briefly and clipping the
observation would hide it from the agent. Gymnasium's checker warns about this;
the warning is expected.

## Actions — `action_mode`

| mode | space | meaning |
|---|---|---|
| `discrete` | `Discrete(15)` | 5 steering positions × {accelerate, coast, brake} |
| `continuous` (default) | `Box([-1,0,0], [1,1,1])` | steer, accelerate, brake |
| `continuous_full` | `Box(·, (5,))` | the above plus nitro and skid, thresholded at 0.5 |
| `keys` | `MultiBinary(8)` | the keys held: left, right, up, down, nitro, skid, fire, rescue |

The server always accepts the whole control surface — steer, accel, brake,
nitro, skid, fire, rescue — and `stk_gym.actions` decides which subset the agent
gets. Adding an action space is a Python edit.

`keys` is different in kind: the game is started with `--gym-keys` and the
kart is driven by SuperTuxKart's own player controller, which receives one
press or release event per key that changed since the previous step, exactly
as a keyboard would send them. Steering therefore ramps over a few tenths of a
second rather than jumping, a skid takes its direction from the key held when
it started, nitro only burns while accelerating — the game's rules, not this
package's. It is the mode for comparing an agent with a person: a harness that
reads a keyboard and forwards the held keys (fmri-gym does) puts both in front
of the same environment.

## Rewards — `reward_scheme`

Progress is `finished_laps × track_length + distance_down_track`, in metres.
It runs smoothly through the start line, where `distance_down_track` wraps to
zero and `finished_laps` steps from −1 to 0.

| scheme | per step |
|---|---|
| `progress` (default) | Δ metres − 0.05 − 0.5 if off-road, +50 on finishing |
| `progress_only` | Δ metres |
| `sparse` | 50 on finishing, otherwise 0 |

A step covering more than half a lap is paid nothing: that is a rescue or a lap
counter crossing, not driving, and paying for it would teach the agent to
trigger it.

**Do not build a reward on `overall_distance`.** It is the game's own measure and
is computed from the *checkline-validated* distance, which only moves when a
checkline is crossed — it exists to defeat shortcuts, not to reward progress, and
is flat in between. `info["progress_m"]` and `stk_gym.progress_of` use the right
one.

`info` carries `is_success`, `progress_m`, `lap`, `rank`, `speed`, `on_road`,
`wrong_way` and `race_time`.

## Episodes

`step` never returns `truncated=True`: a time limit is `TimeLimit`'s decision,
which `gymnasium.make` applies from the id's `max_episode_steps`. `terminated`
means the kart crossed the finish line.

`reset` restarts the race in place and costs well under a millisecond.
Constructing an environment loads a track and costs about half a second, so keep
environments and reset them.

**The track is fixed for the life of an environment.** Reloading one costs
seconds, so `reset(options={"track": ...})` is refused by name rather than
silently ignored; race a different track with a different environment.

## Several races at once

```python
venv = stk_gym.StkVectorEnv(num_envs=8, track="hacienda")
```

`World`, `RaceManager` and the physics world are process globals in the game, so
this runs one child process per environment rather than one process with several
sessions. A step posts its request to every child before reading any reply, so
the races simulate at the same time; the test suite asserts it is bit-identical
to `SyncVectorEnv` step for step.

For Stable-Baselines3, `stk_gym.sb3.make_sb3_vec_env(num_envs=8, ...)` wraps it
and reconciles the two places the conventions differ: autoreset timing, and
`terminated`/`truncated` versus `done` + `TimeLimit.truncated`.

## Watching a policy drive

```python
env = stk_gym.StkEnv(render_mode="human")
```

There is no second binary: the game decides at run time whether it has a window,
so this simply leaves `--no-graphics` off. `render()` returns `None` because the
game's own window does the drawing. `render_mode="ansi"` returns a status line
instead.

## Frames

```python
env = stk_gym.StkEnv(render_mode="rgb_array", action_mode="keys",
                     frame_skip=2, screensize=(640, 360))
obs, info = env.reset(seed=1)
frame = env.render()                     # (360, 640, 3) uint8, HUD included
```

With `render_mode="rgb_array"` (or `obs_mode="pixels"`, which makes the frame
the observation) every `reset` and `step` brings back the frame the game drew
for that state, read from the back buffer before it is presented. The game
renders into a window that is created hidden (`--gym-hidden`): it is never
mapped, so the window manager never resizes it and the frame is exactly the
requested `screensize`; vsync is off, so a step never waits for a monitor
nobody sees; and the game's saved settings are left alone. Pass `hidden=False`
to watch the window as well.

The frame is the game's own rendering, so it needs a real OpenGL display
(`--no-graphics` cannot draw, Vulkan cannot read back): the handshake reports
`frame_supported`, and the constructor refuses to start without it. Each step
then costs the render plus about 700 KB through the pipe. Measured on an Intel
Arc (Meteor Lake) laptop, 640×360, four karts, `frame_skip=2` (60 steps/s):
2.5 ms per step mean, 2.9 ms p95, 8 ms max over 600 steps, against a 16.7 ms
refresh; the same step without the frame costs 1 ms. `examples/measure_frames.py`
prints those numbers for the machine it runs on.

## A person at the wheel

```python
session = stk_gym.HumanSession(track="hacienda", laps=3, num_karts=4, fullscreen=True)
session.reset(seed=1)
while not session.state()["finished"]:
    ...                                   # sample at whatever rate you like
session.close()
```

`--gym-human` turns the server round. The game opens its usual window, reads
the keyboard or gamepad itself and runs on its own clock, with its frame pacing
and sound; the protocol only *reports*. `state()` returns the race as of the
game's next frame (so one frame of latency), including a `controls` object —
`steer`, `accel`, `brake`, `nitro`, `skid`, `fire`, `rescue`, `look_back` —
which is the record of what the person did; `sample()` flattens that state to
one row of floats (`SAMPLE_FIELDS`, always all of them, NaN where the game
reported nothing), the shape a per-sample log wants; `reset()` restarts the
race. There
is no `step`: the game refuses it in this mode, so a caller cannot mistake a
race that runs by itself for one it is driving. The ready-set-go countdown is
kept unless `race_now=True`. This is the shape an experiment harness wants:
present the game to a participant, sample the trajectory at its own rate.

## Reproducibility

A run reproduces exactly: the same seeds and the same actions from process
start give the same trajectory, bit for bit, in a different process on the same
machine.

"Seeds", plural. `reset(seed=...)` seeds the AI's random draws, the item boxes
and the powerup draws for the episode. The choice of the AI karts themselves
is made once, before the first reset, from the game's launch seed:
`StkEnv(seed=...)` passes `--seed=...` on the command line, and a run with
opponents is only reproducible with it.

Individual episodes *within* a run are not bit-identical to each other. A
restart does not restore the physics world completely, and how much the world
ran before the restart is what decides the difference: measured on `hacienda`,
two episodes with the same seed and actions start about 0.7 mm apart and drift
to about 2 cm over 300 steps. Episodes preceded by identical histories are
identical.

The seed does less here than in a procedurally generated environment, but it is
not inert. The grid, the track and the driveline are fixed, so a policy that
only steers sees essentially no variation - a trained policy measured over six
seeds lapped in 80.3 s every time. Seeding changes the game's random draws, so
anything that picks things up does vary: the built-in AI, over the same six
seeds, lapped `hacienda` in 89.7 to 95.5 s.

Within a single process, successive episodes with the same seed give the same
lap time to a tenth of a second; the sub-millimetre drift above is not enough to
change the outcome.

## Talking to the game yourself

The protocol is meant to be driveable by hand:

```sh
printf '{"id":1,"cmd":"hello"}\n{"id":2,"cmd":"reset","seed":1}\n{"id":3,"cmd":"step","action":{"steer":0,"accel":1}}\n' \
  | ./build/bin/supertuxkart --gym --no-graphics --track=hacienda 2>/dev/null
```

SuperTuxKart's command line requires the `=`: `--track hacienda` with a space is
parsed as two unrecognised arguments and the default track is started instead.
`server_args` always emits `--track=...`.

One JSON object per line in each direction, paired by `id`. Commands are
`hello`, `reset`, `step`, `state`, `close`, plus `reset_batch` and `step_batch`
for a batch of one. Failures answer `{"ok":false,"kind":"...","error":"..."}`
with `kind` one of `bad_json`, `unknown_cmd`, `no_such_env`, `bad_action`,
`not_reset`, `bad_batch`, `not_supported`.

The one departure from one-object-per-line is the frame. A `reset`, `step` or
`state` request with `"frame": true` is answered by a line that also carries
`"frame": {"width": W, "height": H, "format": "rgb8"}`, followed — after that
line's newline — by exactly `W*H*3` raw bytes: rows top to bottom, RGB. Start
the game with a window (`--gym-hidden` for one that stays off the screen) and a
`--screensize`; the handshake says `frame_supported` and the size. With
`--gym-keys` the action is `{"keys": {"left": true, "up": true}}` — absent keys
are released — and the game's own player controller drives.

The server reports facts and never a reward: the reward scheme, the termination
rule and the observation encoding all live here in Python, so changing any of
them costs an edit rather than a rebuild of the game.

## Tests

```sh
pip install -e "python[dev]"
pytest python/tests -q
```

Every test that is not a pure function starts a real race, so the suite needs the
built binary; without one it skips.
