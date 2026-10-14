# Isaac-L2F

L2F-faithful quadrotor dynamics and RAPTOR-contract environments for
[NVIDIA Isaac Lab 3.0](https://github.com/isaac-sim/IsaacLab) on both Newton and PhysX.

[L2F](https://github.com/rl-tools/l2f) ("Learning to Fly in Seconds", rl-tools) is the
fastest quadrotor dynamics simulator and the training ground of the
[RAPTOR](https://github.com/rl-tools/raptor) foundation policy. It has no rendering and no
collisions. Isaac Lab has GPU rendering, collisions and obstacle scenes, but its own
multirotor support is PhysX-only and uses a different motor model. Isaac-L2F brings L2F's
exact motor and wrench model and the RAPTOR observation/action contract into Isaac Lab, so
that policies trained in L2F fly in Isaac Lab zero-shot, and vision policies can be trained
on top of the same dynamics.

Status: **M0 (dynamics core) done.** The Isaac Lab environments are the next milestone; see
[PLAN.md](PLAN.md).

## The contract

Everything in this package is written to one contract, the one RAPTOR and L2F use:

| | |
|---|---|
| Frames | body FLU (x forward, y left, z up), world z-up |
| Observation (22) | position (3, world) · rotation matrix row-major (9) · linear velocity (3, world) · angular velocity (3, body) · previous action (4) |
| Action (4) | per-motor, normalised to [-1, 1], order front-right, back-right, back-left, front-left |
| Rate | 100 Hz (dt = 0.01 s), one RK4 step per control step |
| Quaternion | (w, x, y, z), body-to-world |

## What is here

`isaac_l2f.dynamics` is a pure-PyTorch, batched, device-agnostic port of the L2F model:

- parameter JSON schema (same keys as L2F / RAPTOR), load, save, stack, select
- first-order motor lag with separate rising and falling time constants
- thrust polynomial `c0 + c1 u + c2 u²`, rotor drag torque, arm moments
- RK4 step, post-integration, observation, reward, termination, initial-state sampling
- the RAPTOR domain-randomisation sampler (thrust-to-weight, torque-to-inertia, mass, size,
  time constants, torque constant, disturbance magnitude)
- `rotor_wrench(params, u)` returning the body-frame force and torque, which is what the
  Isaac Lab environment injects through the wrench composer

Upstream quirks are reproduced on purpose, not fixed: `J` and `J_inv` are stored
independently and used as loaded, the orientation cost is `2·acos(1 − |q_z|)`, and the
initial attitude angle ignores `max_angle`. Golden tests check every step against the
`l2f` pip package.

```python
import torch
from isaac_l2f.dynamics import load_registry, sample_initial_state, step, student_observation

params = load_registry("crazyflie").expand(1024)
gen = torch.Generator().manual_seed(0)
state = sample_initial_state(params, 1024, gen)
action = torch.zeros(1024, 4)
state = step(params, state, action)
obs = student_observation(params, state)   # [1024, 22]
```

## Install

Development (any platform, no GPU needed for the dynamics core and its tests):

```bash
uv sync
uv run pytest -q
```

With Isaac Lab 3.0 (Linux, NVIDIA GPU, Python 3.12). Isaac Lab ships as one distribution on
NVIDIA's package index with Newton included, so the `isaac` extra is all that is needed:

```bash
uv sync --extra isaac
```

Kit-less training uses `physics=newton_mjwarp renderer=newton_renderer`; the PhysX preset
needs an Isaac Sim 6.x install on top.

## Acknowledgements

Dynamics model and parameter format from [rl-tools/l2f](https://github.com/rl-tools/l2f)
(MIT) and the RAPTOR pre-training code in [rl-tools](https://github.com/rl-tools/rl-tools).

## License

MIT, see [LICENSE](LICENSE).
