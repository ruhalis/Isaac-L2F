"""The env core driving a stand-in rigid-body integrator reproduces ``isaac_l2f.dynamics.step``.

``dynamics.step`` is golden-tested against the l2f oracle; here the same transitions are produced
by ``L2FCore`` (motor lag, wrench, observation, reward, termination) around ``rigid_twin`` (the
simulator's role: integrate a held wrench). Parity is measured per step with teacher forcing,
the way the golden tests do it: the core is put into the L2F state, steps once, and its outputs
are compared with L2F's. Closed-loop rollouts of a tumbling quadrotor are chaotic and would
only amplify the one-step numbers.

Two twins:

* the *ideal* twin uses ``J`` and ``J_inv`` as loaded, L2F's own quirk, so the only differences
  are the Isaac stepping scheme (exact lag plus held wrench vs. L2F's coupled RK4): this pins the
  core's frames, wrench, motor model, observation, reward and termination;
* the *simulator* twin inverts the inertia it is given, ``boundary.effective_inertia`` as the
  env authors it, and measures the cost of that decision (the gyroscopic term sees ``inv(J_inv)``
  instead of ``J``). Numbers are recorded in STATUS.md.
"""

from __future__ import annotations

import pytest
import torch

from isaac_l2f.dynamics import State, load_registry, reward, step, student_observation, terminated
from isaac_l2f.tasks.boundary import effective_inertia
from isaac_l2f.tasks.core import L2FCore
from rigid_twin import rigid_rk4, rigid_rk4_world_frozen, rigid_semi_implicit_euler, wrench_to_world

N = 32
PHYSICS_DT = 1 / 400
DECIMATION = 4


def _params():
    return load_registry("crazyflie", dtype=torch.float64).expand(N)


def _core(params, seed=0):
    core = L2FCore(params, N, seed=seed)
    sampled = core.reset()
    core.sync(sampled.position, sampled.orientation, sampled.linear_velocity, sampled.angular_velocity)
    return core, sampled


def _load(core: L2FCore, s: State) -> None:
    """Teacher forcing: put the core into the L2F state ``s``."""
    core.sync(s.position, s.orientation, s.linear_velocity, s.angular_velocity)
    core.state.rpm = s.rpm.clone()
    core.state.last_action = s.last_action.clone()
    core.state.force = s.force.clone()
    core.state.torque = s.torque.clone()


def _core_step(core, params, s, action, hold_dt, holds, inertia, inv_inertia=None):
    """One policy step of the core around the twin from L2F state ``s``; returns (obs, next kinematics, reward, done)."""
    _load(core, s)
    obs = core.observe()
    core.pre_physics_step(action)
    p, q, v, w = s.position, s.orientation, s.linear_velocity, s.angular_velocity
    for _ in range(holds):
        fb, tb, fw, td = core.substep(hold_dt)
        p, q, v, w = rigid_rk4(p, q, v, w, fb, tb, fw, td, params.mass, inertia, params.gravity, hold_dt, inv_inertia)
    core.sync(p, q, v, w)
    return obs, core.state.clone(), core.reward(), core.terminated()


def _l2f_trajectory(params, s0, actions):
    states = [s0]
    for t in range(actions.shape[0]):
        states.append(step(params, states[-1], actions[t]))
    return states


def _teacher_forced_errors(hold_dt, holds, ideal: bool, action_spread=1.0, steps=20, seed=0):
    """Max one-step errors of the core+twin vs L2F over teacher-forced trajectories.

    ``action_spread`` 1.0 draws raw actions over the full range (a tumbling stress regime where
    |ω| reaches tens of rad/s); 0.2 keeps them near hover (the flight regime of the M1 gate).
    """
    params = _params()
    core, s0 = _core(params, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    hover = params.hovering_throttle_relative[0].item() * 2 - 1
    actions = hover + (torch.rand(steps, N, 4, generator=g, dtype=torch.float64) * 2 - 1) * action_spread
    actions = actions.clamp(-1.5, 1.5)  # raw actions may leave [-1, 1]; the motor map clamps
    states = _l2f_trajectory(params, s0, actions)
    inertia = params.J if ideal else effective_inertia(params)
    inv_inertia = params.J_inv if ideal else None
    err = {k: 0.0 for k in ("obs", "position", "orientation", "linear_velocity", "angular_velocity", "rpm", "reward")}
    for t in range(steps):
        s, nxt = states[t], states[t + 1]
        obs, out, r, done = _core_step(core, params, s, actions[t], hold_dt, holds, inertia, inv_inertia)
        err["obs"] = max(err["obs"], (obs - student_observation(params, s)).abs().max().item())
        for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm"):
            err[k] = max(err[k], (getattr(out, k) - getattr(nxt, k)).abs().max().item())
        err["reward"] = max(err["reward"], (r - reward(params, s, actions[t], nxt)).abs().max().item())
        assert torch.equal(done, terminated(params, nxt))
        assert torch.equal(out.last_action, actions[t])
    err["max_omega"] = max(st.angular_velocity.norm(dim=-1).max().item() for st in states)
    return err


def _report(name, err):
    print(f"\n{name}: " + ", ".join(f"{k} {v:.2e}" for k, v in err.items()))


# Bounds below are twice the values measured on 2026-10-18 (recorded in STATUS.md); they pin the
# stepping-scheme discretisation as a regression check, not a parity claim against the oracle.


def test_one_step_parity_ideal_twin_four_substeps():
    """Isaac scheme vs L2F's RK4 with L2F's own inertia quirk: only the stepping scheme differs."""
    err = _teacher_forced_errors(PHYSICS_DT, DECIMATION, ideal=True)
    _report("ideal twin, 4 x 2.5 ms, full-range actions", err)
    assert err["obs"] < 1e-12  # quaternion re-normalisation round-off only
    assert err["reward"] < 1e-12
    assert err["rpm"] < 4e-6  # exact lag vs RK4 of the lag (h / tau ~ 0.18)
    assert err["position"] < 1.3e-6
    assert err["orientation"] < 3.4e-5
    assert err["linear_velocity"] < 1.5e-5
    assert err["angular_velocity"] < 2.5e-4


def test_one_step_parity_ideal_twin_single_hold():
    """A backend that runs the decimation internally holds one wrench for 10 ms."""
    err = _teacher_forced_errors(PHYSICS_DT * DECIMATION, 1, ideal=True)
    _report("ideal twin, 1 x 10 ms, full-range actions", err)
    assert err["obs"] < 1e-12
    assert err["rpm"] < 4e-6
    assert err["position"] < 2e-5
    assert err["orientation"] < 5.3e-4
    assert err["linear_velocity"] < 2.3e-4
    assert err["angular_velocity"] < 1.8e-3


def test_one_step_parity_simulator_twin_stress():
    """The env's inertia decision under full-range actions: the gyroscopic term sees inv(J_inv), not J."""
    err = _teacher_forced_errors(PHYSICS_DT, DECIMATION, ideal=False)
    _report("simulator twin (effective inertia), 4 x 2.5 ms, full-range actions", err)
    assert err["position"] < 1.3e-6
    assert err["linear_velocity"] < 1.3e-5
    assert err["orientation"] < 4e-4
    assert err["angular_velocity"] < 2e-1  # |ω| reaches tens of rad/s here; the mismatch grows with ω²


def test_one_step_parity_simulator_twin_raptor_flight():
    """Same decision in the flight regime of the M1 gate: teacher forcing along RAPTOR-flown trajectories."""
    pytest.importorskip("foundation_policy")
    from isaac_l2f.policy import load_raptor

    params = _params()
    core, s0 = _core(params, seed=5)
    policy = load_raptor()
    policy.reset()
    states, actions = [s0], []
    for _ in range(100):
        a = policy.step(student_observation(params, states[-1]).float()).double()
        actions.append(a)
        states.append(step(params, states[-1], a))
    inertia = effective_inertia(params)
    err = {k: 0.0 for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm")}
    for t in range(100):
        s, nxt = states[t], states[t + 1]
        _, out, _, _ = _core_step(core, params, s, actions[t], PHYSICS_DT, DECIMATION, inertia)
        for k in err:
            err[k] = max(err[k], (getattr(out, k) - getattr(nxt, k)).abs().max().item())
    err["max_omega"] = max(st.angular_velocity.norm(dim=-1).max().item() for st in states)
    _report("simulator twin (effective inertia), 4 x 2.5 ms, RAPTOR flight", err)
    # measured 2026-10-18: position 5.6e-07, orientation 2.5e-05, lin vel 1.9e-06, ang vel 1.06e-02 at |ω| up to 7 rad/s
    assert err["position"] < 1.2e-6
    assert err["linear_velocity"] < 4e-6
    assert err["orientation"] < 5e-5
    assert err["angular_velocity"] < 2.2e-2


def test_closed_loop_balanced_motors_ideal_twin():
    """Constant, balanced motor state (no control torque, no chaos): 100 steps closed loop stay at RK4 truncation."""
    params = _params()
    core, s0 = _core(params, seed=1)
    u = params.hovering_throttle_relative[:, None].expand(N, 4)
    s0.rpm = u.clone()
    _load(core, s0)
    actions = (u * 2 - 1).unsqueeze(0).repeat(100, 1, 1)  # command equals state: the lag is at rest
    states = _l2f_trajectory(params, s0, actions)
    p, q, v, w = s0.position, s0.orientation, s0.linear_velocity, s0.angular_velocity
    err_obs = 0.0
    for t in range(100):
        err_obs = max(err_obs, (core.observe() - student_observation(params, states[t])).abs().max().item())
        core.pre_physics_step(actions[t])
        for _ in range(DECIMATION):
            fb, tb, fw, td = core.substep(PHYSICS_DT)
            p, q, v, w = rigid_rk4(p, q, v, w, fb, tb, fw, td, params.mass, params.J, params.gravity, PHYSICS_DT, params.J_inv)
        core.sync(p, q, v, w)
        assert torch.equal(core.terminated(), terminated(params, states[t + 1]))
    assert torch.allclose(core.state.rpm, u, atol=1e-12)
    end = states[-1]
    print(f"\nbalanced motors, 100 steps closed loop: max|obs err| {err_obs:.2e}")
    assert err_obs < 1e-7, err_obs
    assert torch.allclose(core.state.position, end.position, atol=1e-7)
    assert torch.allclose(core.state.angular_velocity, end.angular_velocity, atol=1e-6)


def _backend_twin_flight_errors(integrator):
    """Max one-step errors of the core around a world-frozen-wrench backend twin, teacher-forced along RAPTOR flight."""
    pytest.importorskip("foundation_policy")
    from isaac_l2f.policy import load_raptor

    params = _params()
    core, s0 = _core(params, seed=5)
    policy = load_raptor()
    policy.reset()
    states, actions = [s0], []
    for _ in range(100):
        a = policy.step(student_observation(params, states[-1]).float()).double()
        actions.append(a)
        states.append(step(params, states[-1], a))
    inertia = effective_inertia(params)
    err = {k: 0.0 for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm")}
    for t in range(100):
        s, nxt = states[t], states[t + 1]
        _load(core, s)
        core.pre_physics_step(actions[t])
        p, q, v, w = s.position, s.orientation, s.linear_velocity, s.angular_velocity
        for _ in range(DECIMATION):
            fb, tb, fw, td = core.substep(PHYSICS_DT)
            force_w, torque_w = wrench_to_world(q, fb, tb, fw, td)  # once per substep, start orientation
            p, q, v, w = integrator(p, q, v, w, force_w, torque_w, params.mass, inertia, params.gravity, PHYSICS_DT)
        core.sync(p, q, v, w)
        out = core.state
        for k in err:
            err[k] = max(err[k], (getattr(out, k) - getattr(nxt, k)).abs().max().item())
    err["max_omega"] = max(st.angular_velocity.norm(dim=-1).max().item() for st in states)
    return err


def test_one_step_parity_backend_twin_semi_implicit_euler_raptor_flight():
    """Newton/MJWarp ``implicitfast`` = semi-implicit Euler with the wrench frozen in world per 2.5 ms substep.

    Documents the backend's discretisation cost along RAPTOR flight, not a defect of the core.
    """
    err = _backend_twin_flight_errors(rigid_semi_implicit_euler)
    _report("backend twin (semi-implicit Euler, world-frozen wrench), 4 x 2.5 ms, RAPTOR flight", err)
    # measured 2026-10-26: position 1.12e-04, orientation 8.74e-04, lin vel 6.08e-04, ang vel 1.06e-02 at |ω| up to 7 rad/s
    assert err["position"] < 2.3e-4
    assert err["orientation"] < 1.8e-3
    assert err["linear_velocity"] < 1.3e-3
    assert err["angular_velocity"] < 2.2e-2


def test_one_step_parity_backend_twin_rk4_world_frozen_raptor_flight():
    """Newton/MJWarp with the ``rk4`` option: RK4 with the same world-frozen wrench (no per-stage re-rotation).

    Documents the backend's discretisation cost along RAPTOR flight, not a defect of the core.
    """
    err = _backend_twin_flight_errors(rigid_rk4_world_frozen)
    _report("backend twin (RK4, world-frozen wrench), 4 x 2.5 ms, RAPTOR flight", err)
    # measured 2026-10-26: position 2.74e-06, orientation 2.54e-05, lin vel 5.97e-04, ang vel 1.02e-02 at |ω| up to 7 rad/s
    assert err["position"] < 5.5e-6
    assert err["orientation"] < 5.1e-5
    assert err["linear_velocity"] < 1.2e-3
    assert err["angular_velocity"] < 2.1e-2
