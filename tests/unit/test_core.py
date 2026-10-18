"""L2FCore: reset, sync, observation, motor/wrench substeps, hooks, reward and termination."""

from __future__ import annotations

import pytest
import torch

from isaac_l2f.dynamics import (
    action_to_command,
    load_registry,
    motor_step_closed_form,
    privileged_observation,
    reward,
    rotor_thrust_mean,
    student_observation,
    terminated,
    wrench_from_thrust,
)
from isaac_l2f.dynamics import quaternion as quat
from isaac_l2f.tasks.core import L2FCore, assert_rates

N = 8
DT = 1 / 400
FIELDS = ("position", "orientation", "linear_velocity", "angular_velocity", "rpm", "last_action", "force", "torque")


def _core(seed=0):
    params = load_registry("crazyflie", dtype=torch.float64)
    core = L2FCore(params, N, seed=seed)
    return core.params, core


def _rand_kin(n=N, seed=1, scale=1.0):
    g = torch.Generator()
    g.manual_seed(seed)
    p = torch.randn(n, 3, generator=g, dtype=torch.float64) * scale
    q = quat.normalize(torch.randn(n, 4, generator=g, dtype=torch.float64))
    v = torch.randn(n, 3, generator=g, dtype=torch.float64)
    w = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return p, q, v, w


def test_construction_expands_params_and_rejects_mismatch():
    params = load_registry("crazyflie", dtype=torch.float64)
    assert L2FCore(params, 8).params.n == 8
    with pytest.raises(ValueError):
        L2FCore(params.expand(3), 8)


def test_assert_rates():
    params, _ = _core()
    assert_rates(1 / 400, 4, params)
    with pytest.raises(ValueError):
        assert_rates(1 / 200, 4, params)
    with pytest.raises(ValueError):
        assert_rates(1 / 400, 3, params)


def test_reset_subset_only_touches_given_rows():
    _, core = _core()
    core.reset()
    snap = core.state.clone()
    ids = torch.tensor([1, 4, 6])
    sampled = core.reset(ids)
    others = torch.tensor([0, 2, 3, 5, 7])
    for f in FIELDS:
        cur, old, smp = getattr(core.state, f), getattr(snap, f), getattr(sampled, f)
        assert torch.equal(cur[ids], smp), f
        assert torch.equal(cur[others], old[others]), f
    assert (sampled.last_action == 0).all()
    assert sampled.position.abs().max() <= 0.5
    assert sampled.linear_velocity.abs().max() <= 1
    assert sampled.angular_velocity.abs().max() <= 1
    torch.testing.assert_close(sampled.orientation.norm(dim=-1), torch.ones(3, dtype=torch.float64), atol=1e-12, rtol=0)
    assert sampled.rpm.min() >= 0 and sampled.rpm.max() <= 1
    assert (sampled.force == 0).all() and (sampled.torque == 0).all()
    assert torch.equal(core.u_cmd[ids], sampled.rpm)
    assert (core.action[ids] == 0).all()


def test_fresh_envs_keep_sampled_kinematics_on_next_sync():
    _, core = _core()
    core.reset()
    core.sync(*_rand_kin(seed=0))
    sampled = core.reset(torch.tensor([2]))
    p, q, v, w = _rand_kin(seed=2)
    core.sync(p, q, v, w)
    names = ("position", "orientation", "linear_velocity", "angular_velocity")
    for name, given in zip(names, (p, q, v, w)):
        cur = getattr(core.state, name)
        torch.testing.assert_close(cur[2], getattr(sampled, name)[0], atol=1e-12, rtol=0)
        mask = torch.arange(N) != 2
        torch.testing.assert_close(cur[mask], given[mask], atol=1e-12, rtol=0)
    p2, q2, v2, w2 = _rand_kin(seed=3)
    core.sync(p2, q2, v2, w2)
    torch.testing.assert_close(core.state.position, p2, atol=1e-12, rtol=0)
    torch.testing.assert_close(core.state.orientation, q2, atol=1e-12, rtol=0)


def test_sync_normalises_orientation():
    _, core = _core()
    core.reset()
    core.sync(*_rand_kin(seed=0))  # consume the fresh-reset flag
    p, q, v, w = _rand_kin()
    core.sync(p, q * 3, v, w)
    torch.testing.assert_close(core.state.orientation.norm(dim=-1), torch.ones(N, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(core.state.orientation, q, atol=1e-12, rtol=0)


def test_observation_and_reference_hook():
    _, core = _core()
    core.reset()
    core.sync(*_rand_kin())
    obs = core.observe()
    assert obs.shape == (N, 22)
    assert torch.equal(obs, student_observation(core.params, core.state))
    g = torch.Generator()
    g.manual_seed(9)
    pr = torch.randn(N, 3, generator=g, dtype=torch.float64)
    vr = torch.randn(N, 3, generator=g, dtype=torch.float64)
    core.set_reference(position=pr, velocity=vr)
    diff = obs - core.observe()
    torch.testing.assert_close(diff[:, 0:3], pr, atol=1e-12, rtol=0)
    torch.testing.assert_close(diff[:, 12:15], vr, atol=1e-12, rtol=0)
    rest = torch.ones(22, dtype=torch.bool)
    rest[0:3] = False
    rest[12:15] = False
    assert (diff[:, rest] == 0).all()


def test_set_reference_env_ids_and_none_leaves_other_part():
    _, core = _core()
    core.set_reference(position=torch.tensor([1.0, 2.0, 3.0]), env_ids=torch.tensor([2, 5]))
    expected = torch.zeros(N, 3, dtype=torch.float64)
    expected[[2, 5]] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert torch.equal(core.position_ref, expected)
    assert (core.velocity_ref == 0).all()


def test_privileged_observation_matches_dynamics():
    _, core = _core()
    core.reset()
    core.sync(*_rand_kin())
    core.set_reference(position=torch.tensor([0.1, 0.2, 0.3]))
    out = core.observe_privileged()
    assert out.shape == (N, 26)
    expected = privileged_observation(
        core.params, core.state, position_ref=core.position_ref, velocity_ref=core.velocity_ref
    )
    assert torch.equal(out, expected)


def test_previous_action_is_raw_and_command_is_clamped():
    _, core = _core()
    core.reset()
    core.sync(*_rand_kin())
    action = torch.rand(N, 4, dtype=torch.float64) * 2 - 1
    action[0, 0], action[3, 2] = 1.7, -2.3
    core.pre_physics_step(action)
    assert torch.equal(core.observe()[:, 18:22], action)
    assert torch.equal(core.action, action)
    assert torch.equal(core.u_cmd, action_to_command(core.params, torch.clamp(action, -1, 1)))


def test_substep_composes_motor_lag_and_wrench():
    _, core = _core()
    core.reset()
    a = torch.rand(N, 4, dtype=torch.float64) * 2 - 1
    core.pre_physics_step(a)
    rpm0 = core.state.rpm.clone()
    for k in range(1, 5):
        out = core.substep(DT)
        assert len(out) == 4 and all(t.shape == (N, 3) for t in out)
        if k == 1:
            thrust = rotor_thrust_mean(core.params, rpm0, core.u_cmd, DT) * core.effectiveness
            f, t = wrench_from_thrust(core.params, thrust)
            assert torch.equal(out[0], f) and torch.equal(out[1], t)
            assert torch.equal(out[2], core.state.force) and torch.equal(out[3], core.state.torque)
            assert (out[2] == 0).all() and (out[3] == 0).all()
        torch.testing.assert_close(
            core.state.rpm, motor_step_closed_form(core.params, rpm0, core.u_cmd, k * DT), atol=1e-12, rtol=0
        )


def test_motor_effectiveness_hook():
    _, core = _core()
    core.reset()
    hover = core.params.hovering_throttle_relative[:, None].expand(N, 4)
    core.state.rpm = hover.clone()
    core.pre_physics_step(hover * 2 - 1)
    c0, c1, c2 = core.params.rotor_thrust_coefficients[0, 0]
    u = hover[0, 0]
    thrust = c0 + c1 * u + c2 * u * u

    f, t, _, _ = core.substep(DT)
    assert t.abs().max() <= 1e-12
    torch.testing.assert_close(f[:, 2], (4 * thrust).expand(N), atol=1e-12, rtol=0)
    assert f[:, :2].abs().max() <= 1e-12

    core.state.rpm = hover.clone()
    core.set_motor_effectiveness(torch.tensor([1.0, 1.0, 1.0, 0.5]))
    f2, t2, _, _ = core.substep(DT)
    torch.testing.assert_close(f2[:, 2], (3.5 * thrust).expand(N), atol=1e-12, rtol=0)
    assert (t2[:, 0].abs() > 0).all() and (t2[:, 1].abs() > 0).all() and (t2[:, 2].abs() > 0).all()

    core.set_motor_effectiveness(1.0)
    assert (core.effectiveness == 1).all()
    core.set_motor_effectiveness(torch.tensor([0.0, 1.0, 1.0, 1.0]), env_ids=torch.tensor([3]))
    expected = torch.ones(N, 4, dtype=torch.float64)
    expected[3, 0] = 0.0
    assert torch.equal(core.effectiveness, expected)


def test_disturbance_wrench_hook():
    _, core = _core()
    core.reset()
    core.pre_physics_step(torch.zeros(N, 4, dtype=torch.float64))
    core.set_disturbance_wrench(
        force=torch.tensor([0.0, 0.0, 0.1]), torque=torch.tensor([1e-3, 0.0, 0.0]), env_ids=torch.tensor([3])
    )
    _, _, fw, tb = core.substep(DT)
    ef = torch.zeros(N, 3, dtype=torch.float64)
    et = torch.zeros(N, 3, dtype=torch.float64)
    ef[3] = torch.tensor([0.0, 0.0, 0.1])
    et[3] = torch.tensor([1e-3, 0.0, 0.0])
    assert torch.equal(fw, ef) and torch.equal(tb, et)

    core.set_disturbance_wrench(torque=torch.tensor([0.0, 2e-3, 0.0]), env_ids=torch.tensor([3]))
    et[3] = torch.tensor([0.0, 2e-3, 0.0])
    assert torch.equal(core.state.force, ef) and torch.equal(core.state.torque, et)

    core.reset(torch.tensor([3]))
    assert (core.state.force == 0).all() and (core.state.torque == 0).all()


def test_reward_and_termination_reuse_dynamics():
    _, core = _core()
    with pytest.raises(RuntimeError):
        core.reward()
    core.reset()
    core.sync(*_rand_kin(scale=0.1))
    prev = core.state.clone()
    a = torch.rand(N, 4, dtype=torch.float64) * 2 - 1
    core.pre_physics_step(a)
    p, q, v, w = _rand_kin(seed=5, scale=0.1)
    p[[1, 5]] = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    core.sync(p, q, v, w)
    assert torch.equal(core.reward(), reward(core.params, prev, a, core.state))
    term = core.terminated()
    assert torch.equal(term, terminated(core.params, core.state))
    assert term[[1, 5]].all()


def test_float32_smoke():
    core = L2FCore(load_registry("crazyflie", dtype=torch.float32), 4)
    s = core.reset()
    core.sync(s.position, s.orientation, s.linear_velocity, s.angular_velocity)
    core.pre_physics_step(torch.zeros(4, 4))
    for _ in range(4):
        core.substep(DT)
    obs = core.observe()
    assert obs.shape == (4, 22) and obs.dtype == torch.float32
    assert core.reward().shape == (4,)
