import math

import torch

from isaac_l2f.dynamics import (
    State,
    dxdt,
    motor_step_closed_form,
    quaternion as quat,
    rotor_wrench,
    sample_initial_state,
    sample_parameters,
)


def test_wrench_dxdt_consistency(cf64, gen) -> None:
    n = 6
    p = sample_parameters(cf64, n, gen)
    s = sample_initial_state(p, n, gen)
    s.rpm = torch.rand(n, 4, generator=gen, dtype=torch.float64)
    s.angular_velocity = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    s.force = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    s.torque = torch.randn(n, 3, generator=gen, dtype=torch.float64)
    u_cmd = torch.rand(n, 4, generator=gen, dtype=torch.float64)
    d = dxdt(p, s, u_cmd, s.force, s.torque)
    f, tau = rotor_wrench(p, s.rpm)
    m = p.mass[:, None]
    v_dot = quat.rotate(s.orientation, f) / m + p.gravity + s.force / m
    w = s.angular_velocity
    w_dot = torch.einsum(
        "nij,nj->ni", p.J_inv, tau - torch.cross(w, torch.einsum("nij,nj->ni", p.J, w), dim=-1) + s.torque
    )
    assert torch.allclose(d.linear_velocity, v_dot, rtol=0, atol=1e-10 * max(1.0, float(v_dot.abs().max())))
    assert torch.allclose(d.angular_velocity, w_dot, rtol=1e-10, atol=1e-10)


def test_motor_closed_form_vs_rk4(cf64) -> None:
    dt = 0.01
    p = cf64.expand(4)
    u = torch.tensor([[0.1, 0.5, 0.9, 0.3]] * 4, dtype=torch.float64)
    cmd = torch.tensor([[0.8, 0.2, 0.95, 0.3]] * 4, dtype=torch.float64)
    tau = torch.where(cmd >= u, p.rotor_time_constants_rising, p.rotor_time_constants_falling)
    h = dt
    f = lambda y: (cmd - y) / tau  # noqa: E731
    k1 = f(u)
    k2 = f(u + h / 2 * k1)
    k3 = f(u + h / 2 * k2)
    k4 = f(u + h * k3)
    rk = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    cf = motor_step_closed_form(p, u, cmd, dt)
    tol = 2 * (dt / float(tau.min())) ** 5 / 120 + 1e-12
    assert float((rk - cf).abs().max()) <= tol


def test_hover_throttle(cf64) -> None:
    p = cf64
    c0, c1, c2 = (float(x) for x in p.rotor_thrust_coefficients[0, 0])
    target = float(p.mass[0]) * 9.81 / 4
    u = (-c1 + math.sqrt(c1**2 - 4 * c2 * (c0 - target))) / (2 * c2)
    s = State.zeros(1, dtype=torch.float64)
    s.rpm = torch.full((1, 4), u, dtype=torch.float64)
    d = dxdt(p, s, s.rpm, s.force, s.torque)
    assert abs(float(d.linear_velocity[0, 2])) < 1e-6
    assert float(d.angular_velocity.abs().max()) < 1e-6


def test_effectiveness_zero_removes_rotor(cf64) -> None:
    p = cf64
    u = torch.full((1, 4), 0.6, dtype=torch.float64)
    eff = torch.ones(1, 4, dtype=torch.float64)
    eff[0, 1] = 0.0
    f, tau = rotor_wrench(p, u, eff)
    f0, tau0 = rotor_wrench(p, u)
    thrust = float(p.rotor_thrust_coefficients[0, 1] @ torch.tensor([1.0, 0.6, 0.36], dtype=torch.float64))
    expected_f = f0[0] - thrust * p.rotor_thrust_directions[0, 1]
    expected_t = tau0[0] - (
        p.rotor_torque_directions[0, 1] * p.rotor_torque_constants[0, 1] * thrust
        + torch.cross(p.rotor_positions[0, 1], thrust * p.rotor_thrust_directions[0, 1], dim=-1)
    )
    assert torch.allclose(f[0], expected_f, atol=1e-12)
    assert torch.allclose(tau[0], expected_t, atol=1e-12)
    assert torch.allclose(rotor_wrench(p, u, torch.ones_like(u))[0], f0)
