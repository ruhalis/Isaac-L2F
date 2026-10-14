"""L2F rigid-body quadrotor model: rotor wrench, continuous dynamics, RK4 env step.

Frames: position/linear velocity/force WORLD; angular velocity/torque BODY (FLU); quaternion
(w,x,y,z) body->world. Rotor order [front-right, back-right, back-left, front-left].
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from . import quaternion as quat
from .params import Params
from .state import State


class Derivative(NamedTuple):
    """Time derivative of the integrated state (p, q, v, omega, u)."""

    position: Tensor
    orientation: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    rpm: Tensor


def rotor_thrust(params: Params, u: Tensor) -> Tensor:
    """Per-rotor thrust ``[N,4]`` (N): ``c0 + c1 u + c2 u^2``."""
    c = params.rotor_thrust_coefficients
    return c[..., 0] + c[..., 1] * u + c[..., 2] * u * u


def rotor_wrench(params: Params, u: Tensor, effectiveness: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Body-frame force ``[N,3]`` (N) and torque ``[N,3]`` (N m) about the CoM from motor state ``u`` ``[N,4]``.

    ``effectiveness`` ``[N,4]`` (optional) multiplies each rotor's thrust before the sums.
    """
    thrust = rotor_thrust(params, u)
    if effectiveness is not None:
        thrust = thrust * effectiveness
    f_i = thrust[..., None] * params.rotor_thrust_directions  # [N,4,3]
    drag = params.rotor_torque_directions * (params.rotor_torque_constants * thrust)[..., None]
    tau_i = drag + torch.cross(params.rotor_positions, f_i, dim=-1)
    return f_i.sum(1), tau_i.sum(1)


def _tau(params: Params, u: Tensor, u_cmd: Tensor) -> Tensor:
    return torch.where(u_cmd >= u, params.rotor_time_constants_rising, params.rotor_time_constants_falling)


def dxdt(
    params: Params,
    state: State,
    u_cmd: Tensor,
    force: Tensor,
    torque: Tensor,
    effectiveness: Tensor | None = None,
) -> Derivative:
    """Continuous dynamics. ``u = state.rpm``; ``u_cmd`` ``[N,4]`` is the motor command in action-limit units.

    ``force`` is a world-frame disturbance (N), ``torque`` a body-frame disturbance (N m).
    """
    q, w, u = state.orientation, state.angular_velocity, state.rpm
    f_body, tau_body = rotor_wrench(params, u, effectiveness)
    m = params.mass[:, None]
    dv = quat.rotate(q, f_body) / m + params.gravity + force / m
    jw = torch.einsum("nij,nj->ni", params.J, w)
    dw = torch.einsum("nij,nj->ni", params.J_inv, tau_body - torch.cross(w, jw, dim=-1) + torque)
    du = (u_cmd - u) / _tau(params, u, u_cmd)
    return Derivative(state.linear_velocity, quat.derivative(q, w), dv, dw, du)


def motor_step_closed_form(params: Params, u: Tensor, u_cmd: Tensor, dt: Tensor | float) -> Tensor:
    """Exact first-order lag over ``dt`` with rising/falling rule, clamped to the action limit."""
    dt_t = torch.as_tensor(dt, dtype=u.dtype, device=u.device)
    if dt_t.dim() == 1:
        dt_t = dt_t[:, None]
    out = u + (u_cmd - u) * (1 - torch.exp(-dt_t / _tau(params, u, u_cmd)))
    return torch.clamp(out, params.action_min[:, None], params.action_max[:, None])


def _advance(s: State, d: Derivative, h: Tensor) -> State:
    out = s.clone()
    out.position = s.position + h * d.position
    out.orientation = s.orientation + h * d.orientation
    out.linear_velocity = s.linear_velocity + h * d.linear_velocity
    out.angular_velocity = s.angular_velocity + h * d.angular_velocity
    out.rpm = s.rpm + h * d.rpm
    return out


def rk4_step(
    params: Params,
    state: State,
    u_cmd: Tensor,
    dt: Tensor | float,
    effectiveness: Tensor | None = None,
) -> State:
    """One classic RK4 step with command and disturbances held fixed; no normalisation/clamping."""
    dt_t = torch.as_tensor(dt, dtype=state.position.dtype, device=state.position.device)
    h = dt_t[:, None] if dt_t.dim() == 1 else dt_t

    def f(s: State) -> Derivative:
        return dxdt(params, s, u_cmd, state.force, state.torque, effectiveness)

    k1 = f(state)
    k2 = f(_advance(state, k1, h / 2))
    k3 = f(_advance(state, k2, h / 2))
    k4 = f(_advance(state, k3, h))
    out = state.clone()
    for name in Derivative._fields:
        y = getattr(state, name)
        inc = h / 6 * getattr(k1, name) + h / 3 * getattr(k2, name) + h / 3 * getattr(k3, name) + h / 6 * getattr(k4, name)
        setattr(out, name, y + inc)
    return out


def action_to_command(params: Params, action: Tensor) -> Tensor:
    """Map clamped action in [-1,1] to motor command in action-limit units."""
    lo, hi = params.action_min[:, None], params.action_max[:, None]
    half = (hi - lo) / 2
    return action * half + lo + half


def step(
    params: Params,
    state: State,
    action: Tensor,
    dt: Tensor | float | None = None,
    generator: torch.Generator | None = None,
    effectiveness: Tensor | None = None,
) -> State:
    """One env step: action [N,4] raw in [-1,1] -> next state (RK4 + normalise + clamp)."""
    if dt is None:
        dt = params.dt
    a = action
    std = params.scalar("mdp.action_noise.normalized_rpm")
    if generator is not None and bool((std > 0).any()):
        noise = torch.randn(a.shape, generator=generator, device=generator.device, dtype=a.dtype).to(a.device)
        a = a + noise * std[:, None]
    a = torch.clamp(a, -1.0, 1.0)
    u_cmd = action_to_command(params, a)
    nxt = rk4_step(params, state, u_cmd, dt, effectiveness)
    nxt.orientation = quat.normalize(nxt.orientation)
    nxt.last_action = action.clone()
    nxt.rpm = torch.clamp(nxt.rpm, params.action_min[:, None], params.action_max[:, None])
    nxt.force = state.force.clone()
    nxt.torque = state.torque.clone()
    return nxt
