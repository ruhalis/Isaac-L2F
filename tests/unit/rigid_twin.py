"""A torch rigid-body integrator standing in for the simulator in CPU tests of the env core.

Integrates (p, q, v, ω) with RK4 under a wrench held constant over ``dt``, exactly what a physics
backend does with the composed wrench between two ``write_data_to_sim`` calls. It knows nothing
about motors. By default it inverts the inertia it is given, like a simulator; passing
``inv_inertia`` separately reproduces L2F's "J and J_inv as loaded" quirk exactly, which isolates
the core's frames, wrench and motor model from the inertia decision made at the Isaac boundary.
"""

from __future__ import annotations

import torch
from torch import Tensor

from isaac_l2f.dynamics import quaternion as quat


def rigid_rk4(
    position: Tensor,
    orientation: Tensor,
    linear_velocity: Tensor,
    angular_velocity: Tensor,
    force_b: Tensor,
    torque_b: Tensor,
    force_w: Tensor,
    torque_b_dist: Tensor,
    mass: Tensor,
    inertia: Tensor,
    gravity: Tensor,
    dt: float,
    inv_inertia: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One RK4 step of the free rigid body; ``inertia`` ``[N,3,3]`` is inverted here unless ``inv_inertia`` is given."""
    if inv_inertia is None:
        inv_inertia = torch.linalg.inv(inertia)
    m = mass[:, None]

    def f(p: Tensor, q: Tensor, v: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        dv = quat.rotate(q, force_b) / m + gravity + force_w / m
        jw = torch.einsum("nij,nj->ni", inertia, w)
        dw = torch.einsum("nij,nj->ni", inv_inertia, torque_b - torch.cross(w, jw, dim=-1) + torque_b_dist)
        return v, quat.derivative(q, w), dv, dw

    y = (position, orientation, linear_velocity, angular_velocity)
    k1 = f(*y)
    k2 = f(*(a + dt / 2 * b for a, b in zip(y, k1)))
    k3 = f(*(a + dt / 2 * b for a, b in zip(y, k2)))
    k4 = f(*(a + dt * b for a, b in zip(y, k3)))
    out = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4)]
    out[1] = quat.normalize(out[1])
    return out[0], out[1], out[2], out[3]


def wrench_to_world(
    orientation: Tensor, force_b: Tensor, torque_b: Tensor, force_w: Tensor, torque_b_dist: Tensor
) -> tuple[Tensor, Tensor]:
    """Rotate the core's body-frame wrench to world with the start-of-substep orientation, as Isaac Lab does once per substep.

    Returns ``(force_w_total, torque_w_total)``; the world disturbance force is added, the body disturbance torque is rotated.
    """
    return quat.rotate(orientation, force_b) + force_w, quat.rotate(orientation, torque_b + torque_b_dist)


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_integrate_body(q: Tensor, omega_body: Tensor, dt: float) -> Tensor:
    """``q ⊗ exp(ω dt / 2)`` for a body-frame angular velocity (MuJoCo ``mju_quatIntegrate``)."""
    norm = omega_body.norm(dim=-1, keepdim=True)
    angle = norm * dt
    axis = torch.where(norm > 0, omega_body / norm.clamp_min(1e-300), torch.zeros_like(omega_body))
    dq = torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)
    return quat.normalize(_quat_mul(q, dq))


def rigid_semi_implicit_euler(
    position: Tensor,
    orientation: Tensor,
    linear_velocity: Tensor,
    angular_velocity: Tensor,
    force_w: Tensor,
    torque_w: Tensor,
    mass: Tensor,
    inertia: Tensor,
    gravity: Tensor,
    dt: float,
    inv_inertia: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One semi-implicit Euler substep with the WORLD-frame wrench ``force_w``/``torque_w`` held fixed (Newton/MJWarp ``implicitfast`` without damping).

    Velocities advance first (``v += a dt``, ``ω += α dt``, the torque mapped to the body with the start orientation),
    then position and orientation advance with the new velocities.
    """
    if inv_inertia is None:
        inv_inertia = torch.linalg.inv(inertia)
    torque_b = torch.einsum("nji,nj->ni", quat.rotation_matrix(orientation), torque_w)
    jw = torch.einsum("nij,nj->ni", inertia, angular_velocity)
    alpha = torch.einsum("nij,nj->ni", inv_inertia, torque_b - torch.cross(angular_velocity, jw, dim=-1))
    v = linear_velocity + dt * (force_w / mass[:, None] + gravity)
    w = angular_velocity + dt * alpha
    return position + dt * v, _quat_integrate_body(orientation, w, dt), v, w


def rigid_rk4_world_frozen(
    position: Tensor,
    orientation: Tensor,
    linear_velocity: Tensor,
    angular_velocity: Tensor,
    force_w: Tensor,
    torque_w: Tensor,
    mass: Tensor,
    inertia: Tensor,
    gravity: Tensor,
    dt: float,
    inv_inertia: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One RK4 step with the WORLD-frame wrench held fixed across the stages (no per-stage re-rotation), as MJWarp's ``rk4`` does."""
    if inv_inertia is None:
        inv_inertia = torch.linalg.inv(inertia)
    m = mass[:, None]

    def f(p: Tensor, q: Tensor, v: Tensor, w: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        torque_b = torch.einsum("nji,nj->ni", quat.rotation_matrix(q), torque_w)
        jw = torch.einsum("nij,nj->ni", inertia, w)
        dw = torch.einsum("nij,nj->ni", inv_inertia, torque_b - torch.cross(w, jw, dim=-1))
        return v, quat.derivative(q, w), force_w / m + gravity, dw

    y = (position, orientation, linear_velocity, angular_velocity)
    k1 = f(*y)
    k2 = f(*(a + dt / 2 * b for a, b in zip(y, k1)))
    k3 = f(*(a + dt / 2 * b for a, b in zip(y, k2)))
    k4 = f(*(a + dt * b for a, b in zip(y, k3)))
    out = [a + dt / 6 * (b1 + 2 * b2 + 2 * b3 + b4) for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4)]
    out[1] = quat.normalize(out[1])
    return out[0], out[1], out[2], out[3]
