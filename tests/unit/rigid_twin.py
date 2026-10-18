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
