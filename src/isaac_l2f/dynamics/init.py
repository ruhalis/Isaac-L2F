"""Initial-state sampling (L2F ``sample_initial_state`` semantics)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .params import DISTILLATION_INIT, Params
from .state import State

__all__ = ["DISTILLATION_INIT", "sample_initial_state"]


def sample_initial_state(params: Params, n: int, generator: torch.Generator) -> State:
    """Sample ``n`` initial states (``params`` has N==n or N==1, broadcast).

    Note: upstream ignores ``mdp.init.max_angle`` and draws the tilt angle from U(0, 1) rad; this is
    reproduced deliberately. Disturbances: force ~ N(mean, std) (world); torque xy ~ N(mean, std),
    z ~ N(mean, std/100) (body).
    """
    dev, dt = params.device, params.dtype
    gd = generator.device

    def U(*shape: int) -> Tensor:
        return torch.rand(shape, generator=generator, device=gd, dtype=dt).to(dev)

    def G(*shape: int) -> Tensor:
        return torch.randn(shape, generator=generator, device=gd, dtype=dt).to(dev)

    def sc(key: str) -> Tensor:
        v = params.scalar(key)
        return v if v.shape[0] == n else v.expand(n)

    init = "mdp.init."
    guided = (U(n) < sc(init + "guidance"))[:, None]
    pos = (U(n, 3) * 2 - 1) * sc(init + "max_position")[:, None]
    phi = 2 * math.pi * U(n)
    cos_t = 1 - 2 * U(n)
    sin_t = torch.sqrt(torch.clamp(1 - cos_t**2, min=0.0))
    axis = torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), cos_t], -1)
    ang = U(n)  # U(0,1) rad, max_angle ignored (upstream quirk)
    quat = torch.cat([torch.cos(ang / 2)[:, None], axis * torch.sin(ang / 2)[:, None]], -1)
    vel = (U(n, 3) * 2 - 1) * sc(init + "max_linear_velocity")[:, None]
    angvel = (U(n, 3) * 2 - 1) * sc(init + "max_angular_velocity")[:, None]

    ident = torch.tensor([1.0, 0, 0, 0], device=dev, dtype=dt).expand(n, 4)
    zero3 = torch.zeros(n, 3, device=dev, dtype=dt)
    pos = torch.where(guided, zero3, pos)
    quat = torch.where(guided, ident, quat)
    vel = torch.where(guided, zero3, vel)
    angvel = torch.where(guided, zero3, angvel)

    amin = params.action_min if params.n == n else params.action_min.expand(n)
    amax = params.action_max if params.n == n else params.action_max.expand(n)
    span = amax - amin
    mn, mx = sc(init + "min_rpm"), sc(init + "max_rpm")
    rel = (params.scalar(init + "relative_rpm") if params.n == n else params.scalar(init + "relative_rpm").expand(n)) > 0.5
    low = torch.where(rel, (mn + 1) / 2 * span + amin, mn)
    high = torch.where(rel, (mx + 1) / 2 * span + amin, mx)
    rpm = low[:, None] + (high - low)[:, None] * U(n, 4)

    fm, fs = sc("disturbances.random_force.mean"), sc("disturbances.random_force.std")
    tm, ts = sc("disturbances.random_torque.mean"), sc("disturbances.random_torque.std")
    force = fm[:, None] + fs[:, None] * G(n, 3)
    tscale = torch.stack([ts, ts, ts / 100], -1)
    torque = tm[:, None] + tscale * G(n, 3)
    return State(pos, quat, vel, angvel, rpm, torch.zeros(n, 4, device=dev, dtype=dt), force, torque)
