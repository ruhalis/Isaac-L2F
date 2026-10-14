"""Reward and termination (origin target, zero target velocity)."""

from __future__ import annotations

import torch
from torch import Tensor

from .params import Params
from .state import State


def terminated(params: Params, state: State) -> Tensor:
    """Bool ``[N]``: per-axis absolute thresholds on position, linear and angular velocity (if enabled)."""
    s = params.scalar
    pos = (state.position.abs() > s("mdp.termination.position_threshold")[:, None]).any(-1)
    vel = (state.linear_velocity.abs() > s("mdp.termination.linear_velocity_threshold")[:, None]).any(-1)
    ang = (state.angular_velocity.abs() > s("mdp.termination.angular_velocity_threshold")[:, None]).any(-1)
    return params.flag("mdp.termination.enabled") & (pos | vel | ang)


def reward(params: Params, state: State, action: Tensor, next_state: State) -> Tensor:
    """Reward ``[N]`` for the transition ``state --action--> next_state`` (action raw in [-1,1])."""
    s = params.scalar
    k = "mdp.reward."
    dt = params.dt
    n = torch.linalg.norm
    qz = state.orientation[:, 3]
    orient = 2 * torch.acos(torch.clamp(1 - qz.abs(), -1.0, 1.0))
    pos = n(state.position, dim=-1)
    clip = s(k + "position_clip")
    pos = torch.where(clip > 0, torch.minimum(pos, clip), pos)
    linvel = n(state.linear_velocity, dim=-1)
    angvel = n(state.angular_velocity, dim=-1)
    linacc = n(next_state.linear_velocity - state.linear_velocity, dim=-1) / dt
    angacc = n(next_state.angular_velocity - state.angular_velocity, dim=-1) / dt
    act = n((action + 1) / 2 - params.hovering_throttle_relative[:, None], dim=-1) ** 2
    dact = n(action - state.last_action, dim=-1)
    weighted = (
        s(k + "position") * pos
        + s(k + "orientation") * orient
        + s(k + "linear_velocity") * linvel
        + s(k + "angular_velocity") * angvel
        + s(k + "linear_acceleration") * linacc
        + s(k + "angular_acceleration") * angacc
        + s(k + "action") * act
        + s(k + "d_action") * dact
    )
    r = -s(k + "scale") * weighted + s(k + "constant")
    r = torch.where(params.flag(k + "non_negative"), torch.clamp(r, min=0.0), r)
    return torch.where(terminated(params, next_state), s(k + "termination_penalty"), r)
