"""Observation builders.

The first three dims are position relative to a reference position and dims 12:15 are the world-frame
linear velocity relative to a reference velocity. The reference defaults to the origin / zero velocity.
Trajectory references and any position clamp live downstream of this package.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import quaternion as quat
from .params import Params
from .state import State


def _noise(x: Tensor, std: Tensor, generator: torch.Generator | None) -> Tensor:
    if generator is None or not bool((std > 0).any()):
        return x
    n = torch.randn(x.shape, generator=generator, device=generator.device, dtype=x.dtype).to(x.device)
    return x + n * std[:, None]


def _core(
    params: Params,
    state: State,
    generator: torch.Generator | None,
    position_ref: Tensor | None = None,
    velocity_ref: Tensor | None = None,
) -> Tensor:
    r = quat.rotation_matrix(state.orientation).reshape(-1, 9)
    pos = state.position if position_ref is None else state.position - position_ref
    vel = state.linear_velocity if velocity_ref is None else state.linear_velocity - velocity_ref
    g = "mdp.observation_noise."
    return torch.cat(
        [
            _noise(pos, params.scalar(g + "position"), generator),
            _noise(r, params.scalar(g + "orientation"), generator),
            _noise(vel, params.scalar(g + "linear_velocity"), generator),
            _noise(state.angular_velocity, params.scalar(g + "angular_velocity"), generator),
            state.last_action,
        ],
        -1,
    )


def student_observation(
    params: Params,
    state: State,
    generator: torch.Generator | None = None,
    *,
    position_ref: Tensor | None = None,
    velocity_ref: Tensor | None = None,
) -> Tensor:
    """``[N,22]``: position - ``position_ref`` (3) | R row-major (9) | linear vel (world) - ``velocity_ref`` (3) |
    angular vel (body) (3) | last action (raw, as given to ``step``; 4).

    ``position_ref`` / ``velocity_ref`` are ``[N,3]`` world-frame tensors; ``None`` means origin / zero velocity.
    Noise is applied to the relative quantities.
    """
    return _core(params, state, generator, position_ref, velocity_ref)


def privileged_observation(
    params: Params,
    state: State,
    action_history: Tensor | None = None,
    *,
    position_ref: Tensor | None = None,
    velocity_ref: Tensor | None = None,
) -> Tensor:
    """``[N,26]``: noiseless student observation (reference-relative as above) + normalised rotor speeds in
    [-1,1] (``action_history`` unused)."""
    lo, hi = params.action_min[:, None], params.action_max[:, None]
    rotor = (state.rpm - lo) / (hi - lo) * 2 - 1
    return torch.cat([_core(params, state, None, position_ref, velocity_ref), rotor], -1)
