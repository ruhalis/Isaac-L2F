"""Observation builders. Targets are always origin/zero velocity (no trajectories in this package)."""

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


def _core(params: Params, state: State, generator: torch.Generator | None) -> Tensor:
    r = quat.rotation_matrix(state.orientation).reshape(-1, 9)
    g = "mdp.observation_noise."
    return torch.cat(
        [
            _noise(state.position, params.scalar(g + "position"), generator),
            _noise(r, params.scalar(g + "orientation"), generator),
            _noise(state.linear_velocity, params.scalar(g + "linear_velocity"), generator),
            _noise(state.angular_velocity, params.scalar(g + "angular_velocity"), generator),
            state.last_action,
        ],
        -1,
    )


def student_observation(params: Params, state: State, generator: torch.Generator | None = None) -> Tensor:
    """``[N,22]``: position (world) | R row-major (9) | linear vel (world) | angular vel (body) | last action (raw)."""
    return _core(params, state, generator)


def privileged_observation(params: Params, state: State, action_history: Tensor | None = None) -> Tensor:
    """``[N,26]``: noiseless student observation + normalised rotor speeds in [-1,1] (``action_history`` unused)."""
    lo, hi = params.action_min[:, None], params.action_max[:, None]
    rotor = (state.rpm - lo) / (hi - lo) * 2 - 1
    return torch.cat([_core(params, state, None), rotor], -1)
