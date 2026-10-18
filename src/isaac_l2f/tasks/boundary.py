"""Rules at the Isaac Lab boundary: quaternion order and the inertia the simulator must carry.

Everything on the torch side of this package uses the contract's conventions (quaternion
(w, x, y, z), FLU body frame, world z up). Isaac Lab 3.0 hands out and expects (x, y, z, w);
the environment converts with the two helpers here and nowhere else.
"""

from __future__ import annotations

import torch
from torch import Tensor

from isaac_l2f.dynamics import Params

_XYZW_TO_WXYZ = (3, 0, 1, 2)
_WXYZ_TO_XYZW = (1, 2, 3, 0)


def quat_xyzw_to_wxyz(q: Tensor) -> Tensor:
    """``[..., 4]`` (x, y, z, w) -> (w, x, y, z)."""
    return q[..., list(_XYZW_TO_WXYZ)]


def quat_wxyz_to_xyzw(q: Tensor) -> Tensor:
    """``[..., 4]`` (w, x, y, z) -> (x, y, z, w)."""
    return q[..., list(_WXYZ_TO_XYZW)]


def effective_inertia(params: Params) -> Tensor:
    """``[N, 3, 3]`` inertia tensor to author in the simulator: the inverse of ``J_inv`` as loaded.

    L2F integrates ``dω = J_inv (τ − ω × J ω)`` with ``J`` and ``J_inv`` both read from the parameter
    file, and the registry's ``J_inv`` is not the inverse of its ``J`` (upstream quirk, pinned by the
    golden tests). A simulator inverts the inertia it is given, so only one of the two can be
    reproduced: the ``J_inv τ`` term dominates the rotational response and is matched exactly by
    authoring ``inv(J_inv)``; the gyroscopic term ``ω × J ω`` then carries the few-percent
    mismatch between ``J`` and ``inv(J_inv)``.
    """
    return torch.linalg.inv(params.J_inv)


def principal_inertia(params: Params, tol: float = 1e-12) -> Tensor:
    """``[N, 3]`` diagonal of :func:`effective_inertia`; raises if any env's tensor is not diagonal.

    Both backends take a diagonal inertia in the body frame for a principal-axis-aligned body, which
    every L2F registry airframe is.
    """
    inertia = effective_inertia(params)
    diag = torch.diagonal(inertia, dim1=-2, dim2=-1)
    off = (inertia - torch.diag_embed(diag)).abs().amax(dim=(-2, -1))
    if bool((off > tol).any()):
        raise ValueError(f"effective inertia is not diagonal (max off-diagonal {off.max().item():.3e})")
    return diag
