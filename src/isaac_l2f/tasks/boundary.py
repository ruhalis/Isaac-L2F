"""Rules at the Isaac Lab boundary: quaternion order and the inertia the simulator must carry.

Everything on the torch side of this package uses the contract's conventions (quaternion
(w, x, y, z), FLU body frame, world z up). Isaac Lab 3.0 hands out and expects (x, y, z, w);
the environment converts with the two helpers here and nowhere else.
"""

from __future__ import annotations

import torch
from torch import Tensor

from isaac_l2f.dynamics import Params
from isaac_l2f.dynamics.quaternion import rotation_matrix

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

    Newton takes a diagonal inertia in the body frame for a principal-axis-aligned body, which
    every L2F registry airframe is.
    """
    inertia = effective_inertia(params)
    diag = torch.diagonal(inertia, dim1=-2, dim2=-1)
    off = (inertia - torch.diag_embed(diag)).abs().amax(dim=(-2, -1))
    if bool((off > tol).any()):
        raise ValueError(f"effective inertia is not diagonal (max off-diagonal {off.max().item():.3e})")
    return diag


def parallel_axis(offset: Tensor) -> Tensor:
    """``[..., 3]`` -> ``[..., 3, 3]``: ``|d|^2 I - d d^T``, the parallel-axis term per unit mass."""
    eye = torch.eye(3, dtype=offset.dtype, device=offset.device)
    return (offset * offset).sum(dim=-1)[..., None, None] * eye - offset[..., :, None] * offset[..., None, :]


def bodies_in_root_frame(
    link_pos_w: Tensor, link_quat_wxyz: Tensor, com_pos_b: Tensor, inertia_b: Tensor, root: int
) -> tuple[Tensor, Tensor]:
    """Each body's CoM position ``[N, B, 3]`` and inertia about its own CoM ``[N, B, 3, 3]`` in the root link frame.

    Inputs: link positions ``[N, B, 3]`` and quaternions ``[N, B, 4]`` (w, x, y, z) in the world, CoM positions
    ``[N, B, 3]`` and CoM inertias ``[N, B, 3, 3]`` in each body's link frame. The origin is the root link origin.
    """
    R = rotation_matrix(link_quat_wxyz)
    R_root = R[:, root]
    com_w = link_pos_w + torch.einsum("nbij,nbj->nbi", R, com_pos_b)
    r = torch.einsum("nji,nbj->nbi", R_root, com_w - link_pos_w[:, root : root + 1])
    R_rel = torch.einsum("nji,nbjk->nbik", R_root, R)
    inertia = R_rel @ inertia_b @ R_rel.transpose(-1, -2)
    return r, inertia


def composite_inertia(masses: Tensor, coms: Tensor, inertias: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``[N, B]``, ``[N, B, 3]``, ``[N, B, 3, 3]`` -> total mass ``[N]``, composite CoM ``[N, 3]``, inertia about it ``[N, 3, 3]``."""
    total = masses.sum(dim=1)
    com = (masses[..., None] * coms).sum(dim=1) / total[:, None]
    inertia = (inertias + masses[..., None, None] * parallel_axis(coms - com[:, None])).sum(dim=1)
    return total, com, inertia


def root_for_composite(
    target_mass: Tensor,
    target_inertia: Tensor,
    root_com: Tensor,
    other_masses: Tensor,
    other_coms: Tensor,
    other_inertias: Tensor,
) -> tuple[Tensor, Tensor]:
    """Root mass ``[N]`` and root inertia about the root CoM ``[N, 3, 3]`` making the composite match the target.

    ``target_mass`` ``[N]`` and ``target_inertia`` ``[N, 3, 3]`` (about the composite CoM) are the L2F body;
    ``root_com`` ``[N, 3]`` and the others (``[N, P]``, ``[N, P, 3]``, ``[N, P, 3, 3]``, P may be 0) share one frame.
    The root CoM is not moved, so the composite CoM ends up at ``root_com + sum m_p (r_p - root_com) / target_mass``,
    a residual offset (about 3e-5 m for the Crazyflie). Raises ValueError if the root mass is not positive or the
    root inertia is not positive definite.
    """
    m_root = target_mass - other_masses.sum(dim=1)
    if bool((m_root <= 0).any()):
        raise ValueError(f"others outweigh the target mass (root mass {m_root.min().item():.3e} kg)")
    com = root_com + (other_masses[..., None] * (other_coms - root_com[:, None])).sum(dim=1) / target_mass[:, None]
    inertia = (
        target_inertia
        - m_root[:, None, None] * parallel_axis(root_com - com)
        - (other_inertias + other_masses[..., None, None] * parallel_axis(other_coms - com[:, None])).sum(dim=1)
    )
    inertia = 0.5 * (inertia + inertia.transpose(-1, -2))
    if bool((torch.linalg.eigvalsh(inertia)[:, 0] <= 0).any()):
        raise ValueError("root inertia is not positive definite")
    return m_root, inertia
