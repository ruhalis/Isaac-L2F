"""Quaternion helpers. Convention: (w, x, y, z), body -> world, batched ``[N, 4]``."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize(q: Tensor) -> Tensor:
    """Return ``q / ||q||`` along the last dim."""
    return q / torch.linalg.norm(q, dim=-1, keepdim=True)


def rotation_matrix(q: Tensor) -> Tensor:
    """Body->world rotation matrix ``[N, 3, 3]`` (row-major layout) from ``q`` ``[N, 4]``."""
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y], -1)
    row1 = torch.stack([2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x], -1)
    row2 = torch.stack([2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y], -1)
    return torch.stack([row0, row1, row2], -2)


def rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate body-frame vectors ``v`` ``[N, 3]`` into the world frame."""
    return torch.einsum("nij,nj->ni", rotation_matrix(q), v)


def derivative(q: Tensor, omega_body: Tensor) -> Tensor:
    """Quaternion time derivative ``[N, 4]`` for body-frame angular velocity ``[N, 3]`` (rad/s)."""
    w, x, y, z = q.unbind(-1)
    wx, wy, wz = omega_body.unbind(-1)
    return 0.5 * torch.stack(
        [
            -x * wx - y * wy - z * wz,
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
        ],
        -1,
    )
