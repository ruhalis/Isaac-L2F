"""Isaac Lab boundary: quaternion order and the inertia the simulator carries."""

from __future__ import annotations

import math

import pytest
import torch

from isaac_l2f.dynamics import quaternion
from isaac_l2f.tasks.boundary import (
    effective_inertia,
    principal_inertia,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)


def test_quaternion_order_round_trip(gen):
    q = torch.randn(5, 4, generator=gen, dtype=torch.float64)
    assert torch.equal(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q)
    assert torch.equal(quat_xyzw_to_wxyz(torch.tensor([0.0, 0.0, 0.0, 1.0])), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_xyzw_yaw_quaternion_is_rotation_about_z():
    theta = 0.7
    q_xyzw = torch.tensor([[0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)]], dtype=torch.float64)
    R = quaternion.rotation_matrix(quat_xyzw_to_wxyz(q_xyzw))
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    torch.testing.assert_close(R[0], expected, atol=1e-12, rtol=0)


def test_effective_inertia_is_inverse_of_j_inv_and_differs_from_j(cf64):
    inertia = effective_inertia(cf64)
    torch.testing.assert_close(inertia, torch.linalg.inv(cf64.J_inv))
    diag = torch.diagonal(inertia, dim1=-2, dim2=-1)
    assert torch.equal(inertia, torch.diag_embed(diag)) or (inertia - torch.diag_embed(diag)).abs().max() < 1e-15
    j_diag = torch.diagonal(cf64.J, dim1=-2, dim2=-1)
    assert ((diag - j_diag).abs() / j_diag).max() > 0.01
    p = principal_inertia(cf64)
    assert p.shape == (1, 3)
    torch.testing.assert_close(p, diag)


def test_principal_inertia_rejects_off_diagonal(cf64):
    p = cf64.clone()
    p.J_inv[0, 0, 1] = 1e3
    with pytest.raises(ValueError):
        principal_inertia(p)
