"""Isaac Lab boundary: quaternion order and the inertia the simulator carries."""

from __future__ import annotations

import math

import pytest
import torch

from isaac_l2f.dynamics import quaternion
from isaac_l2f.tasks.boundary import (
    bodies_in_root_frame,
    composite_inertia,
    effective_inertia,
    principal_inertia,
    quat_wxyz_to_xyzw,
    parallel_axis,
    quat_xyzw_to_wxyz,
    root_for_composite,
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


def _spd(n, scale, gen):
    a = torch.randn(n, 3, 3, generator=gen, dtype=torch.float64)
    return scale * (a @ a.transpose(-1, -2) + 3 * torch.eye(3, dtype=torch.float64))


def _random_rotations(n, gen):
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return quaternion.rotation_matrix(q / q.norm(dim=-1, keepdim=True))


def test_parallel_axis():
    d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected = 14 * torch.eye(3, dtype=torch.float64) - torch.outer(d, d)
    torch.testing.assert_close(parallel_axis(d), expected)


def test_root_for_composite_round_trip(gen):
    n, p = 8, 4
    d = torch.float64
    target_mass = 0.03 + 0.005 * torch.rand(n, generator=gen, dtype=d)
    target_inertia = _spd(n, 1e-5, gen)
    root_com = 1e-3 * torch.randn(n, 3, generator=gen, dtype=d)
    m = 1e-5 + (1e-3 - 1e-5) * torch.rand(n, p, generator=gen, dtype=d)
    c = 0.05 * (2 * torch.rand(n, p, 3, generator=gen, dtype=d) - 1)
    R = _random_rotations(n * p, gen)
    inertia_p = (R @ _spd(n * p, 1e-8, gen) @ R.transpose(-1, -2)).reshape(n, p, 3, 3)
    m_root, i_root = root_for_composite(target_mass, target_inertia, root_com, m, c, inertia_p)
    total, _, inertia = composite_inertia(
        torch.cat([m_root[:, None], m], dim=1),
        torch.cat([root_com[:, None], c], dim=1),
        torch.cat([i_root[:, None], inertia_p], dim=1),
    )
    torch.testing.assert_close(total, target_mass, atol=0, rtol=1e-10)
    torch.testing.assert_close(inertia, target_inertia, atol=0, rtol=1e-10)


def test_root_for_composite_no_others_returns_target(gen):
    target_inertia = _spd(3, 1e-5, gen)
    target_mass = torch.full((3,), 0.03, dtype=torch.float64)
    m_root, i_root = root_for_composite(
        target_mass,
        target_inertia,
        torch.zeros(3, 3, dtype=torch.float64),
        torch.zeros(3, 0, dtype=torch.float64),
        torch.zeros(3, 0, 3, dtype=torch.float64),
        torch.zeros(3, 0, 3, 3, dtype=torch.float64),
    )
    torch.testing.assert_close(m_root, target_mass)
    torch.testing.assert_close(i_root, target_inertia)


def test_root_for_composite_infeasible_raises():
    d = torch.float64
    with pytest.raises(ValueError):
        root_for_composite(
            torch.tensor([0.03], dtype=d),
            1e-5 * torch.eye(3, dtype=d)[None],
            torch.zeros(1, 3, dtype=d),
            torch.tensor([[0.02, 0.02]], dtype=d),
            torch.zeros(1, 2, 3, dtype=d),
            torch.zeros(1, 2, 3, 3, dtype=d),
        )


def test_bodies_in_root_frame_hand_case():
    d = torch.float64
    s = math.sqrt(0.5)
    # root yawed 90 degrees at (1, 2, 3); body 1 unrotated, 0.1 m ahead of it in world x, CoM 0.01 m up its link z
    pos = torch.tensor([[[1.0, 2.0, 3.0], [1.1, 2.0, 3.0]]], dtype=d)
    quat_wxyz = torch.tensor([[[s, 0.0, 0.0, s], [1.0, 0.0, 0.0, 0.0]]], dtype=d)
    com = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]], dtype=d)
    inertia = torch.zeros(1, 2, 3, 3, dtype=d)
    inertia[0, 0] = torch.eye(3, dtype=d)
    inertia[0, 1] = torch.diag(torch.tensor([1e-6, 2e-6, 3e-6], dtype=d))
    r, i = bodies_in_root_frame(pos, quat_wxyz, com, inertia, root=0)
    torch.testing.assert_close(r[0, 0], torch.zeros(3, dtype=d), atol=1e-15, rtol=0)
    torch.testing.assert_close(r[0, 1], torch.tensor([0.0, -0.1, 0.01], dtype=d), atol=1e-12, rtol=0)
    torch.testing.assert_close(i[0, 0], torch.eye(3, dtype=d), atol=1e-12, rtol=0)
    torch.testing.assert_close(i[0, 1], torch.diag(torch.tensor([2e-6, 1e-6, 3e-6], dtype=d)), atol=1e-18, rtol=0)


def test_composite_inertia_single_body(gen):
    d = torch.float64
    m = torch.rand(2, 1, generator=gen, dtype=d) + 0.1
    c = torch.randn(2, 1, 3, generator=gen, dtype=d)
    inertia = _spd(2, 1e-5, gen)[:, None]
    total, com, comp = composite_inertia(m, c, inertia)
    torch.testing.assert_close(total, m[:, 0])
    torch.testing.assert_close(com, c[:, 0])
    torch.testing.assert_close(comp, inertia[:, 0])
