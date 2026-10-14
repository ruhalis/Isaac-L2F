import torch

from isaac_l2f.dynamics import quaternion as quat


def _rand_q(n: int, g: torch.Generator) -> torch.Tensor:
    return quat.normalize(torch.randn(n, 4, generator=g, dtype=torch.float64))


def test_identity() -> None:
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    assert torch.allclose(quat.rotation_matrix(q)[0], torch.eye(3, dtype=torch.float64))
    v = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    assert torch.allclose(quat.rotate(q, v), v)
    assert torch.allclose(quat.derivative(q, torch.zeros(1, 3, dtype=torch.float64)), torch.zeros(1, 4, dtype=torch.float64))


def test_matrix_matches_rotate_and_orthonormal(gen: torch.Generator) -> None:
    q = _rand_q(16, gen)
    v = torch.randn(16, 3, generator=gen, dtype=torch.float64)
    r = quat.rotation_matrix(q)
    assert torch.allclose(quat.rotate(q, v), (r @ v[..., None])[..., 0])
    assert torch.allclose(r @ r.transpose(-1, -2), torch.eye(3, dtype=torch.float64).expand(16, 3, 3), atol=1e-12)
    assert torch.allclose(torch.linalg.det(r), torch.ones(16, dtype=torch.float64))


def test_z_rotation_90deg() -> None:
    c = 2**-0.5
    q = torch.tensor([[c, 0, 0, c]], dtype=torch.float64)
    out = quat.rotate(q, torch.tensor([[1.0, 0, 0]], dtype=torch.float64))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64), atol=1e-12)


def test_constant_omega_integration_matches_composition(gen: torch.Generator) -> None:
    """Body-rate integration with small dt equals R(q0) @ expm([w]x t)."""
    q0 = _rand_q(4, gen)
    w = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    t, steps = 0.1, 2000
    h = t / steps
    q = q0.clone()
    for _ in range(steps):
        k1 = quat.derivative(q, w)
        k2 = quat.derivative(q + h / 2 * k1, w)
        k3 = quat.derivative(q + h / 2 * k2, w)
        k4 = quat.derivative(q + h * k3, w)
        q = q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    z = torch.zeros(4, dtype=torch.float64)
    skew = torch.stack(
        [torch.stack([z, -w[:, 2], w[:, 1]], -1), torch.stack([w[:, 2], z, -w[:, 0]], -1), torch.stack([-w[:, 1], w[:, 0], z], -1)],
        -2,
    )
    expected = quat.rotation_matrix(q0) @ torch.linalg.matrix_exp(skew * t)
    assert torch.allclose(quat.rotation_matrix(quat.normalize(q)), expected, atol=1e-9)
