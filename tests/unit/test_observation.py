import torch

from isaac_l2f.dynamics import (
    load_registry,
    privileged_observation,
    sample_initial_state,
    student_observation,
)

N = 6


def _batch():
    p = load_registry("crazyflie", dtype=torch.float64).expand(N)
    g = torch.Generator().manual_seed(3)
    s = sample_initial_state(p, N, g)
    s.last_action = torch.rand(N, 4, generator=g, dtype=torch.float64) * 2 - 1
    s.rpm = torch.rand(N, 4, generator=g, dtype=torch.float64)
    pr = torch.randn(N, 3, generator=g, dtype=torch.float64)
    vr = torch.randn(N, 3, generator=g, dtype=torch.float64)
    return p, s, pr, vr


def test_shapes() -> None:
    p, s, _, _ = _batch()
    assert student_observation(p, s).shape == (N, 22)
    assert privileged_observation(p, s).shape == (N, 26)


def test_zero_reference_is_identity() -> None:
    p, s, _, _ = _batch()
    z = torch.zeros(N, 3, dtype=torch.float64)
    assert torch.equal(student_observation(p, s), student_observation(p, s, position_ref=z, velocity_ref=z))
    assert torch.equal(privileged_observation(p, s), privileged_observation(p, s, position_ref=z, velocity_ref=z))
    assert torch.equal(student_observation(p, s)[:, :3], s.position)
    assert torch.equal(student_observation(p, s)[:, 12:15], s.linear_velocity)


def test_reference_shifts_only_position_and_velocity() -> None:
    p, s, pr, vr = _batch()
    for fn in (student_observation, privileged_observation):
        base = fn(p, s)
        shifted = fn(p, s, position_ref=pr, velocity_ref=vr)
        expected = base.clone()
        expected[:, 0:3] -= pr
        expected[:, 12:15] -= vr
        assert torch.allclose(shifted, expected, atol=1e-12, rtol=0)
        untouched = [i for i in range(base.shape[1]) if not (0 <= i < 3 or 12 <= i < 15)]
        assert torch.equal(shifted[:, untouched], base[:, untouched])
