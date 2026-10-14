import torch

from isaac_l2f.dynamics import Params, rotor_wrench, sample_parameters


def test_dr_ranges(cf64) -> None:
    g = torch.Generator().manual_seed(0)
    n = 256
    p = sample_parameters(cf64, n, g)
    amax = p.action_max[:, None]
    c = p.rotor_thrust_coefficients
    thrust_max = (c[..., 0] + c[..., 1] * amax + c[..., 2] * amax**2)
    t2w = thrust_max.sum(1) / (p.mass * 9.81)
    assert t2w.min() >= 1.5 - 1e-6 and t2w.max() <= 5 + 1e-6
    assert p.mass.min() >= 0.02 and p.mass.max() <= 5.0
    tti = p.rotor_positions[:, 0, 0].abs() * 2**0.5 * (thrust_max.sum(1) / 4) / p.J[:, 0, 0]
    assert tti.min() >= 40 - 1e-6 and tti.max() <= 1200 + 1e-6
    assert p.rotor_time_constants_rising.min() >= 0.03 and p.rotor_time_constants_rising.max() <= 0.1
    assert p.rotor_time_constants_falling.min() >= 0.03 and p.rotor_time_constants_falling.max() <= 0.3
    assert p.rotor_torque_constants.min() >= 0.005 and p.rotor_torque_constants.max() <= 0.05
    assert (p.rotor_time_constants_rising == p.rotor_time_constants_rising[:, :1]).all()
    for i in range(3):
        ratio = (p.J[:, i, i] / cf64.J[0, i, i]) / (cf64.J_inv[0, i, i] / p.J_inv[:, i, i])
        assert torch.allclose(ratio, torch.ones_like(ratio), rtol=1e-9)
    assert (p.scalar("disturbances.random_force.std") >= 0).all()
    maxd = torch.linalg.norm(p.rotor_positions, dim=-1).amax(-1)
    assert torch.allclose(p.scalar("mdp.termination.position_threshold"), 20 * maxd)
    f, _ = rotor_wrench(p, torch.ones(n, 4, dtype=torch.float64))
    assert torch.isfinite(f).all()


def test_json_roundtrip_exact(cf64) -> None:
    g = torch.Generator().manual_seed(0)
    p = sample_parameters(cf64, 16, g)
    dicts = p.to_json_dicts()
    q = Params.stack([Params.from_json(d, dtype=torch.float64) for d in dicts])
    assert q.to_json_dicts() == dicts
    for f in ("rotor_positions", "rotor_thrust_coefficients", "J", "J_inv", "mass", "rotor_torque_constants"):
        assert torch.equal(getattr(p, f), getattr(q, f))
    for k, v in p.scalars.items():
        assert torch.equal(v, q.scalars[k]), k


def test_registry_roundtrip(cf64) -> None:
    d = cf64.to_json_dicts()[0]
    assert d["dynamics"]["action_limit"] == {"min": 0.0, "max": 1.0}
    assert d["mdp"]["termination"]["enabled"] is True
    assert d["trajectory"]["MIXTURE_N"] == 2
