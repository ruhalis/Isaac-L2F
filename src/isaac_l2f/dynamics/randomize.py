"""Domain randomisation of airframe parameters (L2F ``sample_parameters`` recipe)."""

from __future__ import annotations

import torch
from torch import Tensor

from .params import Params


def sample_parameters(nominal: Params, n: int, generator: torch.Generator) -> Params:
    """Sample ``n`` randomised airframes from an N=1 ``nominal`` using its ``domain_randomization`` ranges."""
    if nominal.n != 1:
        raise ValueError("nominal must have N == 1")
    dev, dt = nominal.device, nominal.dtype
    gd = generator.device

    def U(lo: Tensor | float, hi: Tensor | float, *shape: int) -> Tensor:
        r = torch.rand(shape or (n,), generator=generator, device=gd, dtype=dt).to(dev)
        return lo + (hi - lo) * r

    def dr(k: str) -> float:
        return float(nominal.scalar("domain_randomization." + k)[0])

    p = nominal.expand(n)
    m_nom = nominal.mass[0]
    g_norm = torch.linalg.norm(nominal.gravity[0])
    amax = nominal.action_max[0]
    c = nominal.rotor_thrust_coefficients[0]
    t2w_nom = (c[:, 0] + c[:, 1] * amax + c[:, 2] * amax**2).sum() / (m_nom * g_norm)

    # 1-2: thrust-to-weight and mass
    t2w = U(dr("thrust_to_weight_min"), dr("thrust_to_weight_max"))
    f_t2w = t2w / t2w_nom
    mmin, mmax = dr("mass_min"), dr("mass_max")
    size = U(mmin ** (1 / 3), mmax ** (1 / 3))
    m_new = torch.clamp(size**3, mmin, mmax)
    scale_rel = (m_new / m_nom) ** (1 / 3)
    f_mass = m_new / m_nom
    p.mass = m_new

    # 3: thrust coefficients
    p.rotor_thrust_coefficients = p.rotor_thrust_coefficients * (f_t2w * f_mass)[:, None, None]

    # 4: torque-to-inertia
    max_thrust = t2w * m_new * g_norm / 4
    max_torque = nominal.rotor_positions[0, 0, 0].abs() * (2**0.5) * max_thrust
    tti_nom = max_torque / nominal.J[0, 0, 0]
    tti = U(dr("torque_to_inertia_min"), dr("torque_to_inertia_max"))
    f_tti = tti / tti_nom

    # 5: rotor distance factor
    dev_f = dr("mass_size_deviation")
    f = -dev_f + dev_f * torch.randn((n,), generator=generator, device=gd, dtype=dt).to(dev)
    size_factor = torch.where(f < 0, 1 / (1 - f), 1 + f)
    rdf = scale_rel * size_factor

    # 6: inertia and geometry
    inertia_factor = f_tti / rdf
    eye = torch.eye(3, device=dev, dtype=dt)
    p.J = p.J - p.J * eye * (1 - 1 / inertia_factor)[:, None, None]
    p.J_inv = p.J_inv + p.J_inv * eye * (inertia_factor - 1)[:, None, None]
    p.rotor_positions = p.rotor_positions * rdf[:, None, None]
    max_dist = torch.linalg.norm(p.rotor_positions, dim=-1).amax(-1)
    p.scalars["mdp.termination.position_threshold"] = 20 * max_dist
    p.scalars["mdp.init.max_position"] = 10 * max_dist

    # 7: torque constant (shared by all rotors)
    k = U(dr("rotor_torque_constant_min"), dr("rotor_torque_constant_max"))
    p.rotor_torque_constants = k[:, None].repeat(1, 4)

    # 8: disturbance force std
    p.scalars["disturbances.random_force.std"] = U(0.0, 1.0) * (t2w - 1) * dr("disturbance_force_max") * t2w * m_new / 3
    p.scalars["disturbances.random_force.mean"] = torch.zeros(n, device=dev, dtype=dt)

    # 9: time constants (shared by all rotors)
    rise = U(dr("rotor_time_constant_rising_min"), dr("rotor_time_constant_rising_max"))
    fall = U(dr("rotor_time_constant_falling_min"), dr("rotor_time_constant_falling_max"))
    p.rotor_time_constants_rising = rise[:, None].repeat(1, 4)
    p.rotor_time_constants_falling = fall[:, None].repeat(1, 4)
    p.scalars = dict(p.scalars)
    return p
