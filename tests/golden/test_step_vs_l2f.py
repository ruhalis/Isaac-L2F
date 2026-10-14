import numpy as np
import torch

from isaac_l2f.dynamics import load_registry, sample_initial_state, sample_parameters, step
from oracle import NB, Oracle


def _rand_action(g: torch.Generator) -> torch.Tensor:
    return (torch.rand(NB, 4, generator=g) * 2 - 1).float()


def _check(ours, o: Oracle, rtol: float, atol: float) -> None:
    ref = o.read()
    for f in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm"):
        np.testing.assert_allclose(getattr(ours, f).double().numpy(), ref[f], rtol=rtol, atol=atol, err_msg=f)


def _setup(p, seed: int):
    g = torch.Generator().manual_seed(seed)
    s = sample_initial_state(p, NB, g)
    o = Oracle(p.to_json_dicts())
    o.set_state(s)
    return s, o, g


def test_single_step_crazyflie() -> None:
    p = load_registry("crazyflie").expand(NB)
    s, o, g = _setup(p, 1)
    a = _rand_action(g)
    _, dts = o.step(a)
    nxt = step(p, s, a)
    np.testing.assert_allclose(p.dt.double().numpy(), dts, rtol=1e-6)
    _check(nxt, o, 1e-4, 1e-5)


def test_rollout_20_crazyflie() -> None:
    p = load_registry("crazyflie").expand(NB)
    s, o, g = _setup(p, 2)
    for _ in range(20):
        a = _rand_action(g)
        o.step(a)
        s = step(p, s, a)
    _check(s, o, 1e-3, 1e-3)


def test_single_step_sampled_params() -> None:
    nominal = load_registry("crazyflie")
    g = torch.Generator().manual_seed(3)
    p = sample_parameters(nominal, NB, g)
    s = sample_initial_state(p, NB, g)  # uses the scaled max_position
    assert (s.position.abs() < p.scalar("mdp.termination.position_threshold")[:, None]).all()
    s.force.zero_()  # oracle's internal disturbance cannot be injected; sampled std > 0 here
    s.torque.zero_()
    o = Oracle(p.to_json_dicts())
    o.set_state(s)
    a = _rand_action(g)
    _, dts = o.step(a)
    nxt = step(p, s, a)
    np.testing.assert_allclose(p.dt.double().numpy(), dts, rtol=1e-6)
    _check(nxt, o, 1e-4, 1e-5)
