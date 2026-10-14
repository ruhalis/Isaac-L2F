import numpy as np
import torch

from isaac_l2f.dynamics import load_registry, reward, sample_initial_state, step, student_observation, terminated
from oracle import NB, Oracle


def _batch():
    p = load_registry("crazyflie").expand(NB)
    g = torch.Generator().manual_seed(7)
    s = sample_initial_state(p, NB, g)
    # push some envs beyond thresholds
    s.position[0, 0] = 1.5
    s.position[1, 2] = -1.2
    s.linear_velocity[2, 1] = 2.5
    s.angular_velocity[3, 0] = -40.0
    return p, s, g


def test_observation_matches() -> None:
    p, s, g = _batch()
    s.last_action = torch.rand(NB, 4, generator=g) * 2 - 1
    o = Oracle(p.to_json_dicts())
    o.set_state(s)
    ref = o.observe()
    ours = student_observation(p, s)
    assert ours.shape == (NB, 22)
    np.testing.assert_allclose(ours[:, :18].numpy(), ref[:, :18], rtol=1e-5, atol=1e-6)


def test_terminated_matches() -> None:
    p, s, _ = _batch()
    o = Oracle(p.to_json_dicts())
    o.set_state(s)
    ref = o.terminated()
    ours = terminated(p, s).numpy()
    assert ref.any() and not ref.all()
    np.testing.assert_array_equal(ours, ref)


def test_reward_two_step_protocol() -> None:
    p, s, g = _batch()
    s.position[:] = s.position.clamp(-0.9, 0.9)  # start non-terminated
    s.linear_velocity.clamp_(-1.5, 1.5)
    s.angular_velocity.clamp_(-10, 10)
    o = Oracle(p.to_json_dicts())
    o.set_state(s)
    a1 = (torch.rand(NB, 4, generator=g) * 2 - 1).float()
    o.step(a1)
    s1 = step(p, s, a1)
    # force a terminated transition in env 0 on both sims
    o.set_state(_force(s1))
    s1 = _force(s1)
    a2 = (torch.rand(NB, 4, generator=g) * 2 - 1).float()
    prev, _ = o.step(a2)
    s2 = step(p, s1, a2)
    ref = o.reward(prev, a2, o.state)
    ours = reward(p, s1, a2, s2).numpy()
    assert ref[0] == -100.0 and ours[0] == -100.0
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-4)


def _force(s):
    s = s.clone()
    s.position[0, 0] = 0.999
    s.linear_velocity[0, 0] = 5.0
    return s
