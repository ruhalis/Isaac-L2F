import h5py
import importlib.resources
import numpy as np
import pytest
import torch

from isaac_l2f.dynamics import load_registry, sample_initial_state, step, student_observation, terminated
from isaac_l2f.dynamics.quaternion import rotation_matrix
from isaac_l2f.policy import export_raptor_torchscript, load_raptor, load_torchscript

foundation_policy = pytest.importorskip("foundation_policy")


def random_obs(T, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T * B, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    rot = rotation_matrix(q).reshape(T, B, 9)
    pos = (torch.rand(T, B, 3, generator=g) * 2 - 1) * 0.5
    vel = torch.rand(T, B, 3, generator=g) * 2 - 1
    ang = torch.rand(T, B, 3, generator=g) * 2 - 1
    act = torch.rand(T, B, 4, generator=g) * 2 - 1
    return torch.cat([pos, rot, vel, ang, act], -1).float()


def numpy_run(obs):
    ref = foundation_policy.Raptor()
    ref.reset()
    return np.stack([ref.evaluate_step(o.numpy()) for o in obs], 0)


def test_matches_numpy_package():
    obs = random_obs(20, 5)
    expected = numpy_run(obs)
    policy = load_raptor()
    ours = torch.stack([policy.step(o) for o in obs]).numpy()
    diff = np.abs(ours - expected).max()
    print("max abs diff vs numpy:", diff)
    assert diff < 1e-5


def test_checkpoint_self_test():
    load_raptor(verify=True)
    path = importlib.resources.files("foundation_policy").joinpath("blob", "checkpoint.h5")
    with h5py.File(str(path), "r") as f:
        x = f["example"]["input"][()]
        y = f["example"]["output"][()]
    policy = load_raptor()
    policy.reset()
    outs = np.stack([policy.step(torch.as_tensor(x[t])).numpy() for t in range(x.shape[0])], 0)
    assert np.abs(outs - y).mean() < 1e-6


def test_masked_reset_reseeds_h0():
    obs = random_obs(15, 4, seed=1)
    policy = load_raptor()
    for t in range(10):
        policy.step(obs[t])
    policy.reset(mask=torch.tensor([True, False, True, False]))
    ours = torch.stack([policy.step(obs[t]) for t in range(10, 15)]).numpy()
    full = numpy_run(obs)[10:15]
    fresh = numpy_run(obs[10:15])
    for i in (0, 2):
        np.testing.assert_allclose(ours[:, i], fresh[:, i], atol=1e-5)
    for i in (1, 3):
        np.testing.assert_allclose(ours[:, i], full[:, i], atol=1e-5)


def test_torchscript_export_roundtrip(tmp_path):
    p = tmp_path / "raptor.pt"
    export_raptor_torchscript(p)
    a, b = load_torchscript(p), load_raptor()
    assert a.hidden_size == 16
    assert torch.equal(a.h0, b.h0)
    obs = random_obs(5, 3, seed=2)
    for o in obs:
        assert torch.allclose(a.step(o), b.step(o), atol=1e-6)


def hover_rollout(steps=500, init=None):
    params = load_registry("crazyflie", dtype=torch.float32)
    if init:
        params = params.with_init(**init)
    params = params.expand(16)
    g = torch.Generator().manual_seed(0)
    state = sample_initial_state(params, 16, g)
    policy = load_raptor()
    policy.reset()
    term, actions = [], []
    for _ in range(steps):
        obs = student_observation(params, state)
        action = policy.step(obs)
        state = step(params, state, action)
        term.append(terminated(params, state))
        actions.append(action)
    return state, torch.stack(term), torch.stack(actions)


@pytest.fixture(scope="module")
def rollout():
    return hover_rollout()


def first_terminated(term):
    return [(term[:, i].nonzero()[0].item() if term[:, i].any() else None) for i in range(term.shape[1])]


def test_raptor_hovers_from_gentle_init():
    """RAPTOR recovers and hovers from a reduced L2F init box; the full box is covered by the survival test below."""
    # Measured steady-state hover offset: all envs settle near z = -0.052 m on the crazyflie registry,
    # so the 0.1 m threshold includes it.
    state, term, _ = hover_rollout(init=dict(max_position=0.3, max_linear_velocity=0.5, max_angular_velocity=0.5))
    pos = state.position.norm(dim=-1).max().item()
    vel = state.linear_velocity.norm(dim=-1).max().item()
    print("gentle init: max |pos|", pos, "max |vel|", vel, "first terminated", first_terminated(term))
    assert not term[100:].any()
    assert pos < 0.1
    assert vel < 0.1


def test_raptor_recovers_most_full_init_states(rollout):
    state, term, _ = rollout
    first = first_terminated(term)
    ever = term.any(0)
    print("full init: terminated envs", int(ever.sum()), "first terminated step per env", first)
    assert (~ever).sum().item() >= 14
    ok = ~ever
    assert state.position[ok].norm(dim=-1).max().item() < 0.1
    assert state.linear_velocity[ok].norm(dim=-1).max().item() < 0.1


def test_raw_output_exceeds_unit_range(rollout):
    _, _, actions = rollout
    m = actions[:20].abs().max().item()
    print("max |action| in first 20 steps:", m)
    assert m > 1.0
