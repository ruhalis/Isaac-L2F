"""Sim-to-sim tooling: references, recordings, the pure-L2F rollout/replay and the comparison."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from isaac_l2f.dynamics import load_registry, sample_initial_state, step, student_observation
from isaac_l2f.sim2sim import (
    Recording,
    compare,
    fault_effectiveness,
    figure_eight,
    format_report,
    one_step_errors,
    replay_in_l2f,
    rollout_l2f,
)

N = 6


class HoverPolicy:
    def reset(self, mask=None):
        pass

    def step(self, obs):
        return torch.full((obs.shape[0], 4), 2 * 0.806 - 1, dtype=obs.dtype)


def _setup(seed=0):
    params = load_registry("crazyflie").expand(N)
    g = torch.Generator().manual_seed(seed)
    return params, sample_initial_state(params, N, g)


def test_figure_eight_velocity_is_derivative():
    amp, period, height = 0.3, 2.0, 0.7
    ref = figure_eight(amp, period, height, dtype=torch.float64)
    h = 1e-4
    for t in (0.0, 0.31, 0.77, 1.5, 1.93):
        pp, _ = ref(t + h)
        pm, _ = ref(t - h)
        _, v = ref(t)
        torch.testing.assert_close((pp - pm) / (2 * h), v, rtol=0, atol=1e-6)
    p0, _ = ref(0.0)
    torch.testing.assert_close(p0, torch.tensor([0.0, 0.0, height], dtype=torch.float64))
    ts = np.linspace(0.0, period, 400)
    assert max(float(ref(float(t))[0][:2].abs().max()) for t in ts) <= amp + 1e-12


def test_fault_effectiveness():
    eff = fault_effectiveness(3, 2, 0.5)
    expected = torch.ones(3, 4)
    expected[:, 2] = 0.5
    torch.testing.assert_close(eff, expected)
    torch.testing.assert_close(fault_effectiveness(3, None, 0.5), torch.ones(3, 4))


def test_rollout_shapes_and_initial_state():
    params, init = _setup()
    T = 30
    rec = rollout_l2f(params, init, HoverPolicy(), T)
    assert rec.obs.shape == (T, N, 22)
    assert rec.actions.shape == (T, N, 4)
    assert rec.rewards.shape == (T, N)
    assert rec.terminated.shape == (T, N) and rec.terminated.dtype == bool
    for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm", "position_ref", "velocity_ref"):
        assert getattr(rec, k).shape[:2] == (T + 1, N), k
    assert rec.effectiveness.shape == (T, N, 4)
    np.testing.assert_array_equal(rec.effectiveness, np.ones((T, N, 4)))
    s0 = rec.initial_state()
    for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm"):
        torch.testing.assert_close(getattr(s0, k), getattr(init, k), rtol=0, atol=0)
    assert rec.meta["backend"] == "l2f"
    np.testing.assert_array_equal(rec.state(5).last_action.numpy(), rec.actions[4])
    assert not rec.state(0).last_action.any()


def test_rollout_matches_manual_loop():
    params, init = _setup()
    T = 25
    policy = HoverPolicy()
    rec = rollout_l2f(params, init, policy, T)
    s = init.clone()
    zeros = torch.zeros(N, 3)
    for t in range(T):
        obs = student_observation(params, s, position_ref=zeros, velocity_ref=zeros)
        s = step(params, s, policy.step(obs), effectiveness=torch.ones(N, 4))
        torch.testing.assert_close(s.position, torch.as_tensor(rec.position[t + 1]), rtol=0, atol=1e-6)


def test_recording_roundtrip(tmp_path):
    params, init = _setup()
    rec = rollout_l2f(params, init, HoverPolicy(), 10)
    path = tmp_path / "rec.npz"
    rec.save(path)
    back = Recording.load(path)
    for k in Recording._ARRAYS:
        np.testing.assert_array_equal(getattr(back, k), getattr(rec, k), err_msg=k)
    assert back.meta == rec.meta


def test_alive_masks_after_first_termination():
    params, init = _setup()
    T = 20
    rec = rollout_l2f(params, init, HoverPolicy(), T)
    term = np.zeros_like(rec.terminated)
    term[10, 2] = True
    rec = dataclasses.replace(rec, terminated=term)
    alive = rec.alive()
    assert alive.shape == (T + 1, N)
    assert alive[:11, 2].all()
    assert not alive[11:, 2].any()
    others = np.delete(alive, 2, axis=1)
    assert others.all()


def test_replay_reproduces_l2f_recording():
    params, init = _setup()
    rec = rollout_l2f(
        params, init, HoverPolicy(), 40,
        reference=figure_eight(0.3, 2.0),
        effectiveness=lambda t: fault_effectiveness(N, 1, 0.5) if t >= 20 else torch.ones(N, 4),
    )  # fmt: skip
    rep = replay_in_l2f(rec, params, HoverPolicy())
    res = compare(rep, rec)
    assert res["position_error_max"] < 1e-5
    assert res["action_error_max"] == 0
    np.testing.assert_array_equal(rep.position_ref, rec.position_ref)
    np.testing.assert_array_equal(rep.effectiveness, rec.effectiveness)


def test_one_step_errors_vanish_on_l2f_recording():
    params, init = _setup()
    rec = rollout_l2f(params, init, HoverPolicy(), 30)
    errs = one_step_errors(rec, params)
    for k, v in errs.items():
        if k != "max_omega":
            assert v < 1e-5, (k, v)
    assert errs["max_omega"] > 0


def test_compare_reports_tracking_and_termination():
    params, init_a = _setup(0)
    _, init_b = _setup(1)
    a = rollout_l2f(params, init_a, HoverPolicy(), 20)
    b = rollout_l2f(params, init_b, HoverPolicy(), 20)
    res = compare(a, b)
    for k in ("position_error_max", "position_error_rms", "tracking_error_reference_rms", "tracking_error_other_rms",
              "alive_both_at_end", "terminated_reference", "terminated_other"):  # fmt: skip
        assert k in res, k
    c = rollout_l2f(params, init_a, HoverPolicy(), 15)
    with pytest.raises(ValueError):
        compare(a, c)


def test_format_report():
    params, init = _setup()
    rec = rollout_l2f(params, init, HoverPolicy(), 10)
    res = compare(rec, rec)
    text = format_report("title", res)
    assert isinstance(text, str)
    for k in res:
        assert k in text
