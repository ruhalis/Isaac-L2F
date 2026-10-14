"""Parity sweep of isaac_l2f.dynamics against the l2f pip oracle on many randomised airframes.

Usage: uv run python scripts/parity_sweep.py
Reports max abs error per state field for one step and a 20-step rollout, observation and termination parity.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "golden"))
import numpy as np, torch
from isaac_l2f.dynamics import load_registry, sample_parameters, sample_initial_state, step, student_observation, terminated
from oracle import NB, Oracle
F = ("position", "orientation", "linear_velocity", "angular_velocity", "rpm")
nominal = load_registry("crazyflie")
g = torch.Generator().manual_seed(0)
P = sample_parameters(nominal, 128, g)
e1 = {f: 0.0 for f in F}; e20 = {f: 0.0 for f in F}; ok1 = ok20 = True
eo = 0.0; okobs = True; okterm = True; nterm = 0
ra = lambda: (torch.rand(NB, 4, generator=g) * 2 - 1).float()
for b in range(16):
    p = P.select(slice(b * 8, b * 8 + 8))
    s = sample_initial_state(p, NB, g); s.force.zero_(); s.torque.zero_()
    # single step
    o = Oracle(p.to_json_dicts()); o.set_state(s)
    a = ra(); o.step(a); nx = step(p, s, a); ref = o.read()
    for f in F:
        d = getattr(nx, f).double().numpy(); e1[f] = max(e1[f], np.abs(d - ref[f]).max())
        ok1 &= np.allclose(d, ref[f], rtol=1e-4, atol=1e-5)
    # rollout
    o = Oracle(p.to_json_dicts()); o.set_state(s); c = s
    for _ in range(20):
        a = ra(); o.step(a); c = step(p, c, a)
    ref = o.read()
    for f in F:
        d = getattr(c, f).double().numpy(); e20[f] = max(e20[f], np.abs(d - ref[f]).max())
        ok20 &= np.allclose(d, ref[f], rtol=1e-3, atol=1e-3)
    # obs / terminated, with some beyond thresholds
    t = s.clone()
    t.position[0, 0] = 1.5; t.position[1, 2] = -1.2; t.linear_velocity[2, 1] = 2.5; t.angular_velocity[3, 0] = -40.0
    t.last_action = torch.rand(NB, 4, generator=g) * 2 - 1
    o = Oracle(p.to_json_dicts()); o.set_state(t)
    ro = o.observe(); ours = student_observation(p, t).numpy()
    eo = max(eo, np.abs(ours[:, :18] - ro[:, :18]).max())
    okobs &= np.allclose(ours[:, :18], ro[:, :18], rtol=1e-5, atol=1e-6)
    rt = o.terminated(); ot = terminated(p, t).numpy(); nterm += int(rt.sum())
    okterm &= bool((rt == ot).all())
print("single-step max abs err:", e1, "ALL PASS:", ok1)
print("20-step rollout max abs err:", e20, "ALL PASS:", ok20)
print("obs max abs err:", eo, "PASS:", okobs)
print("terminated exact:", okterm, "n_terminated_oracle:", nterm)
