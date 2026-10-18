"""Sim-to-sim tooling: recordings, the pure-L2F reference rollout and the comparison.

A :class:`Recording` is the same whether it comes from an Isaac Lab backend
(``scripts/fly_raptor.py``) or from pure L2F (:func:`rollout_l2f`): contract observations,
raw actions, rewards, terminations, the contract state after every step (index 0 is the state
after reset), the reference and the rotor effectiveness that were active, plus metadata.
:func:`compare` reports the closed-loop position error of a recording against a pure-L2F
rollout of the same policy from the same initial states, and :func:`one_step_errors` the
teacher-forced one-step error of each recorded transition against ``isaac_l2f.dynamics.step``,
which is the number that is not amplified by closed-loop chaos.

The figure-eight here is the M1 gate's moving reference (``v_ref`` equal to the trajectory
velocity); trajectory references for training live downstream.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from isaac_l2f.dynamics import Params, State, reward, step, student_observation, terminated
from isaac_l2f.policy import Policy

ReferenceFn = Callable[[float], tuple[Tensor, Tensor]]
"""``t (s) -> (p_ref [3], v_ref [3])`` in the world frame."""


# ---------------------------------------------------------------------------- references
def hover_reference(device=None, dtype=torch.float32) -> ReferenceFn:
    """Origin, zero velocity (the neutral default)."""
    zero = torch.zeros(3, device=device, dtype=dtype)

    def ref(t: float) -> tuple[Tensor, Tensor]:
        return zero, zero

    return ref


def figure_eight(amplitude: float = 0.5, period: float = 4.0, height: float = 0.0, device=None, dtype=torch.float32) -> ReferenceFn:
    """Lemniscate in the x-y plane: ``x = A sin(wt)``, ``y = (A/2) sin(2wt)``, ``z = height``; ``v_ref`` is its derivative."""
    w = 2 * math.pi / period

    def ref(t: float) -> tuple[Tensor, Tensor]:
        p = torch.tensor([amplitude * math.sin(w * t), 0.5 * amplitude * math.sin(2 * w * t), height], device=device, dtype=dtype)
        v = torch.tensor([amplitude * w * math.cos(w * t), amplitude * w * math.cos(2 * w * t), 0.0], device=device, dtype=dtype)
        return p, v

    return ref


def fault_effectiveness(num_envs: int, rotor: int | None, effectiveness: float, device=None, dtype=torch.float32) -> Tensor:
    """``[N, 4]`` ones with ``rotor`` (0 FR, 1 BR, 2 BL, 3 FL) scaled to ``effectiveness``; all ones when ``rotor`` is None."""
    eff = torch.ones(num_envs, 4, device=device, dtype=dtype)
    if rotor is not None:
        eff[:, rotor] = effectiveness
    return eff


# ---------------------------------------------------------------------------- recording
@dataclass
class Recording:
    """One closed-loop rollout; arrays are numpy, time-major, contract conventions (wxyz, FLU, env-local position)."""

    obs: np.ndarray  # [T, N, 22]
    actions: np.ndarray  # [T, N, 4] raw
    rewards: np.ndarray  # [T, N]
    terminated: np.ndarray  # [T, N] bool
    position: np.ndarray  # [T+1, N, 3]
    orientation: np.ndarray  # [T+1, N, 4]
    linear_velocity: np.ndarray  # [T+1, N, 3]
    angular_velocity: np.ndarray  # [T+1, N, 3]
    rpm: np.ndarray  # [T+1, N, 4]
    position_ref: np.ndarray  # [T+1, N, 3]
    velocity_ref: np.ndarray  # [T+1, N, 3]
    effectiveness: np.ndarray  # [T, N, 4] active during step t
    meta: dict[str, Any] = field(default_factory=dict)

    _ARRAYS = ("obs", "actions", "rewards", "terminated", "position", "orientation", "linear_velocity",
               "angular_velocity", "rpm", "position_ref", "velocity_ref", "effectiveness")  # fmt: skip

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.actions.shape[1])

    def initial_state(self, device=None, dtype=torch.float32) -> State:
        """Contract state after reset (index 0), with zero previous action and zero disturbance."""
        n = self.num_envs
        kw = dict(device=device, dtype=dtype)
        return State(
            torch.as_tensor(self.position[0], **kw),
            torch.as_tensor(self.orientation[0], **kw),
            torch.as_tensor(self.linear_velocity[0], **kw),
            torch.as_tensor(self.angular_velocity[0], **kw),
            torch.as_tensor(self.rpm[0], **kw),
            torch.zeros(n, 4, **kw),
            torch.zeros(n, 3, **kw),
            torch.zeros(n, 3, **kw),
        )

    def state(self, t: int, device=None, dtype=torch.float32) -> State:
        """Contract state at index ``t`` (after step ``t - 1``), previous action = ``actions[t - 1]``."""
        kw = dict(device=device, dtype=dtype)
        n = self.num_envs
        last = torch.as_tensor(self.actions[t - 1], **kw) if t > 0 else torch.zeros(n, 4, **kw)
        return State(
            torch.as_tensor(self.position[t], **kw),
            torch.as_tensor(self.orientation[t], **kw),
            torch.as_tensor(self.linear_velocity[t], **kw),
            torch.as_tensor(self.angular_velocity[t], **kw),
            torch.as_tensor(self.rpm[t], **kw),
            last,
            torch.zeros(n, 3, **kw),
            torch.zeros(n, 3, **kw),
        )

    def alive(self) -> np.ndarray:
        """``[T+1, N]`` bool: env never terminated before index t (index 0 always True)."""
        term = np.cumsum(self.terminated.astype(np.int64), axis=0) > 0
        return np.concatenate([np.zeros((1, self.num_envs), dtype=bool), term], axis=0) == 0

    def save(self, path: str | Path) -> None:
        arrays = {k: getattr(self, k) for k in self._ARRAYS}
        np.savez_compressed(path, meta=np.array(repr(self.meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        import ast

        with np.load(path) as z:
            arrays = {k: z[k] for k in cls._ARRAYS}
            meta = ast.literal_eval(str(z["meta"]))
        return cls(**arrays, meta=meta)


class Recorder:
    """Collects a :class:`Recording` step by step from torch tensors (any device)."""

    def __init__(self, meta: dict[str, Any] | None = None):
        self._buf: dict[str, list[np.ndarray]] = {k: [] for k in Recording._ARRAYS}
        self.meta = dict(meta or {})

    @staticmethod
    def _np(x: Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().copy()

    def state(self, s: State, p_ref: Tensor, v_ref: Tensor) -> None:
        """Record the contract state and the reference active at this tick (call once after reset, then after every step)."""
        for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm"):
            self._buf[k].append(self._np(getattr(s, k)))
        self._buf["position_ref"].append(self._np(p_ref))
        self._buf["velocity_ref"].append(self._np(v_ref))

    def step(self, obs: Tensor, action: Tensor, rew: Tensor, term: Tensor, effectiveness: Tensor) -> None:
        self._buf["obs"].append(self._np(obs))
        self._buf["actions"].append(self._np(action))
        self._buf["rewards"].append(self._np(rew))
        self._buf["terminated"].append(self._np(term).astype(bool))
        self._buf["effectiveness"].append(self._np(effectiveness))

    def finish(self) -> Recording:
        arrays = {k: np.stack(v) for k, v in self._buf.items()}
        rec = Recording(**arrays, meta=dict(self.meta))
        rec.meta.setdefault("steps", rec.steps)
        rec.meta.setdefault("num_envs", rec.num_envs)
        return rec


# ---------------------------------------------------------------------------- pure L2F rollout
def rollout_l2f(
    params: Params,
    init: State,
    policy: Policy,
    steps: int,
    *,
    reference: ReferenceFn | None = None,
    effectiveness: Tensor | Callable[[int], Tensor] | None = None,
    dt: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Recording:
    """Closed-loop pure-L2F rollout of ``policy`` from ``init`` (``params`` has ``N == init.n`` rows).

    ``reference(t)`` sets ``(p_ref, v_ref)`` for the observation at time ``t``; ``effectiveness`` is
    a fixed ``[N, 4]`` tensor or a function of the step index. Terminated envs are not reset (the
    comparison masks them); the recording therefore holds exactly one episode per env.
    """
    n = init.n
    dev, dtype = init.position.device, init.position.dtype
    if dt is None:
        dt = float(params.dt[0])
    reference = reference or hover_reference(dev, dtype)
    eff_fn = effectiveness if callable(effectiveness) else (lambda t, e=effectiveness: e)
    policy.reset()
    rec = Recorder({"backend": "l2f", "dt": dt, "hold_dt": dt, **(meta or {})})
    s = init.clone()
    p_ref, v_ref = reference(0.0)
    rec.state(s, p_ref.expand(n, 3), v_ref.expand(n, 3))
    t0 = time.perf_counter()
    for t in range(steps):
        obs = student_observation(params, s, position_ref=p_ref.expand(n, 3), velocity_ref=v_ref.expand(n, 3))
        action = policy.step(obs)
        eff = eff_fn(t)
        eff = torch.ones(n, 4, device=dev, dtype=dtype) if eff is None else eff.to(device=dev, dtype=dtype)
        nxt = step(params, s, action, effectiveness=eff)
        rec.step(obs, action, reward(params, s, action, nxt), terminated(params, nxt), eff)
        s = nxt
        p_ref, v_ref = reference((t + 1) * dt)
        rec.state(s, p_ref.expand(n, 3), v_ref.expand(n, 3))
    rec.meta["wall_s_per_step"] = (time.perf_counter() - t0) / max(steps, 1)
    return rec.finish()


def replay_in_l2f(rec: Recording, params: Params, policy: Policy, device=None, dtype=torch.float32) -> Recording:
    """Pure-L2F rollout of ``policy`` from ``rec``'s initial states with ``rec``'s reference and effectiveness schedule."""
    dev = device
    p_ref = torch.as_tensor(rec.position_ref, device=dev, dtype=dtype)
    v_ref = torch.as_tensor(rec.velocity_ref, device=dev, dtype=dtype)
    eff = torch.as_tensor(rec.effectiveness, device=dev, dtype=dtype)
    dt = float(rec.meta.get("dt", float(params.dt[0])))
    idx = lambda t: min(int(round(t / dt)), rec.steps)  # noqa: E731

    def reference(t: float) -> tuple[Tensor, Tensor]:
        i = idx(t)
        return p_ref[i, 0], v_ref[i, 0]

    if not np.allclose(rec.position_ref, rec.position_ref[:, :1]) or not np.allclose(rec.velocity_ref, rec.velocity_ref[:, :1]):
        raise ValueError("replay_in_l2f supports one shared reference across envs")
    return rollout_l2f(
        params, rec.initial_state(dev, dtype), policy, rec.steps,
        reference=reference, effectiveness=lambda t: eff[t], dt=dt, meta={"replay_of": rec.meta.get("backend", "?")},
    )  # fmt: skip


# ---------------------------------------------------------------------------- comparison
def _masked_max_rms(err: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """``err`` ``[T, N]`` with ``mask`` ``[T, N]``; returns (max, rms) over the masked entries."""
    vals = err[mask]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.max()), float(np.sqrt(np.mean(vals**2)))


def compare(reference: Recording, other: Recording) -> dict[str, Any]:
    """Closed-loop comparison of ``other`` against ``reference`` (same init, policy, schedules).

    Position error is masked to envs alive in both recordings; tracking error is each recording's
    distance to its own reference. Returns a flat dict of numbers.
    """
    if reference.steps != other.steps or reference.num_envs != other.num_envs:
        raise ValueError("recordings differ in steps or num_envs")
    alive = reference.alive() & other.alive()
    pos_err = np.linalg.norm(other.position - reference.position, axis=-1)  # [T+1, N]
    vel_err = np.linalg.norm(other.linear_velocity - reference.linear_velocity, axis=-1)
    act_err = np.abs(other.actions - reference.actions).max(axis=-1)  # [T, N]
    out: dict[str, Any] = {
        "steps": reference.steps,
        "num_envs": reference.num_envs,
        "alive_both_at_end": int(alive[-1].sum()),
        "terminated_reference": int((~reference.alive()[-1]).sum()),
        "terminated_other": int((~other.alive()[-1]).sum()),
    }
    out["position_error_max"], out["position_error_rms"] = _masked_max_rms(pos_err, alive)
    out["position_error_final_max"] = float(pos_err[-1][alive[-1]].max()) if alive[-1].any() else float("nan")
    out["velocity_error_max"], out["velocity_error_rms"] = _masked_max_rms(vel_err, alive)
    out["action_error_max"], out["action_error_rms"] = _masked_max_rms(act_err, alive[:-1])
    for name, rec in (("reference", reference), ("other", other)):
        track = np.linalg.norm(rec.position - rec.position_ref, axis=-1)
        out[f"tracking_error_{name}_max"], out[f"tracking_error_{name}_rms"] = _masked_max_rms(track, rec.alive())
    return out


def one_step_errors(rec: Recording, params: Params, device=None, dtype=torch.float64) -> dict[str, float]:
    """Teacher-forced one-step error of every recorded transition against ``isaac_l2f.dynamics.step``.

    For each step ``t`` the recorded state ``t`` and action ``t`` (and the active effectiveness)
    are stepped once in pure L2F and compared with the recorded state ``t + 1``; envs are masked
    after their first termination. The maximum over steps and envs is returned per field, plus
    the max |ω| seen, which sets the scale of the gyroscopic-term mismatch.
    """
    p = params.to(device=device, dtype=dtype)
    alive = rec.alive()
    errs = {k: 0.0 for k in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm")}
    max_omega = 0.0
    for t in range(rec.steps):
        mask = torch.as_tensor(alive[t + 1], device=device)
        if not bool(mask.any()):
            break
        s = rec.state(t, device, dtype)
        nxt_rec = rec.state(t + 1, device, dtype)
        nxt = step(p, s, torch.as_tensor(rec.actions[t], device=device, dtype=dtype),
                   effectiveness=torch.as_tensor(rec.effectiveness[t], device=device, dtype=dtype))  # fmt: skip
        for k in errs:
            d = (getattr(nxt, k) - getattr(nxt_rec, k)).abs().amax(dim=-1)[mask]
            errs[k] = max(errs[k], float(d.max()))
        max_omega = max(max_omega, float(s.angular_velocity.norm(dim=-1)[mask].max()))
    errs["max_omega"] = max_omega
    return errs


def format_report(title: str, numbers: dict[str, Any]) -> str:
    width = max(len(k) for k in numbers)
    lines = [title] + [f"  {k:<{width}}  {v:.3e}" if isinstance(v, float) else f"  {k:<{width}}  {v}" for k, v in numbers.items()]
    return "\n".join(lines)


__all__ = [
    "Recorder", "Recording", "ReferenceFn", "compare", "fault_effectiveness", "figure_eight", "format_report",
    "hover_reference", "one_step_errors", "replay_in_l2f", "rollout_l2f",
]  # fmt: skip
