"""Batched L2F airframe/MDP parameters with JSON (de)serialisation.

All per-airframe fields carry a leading batch dim ``N``. Units: rotor positions m (body frame),
thrust coefficients N (``c0 + c1 u + c2 u^2``), torque constants m, time constants s, mass kg,
gravity m/s^2 (world), inertia kg m^2. Scalar config values (``integration``, ``mdp``,
``disturbances``, ``domain_randomization`` sections) are stored as ``[N]`` tensors keyed by dotted
path, e.g. ``"mdp.reward.scale"``. The ``trajectory`` block is parsed and round-tripped only
(Langevin trajectories are out of scope; the target is always the origin with zero velocity).
"""

from __future__ import annotations

import copy
import importlib.resources as resources
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

_TENSOR_FIELDS = (
    "rotor_positions",
    "rotor_thrust_directions",
    "rotor_torque_directions",
    "rotor_thrust_coefficients",
    "rotor_torque_constants",
    "rotor_time_constants_rising",
    "rotor_time_constants_falling",
    "mass",
    "gravity",
    "J",
    "J_inv",
    "hovering_throttle_relative",
    "action_min",
    "action_max",
)
_SCALAR_SECTIONS = ("integration", "mdp", "disturbances", "domain_randomization")

DISTILLATION_INIT: dict[str, float | bool] = dict(
    guidance=0.1,
    max_position=0.5,
    max_angle=math.pi / 2,
    max_linear_velocity=1.0,
    max_angular_velocity=1.0,
    relative_rpm=True,
    min_rpm=-1.0,
    max_rpm=0.0,
)


def _leaves(d: dict[str, Any], prefix: str) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend(_leaves(v, f"{prefix}.{k}"))
        else:
            out.append((f"{prefix}.{k}", v))
    return out


def _set_path(d: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


@dataclass
class Params:
    """Batched airframe + MDP parameters (see module docstring)."""

    rotor_positions: Tensor  # [N,4,3]
    rotor_thrust_directions: Tensor  # [N,4,3]
    rotor_torque_directions: Tensor  # [N,4,3]
    rotor_thrust_coefficients: Tensor  # [N,4,3]
    rotor_torque_constants: Tensor  # [N,4]
    rotor_time_constants_rising: Tensor  # [N,4]
    rotor_time_constants_falling: Tensor  # [N,4]
    mass: Tensor  # [N]
    gravity: Tensor  # [N,3]
    J: Tensor  # [N,3,3]
    J_inv: Tensor  # [N,3,3] (stored independently of J)
    hovering_throttle_relative: Tensor  # [N]
    action_min: Tensor  # [N]
    action_max: Tensor  # [N]
    scalars: dict[str, Tensor] = field(default_factory=dict)  # dotted path -> [N]
    bool_keys: frozenset[str] = frozenset()
    template: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)  # one per sample

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        """Batch size."""
        return int(self.mass.shape[0])

    @property
    def device(self) -> torch.device:
        return self.mass.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mass.dtype

    def scalar(self, key: str) -> Tensor:
        """Per-env scalar config value ``[N]`` by dotted path, e.g. ``"mdp.reward.scale"``."""
        return self.scalars[key]

    def flag(self, key: str) -> Tensor:
        """Per-env boolean config value ``[N]``."""
        return self.scalars[key] > 0.5

    @property
    def dt(self) -> Tensor:
        """Integration step ``[N]`` in seconds."""
        return self.scalars["integration.dt"]

    def _replace(self, **kw: Any) -> "Params":
        d = {f: getattr(self, f) for f in _TENSOR_FIELDS}
        d.update(scalars=dict(self.scalars), bool_keys=self.bool_keys, template=self.template, trajectory=self.trajectory)
        d.update(kw)
        return Params(**d)

    def _map(self, fn: Any) -> "Params":
        kw: dict[str, Any] = {f: fn(getattr(self, f)) for f in _TENSOR_FIELDS}
        kw["scalars"] = {k: fn(v) for k, v in self.scalars.items()}
        return self._replace(**kw)

    def clone(self) -> "Params":
        """Deep copy of all tensors."""
        return self._map(lambda t: t.clone())

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "Params":
        """Move/cast all tensors."""
        return self._map(lambda t: t.to(device=device, dtype=dtype))

    def select(self, idx: Tensor | slice | list[int]) -> "Params":
        """Index the batch dim (``idx`` is a slice, index list or index/bool tensor)."""
        out = self._map(lambda t: t[idx])
        if isinstance(idx, slice):
            traj = self.trajectory[idx]
        else:
            ids = torch.as_tensor(idx)
            if ids.dtype == torch.bool:
                ids = ids.nonzero().flatten()
            traj = [self.trajectory[int(i)] for i in ids.flatten()]
        return out._replace(trajectory=traj)

    def expand(self, n: int) -> "Params":
        """Repeat a single airframe (N=1) into a batch of ``n`` identical airframes."""
        if self.n != 1:
            raise ValueError("expand() requires N == 1")
        out = self._map(lambda t: t.repeat(n, *([1] * (t.dim() - 1))).clone())
        return out._replace(trajectory=[copy.deepcopy(self.trajectory[0]) for _ in range(n)])

    @staticmethod
    def stack(items: list["Params"]) -> "Params":
        """Concatenate parameter batches along the env dim."""
        first = items[0]
        kw: dict[str, Any] = {f: torch.cat([getattr(p, f) for p in items], 0) for f in _TENSOR_FIELDS}
        kw["scalars"] = {k: torch.cat([p.scalars[k] for p in items], 0) for k in first.scalars}
        kw["trajectory"] = [copy.deepcopy(t) for p in items for t in p.trajectory]
        return first._replace(**kw)

    def with_init(self, **overrides: float | bool) -> "Params":
        """Copy with ``mdp.init.*`` values replaced (e.g. ``with_init(**DISTILLATION_INIT)``)."""
        out = self.clone()
        for k, v in overrides.items():
            key = f"mdp.init.{k}"
            if key not in out.scalars:
                raise KeyError(key)
            out.scalars[key] = torch.full_like(out.scalars[key], float(v))
        return out

    # -------------------------------------------------------------------- JSON
    @staticmethod
    def from_json(
        src: dict[str, Any] | str | Path,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Params":
        """Build an N=1 ``Params`` from a JSON dict or a path to a JSON file."""
        d = json.loads(Path(src).read_text()) if not isinstance(src, dict) else copy.deepcopy(src)
        dyn = d["dynamics"]

        def t(x: Any) -> Tensor:
            return torch.tensor([x], dtype=torch.float64).to(device=device, dtype=dtype)

        scalars: dict[str, Tensor] = {}
        bools: set[str] = set()
        template: dict[str, Any] = {}
        for sec in _SCALAR_SECTIONS:
            if sec not in d:
                continue
            template[sec] = copy.deepcopy(d[sec])
            for key, v in _leaves(d[sec], sec):
                if isinstance(v, bool):
                    bools.add(key)
                scalars[key] = t(float(v))
        template["dynamics"] = {}
        for k in dyn:
            if k not in _TENSOR_FIELDS and k != "action_limit":
                template["dynamics"][k] = copy.deepcopy(dyn[k])
        return Params(
            rotor_positions=t(dyn["rotor_positions"]),
            rotor_thrust_directions=t(dyn["rotor_thrust_directions"]),
            rotor_torque_directions=t(dyn["rotor_torque_directions"]),
            rotor_thrust_coefficients=t(dyn["rotor_thrust_coefficients"]),
            rotor_torque_constants=t(dyn["rotor_torque_constants"]),
            rotor_time_constants_rising=t(dyn["rotor_time_constants_rising"]),
            rotor_time_constants_falling=t(dyn["rotor_time_constants_falling"]),
            mass=t(dyn["mass"]),
            gravity=t(dyn["gravity"]),
            J=t(dyn["J"]),
            J_inv=t(dyn["J_inv"]),
            hovering_throttle_relative=t(dyn["hovering_throttle_relative"]),
            action_min=t(dyn["action_limit"]["min"]),
            action_max=t(dyn["action_limit"]["max"]),
            scalars=scalars,
            bool_keys=frozenset(bools),
            template=template,
            trajectory=[copy.deepcopy(d.get("trajectory", {}))],
        )

    def to_json_dicts(self) -> list[dict[str, Any]]:
        """Inverse of ``from_json``: one dict per env, float64 values, same key layout."""
        out: list[dict[str, Any]] = []
        for i in range(self.n):
            d: dict[str, Any] = copy.deepcopy(self.template)
            dyn = d["dynamics"]
            for f in _TENSOR_FIELDS:
                if f in ("action_min", "action_max"):
                    continue
                dyn[f] = getattr(self, f)[i].detach().cpu().double().tolist()
            dyn["action_limit"] = {
                "min": float(self.action_min[i].double()),
                "max": float(self.action_max[i].double()),
            }
            for key, v in self.scalars.items():
                val: Any = bool(v[i] > 0.5) if key in self.bool_keys else float(v[i].double())
                _set_path(d, key, val)
            d["trajectory"] = copy.deepcopy(self.trajectory[i])
            out.append(d)
        return out


def load_registry(name: str, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> Params:
    """Load a bundled airframe (e.g. ``"crazyflie"``) as an N=1 ``Params``."""
    text = resources.files("isaac_l2f.dynamics").joinpath(f"registry/{name}.json").read_text()
    return Params.from_json(json.loads(text), device=device, dtype=dtype)
