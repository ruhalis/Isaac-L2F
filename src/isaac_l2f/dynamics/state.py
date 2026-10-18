"""Batched rigid-body + motor state."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_FIELDS = ("position", "orientation", "linear_velocity", "angular_velocity", "rpm", "last_action", "force", "torque")


@dataclass
class State:
    """Quadrotor state. Position/velocity/force in WORLD frame; angular velocity/torque in BODY frame.

    ``orientation`` is (w,x,y,z) body->world; ``rpm`` is motor state in action-limit units
    (``[0,1]`` = normalised throttle); ``last_action`` is the raw (unclamped) action passed to ``step``, matching the l2f history channel; ``force`` is a
    world-frame disturbance force (N); ``torque`` a body-frame disturbance torque (N m).
    """

    position: Tensor  # [N,3]
    orientation: Tensor  # [N,4]
    linear_velocity: Tensor  # [N,3]
    angular_velocity: Tensor  # [N,3]
    rpm: Tensor  # [N,4]
    last_action: Tensor  # [N,4]
    force: Tensor  # [N,3]
    torque: Tensor  # [N,3]

    @staticmethod
    def zeros(n: int, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> "State":
        """Identity-orientation state at rest."""
        kw = dict(device=device, dtype=dtype)
        q = torch.zeros(n, 4, **kw)
        q[:, 0] = 1.0
        return State(
            torch.zeros(n, 3, **kw), q, torch.zeros(n, 3, **kw), torch.zeros(n, 3, **kw),
            torch.zeros(n, 4, **kw), torch.zeros(n, 4, **kw), torch.zeros(n, 3, **kw), torch.zeros(n, 3, **kw),
        )  # fmt: skip

    @property
    def n(self) -> int:
        return int(self.position.shape[0])

    def clone(self) -> "State":
        return State(*(getattr(self, f).clone() for f in _FIELDS))

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "State":
        return State(*(getattr(self, f).to(device=device, dtype=dtype) for f in _FIELDS))

    def __getitem__(self, idx: Tensor | slice | list[int]) -> "State":
        """Index the env dim."""
        return State(*(getattr(self, f)[idx] for f in _FIELDS))

    def where(self, mask: Tensor, other: "State") -> "State":
        """Per-env select: ``self`` where ``mask`` ``[N]`` is True else ``other``."""
        return State(*(torch.where(mask[:, None], getattr(self, f), getattr(other, f)) for f in _FIELDS))

    def scatter_(self, idx: Tensor, other: "State") -> None:
        """In-place write ``other`` into envs ``idx``."""
        for f in _FIELDS:
            getattr(self, f)[idx] = getattr(other, f)
