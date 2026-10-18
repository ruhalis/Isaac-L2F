"""Policy protocol for the Isaac-L2F contract (obs [N, 22] -> action [N, 4], 100 Hz)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Policy(Protocol):
    """Batched 100 Hz policy on the Isaac-L2F contract.

    obs [N, 22] -> action [N, 4] (raw; the env clamps). The policy owns its recurrent state:
    reset(mask) re-seeds it for the envs where mask is True (all envs when mask is None);
    step() advances it by one policy tick. Stateless policies implement reset as a no-op.
    """

    def reset(self, mask: Tensor | None = None) -> None: ...

    def step(self, obs: Tensor) -> Tensor: ...


class RecurrentPolicy:
    """Helper base: owns a ``[N, H]`` hidden state with per-env reset.

    Subclasses implement ``_forward(obs, hidden) -> (action, hidden)``. ``hidden`` is allocated
    lazily on the first ``step`` from ``h0``; ``hidden_size == 0`` gives a ``[N, 0]`` state so
    stateless policies reuse this class.
    """

    def __init__(self, hidden_size: int = 0, h0: Tensor | None = None):
        self.hidden_size = int(hidden_size)
        if h0 is None:
            h0 = torch.zeros(self.hidden_size)
        h0 = torch.as_tensor(h0).detach().reshape(-1)
        if h0.numel() != self.hidden_size:
            raise ValueError(f"h0 has {h0.numel()} elements, expected hidden_size={self.hidden_size}")
        self.h0: Tensor = h0
        self.hidden: Tensor | None = None

    def _forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _alloc(self, obs: Tensor) -> Tensor:
        h0 = self.h0.to(device=obs.device, dtype=obs.dtype)
        return h0.unsqueeze(0).repeat(obs.shape[0], 1).contiguous()

    def reset(self, mask: Tensor | None = None) -> None:
        if self.hidden is None:
            return
        if mask is None:
            self.hidden = self._alloc_like(self.hidden)
            return
        mask = mask.to(device=self.hidden.device, dtype=torch.bool)
        h0 = self.h0.to(device=self.hidden.device, dtype=self.hidden.dtype)
        self.hidden[mask] = h0

    def _alloc_like(self, hidden: Tensor) -> Tensor:
        h0 = self.h0.to(device=hidden.device, dtype=hidden.dtype)
        return h0.unsqueeze(0).repeat(hidden.shape[0], 1).contiguous()

    def step(self, obs: Tensor) -> Tensor:
        if (
            self.hidden is None
            or self.hidden.shape[0] != obs.shape[0]
            or self.hidden.device != obs.device
            or self.hidden.dtype != obs.dtype
        ):
            self.hidden = self._alloc(obs)
        with torch.no_grad():
            action, hidden = self._forward(obs, self.hidden)
        self.hidden = hidden
        return action
