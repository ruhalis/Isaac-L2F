"""TorchScript policy loader.

Export convention (so downstream repos can export into it): script a module with

    forward(obs: Tensor[N, 22], hidden: Tensor[N, H]) -> tuple[Tensor[N, 4], Tensor[N, H]]

register ``h0`` as a ``[H]`` buffer and set an integer attribute ``hidden_size``. A feed-forward
policy uses ``hidden_size = 0`` (``h0`` of shape ``[0]``) and returns ``hidden`` unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .protocol import RecurrentPolicy


def _module_attr(module: torch.nn.Module, name: str):
    try:
        return getattr(module, name)
    except (AttributeError, RuntimeError):
        return None


class TorchScriptPolicy(RecurrentPolicy):
    def __init__(self, module: torch.nn.Module, hidden_size: int | None = None, h0: Tensor | None = None):
        mod_h0 = _module_attr(module, "h0")
        if h0 is None and isinstance(mod_h0, Tensor):
            h0 = mod_h0
        if hidden_size is None:
            attr = _module_attr(module, "hidden_size")
            if attr is not None:
                hidden_size = int(attr)
            elif h0 is not None:
                hidden_size = int(torch.as_tensor(h0).numel())
        if hidden_size is None:
            raise ValueError(
                "cannot determine hidden size: pass hidden_size=, or give the module a "
                "`hidden_size` attribute or an `h0` buffer"
            )
        super().__init__(hidden_size, h0)
        self.module = module.eval()

    def _forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        action, hidden = self.module(obs, hidden)
        return action, hidden


def load_torchscript(path, map_location="cpu", **kwargs) -> TorchScriptPolicy:
    module = torch.jit.load(str(path), map_location=map_location)
    return TorchScriptPolicy(module, **kwargs)
