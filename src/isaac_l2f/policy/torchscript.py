"""TorchScript policy loader.

Export convention (so downstream repos can export into it): script a module with

    forward(obs: Tensor[N, 22], hidden: Tensor[N, H]) -> tuple[Tensor[N, 4], Tensor[N, H]]

register ``h0`` as a ``[H]`` buffer and set an integer attribute ``hidden_size``. A feed-forward
policy uses ``hidden_size = 0`` (``h0`` of shape ``[0]``) and returns ``hidden`` unchanged.

Isaac Lab / rsl-rl feed-forward exports (``isaaclab play`` writes ``exported/policy.pt`` with
``forward(obs) -> action``) load as hidden size 0; rsl-rl recurrent exports are rejected.
"""

from __future__ import annotations

import inspect

import torch
from torch import Tensor

from .protocol import RecurrentPolicy


def _module_attr(module: torch.nn.Module, name: str):
    try:
        return getattr(module, name)
    except (AttributeError, RuntimeError):
        return None


def _forward_arity(module: torch.nn.Module) -> int | None:
    """Number of forward inputs excluding self, or None if it cannot be determined."""
    try:
        if isinstance(module, torch.jit.ScriptModule):
            return len(module.forward.schema.arguments) - 1
        return len(inspect.signature(module.forward).parameters)
    except (AttributeError, RuntimeError, TypeError, ValueError):
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
        self._single_input = False
        if hidden_size is None and _forward_arity(module) == 1:
            if _module_attr(module, "hidden_state") is not None:
                raise ValueError(
                    "recurrent rsl-rl exports keep a batch-1 hidden state inside the module and cannot "
                    "serve N envs through the protocol; export with the protocol convention instead"
                )
            hidden_size = 0
            self._single_input = True
        if hidden_size is None:
            raise ValueError(
                "cannot determine hidden size: pass hidden_size=, or give the module a "
                "`hidden_size` attribute or an `h0` buffer"
            )
        super().__init__(hidden_size, h0)
        self.module = module.eval()

    def _forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if self._single_input:
            return self.module(obs), hidden
        action, hidden = self.module(obs, hidden)
        return action, hidden


def load_torchscript(path, map_location="cpu", **kwargs) -> TorchScriptPolicy:
    module = torch.jit.load(str(path), map_location=map_location)
    return TorchScriptPolicy(module, **kwargs)
