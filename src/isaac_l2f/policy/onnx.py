"""ONNX policy loader (needs the ``onnx`` extra: ``pip install isaac-l2f[onnx]``).

Graph convention: inputs ``obs`` [N, 22] and ``hidden`` [N, H]; outputs ``action`` [N, 4] and
``hidden_out`` [N, H], with a dynamic batch axis.

Isaac Lab / rsl-rl feed-forward exports (``isaaclab play`` writes ``exported/policy.onnx`` with input
``obs``, output ``actions``, batch 1) load as hidden size 0; rsl-rl recurrent exports are rejected.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .protocol import RecurrentPolicy


def _import_ort():
    try:
        import onnxruntime
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for ONNX policies; install the extra: pip install 'isaac-l2f[onnx]'"
        ) from e
    return onnxruntime


class OnnxPolicy(RecurrentPolicy):
    def __init__(self, session, hidden_size: int | None = None, h0: Tensor | None = None):
        _import_ort()
        inputs = session.get_inputs()
        self._single_input = not any(i.name == "hidden" for i in inputs)
        self._row_loop = False
        if self._single_input:
            names = [o.name for o in session.get_outputs()]
            self._out_name = "actions" if "actions" in names else "action" if "action" in names else names[0]
            obs_shape = next(i.shape for i in inputs if i.name == "obs")
            self._row_loop = bool(obs_shape) and isinstance(obs_shape[0], int) and obs_shape[0] == 1
            if hidden_size is None:
                hidden_size = 0
        if hidden_size is None:
            shape = next(i.shape for i in session.get_inputs() if i.name == "hidden")
            if len(shape) == 2 and isinstance(shape[1], int):
                hidden_size = shape[1]
            else:
                raise ValueError("hidden size is not static in the ONNX graph; pass hidden_size=")
        super().__init__(hidden_size, h0)
        self.session = session

    def _forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if self._single_input:
            x = obs.detach().cpu().numpy().astype(np.float32, copy=False)
            if self._row_loop and x.shape[0] != 1:
                rows = [self.session.run([self._out_name], {"obs": x[i : i + 1]})[0] for i in range(x.shape[0])]
                action = np.concatenate(rows, axis=0)
            else:
                action = self.session.run([self._out_name], {"obs": x})[0]
            return torch.from_numpy(action).to(device=obs.device, dtype=obs.dtype), hidden
        feeds = {
            "obs": obs.detach().cpu().numpy().astype(np.float32, copy=False),
            "hidden": hidden.detach().cpu().numpy().astype(np.float32, copy=False),
        }
        action, hidden_out = self.session.run(["action", "hidden_out"], feeds)
        return (
            torch.from_numpy(action).to(device=obs.device, dtype=obs.dtype),
            torch.from_numpy(hidden_out).to(device=obs.device, dtype=obs.dtype),
        )


def load_onnx(path, providers=None, **kwargs) -> OnnxPolicy:
    ort = _import_ort()
    session = ort.InferenceSession(str(path), providers=providers or ["CPUExecutionProvider"])
    return OnnxPolicy(session, **kwargs)
