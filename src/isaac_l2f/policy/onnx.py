"""ONNX policy loader (needs the ``onnx`` extra: ``pip install isaac-l2f[onnx]``).

Graph convention: inputs ``obs`` [N, 22] and ``hidden`` [N, H]; outputs ``action`` [N, 4] and
``hidden_out`` [N, H], with a dynamic batch axis.
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
        if hidden_size is None:
            shape = next(i.shape for i in session.get_inputs() if i.name == "hidden")
            if len(shape) == 2 and isinstance(shape[1], int):
                hidden_size = shape[1]
            else:
                raise ValueError("hidden size is not static in the ONNX graph; pass hidden_size=")
        super().__init__(hidden_size, h0)
        self.session = session

    def _forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
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
