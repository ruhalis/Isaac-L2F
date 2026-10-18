"""Adapter for the RAPTOR foundation policy (``foundation-policy`` pip package).

Network: Dense(22->16, ReLU) -> GRU(16) with a learned initial state -> Dense(16->4, identity).
The maths is a line-by-line torch port of the package's numpy code (gate order [r, z, n],
reset gate applied to the hidden-side candidate term, ``h' = z * h + (1 - z) * n``).

Latent-advance rule: the GRU state advances exactly once per ``step()``, and ``step()`` must be
called once per 10 ms policy tick, the only rate the package documents
(``simulation.step(action) # simulation dt=10 ms``). At any other rate the recurrent state is
off-distribution, so the policy runs inside the env at the contract's 100 Hz and nowhere else.

The output is raw and unclamped (it reaches about +-3.3). The env clamps it to [-1, 1] before the
motor map and stores the raw value as the previous action in the observation.

Licence: ``foundation-policy`` 1.0.2 is MIT (PyPI ``license`` field and ``pyproject.toml``; the
sdist ships no LICENSE file; the GitHub repo's LICENSE is MIT, Copyright (c) 2025 Jonas
Eschmann). The weights come from the ``foundation-model-blob`` git submodule with no separate
notice found. This adapter reads the checkpoint from the installed package at runtime and
redistributes nothing.

``h5py`` and ``foundation_policy`` are imported lazily inside the loader only.
"""

from __future__ import annotations

import importlib.resources

import torch
from torch import Tensor

from .torchscript import TorchScriptPolicy

__all__ = ["RaptorNet", "load_raptor", "export_raptor_torchscript"]


class RaptorNet(torch.nn.Module):
    """TorchScript-able RAPTOR actor: ``forward(obs [N, 22], hidden [N, 16]) -> (action [N, 4], hidden)``."""

    def __init__(self, w0, b0, w_ih, w_hh, b_ih, b_hh, w2, b2, h0):
        super().__init__()
        self.hidden_size = 16

        def f(t):
            return torch.as_tensor(t, dtype=torch.float32).clone()

        self.register_buffer("w0", f(w0))
        self.register_buffer("b0", f(b0).reshape(-1))
        self.register_buffer("w_ih", f(w_ih))
        self.register_buffer("w_hh", f(w_hh))
        self.register_buffer("b_ih", f(b_ih).reshape(-1))
        self.register_buffer("b_hh", f(b_hh).reshape(-1))
        self.register_buffer("w2", f(w2))
        self.register_buffer("b2", f(b2).reshape(-1))
        self.register_buffer("h0", f(h0).reshape(-1))

    def forward(self, obs: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        h = 16
        x = torch.relu(obs @ self.w0.t() + self.b0)
        wi = x @ self.w_ih.t()
        wh = hidden @ self.w_hh.t()
        rz = torch.sigmoid(wh[:, : 2 * h] + wi[:, : 2 * h] + self.b_hh[: 2 * h] + self.b_ih[: 2 * h])
        r = rz[:, :h]
        z = rz[:, h:]
        n = torch.tanh(r * (wh[:, 2 * h :] + self.b_hh[2 * h :]) + wi[:, 2 * h :] + self.b_ih[2 * h :])
        new_hidden = z * hidden + (1 - z) * n
        action = new_hidden @ self.w2.t() + self.b2
        return action, new_hidden


def _default_path():
    return importlib.resources.files("foundation_policy").joinpath("blob", "checkpoint.h5")


def _read_checkpoint(path=None):
    import h5py  # lazy

    if path is None:
        path = _default_path()
    with h5py.File(str(path), "r") as f:
        a = f["actor"]["layers"]

        def g(*keys):
            node = a
            for k in keys:
                node = node[k]
            return node["parameters"][()]

        net = RaptorNet(
            g("0", "weights"), g("0", "biases"),
            g("1", "weights_input"), g("1", "weights_hidden"),
            g("1", "biases_input"), g("1", "biases_hidden"),
            g("2", "weights"), g("2", "biases"),
            g("1", "initial_hidden_state"),
        )
        example_in = torch.as_tensor(f["example"]["input"][()], dtype=torch.float32)
        example_out = torch.as_tensor(f["example"]["output"][()], dtype=torch.float32)
    return net.eval(), example_in, example_out


def _run_example(net: RaptorNet, example_in: Tensor) -> Tensor:
    """Run ``[T, B, 22]`` through the net from the learned initial state."""
    hidden = net.h0.unsqueeze(0).repeat(example_in.shape[1], 1)
    outs = []
    with torch.no_grad():
        for t in range(example_in.shape[0]):
            a, hidden = net(example_in[t], hidden)
            outs.append(a)
    return torch.stack(outs, 0)


def load_raptor(path=None, device="cpu", verify: bool = True) -> TorchScriptPolicy:
    """Load the RAPTOR checkpoint (default: the installed ``foundation_policy`` package's)."""
    net, example_in, example_out = _read_checkpoint(path)
    if verify:
        diff = (_run_example(net, example_in) - example_out).abs().mean().item()
        if not diff <= 1e-5:
            raise RuntimeError(f"RAPTOR checkpoint self-test failed: mean|diff| = {diff:.3e} > 1e-5")
    net = net.to(device)
    return TorchScriptPolicy(net, hidden_size=16, h0=net.h0)


def export_raptor_torchscript(path_out, source=None) -> None:
    """Script the RAPTOR net and save it so ``load_torchscript(path_out)`` reloads it."""
    net, _, _ = _read_checkpoint(source)
    torch.jit.script(net).save(str(path_out))
