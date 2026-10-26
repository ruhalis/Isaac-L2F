import subprocess
import sys

import pytest
import torch
from torch import nn

from isaac_l2f.policy import OnnxPolicy, Policy, TorchScriptPolicy, load_onnx, load_torchscript

H = 8


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = H
        self.inp = nn.Linear(22, H)
        self.cell = nn.GRUCell(H, H)
        self.out = nn.Linear(H, 4)
        self.register_buffer("h0", torch.linspace(-0.5, 0.7, H))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.inp(obs))
        h = self.cell(x, hidden)
        return self.out(h), h


class FF(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 0
        self.lin = nn.Linear(22, 4)
        self.register_buffer("h0", torch.zeros(0))

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lin(obs), hidden


@pytest.fixture
def net():
    torch.manual_seed(0)
    return Net().eval()


def mk(net, **kw):
    return TorchScriptPolicy(torch.jit.script(net), **kw)


def test_protocol_and_hidden_carried(net):
    p = mk(net)
    assert isinstance(p, Policy)
    obs = torch.randn(3, 22)
    a1 = p.step(obs)
    assert a1.shape == (3, 4)
    a2 = p.step(obs)
    assert not torch.allclose(a1, a2)
    fresh = mk(net)
    assert torch.allclose(fresh.step(obs), a1)


def test_masked_reset(net):
    p = mk(net)
    obs = torch.randn(3, 22)
    p.step(obs)
    p.step(obs)
    before = p.hidden.clone()
    p.reset(torch.tensor([True, False, False]))
    assert torch.equal(p.hidden[0], p.h0)
    assert torch.equal(p.hidden[1:], before[1:])
    a = p.step(obs)
    fresh = mk(net)
    assert torch.allclose(a[0], fresh.step(obs)[0], atol=1e-6)


def test_reset_none_and_before_step(net):
    p = mk(net)
    p.reset()
    p.reset(torch.tensor([True, False]))
    obs = torch.randn(3, 22)
    p.step(obs)
    p.step(obs)
    p.reset(None)
    assert torch.equal(p.hidden, p.h0.expand(3, H))


def test_hidden_size_inference_and_override(net):
    assert mk(net).hidden_size == H
    p = mk(net, hidden_size=H, h0=torch.zeros(H))
    assert torch.equal(p.h0, torch.zeros(H))
    with pytest.raises(ValueError):
        mk(net, hidden_size=H + 1)  # h0 length mismatch
    with pytest.raises(ValueError):
        TorchScriptPolicy(nn.Bilinear(2, 2, 2))  # two inputs, no hidden size info


def test_feedforward_hidden_size_zero():
    p = TorchScriptPolicy(torch.jit.script(FF().eval()))
    assert p.hidden_size == 0
    a = p.step(torch.randn(5, 22))
    assert a.shape == (5, 4)
    assert p.hidden.shape == (5, 0)
    p.reset(torch.ones(5, dtype=torch.bool))


def test_realloc_on_batch_change(net):
    p = mk(net)
    p.step(torch.randn(3, 22))
    assert p.step(torch.randn(5, 22)).shape == (5, 4)
    assert p.hidden.shape == (5, H)


def test_save_load_roundtrip(net, tmp_path):
    path = tmp_path / "pol.pt"
    torch.jit.save(torch.jit.script(net), str(path))
    p, q = load_torchscript(path), mk(net)
    obs = torch.randn(4, 22)
    for _ in range(3):
        assert torch.allclose(p.step(obs), q.step(obs), atol=1e-6)


def test_onnx_matches_torchscript(net, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    path = tmp_path / "pol.onnx"
    torch.onnx.export(
        net, (torch.randn(2, 22), torch.zeros(2, H)), str(path),
        input_names=["obs", "hidden"], output_names=["action", "hidden_out"],
        dynamic_axes={k: {0: "batch"} for k in ("obs", "hidden", "action", "hidden_out")},
        dynamo=False,
    )
    o = load_onnx(path, hidden_size=H, h0=net.h0)
    t = mk(net)
    torch.manual_seed(1)
    for i in range(5):
        if i == 3:
            m = torch.tensor([False, True, False])
            o.reset(m)
            t.reset(m)
        obs = torch.randn(3, 22)
        assert torch.allclose(o.step(obs), t.step(obs), atol=1e-5)
        assert torch.allclose(o.hidden, t.hidden, atol=1e-5)
    assert isinstance(o, OnnxPolicy) and isinstance(o, Policy)


class FFSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(22, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class RecSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(22, 4)
        self.register_buffer("hidden_state", torch.zeros(1, 1, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_isaaclab_feedforward_torchscript(tmp_path):
    torch.manual_seed(0)
    m = FFSingle().eval()
    path = tmp_path / "policy.pt"
    torch.jit.save(torch.jit.script(m), str(path))
    p = load_torchscript(path)
    assert p.hidden_size == 0
    obs = torch.randn(8, 22)
    a = p.step(obs)
    assert torch.equal(a, m(obs))
    p.reset(torch.tensor([True] + [False] * 7))
    assert torch.equal(p.step(obs), a)


def test_recurrent_rslrl_torchscript_rejected(tmp_path):
    path = tmp_path / "policy.pt"
    torch.jit.save(torch.jit.script(RecSingle().eval()), str(path))
    with pytest.raises(ValueError, match="recurrent"):
        load_torchscript(path)


def _rslrl_mlp(norm):
    pytest.importorskip("tensordict")
    rsl_models = pytest.importorskip("rsl_rl.models")
    from tensordict import TensorDict

    torch.manual_seed(0)
    model = rsl_models.MLPModel(
        TensorDict({"policy": torch.zeros(8, 22)}, batch_size=[8]),
        {"policy": ["policy"]},
        "policy",
        4,
        hidden_dims=[64, 64],
        activation="elu",
        obs_normalization=norm,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    if norm:
        model.update_normalization(TensorDict({"policy": torch.randn(64, 22) * 3 + 1}, batch_size=[64]))
    return model.eval()


@pytest.mark.parametrize("norm", [False, True])
def test_rslrl_mlp_jit_export(tmp_path, norm):
    model = _rslrl_mlp(norm)
    path = tmp_path / "policy.pt"
    jit = torch.jit.script(model.as_jit())
    jit.save(str(path))
    p = load_torchscript(path)
    assert p.hidden_size == 0
    obs = torch.randn(8, 22)
    with torch.no_grad():
        expected = model.as_jit()(obs)
    assert torch.allclose(p.step(obs), expected, atol=1e-6)


@pytest.mark.parametrize("norm", [False, True])
def test_rslrl_mlp_onnx_export(tmp_path, norm):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    model = _rslrl_mlp(norm)
    path = tmp_path / "policy.onnx"
    onnx_model = model.as_onnx(False)
    onnx_model.to("cpu")
    onnx_model.eval()
    torch.onnx.export(
        onnx_model, onnx_model.get_dummy_inputs(), str(path), export_params=True, opset_version=18,
        input_names=onnx_model.input_names, output_names=onnx_model.output_names,
    )
    o = load_onnx(path)
    assert o.hidden_size == 0
    obs = torch.randn(8, 22)
    with torch.no_grad():
        expected = model.as_jit()(obs)
    assert torch.allclose(o.step(obs), expected, atol=1e-5)


def test_import_does_not_load_onnxruntime():
    code = "import sys, isaac_l2f.policy; sys.exit(int('onnxruntime' in sys.modules))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # not `uv run`: it re-syncs the venv
    assert r.returncode == 0, r.stderr
