"""Gym registration of the Isaac-L2F tasks is importable and checkable without Isaac Lab."""

from __future__ import annotations

import ast
import importlib.util
import sys
from importlib.metadata import entry_points
from pathlib import Path

import gymnasium as gym

import isaac_l2f.tasks as tasks

HOVER = "isaac_l2f.tasks.hover"


def test_hover_registered_without_isaac_lab():
    spec = gym.spec(tasks.HOVER_TASK_ID)
    assert tasks.HOVER_TASK_ID == "Isaac-L2F-Hover-Direct"  # Isaac Lab 3.0 ids carry no -v0
    assert spec.entry_point == f"{HOVER}.hover_env:HoverEnv"
    kw = spec.kwargs
    assert kw["env_cfg_entry_point"] == f"{HOVER}.hover_env_cfg:HoverEnvCfg"
    assert kw["rsl_rl_cfg_entry_point"] == f"{HOVER}.agents.rsl_rl_ppo_cfg:HoverPPORunnerCfg"
    assert kw["default_agent"] == "rsl_rl"
    for target in (spec.entry_point, kw["env_cfg_entry_point"], kw["rsl_rl_cfg_entry_point"]):
        module = target.split(":")[0]
        assert importlib.util.find_spec(module) is not None, module  # located, not imported (needs Isaac Lab)
    assert "isaaclab" not in sys.modules


def test_isaaclab_tasks_entry_point():
    eps = entry_points(group="isaaclab.tasks")
    assert any(ep.name == "isaac_l2f" and ep.value == "isaac_l2f.tasks" for ep in eps)


def test_isaac_facing_modules_compile():
    """The Linux-only modules cannot be imported here, but they must at least be valid Python."""
    root = Path(importlib.util.find_spec(HOVER).origin).parent
    for name in ("hover_env.py", "hover_env_cfg.py", "agents/rsl_rl_ppo_cfg.py"):
        src = (root / name).read_text()
        compile(src, str(root / name), "exec")


ALLOWED_IMPORTS = {
    "isaac_l2f",
    "torch",
    "numpy",
    "gymnasium",
    "h5py",
    "onnxruntime",
    "isaaclab",
    "isaaclab_newton",
    "isaaclab_rl",
    "isaaclab_tasks",
}
FORBIDDEN_IMPORTS = {"isaaclab_physx", "isaaclab_ov", "isaaclab_contrib"}


def test_package_imports_are_allowlisted():
    """Repo code imports only the stdlib and an allowlist (no PhysX/OV/contrib, no downstream packages)."""
    root = Path(importlib.util.find_spec("isaac_l2f").origin).parent
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top not in FORBIDDEN_IMPORTS, f"{path}: forbidden import {name}"
                assert top in sys.stdlib_module_names or top in ALLOWED_IMPORTS, f"{path}: unexpected import {name}"
