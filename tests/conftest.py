"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from isaac_l2f.dynamics import Params, load_registry  # noqa: E402


@pytest.fixture
def gen() -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(1234)
    return g


@pytest.fixture
def cf64() -> Params:
    return load_registry("crazyflie", dtype=torch.float64)


@pytest.fixture
def cf32() -> Params:
    return load_registry("crazyflie", dtype=torch.float32)
