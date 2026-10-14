"""Helper wrapping the pip ``l2f`` package (vector8) as a parity oracle."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from l2f import vector8 as v8

import l2f
from isaac_l2f.dynamics import State

NB = 8


def _quiet(d: dict[str, Any]) -> dict[str, Any]:
    d = json.loads(json.dumps(d))
    for blk in ("random_force", "random_torque"):
        d["disturbances"][blk]["std"] = 0.0
        d["disturbances"][blk]["mean"] = 0.0
    for k in d["mdp"]["observation_noise"]:
        d["mdp"]["observation_noise"][k] = 0.0
    d["mdp"]["action_noise"]["normalized_rpm"] = 0.0
    return d


class Oracle:
    """8-env L2F vector simulator loaded with the given JSON dicts."""

    def __init__(self, dicts: list[dict[str, Any]]) -> None:
        assert len(dicts) == NB
        self.dev = l2f.Device()
        self.env = v8.VectorEnvironment()
        self.rng = v8.VectorRng()
        self.params = v8.VectorParameters()
        v8.initialize_environment(self.dev, self.env)
        v8.initialize_rng(self.dev, self.rng, 0)
        v8.sample_initial_parameters(self.dev, self.env, self.params, self.rng)
        single = l2f.Environment()
        l2f.initialize_environment(self.dev, single)
        for i, d in enumerate(dicts):
            l2f.parameters_from_json(self.dev, single, json.dumps(_quiet(d)), self.params.parameters[i])
        self.state = v8.VectorState()
        v8.sample_initial_state(self.dev, self.env, self.params, self.state, self.rng)

    def set_state(self, s: State) -> None:
        for i in range(NB):
            o = self.state.states[i]
            o.position = s.position[i].double().tolist()
            o.orientation = s.orientation[i].double().tolist()
            o.linear_velocity = s.linear_velocity[i].double().tolist()
            o.angular_velocity = s.angular_velocity[i].double().tolist()
            o.rpm = s.rpm[i].double().tolist()

    def step(self, action: torch.Tensor) -> tuple[Any, np.ndarray]:
        """Step; the new state becomes the current one. Returns (previous state, dt per env)."""
        prev, nxt = self.state, v8.VectorState()
        dts = v8.step(self.dev, self.env, self.params, prev, action.numpy().astype(np.float32), nxt, self.rng)
        self.prev, self.state = prev, nxt
        return prev, np.asarray(dts)

    def read(self, vs: Any | None = None) -> dict[str, np.ndarray]:
        vs = vs or self.state
        out = {}
        for f in ("position", "orientation", "linear_velocity", "angular_velocity", "rpm"):
            out[f] = np.array([list(getattr(vs.states[i], f)) for i in range(NB)], dtype=np.float64)
        return out

    def reward(self, prev: Any, action: torch.Tensor, nxt: Any) -> np.ndarray:
        r = np.zeros(NB, np.float32)
        v8.reward(self.dev, self.env, self.params, prev, action.numpy().astype(np.float32), nxt, r, self.rng)
        return r

    def terminated(self, vs: Any | None = None) -> np.ndarray:
        t = np.zeros(NB, bool)
        v8.terminated(self.dev, self.env, self.params, vs or self.state, t, self.rng)
        return t

    def observe(self, vs: Any | None = None) -> np.ndarray:
        obs = np.zeros((NB, self.env.OBSERVATION_DIM), np.float32)
        v8.observe(self.dev, self.env, self.params, vs or self.state, obs, self.rng)
        return obs
