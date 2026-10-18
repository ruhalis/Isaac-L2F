"""Simulator-free core of the Isaac-L2F environments.

Everything between the simulator's rigid-body readout and the wrench written back lives here: the
motor lag advanced at the physics rate, the rotor wrench, the 22-dim observation, reward,
termination, the L2F reset distribution and the three per-env hooks (reference, motor
effectiveness, disturbance wrench). Pure torch, batched ``[N, ...]``, device-agnostic; the Isaac
Lab environment is a thin boundary around it and this module has CPU tests without a simulator.

Conventions are the contract's (see ``isaac_l2f.dynamics``): position and linear velocity in the
world frame with position relative to the env origin; quaternion (w, x, y, z) body->world; angular
velocity and torque in the FLU body frame. The simulator applies gravity and integrates the rigid
body with the mass ``params.mass`` and the inertia ``boundary.effective_inertia(params)``; the
wrench returned here never includes gravity.

One policy step at ``params.dt`` (100 Hz) spans ``decimation`` physics substeps, in the order the
Isaac Lab ``DirectRLEnv`` loop calls them::

    core.pre_physics_step(action)                       # raw [N, 4]; stores it as the previous action
    for _ in range(decimation):
        force_b, torque_b, force_w, torque_b_dist = core.substep(physics_dt)   # write, then step the sim
    core.sync(position, quat_wxyz, lin_vel_w, ang_vel_b)                      # readout after the step
    reward, done = core.reward(), core.terminated()
    sampled = core.reset(env_ids)                       # write ``sampled`` to the simulator
    core.sync(...); obs = core.observe()                # envs just reset keep their sampled state
"""

from __future__ import annotations

import torch
from torch import Tensor

from isaac_l2f.dynamics import (
    Params,
    State,
    action_to_command,
    motor_step_closed_form,
    privileged_observation,
    rotor_thrust_mean,
    sample_initial_state,
    student_observation,
    wrench_from_thrust,
)
from isaac_l2f.dynamics import quaternion as quat
from isaac_l2f.dynamics import reward as l2f_reward
from isaac_l2f.dynamics import terminated as l2f_terminated

_KINEMATIC = ("position", "orientation", "linear_velocity", "angular_velocity")


def assert_rates(physics_dt: float, decimation: int, params: Params, tol: float = 1e-9) -> None:
    """Check ``physics_dt * decimation == params.dt`` (the 100 Hz policy step) for every env."""
    dt_max, dt_min = float(params.dt.max()), float(params.dt.min())
    if dt_max - dt_min > tol:
        raise ValueError("params.dt must be the same for all envs")
    if abs(physics_dt * decimation - dt_max) > tol:
        raise ValueError(
            f"physics_dt * decimation = {physics_dt} * {decimation} = {physics_dt * decimation:.6g} s, "
            f"expected the policy step params.dt = {dt_max:.6g} s"
        )


class L2FCore:
    """Batched L2F env logic around an external rigid-body integrator.

    ``params`` has ``N == num_envs`` rows or one row that is broadcast. ``state`` is the contract state
    at the current policy tick: kinematics from the last :meth:`sync`, ``rpm`` the motor state
    advanced by :meth:`substep`, ``last_action`` the raw action of the current step once
    :meth:`pre_physics_step` ran, ``force``/``torque`` the disturbance wrench.

    Hooks, all per env with neutral defaults: ``position_ref``/``velocity_ref`` ``[N, 3]`` (origin,
    zero) subtracted in the observation only; ``effectiveness`` ``[N, 4]`` (ones) scaling each rotor's
    thrust and with it its reaction torque and arm moment; the disturbance wrench (world-frame force,
    body-frame torque) which :meth:`reset` re-draws per episode from the L2F init distribution
    (zero for the registry airframes) and :meth:`set_disturbance_wrench` overrides.
    """

    def __init__(
        self,
        params: Params,
        num_envs: int,
        *,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ):
        if params.n == 1 and num_envs != 1:
            params = params.expand(num_envs)
        elif params.n != num_envs:
            raise ValueError(f"params has {params.n} rows, expected 1 or num_envs={num_envs}")
        self.params = params
        self.num_envs = int(num_envs)
        dev, dtype = params.device, params.dtype
        if generator is None:
            generator = torch.Generator(device=dev)
            generator.manual_seed(seed)
        self.generator = generator
        self.state = State.zeros(self.num_envs, device=dev, dtype=dtype)
        self.action = torch.zeros(self.num_envs, 4, device=dev, dtype=dtype)
        self.u_cmd = self.state.rpm.clone()
        self.position_ref = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)
        self.velocity_ref = torch.zeros(self.num_envs, 3, device=dev, dtype=dtype)
        self.effectiveness = torch.ones(self.num_envs, 4, device=dev, dtype=dtype)
        self._prev: State | None = None
        self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)

    # ------------------------------------------------------------------ properties
    @property
    def device(self) -> torch.device:
        return self.params.device

    @property
    def dtype(self) -> torch.dtype:
        return self.params.dtype

    @property
    def body_mass(self) -> Tensor:
        """``[N]`` mass the simulator must carry."""
        return self.params.mass

    # ------------------------------------------------------------------ hooks
    def _write(self, dst: Tensor, value: Tensor | float | None, env_ids: Tensor | None) -> None:
        if value is None:
            return
        v = torch.as_tensor(value, device=dst.device, dtype=dst.dtype)
        if env_ids is None:
            dst.copy_(v.expand_as(dst))
        else:
            dst[env_ids] = v.expand(int(env_ids.numel()), dst.shape[1]) if v.dim() < 2 else v

    def set_reference(
        self,
        position: Tensor | None = None,
        velocity: Tensor | None = None,
        env_ids: Tensor | None = None,
    ) -> None:
        """World-frame reference ``[k, 3]`` (or ``[3]``) for ``env_ids`` (all when None); None leaves a part unchanged."""
        self._write(self.position_ref, position, env_ids)
        self._write(self.velocity_ref, velocity, env_ids)

    def set_motor_effectiveness(self, effectiveness: Tensor | float, env_ids: Tensor | None = None) -> None:
        """Per-rotor thrust scale ``[k, 4]`` (or ``[4]``, or a scalar) for ``env_ids`` (all when None)."""
        self._write(self.effectiveness, effectiveness, env_ids)

    def set_disturbance_wrench(
        self,
        force: Tensor | None = None,
        torque: Tensor | None = None,
        env_ids: Tensor | None = None,
    ) -> None:
        """World-frame force (N) and body-frame torque (N m) ``[k, 3]`` (or ``[3]``); None leaves a part unchanged."""
        self._write(self.state.force, force, env_ids)
        self._write(self.state.torque, torque, env_ids)

    # ------------------------------------------------------------------ episode
    def reset(self, env_ids: Tensor | None = None) -> State:
        """Draw the L2F initial state for ``env_ids`` (all when None) and return it (``k`` rows).

        Position is relative to the env origin, quaternion (w, x, y, z): the caller writes it to the
        simulator. The motor state, a zero previous action and the per-episode disturbance draw are
        stored here. Until the next :meth:`sync` these envs keep the sampled kinematics, because the
        simulator has not integrated them yet. References and effectiveness are not touched.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = int(env_ids.numel())
        sampled = sample_initial_state(self.params.select(env_ids), k, self.generator)
        self.state.scatter_(env_ids, sampled)
        self.action[env_ids] = 0.0
        self.u_cmd[env_ids] = sampled.rpm
        self._fresh[env_ids] = True
        return sampled

    # ------------------------------------------------------------------ tick
    def sync(self, position: Tensor, orientation: Tensor, linear_velocity: Tensor, angular_velocity: Tensor) -> None:
        """Take the simulator readout in contract convention (``[N, 3]``, ``[N, 4]`` (w, x, y, z), ``[N, 3]`` world, ``[N, 3]`` body).

        Envs reset since the last sync keep their sampled kinematics; the quaternion is normalised,
        as L2F normalises its own after every step.
        """
        new = (position, quat.normalize(orientation), linear_velocity, angular_velocity)
        fresh = self._fresh[:, None] if bool(self._fresh.any()) else None
        for name, val in zip(_KINEMATIC, new):
            old = getattr(self.state, name)
            val = val.to(device=old.device, dtype=old.dtype)
            setattr(self.state, name, val.clone() if fresh is None else torch.where(fresh, old, val))
        self._fresh.zero_()

    def observe(self) -> Tensor:
        """``[N, 22]`` student observation at the current tick (reference-relative, raw previous action)."""
        return student_observation(
            self.params, self.state, self.generator, position_ref=self.position_ref, velocity_ref=self.velocity_ref
        )

    def observe_privileged(self) -> Tensor:
        """``[N, 26]`` noiseless observation plus normalised rotor speeds, for a privileged critic."""
        return privileged_observation(
            self.params, self.state, position_ref=self.position_ref, velocity_ref=self.velocity_ref
        )

    def pre_physics_step(self, action: Tensor) -> None:
        """Take the raw policy action ``[N, 4]``; it becomes the previous action of the next observation.

        The motor command is ``action_to_command`` of the clamped (and, if the airframe says so,
        noised) action, exactly as ``isaac_l2f.dynamics.step`` does it.
        """
        a = action.to(device=self.device, dtype=self.dtype)
        self._prev = self.state.clone()
        self.action = a.clone()
        self.state.last_action = a.clone()
        std = self.params.scalar("mdp.action_noise.normalized_rpm")
        if bool((std > 0).any()):
            noise = torch.randn(a.shape, generator=self.generator, device=self.generator.device, dtype=self.dtype)
            a = a + noise.to(self.device) * std[:, None]
        self.u_cmd = action_to_command(self.params, torch.clamp(a, -1.0, 1.0))

    def substep(self, dt: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance the motor lag by one physics substep ``dt`` and return the wrench to hold over it.

        Returns the body-frame rotor force ``[N, 3]`` and torque about the COM ``[N, 3]``, the
        world-frame disturbance force ``[N, 3]`` and the body-frame disturbance torque ``[N, 3]``.
        The lag is the exact first-order response with the rising/falling time constant, and the
        rotor wrench is built from the time-averaged thrust over the hold
        (:func:`isaac_l2f.dynamics.rotor_thrust_mean`), so a simulator that holds it constant for
        ``dt`` applies the exact impulse of the lagging thrust. ``dt`` is the physics step when the
        backend calls back every substep, or the whole policy step when it runs the decimation
        internally; either way the motor state ends the call at ``t + dt``.
        """
        u0 = self.state.rpm
        thrust = rotor_thrust_mean(self.params, u0, self.u_cmd, dt) * self.effectiveness
        self.state.rpm = motor_step_closed_form(self.params, u0, self.u_cmd, dt)
        force_b, torque_b = wrench_from_thrust(self.params, thrust)
        return force_b, torque_b, self.state.force, self.state.torque

    def reward(self) -> Tensor:
        """``[N]`` L2F reward of the step just completed (pre-step state, its action, synced post-step state)."""
        if self._prev is None:
            raise RuntimeError("reward() needs a pre_physics_step() and a sync() first")
        return l2f_reward(self.params, self._prev, self.action, self.state)

    def terminated(self) -> Tensor:
        """``[N]`` bool L2F termination of the current state."""
        return l2f_terminated(self.params, self.state)
