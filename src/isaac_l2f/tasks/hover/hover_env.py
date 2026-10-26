"""``Isaac-L2F-Hover-Direct``: the L2F hover task as an Isaac Lab 3.0 ``DirectRLEnv``.

The environment is a boundary around :class:`isaac_l2f.tasks.core.L2FCore`: it moves tensors
between the simulator and the core and nothing else. Per physics substep the core advances the
motor lag and returns the rotor wrench, which goes through the backend-neutral permanent wrench
composer (body frame, at the COM); per policy step the core builds the contract observation,
reward and termination from the simulator's COM state and draws the L2F reset distribution.

Boundary rules (Isaac Lab 3.0.0rc1, verified 2026-10-18 on the installed package):

* quaternions are (x, y, z, w) on the Isaac side and (w, x, y, z) in the core
  (:mod:`isaac_l2f.tasks.boundary`);
* the contract position and velocities are COM quantities (``root_com_pos_w``,
  ``root_com_lin_vel_w``, ``root_com_ang_vel_b``); orientation is the link quaternion, which is
  the frame the wrench composer calls "body"; resets write the COM pose and COM velocity, with
  the angular velocity rotated to the world frame as the writer expects;
* the root body carries the L2F mass and ``boundary.effective_inertia(params)`` minus the other
  bodies' (written at start-up with ``set_masses_index`` / ``set_inertias_index``; the props are
  shrunk to ``prop_mass``), so the composite articulation equals the L2F rigid body, with its CoM
  within ~3e-5 m of the root CoM the contract reads;
* gravity is the simulator's; the core's wrench never includes it;
* when the backend runs the decimation loop inside one ``sim.step()`` (Newton with graph-safe
  actuators, ``handles_decimation()``), ``_apply_action`` runs once per policy step and the core
  returns the wrench held over the whole 10 ms; otherwise once per 2.5 ms substep.

Hooks, settable from outside at any time (neutral defaults): :meth:`set_reference`,
:meth:`set_motor_effectiveness`, :meth:`set_disturbance_wrench`; the core itself is ``env.core``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from isaac_l2f.dynamics import load_registry
from isaac_l2f.dynamics import quaternion as quat
from isaac_l2f.tasks.boundary import (
    bodies_in_root_frame,
    composite_inertia,
    effective_inertia,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    root_for_composite,
)
from isaac_l2f.tasks.core import L2FCore, assert_rates

if TYPE_CHECKING:
    from isaac_l2f.tasks.hover.hover_env_cfg import HoverEnvCfg

log = logging.getLogger(__name__)


class HoverEnv(DirectRLEnv):
    """L2F hover with the RAPTOR contract; see the module docstring for the boundary rules."""

    cfg: HoverEnvCfg

    def __init__(self, cfg: HoverEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._root_body_id = self._find_root_body()
        self._prop_body_ids = self._find_prop_bodies()
        self._body_ids = [self._root_body_id]
        self._hold_dt = self.step_dt if self.sim.physics_manager.handles_decimation() else self.physics_dt
        log.info(
            "Isaac-L2F hover: backend %s, bodies %s, root body %d, wrench hold %.4f s (%s)",
            self.scene.physics_backend,
            list(self.robot.body_names),
            self._root_body_id,
            self._hold_dt,
            "backend runs the decimation loop" if self._hold_dt == self.step_dt else "one call per physics substep",
        )
        self._write_body_properties()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, pos, global_paths=())
        cloner.replicate(plan, stage=self.scene.stage)
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func(
            "/World/Light",
            light_cfg,
            orientation=(-0.14644663035869598, -0.3535534143447876, -0.3535534143447876, 0.8535533547401428),
        )

        params = load_registry(self.cfg.airframe, device=self.device)
        if self.cfg.init_overrides:
            params = params.with_init(**self.cfg.init_overrides)
        assert_rates(self.cfg.sim.dt, self.cfg.decimation, params)
        g_sim = torch.tensor(self.cfg.sim.gravity, device=self.device, dtype=params.dtype)
        if not torch.allclose(params.gravity[0], g_sim, atol=1e-6):
            raise ValueError(f"cfg.sim.gravity {tuple(self.cfg.sim.gravity)} differs from the airframe's {params.gravity[0].tolist()}")
        self.core = L2FCore(params, self.scene.num_envs, seed=self.cfg.dynamics_seed)

    def _find_root_body(self) -> int:
        if self.cfg.root_body_name is None:
            return 0
        ids, _ = self.robot.find_bodies(self.cfg.root_body_name)
        if len(ids) != 1:
            raise ValueError(f"root_body_name {self.cfg.root_body_name!r} matched {ids} in {list(self.robot.body_names)}")
        return int(ids[0])

    def _find_prop_bodies(self) -> list[int]:
        if self.cfg.prop_body_regex is None:
            return []
        try:
            ids, _ = self.robot.find_bodies(self.cfg.prop_body_regex)
        except ValueError:
            log.warning("prop_body_regex %r matched no body in %s", self.cfg.prop_body_regex, list(self.robot.body_names))
            return []
        return [int(i) for i in ids if int(i) != self._root_body_id]

    def _write_body_properties(self) -> None:
        """Make the composite articulation (root plus every other body) exactly the L2F rigid body."""
        n = self.scene.num_envs
        data = self.robot.data
        root = self._root_body_id
        q_com = data.body_com_quat_b.torch
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=q_com.device, dtype=q_com.dtype)
        err = torch.minimum((q_com - ident).abs().amax(dim=-1), (q_com + ident).abs().amax(dim=-1))
        if bool((err > 1e-6).any()):
            raise ValueError(f"body CoM frames are not aligned with the link frames (max quaternion error {err.max().item():.3e})")
        if self._prop_body_ids:
            masses = data.body_mass.torch[:, self._prop_body_ids]
            inertias = data.body_inertia.torch[:, self._prop_body_ids].reshape(n, len(self._prop_body_ids), 9)
            target = torch.full_like(masses, self.cfg.prop_mass)
            self.robot.set_masses_index(masses=target, body_ids=self._prop_body_ids)
            self.robot.set_inertias_index(inertias=inertias * (target / masses)[..., None], body_ids=self._prop_body_ids)

        target_mass = self.core.body_mass.to(torch.float64)
        target_inertia = effective_inertia(self.core.params).to(torch.float64)
        others = [i for i in range(self.robot.num_bodies) if i != root]

        def composite() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pose = data.body_link_pose_w.torch.to(torch.float64)
            m = data.body_mass.torch.to(torch.float64)
            com, inertia = bodies_in_root_frame(
                pose[..., :3],
                quat_xyzw_to_wxyz(pose[..., 3:]),
                data.body_com_pos_b.torch.to(torch.float64),
                data.body_inertia.torch.to(torch.float64).reshape(n, -1, 3, 3),
                root,
            )
            return m, com, inertia

        m, com, inertia = composite()
        m_root, i_root = root_for_composite(
            target_mass, target_inertia, com[:, root], m[:, others], com[:, others], inertia[:, others]
        )
        self.robot.set_masses_index(masses=m_root.to(torch.float32).reshape(n, 1), body_ids=[root])
        self.robot.set_inertias_index(inertias=i_root.to(torch.float32).reshape(n, 1, 9), body_ids=[root])

        m, com, inertia = composite()
        total, c_com, c_inertia = composite_inertia(m, com, inertia)
        mass_err = ((total - target_mass) / target_mass).abs()
        scale = torch.diagonal(target_inertia, dim1=-2, dim2=-1).amax(dim=-1)
        inertia_err = (c_inertia - target_inertia).abs().amax(dim=(-2, -1)) / scale
        if bool((mass_err > 1e-5).any()) or bool((inertia_err > 1e-5).any()):
            raise RuntimeError(
                f"composite articulation differs from the L2F body: worst relative mass error {mass_err.max().item():.3e}, "
                f"worst inertia error {inertia_err.max().item():.3e} of the largest diagonal entry"
            )
        offset = (c_com - com[:, root]).norm(dim=-1).max().item()
        log.info("composite CoM offset from the root CoM: max %.3e m over %d envs", offset, n)

    # ------------------------------------------------------------------ hooks
    def set_reference(self, position=None, velocity=None, env_ids=None) -> None:
        """World-frame reference (p_ref, v_ref) subtracted in the observation; see ``L2FCore.set_reference``."""
        self.core.set_reference(position, velocity, env_ids)

    def set_motor_effectiveness(self, effectiveness, env_ids=None) -> None:
        """Per-rotor thrust scale; see ``L2FCore.set_motor_effectiveness``."""
        self.core.set_motor_effectiveness(effectiveness, env_ids)

    def set_disturbance_wrench(self, force=None, torque=None, env_ids=None) -> None:
        """World-frame force and body-frame torque; see ``L2FCore.set_disturbance_wrench``."""
        self.core.set_disturbance_wrench(force, torque, env_ids)

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.core.pre_physics_step(actions)

    def _apply_action(self) -> None:
        force_b, torque_b, force_w, torque_b_dist = self.core.substep(self._hold_dt)
        composer = self.robot.permanent_wrench_composer
        composer.set_forces_and_torques_index(
            forces=force_b.unsqueeze(1),
            torques=(torque_b + torque_b_dist).unsqueeze(1),
            body_ids=self._body_ids,
            is_global=False,
        )
        composer.add_forces_and_torques_index(forces=force_w.unsqueeze(1), body_ids=self._body_ids, is_global=True)

    def _sync_core(self) -> None:
        d = self.robot.data
        self.core.sync(
            d.root_com_pos_w.torch - self.scene.env_origins,
            quat_xyzw_to_wxyz(d.root_link_quat_w.torch),
            d.root_com_lin_vel_w.torch,
            d.root_com_ang_vel_b.torch,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._sync_core()  # first hook after the physics loop: the post-step readout
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.core.terminated(), time_out

    def _get_rewards(self) -> torch.Tensor:
        return self.core.reward()

    def _get_observations(self) -> dict:
        self._sync_core()  # envs reset in this step keep their sampled state inside the core
        return {"policy": self.core.observe()}

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        sampled = self.core.reset(ids)
        pose = torch.cat([sampled.position + self.scene.env_origins[ids], quat_wxyz_to_xyzw(sampled.orientation)], dim=-1)
        ang_vel_w = quat.rotate(sampled.orientation, sampled.angular_velocity)
        velocity = torch.cat([sampled.linear_velocity, ang_vel_w], dim=-1)
        self.robot.write_root_com_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        if self.robot.num_joints > 0:
            rest = torch.zeros(ids.numel(), self.robot.num_joints, device=self.device)
            self.robot.write_joint_position_to_sim_index(position=rest, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=rest, env_ids=env_ids)
