"""``Isaac-L2F-Hover-Direct`` configuration (Isaac Lab 3.0.0rc1 API, verified 2026-10-18).

Imports Isaac Lab, so this module is Linux-only; the registration in ``isaac_l2f.tasks`` refers
to it by string. Physics runs on Newton only: :class:`L2FPhysicsCfg` holds the ``newton_mjwarp``
preset (also the default), selected on the command line with the Hydra-style token
``physics=newton_mjwarp``.
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG

POLICY_HZ = 100
PHYSICS_HZ = 400
DECIMATION = PHYSICS_HZ // POLICY_HZ


@configclass
class L2FPhysicsCfg(PresetCfg):
    """The contract's physics preset: Newton MJWarp, also the default."""

    # No contacts in the hover task (L2F has no ground and no obstacles), so the MJWarp buffers stay
    # small; obstacle scenes (M3) raise njmax/nconmax explicitly.
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3, cone="pyramidal", impratio=1, integrator="implicitfast"),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class HoverEnvCfg(DirectRLEnvCfg):
    """L2F hover on the Crazyflie asset: 22-dim contract observation, 4 raw motor actions, 100 Hz over 400 Hz physics."""

    decimation = DECIMATION
    episode_length_s = 5.0  # L2F hover episode: 500 policy steps
    action_space = 4
    observation_space = 22
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / PHYSICS_HZ,
        render_interval=DECIMATION,
        gravity=(0.0, 0.0, -9.81),  # must equal the airframe's ``gravity``; checked at env init
        physics=L2FPhysicsCfg(),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # Crazyflie USD from Nucleus (not copied). Prop joints are left passive and at rest: their
    # spin in the stock init state is cosmetic and would add gyroscopic terms L2F does not have.
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
    )

    airframe: str = "crazyflie"
    """L2F registry airframe whose parameters drive the dynamics, mass and inertia."""

    dynamics_seed: int = 0
    """Seed of the env core's generator (reset distribution, noise draws)."""

    init_overrides: dict[str, float] | None = None
    """``mdp.init.*`` overrides applied to the airframe (``Params.with_init``), e.g. ``{"max_position": 0.3}``; None = registry box."""

    root_body_name: str | None = None
    """Body that carries the L2F mass, inertia and wrench; None = body index 0 (the articulation root)."""

    prop_body_regex: str | None = "m[1-4]_prop"
    """Prop bodies whose mass is reduced to ``prop_mass`` so the airframe is one L2F rigid body; None = leave them."""

    prop_mass: float = 1.0e-5
    """Mass (kg) written to each prop body; their inertia is scaled by the same ratio."""

    def __post_init__(self):
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.5, 2.5, 1.5))  # only ``eye`` is verified in rc1
