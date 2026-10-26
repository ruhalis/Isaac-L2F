"""Fly a protocol policy (RAPTOR by default) in ``Isaac-L2F-Hover-Direct`` and record the rollout.

Linux + Isaac Lab 3.0 only (kit-less by default). The physics preset is a Hydra-style token::

    uv run --no-sync python scripts/fly_raptor.py --num_envs 16 --steps 500 physics=newton_mjwarp --viz none --out runs/newton_hover.npz
    uv run --no-sync python scripts/fly_raptor.py --num_envs 16 --steps 500 physics=newton_mjwarp --viz none --reference figure8 --out runs/newton_fig8.npz
    uv run --no-sync python scripts/fly_raptor.py --num_envs 16 --steps 500 physics=newton_mjwarp --viz none --fault-rotor 0 --fault-effectiveness 0.5 --fault-step 100 --out runs/newton_fault.npz
    uv run --no-sync python scripts/fly_raptor.py --num_envs 4096 --steps 200 physics=newton_mjwarp --viz none   # throughput only

The recording is compared with pure L2F by ``scripts/sim_to_sim.py compare <npz>`` (runs anywhere).
Pattern follows Isaac Lab 3.0.0 ``isaaclab_rl/entrypoints/simple_agents.py`` (verified 2026-10-26).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401  (Isaac Lab's own registrations; ours come from isaac_l2f.tasks below)
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaac_l2f.tasks  # noqa: F401
from isaac_l2f.policy import load_raptor, load_torchscript
from isaac_l2f.sim2sim import Recorder, fault_effectiveness, figure_eight, hover_reference
from isaac_l2f.tasks import HOVER_TASK_ID


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=HOVER_TASK_ID)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=500, help="policy steps at 100 Hz (one episode, no time-out inside)")
    p.add_argument("--seed", type=int, default=0, help="env core seed (reset distribution)")
    p.add_argument("--policy", default=None, help="TorchScript policy path; default: RAPTOR from foundation-policy")
    p.add_argument("--init", action="append", default=[], metavar="KEY=VALUE",
                   help="mdp.init override, repeatable (max_position=0.3 max_linear_velocity=0.5 ...)")
    p.add_argument("--reference", choices=["hover", "figure8"], default="hover")
    p.add_argument("--fig8-amplitude", type=float, default=0.5)
    p.add_argument("--fig8-period", type=float, default=4.0)
    p.add_argument("--fault-rotor", type=int, default=None, help="rotor index 0 FR, 1 BR, 2 BL, 3 FL")
    p.add_argument("--fault-effectiveness", type=float, default=0.5)
    p.add_argument("--fault-step", type=int, default=100)
    p.add_argument("--out", type=Path, default=None, help="npz recording path")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["none"])
    args, hydra_args = setup_preset_cli(p)  # physics=... tokens stay in hydra_args
    return args, hydra_args


def main() -> None:
    args, hydra_args = parse_args()
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, overrides=hydra_args)
    env_cfg.dynamics_seed = args.seed
    env_cfg.init_overrides = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.init)} or None
    env_cfg.episode_length_s = (args.steps + 10) * env_cfg.sim.dt * env_cfg.decimation  # no time-out while recording
    env_cfg.validate()
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        device, n = torch.device(u.device), u.num_envs  # DirectRLEnv.device is sim.device, a str
        policy = load_torchscript(args.policy, map_location=device) if args.policy else load_raptor(device=device)
        reference = (
            figure_eight(args.fig8_amplitude, args.fig8_period, device=device) if args.reference == "figure8" else hover_reference(device)
        )
        dt = u.step_dt
        fault = fault_effectiveness(n, args.fault_rotor, args.fault_effectiveness, device=device)
        nominal = torch.ones(n, 4, device=device)

        p_ref, v_ref = reference(0.0)
        u.set_reference(p_ref.expand(n, 3), v_ref.expand(n, 3))
        obs_dict, _ = env.reset()
        policy.reset()
        obs = obs_dict["policy"]
        rec = Recorder(
            {
                "backend": u.scene.physics_backend,
                "task": args.task,
                "airframe": u.cfg.airframe,
                "dt": dt,
                "hold_dt": u._hold_dt,
                "seed": args.seed,
                "init_overrides": env_cfg.init_overrides,
                "policy": args.policy or "foundation-policy RAPTOR",
                "reference": args.reference,
                "fault": [args.fault_rotor, args.fault_effectiveness, args.fault_step] if args.fault_rotor is not None else None,
            }
        )
        rec.state(u.core.state, p_ref.expand(n, 3), v_ref.expand(n, 3))

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for t in range(args.steps):
            eff = fault if (args.fault_rotor is not None and t >= args.fault_step) else nominal
            if args.fault_rotor is not None and t == args.fault_step:
                u.set_motor_effectiveness(eff)
            with torch.inference_mode():
                action = policy.step(obs)
            p_ref, v_ref = reference((t + 1) * dt)
            u.set_reference(p_ref.expand(n, 3), v_ref.expand(n, 3))  # reference for the observation after this step
            obs_dict, rew, term, trunc, _ = env.step(action)
            obs = obs_dict["policy"]
            rec.step(obs, action, rew, term, eff)
            rec.state(u.core.state, p_ref.expand(n, 3), v_ref.expand(n, 3))
        torch.cuda.synchronize() if device.type == "cuda" else None
        wall = (time.perf_counter() - t0) / args.steps
        rec.meta["wall_s_per_step"] = wall
        out = rec.finish()
        alive = out.alive()
        track = ((out.position - out.position_ref) ** 2).sum(-1) ** 0.5
        print(f"[fly_raptor] backend {out.meta['backend']}, {n} envs, {args.steps} steps, hold {u._hold_dt:.4f} s")
        print(f"[fly_raptor] {wall * 1e3:.2f} ms/step -> {n / wall:.0f} env-steps/s; terminated {int((~alive[-1]).sum())}/{n}")
        final = f"{track[-1][alive[-1]].max():.3f} m" if alive[-1].any() else "n/a (all terminated)"
        print(f"[fly_raptor] tracking error |p - p_ref|: final max {final}, rms {float((track[alive] ** 2).mean() ** 0.5):.3f} m")
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            out.save(args.out)
            print(f"[fly_raptor] saved {args.out}")
        env.close()


if __name__ == "__main__":
    main()
