"""Sim-to-sim comparison: a recorded Isaac Lab rollout against pure L2F, same policy, seeds and parameters.

Runs anywhere (CPU, no Isaac Lab). Two commands::

    uv run python scripts/sim_to_sim.py compare runs/newton_hover.npz runs/physx_hover.npz
    uv run python scripts/sim_to_sim.py record-l2f --num_envs 16 --steps 500 --reference figure8 --out runs/l2f_fig8.npz

``compare`` replays the policy in pure L2F from each recording's initial states with its reference
and fault schedule, then reports per recording: closed-loop position error vs that replay,
tracking error vs the reference, and the teacher-forced one-step error of the recorded transitions
against ``isaac_l2f.dynamics.step`` (the parity number that closed-loop chaos does not amplify).
``record-l2f`` writes a pure-L2F recording in the same format (also the Mac self-test of this tool).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from isaac_l2f.dynamics import load_registry, sample_initial_state
from isaac_l2f.policy import load_raptor, load_torchscript
from isaac_l2f.sim2sim import (
    Recording,
    compare,
    fault_effectiveness,
    figure_eight,
    format_report,
    hover_reference,
    one_step_errors,
    replay_in_l2f,
    rollout_l2f,
)


def _policy(path: str | None, device: str):
    return load_torchscript(path, map_location=device) if path else load_raptor(device=device)


def cmd_record_l2f(args) -> None:
    device = torch.device(args.device)
    init_overrides = {k: float(v) for k, v in (kv.split("=", 1) for kv in args.init)} or None
    params = load_registry(args.airframe, device=device)
    if init_overrides:
        params = params.with_init(**init_overrides)
    params = params.expand(args.num_envs)
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    init = sample_initial_state(params, args.num_envs, g)
    reference = figure_eight(args.fig8_amplitude, args.fig8_period, device=device) if args.reference == "figure8" else hover_reference(device)
    fault = fault_effectiveness(args.num_envs, args.fault_rotor, args.fault_effectiveness, device=device)
    nominal = torch.ones(args.num_envs, 4, device=device)
    eff = (lambda t: fault if (args.fault_rotor is not None and t >= args.fault_step) else nominal)
    rec = rollout_l2f(
        params, init, _policy(args.policy, args.device), args.steps, reference=reference, effectiveness=eff,
        meta={"airframe": args.airframe, "seed": args.seed, "reference": args.reference, "init_overrides": init_overrides,
              "policy": args.policy or "foundation-policy RAPTOR",
              "fault": [args.fault_rotor, args.fault_effectiveness, args.fault_step] if args.fault_rotor is not None else None},
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rec.save(args.out)
    alive = rec.alive()
    print(f"[record-l2f] {rec.num_envs} envs, {rec.steps} steps, terminated {int((~alive[-1]).sum())}, {rec.meta['wall_s_per_step'] * 1e3:.2f} ms/step; saved {args.out}")


def cmd_compare(args) -> None:
    device = args.device
    results = {}
    for path in args.recordings:
        rec = Recording.load(path)
        airframe = rec.meta.get("airframe", args.airframe)
        params = load_registry(airframe, device=device).expand(rec.num_envs)
        ref = replay_in_l2f(rec, params, _policy(args.policy, device), device=device)
        closed = compare(ref, rec)
        one = one_step_errors(rec, params, device=device)
        backend = rec.meta.get("backend", "?")
        title = f"{path}: {backend}, hold {rec.meta.get('hold_dt', '?')} s, reference {rec.meta.get('reference', '?')}, fault {rec.meta.get('fault')}"
        print(format_report(title + "\n closed loop vs pure-L2F replay", closed))
        print(format_report(" one-step (teacher-forced) vs dynamics.step", one))
        results[str(path)] = {"meta": rec.meta, "closed_loop": closed, "one_step": one}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str))
        print(f"[compare] wrote {args.json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare")
    c.add_argument("recordings", nargs="+", type=Path)
    c.add_argument("--policy", default=None, help="TorchScript policy path; default RAPTOR (must match the recording)")
    c.add_argument("--airframe", default="crazyflie", help="fallback when the recording carries none")
    c.add_argument("--device", default="cpu")
    c.add_argument("--json", type=Path, default=None)
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("record-l2f")
    r.add_argument("--num_envs", type=int, default=16)
    r.add_argument("--steps", type=int, default=500)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--airframe", default="crazyflie")
    r.add_argument("--policy", default=None)
    r.add_argument("--init", action="append", default=[], metavar="KEY=VALUE", help="mdp.init override, repeatable")
    r.add_argument("--reference", choices=["hover", "figure8"], default="hover")
    r.add_argument("--fig8-amplitude", type=float, default=0.5)
    r.add_argument("--fig8-period", type=float, default=4.0)
    r.add_argument("--fault-rotor", type=int, default=None)
    r.add_argument("--fault-effectiveness", type=float, default=0.5)
    r.add_argument("--fault-step", type=int, default=100)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=cmd_record_l2f)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
