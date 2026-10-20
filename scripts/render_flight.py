"""Render a recorded flight (``Recording`` npz) in Isaac Sim from several camera frames.

Replay, not simulation: the recording (from ``scripts/fly_raptor.py`` or ``scripts/sim_to_sim.py
record-l2f``) already holds the flown states, so the Crazyflie USD is placed kinematically at each
recorded pose (physics APIs stripped, the timeline never advances) and rendered with RTX through
Replicator render products. The default renderer is PathTracing (each frame from scratch): RTX
real-time path tracing reprojects its denoiser history with motion vectors that kinematic USD edits
do not produce, which leaves ghost copies of the drone in static cameras. Kit only: the ``isaacsim``
extra on Linux::

    uv sync --extra isaacsim
    FFMPEG=$(uv run --no-project --with imageio-ffmpeg python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')
    OMNI_KIT_ACCEPT_EULA=YES uv run --no-sync python scripts/render_flight.py runs/newton_fig8.npz --env 0 --out runs/renders/fig8 --ffmpeg "$FFMPEG"
    uv run --no-sync python scripts/render_flight.py runs/newton_fig8.npz --env 0 --out runs/renders/fig8 --skip-render --ffmpeg "$FFMPEG"  # re-composite/encode, no Kit

Views (each a different frame of reference):

* ``world_iso``   world frame, fixed oblique camera
* ``world_top``   world frame, top-down (x right, y up), framed on the whole path
* ``world_side``  world frame, side (x right, z up): height hold and the hover sag
* ``chase``       heading frame: follows the drone's position and yaw from behind and above
* ``onboard``     body frame: rigidly attached, looking along body +x (FLU), the horizon tilts with attitude
* ``ref_frame``   reference frame: follows ``p_ref`` (world-aligned), so the motion in the image is the tracking error

Display offset: L2F has no ground and hovers at the origin. Everything is shifted up by
``--ground-depth`` (default 1.05 m, below the 1 m termination box) so a visual-only ground sits at
z = 0; telemetry is reported in the recording's (L2F) frame. The drone is the chosen env only.

Outputs in ``--out``: ``<view>/NNNNN.png`` frames, ``<view>.mp4``, ``composite.mp4`` (3x2, labelled,
real time), ``composite.gif`` (with ``--gif``), ``contact_sheet.png`` and ``stills/``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaac_l2f.sim2sim import Recording

DEFAULT_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"
CRAZYFLIE_SUBPATH = "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"  # same file as isaaclab_assets CRAZYFLIE_CFG
HORIZONTAL_APERTURE = 20.955  # USD default; only the focal/aperture ratio (the FOV) matters here


@dataclass(frozen=True)
class View:
    name: str
    label: str
    hfov_deg: float


VIEWS = {
    v.name: v
    for v in (
        View("world_iso", "world frame - fixed oblique", 40.0),
        View("world_top", "world frame - top (x right, y up)", 35.0),
        View("world_side", "world frame - side (x right, z up)", 35.0),
        View("chase", "heading frame - chase (position + yaw)", 50.0),
        View("onboard", "body frame - onboard, looking along +x", 90.0),
        View("ref_frame", "reference frame - follows p_ref", 45.0),
    )
}

# Camera placement (metres, display frame). The Crazyflie is ~0.11 m across its props.
ISO_DIRECTION = (-1.0, -1.3, 0.9)  # from the path centre towards the camera
FIT_MARGIN = 1.1  # fixed views: path extent x margin
MIN_HALF_EXTENT = 0.15  # fixed views never zoom in closer than this half-extent per axis
FIT_PERCENTILE = 99.5  # fixed views may crop the most extreme 0.5 % of path points
CHASE_OFFSET = (-0.40, 0.0, 0.15)  # heading frame (x along the drone's yaw)
CHASE_LOOK_AHEAD = 0.08  # look this far ahead of the drone (heading frame)
ONBOARD_EYE = (0.085, 0.0, 0.02)  # body frame: a nose camera just ahead of the axis triad (cf2x origin is at its underside)
ONBOARD_PITCH_DEG = 8.0  # looking slightly down
REF_OFFSET = (-0.35, -0.45, 0.25)  # world-aligned, from p_ref
GRID_STEP, GRID_HALF = 0.5, 4.0


# ---------------------------------------------------------------------------- geometry (numpy)
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``[..., 4]`` wxyz -> ``[..., 3, 3]`` rotation (body to world, column vectors)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(q: np.ndarray) -> np.ndarray:
    """Angle between body z and world z."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 camera-to-world in USD's row-vector convention (camera looks down -z, +y up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if np.linalg.norm(np.cross(f, up)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, u, -f, eye
    return m


def _tan_half(view: View, aspect: float) -> tuple[float, float]:
    th = math.tan(math.radians(view.hfov_deg) / 2)
    return th, th / aspect


@dataclass
class Clip:
    """The rendered slice of one env, in the display frame (z shifted by ``ground_depth``)."""

    idx: np.ndarray  # recording indices of the rendered frames
    position: np.ndarray  # [T+1, 3] all recording indices up to the last rendered one
    orientation: np.ndarray  # [T+1, 4] wxyz
    position_ref: np.ndarray  # [T+1, 3]
    t: np.ndarray  # [F] seconds
    error: np.ndarray  # [F] |p - p_ref|
    z_error: np.ndarray  # [F] p_z - p_ref_z
    tilt: np.ndarray  # [F] degrees
    fps: float
    title: str


def load_clip(args) -> Clip:
    rec = Recording.load(args.recording)
    e = args.env
    if not 0 <= e < rec.num_envs:
        raise SystemExit(f"--env {e} out of range (recording has {rec.num_envs} envs)")
    n_valid = int(rec.alive()[:, e].sum())  # states after a termination are the auto-reset ones
    end = n_valid - 1 if args.steps is None else min(n_valid - 1, args.start + args.steps)
    if n_valid < rec.steps + 1:
        print(f"[render] env {e} terminated at step {n_valid - 1}; rendering up to it")
    idx = np.arange(args.start, end + 1, args.stride)
    shift = np.array([0.0, 0.0, args.ground_depth])
    p = rec.position[: end + 1, e].astype(np.float64)
    q = rec.orientation[: end + 1, e].astype(np.float64)
    r = rec.position_ref[: end + 1, e].astype(np.float64)
    dt = float(rec.meta.get("dt", 0.01))
    m = rec.meta
    title = args.title or (
        f"{m.get('policy', 'policy')} | {m.get('task', 'pure L2F')} | {m.get('backend', 'l2f')} | "
        f"reference {m.get('reference', '?')} | env {e} | replayed in Isaac Sim"
    )
    return Clip(
        idx=idx,
        position=p + shift,
        orientation=q / np.linalg.norm(q, axis=-1, keepdims=True),
        position_ref=r + shift,
        t=idx * dt,
        error=np.linalg.norm(p[idx] - r[idx], axis=-1),
        z_error=p[idx, 2] - r[idx, 2],
        tilt=tilt_deg(q[idx]),
        fps=1.0 / (dt * args.stride),
        title=title,
    )


def camera_poses(clip: Clip, view: View, aspect: float) -> np.ndarray:
    """``[F, 4, 4]`` camera-to-world per rendered frame; for ``onboard`` the constant body-frame pose."""
    p, r = clip.position[clip.idx], clip.position_ref[clip.idx]
    n = len(clip.idx)
    if view.name == "onboard":
        pitch = math.radians(ONBOARD_PITCH_DEG)
        eye = np.array(ONBOARD_EYE)
        return look_at(eye, eye + np.array([math.cos(pitch), 0.0, -math.sin(pitch)]))[None]
    if view.name == "chase":
        out = np.empty((n, 4, 4))
        for k, (pk, yk) in enumerate(zip(p, yaw_of(clip.orientation[clip.idx]))):
            c, s = math.cos(yk), math.sin(yk)
            rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            out[k] = look_at(pk + rz @ np.array(CHASE_OFFSET), pk + rz @ np.array([CHASE_LOOK_AHEAD, 0.0, 0.0]))
        return out
    if view.name == "ref_frame":
        return np.stack([look_at(rk + np.array(REF_OFFSET), rk) for rk in r])
    # fixed world-frame views: centred on the reference, the distance that keeps the flown path and the
    # reference (bar the most extreme points) inside the frame with a margin
    last = clip.idx[-1] + 1
    ref = clip.position_ref[:last]
    centre = (ref.min(0) + ref.max(0)) / 2
    pts = np.concatenate([clip.position[:last], ref]) - centre
    direction, up = {
        "world_top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "world_side": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        "world_iso": (ISO_DIRECTION, (0.0, 0.0, 1.0)),
    }[view.name]
    direction = np.asarray(direction) / np.linalg.norm(direction)
    basis = look_at(centre + direction, centre, up)
    x, y, toward = (pts @ basis[0, :3]), (pts @ basis[1, :3]), (pts @ basis[2, :3])
    tx, ty = _tan_half(view, aspect)
    need = np.maximum(FIT_MARGIN * np.abs(x) / tx, FIT_MARGIN * np.abs(y) / ty) + toward
    d = max(float(np.percentile(need, FIT_PERCENTILE)), FIT_MARGIN * MIN_HALF_EXTENT / ty + MIN_HALF_EXTENT)
    return np.repeat(look_at(centre + d * direction, centre, up)[None], n, axis=0)


# ---------------------------------------------------------------------------- Kit render
def render(args, clip: Clip, views: list[View]) -> None:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": args.renderer,
            "width": args.width,
            "height": args.height,
            "anti_aliasing": args.anti_aliasing,
            "denoiser": not args.no_denoiser,
            "samples_per_pixel_per_frame": args.spp,
        }
    )
    app.set_setting("/rtx/pathtracing/totalSpp", args.spp)
    app.set_setting("/rtx/pathtracing/clampSpp", args.spp)
    for kv in args.setting:
        key, value = kv.split("=", 1)
        app.set_setting(key, _setting_value(value))

    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    def material(path: str, rgb, emissive: float = 0.0, roughness: float = 0.6):
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(emissive * c for c in rgb)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def bind(prim, mat) -> None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def box(path: str, centre, size, mat) -> None:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        bind(cube.GetPrim(), mat)

    def matrix(m: np.ndarray):
        return Gf.Matrix4d(*(float(v) for v in m.reshape(-1)))

    def vec3f_array(pts: np.ndarray):
        return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, dtype=np.float32))

    # lights
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(args.dome_intensity)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(args.sun_intensity)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35.0, -20.0, 30.0))

    # visual-only ground (no collider) with a grid, centred under the path
    centre_xy = np.round(clip.position_ref[0, :2] / GRID_STEP) * GRID_STEP
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    s = 30.0
    ground.CreatePointsAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, -s, 0), Gf.Vec3f(s, s, 0), Gf.Vec3f(-s, s, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    ground.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    ground.CreateExtentAttr([Gf.Vec3f(-s, -s, 0), Gf.Vec3f(s, s, 0)])
    UsdGeom.Xformable(ground).AddTranslateOp().Set(Gf.Vec3d(float(centre_xy[0]), float(centre_xy[1]), 0.0))
    bind(ground.GetPrim(), material("/World/Looks/Ground", (0.13, 0.14, 0.16), roughness=0.9))
    grid_mat = material("/World/Looks/Grid", (0.45, 0.47, 0.50), roughness=0.8)
    ticks = np.arange(-GRID_HALF, GRID_HALF + 1e-9, GRID_STEP)
    for k, c in enumerate(ticks):
        w = 0.012 if abs(c - round(c)) < 1e-9 else 0.005
        box(f"/World/Grid/x{k}", (centre_xy[0], centre_xy[1] + c, 0.0005), (2 * GRID_HALF, w, 0.001), grid_mat)
        box(f"/World/Grid/y{k}", (centre_xy[0] + c, centre_xy[1], 0.0005), (w, 2 * GRID_HALF, 0.001), grid_mat)

    # paths: reference (whole clip) and a trail of the flown path that grows per frame
    def path_prim(path: str, pts: np.ndarray, rgb, width: float):
        mat = material(f"/World/Looks/{Path(path).name}", rgb, emissive=args.line_emissive, roughness=0.4)
        if args.path_style == "curves":
            c = UsdGeom.BasisCurves.Define(stage, path)
            c.CreateTypeAttr(UsdGeom.Tokens.linear)
            c.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            c.CreateWidthsAttr([width])
            c.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            c.CreatePointsAttr(vec3f_array(pts))
            c.CreateCurveVertexCountsAttr([len(pts)])
            c.CreateExtentAttr([Gf.Vec3f(*(pts.min(0) - width)), Gf.Vec3f(*(pts.max(0) + width))])
            bind(c.GetPrim(), mat)

            def update(new: np.ndarray) -> None:
                c.GetPointsAttr().Set(vec3f_array(new))
                c.GetCurveVertexCountsAttr().Set([len(new)])
                c.GetExtentAttr().Set([Gf.Vec3f(*(new.min(0) - width)), Gf.Vec3f(*(new.max(0) + width))])

            return update
        inst = UsdGeom.PointInstancer.Define(stage, path)
        proto = UsdGeom.Sphere.Define(stage, f"{path}/Prototypes/dot")
        proto.CreateRadiusAttr(width / 2)
        bind(proto.GetPrim(), mat)
        inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
        inst.CreatePositionsAttr(vec3f_array(pts))
        inst.CreateProtoIndicesAttr([0] * len(pts))

        def update(new: np.ndarray) -> None:
            inst.GetPositionsAttr().Set(vec3f_array(new))
            inst.GetProtoIndicesAttr().Set([0] * len(new))

        return update

    last = clip.idx[-1]
    ref_pts = clip.position_ref[: last + 1]
    if np.ptp(ref_pts, axis=0).max() > 1e-4:
        path_prim("/World/ReferencePath", ref_pts, (1.0, 0.55, 0.05), args.line_width)
    trail_update = path_prim("/World/Trail", clip.position[:2], (0.10, 0.65, 1.0), args.line_width)

    ref_marker = UsdGeom.Sphere.Define(stage, "/World/RefMarker")
    ref_marker.CreateRadiusAttr(0.012)
    bind(ref_marker.GetPrim(), material("/World/Looks/RefMarker", (1.0, 0.25, 0.05), emissive=args.line_emissive))
    ref_op = UsdGeom.Xformable(ref_marker).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)

    # the drone: pose on /World/Drone, the USD referenced below it with physics stripped
    drone = UsdGeom.Xform.Define(stage, "/World/Drone")
    drone_t = drone.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    drone_q = drone.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    usd = args.usd or f"{_asset_root()}/{CRAZYFLIE_SUBPATH}"
    cf = stage.DefinePrim("/World/Drone/cf", "Xform")
    cf.GetReferences().AddReference(usd)
    t0 = time.perf_counter()
    while omni.usd.get_context().get_stage_loading_status()[2] > 0 and time.perf_counter() - t0 < 300:
        app.update()
    if not stage.GetPrimAtPath("/World/Drone/cf/body").IsValid():
        raise SystemExit(f"[render] Crazyflie USD did not load from {usd} (asset root / network?)")
    prims = list(Usd.PrimRange(cf))
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
            continue
        for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    # FLU body axes: x red, y green, z blue, from the deck centre
    axis_len, axis_w, deck = 0.08, 0.003, 0.015
    for name, rgb, centre, size in (
        ("x", (0.9, 0.1, 0.1), (axis_len / 2, 0, deck), (axis_len, axis_w, axis_w)),
        ("y", (0.1, 0.8, 0.1), (0, axis_len / 2, deck), (axis_w, axis_len, axis_w)),
        ("z", (0.15, 0.3, 1.0), (0, 0, deck + axis_len / 2), (axis_w, axis_w, axis_len)),
    ):
        box(f"/World/Drone/axes/{name}", centre, size, material(f"/World/Looks/axis_{name}", rgb, emissive=args.line_emissive))

    # cameras, one render product and rgb annotator each
    aspect = args.width / args.height
    poses, cam_ops, annotators = {}, {}, {}
    for view in views:
        poses[view.name] = camera_poses(clip, view, aspect)
        parent = "/World/Drone" if view.name == "onboard" else "/World/Cameras"
        cam = UsdGeom.Camera.Define(stage, f"{parent}/{view.name}")
        cam.CreateFocalLengthAttr(HORIZONTAL_APERTURE / (2 * math.tan(math.radians(view.hfov_deg) / 2)))
        cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
        cam.CreateVerticalApertureAttr(HORIZONTAL_APERTURE / aspect)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.005 if view.name == "onboard" else 0.02, 500.0))
        cam_ops[view.name] = cam.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        cam_ops[view.name].Set(matrix(poses[view.name][0]))
        rp = rep.create.render_product(str(cam.GetPath()), (args.width, args.height), name=view.name)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(rp)
        annotators[view.name] = ann
        (args.out / view.name).mkdir(parents=True, exist_ok=True)

    rep.orchestrator.set_capture_on_play(False)

    def place(k: int) -> None:
        i = clip.idx[k]
        drone_t.Set(Gf.Vec3d(*map(float, clip.position[i])))
        drone_q.Set(Gf.Quatd(*map(float, clip.orientation[i])))
        ref_op.Set(Gf.Vec3d(*map(float, clip.position_ref[i])))
        trail_update(clip.position[: max(i, 1) + 1])
        for view in views:
            if view.name != "onboard":
                cam_ops[view.name].Set(matrix(poses[view.name][k]))

    place(0)
    for _ in range(args.warmup):
        app.update()
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

    t0 = time.perf_counter()
    for k in range(len(clip.idx)):
        place(k)
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
        for view in views:
            rgb = np.asarray(annotators[view.name].get_data())
            if rgb.size == 0:
                rgb = np.zeros((args.height, args.width, 4), np.uint8)
            Image.fromarray(rgb[..., :3]).save(args.out / view.name / f"{k:05d}.png", compress_level=1)
        if k % 50 == 0:
            print(f"[render] frame {k}/{len(clip.idx)}  {(time.perf_counter() - t0) / (k + 1):.3f} s/frame", flush=True)
    rep.orchestrator.wait_until_complete()
    wall = time.perf_counter() - t0
    print(f"[render] {len(clip.idx)} frames x {len(views)} views in {wall:.1f} s ({wall / len(clip.idx):.3f} s/frame)")

    finish(args, clip, views)  # before app.close(): it ends the process
    app.close()


def _setting_value(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _asset_root() -> str:
    try:
        from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR

        if NUCLEUS_ASSET_ROOT_DIR:
            return NUCLEUS_ASSET_ROOT_DIR
    except Exception as exc:  # noqa: BLE001  (fall back to the 6.1 cloud root)
        print(f"[render] isaaclab asset root unavailable ({exc}); using {DEFAULT_ASSET_ROOT}")
    return DEFAULT_ASSET_ROOT


# ---------------------------------------------------------------------------- composite and encode
def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def composite_frame(args, clip: Clip, views: list[View], k: int, fonts) -> Image.Image:
    big, small = fonts
    cols = min(3, len(views))
    rows = math.ceil(len(views) / cols)
    header = 44
    img = Image.new("RGB", (cols * args.width, header + rows * args.height), (18, 18, 20))
    draw = ImageDraw.Draw(img)
    for n, view in enumerate(views):
        x, y = (n % cols) * args.width, header + (n // cols) * args.height
        img.paste(Image.open(args.out / view.name / f"{k:05d}.png").convert("RGB"), (x, y))
        draw.rectangle([x, y, x + args.width - 1, y + 24], fill=(0, 0, 0))
        draw.text((x + 8, y + 4), f"{view.name}: {view.label}", fill=(235, 235, 235), font=small)
    draw.text((10, 4), clip.title, fill=(240, 240, 240), font=small)
    tele = (
        f"t {clip.t[k]:5.2f} s   |p - p_ref| {clip.error[k]:.3f} m   "
        f"z - z_ref {clip.z_error[k]:+.3f} m   tilt {clip.tilt[k]:4.1f} deg"
    )
    draw.text((10, 22), tele, fill=(255, 200, 80), font=big)
    return img


def finish(args, clip: Clip, views: list[View]) -> None:
    n = len(clip.idx)
    missing = [v.name for v in views if not (args.out / v.name / f"{n - 1:05d}.png").exists()]
    if missing:
        raise SystemExit(f"[render] frames missing for {missing} in {args.out}; render first")
    fonts = (_font(16), _font(14))
    comp_dir = args.out / "composite"
    comp_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        composite_frame(args, clip, views, k, fonts).save(comp_dir / f"{k:05d}.png", compress_level=1)

    keys = np.linspace(0, n - 1, args.keyframes).round().astype(int)
    stills = args.out / "stills"
    stills.mkdir(exist_ok=True)
    cw, ch, lw = args.width // 2, args.height // 2, 90
    sheet = Image.new("RGB", (lw + len(views) * cw, 30 + len(keys) * ch), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    small = fonts[1]
    for c, view in enumerate(views):
        draw.text((lw + c * cw + 6, 8), view.name, fill=(235, 235, 235), font=small)
    for r, k in enumerate(keys):
        shutil.copy(comp_dir / f"{k:05d}.png", stills / f"t{clip.t[k]:06.2f}s.png")
        draw.text((8, 30 + r * ch + ch // 2 - 8), f"t {clip.t[k]:.2f} s", fill=(255, 200, 80), font=small)
        for c, view in enumerate(views):
            cell = Image.open(args.out / view.name / f"{k:05d}.png").convert("RGB").resize((cw, ch), Image.LANCZOS)
            sheet.paste(cell, (lw + c * cw, 30 + r * ch))
    sheet.save(args.out / "contact_sheet.png")
    print(f"[render] composite frames, stills and {args.out / 'contact_sheet.png'} written")

    ffmpeg = _ffmpeg(args.ffmpeg)
    if ffmpeg is None:
        print("[render] no ffmpeg (pass --ffmpeg); frames are left as PNG sequences")
        return
    fps = f"{clip.fps:g}"
    for name, crf in [(v.name, args.crf) for v in views] + [("composite", args.composite_crf)]:
        src, dst = args.out / name / "%05d.png", args.out / f"{name}.mp4"
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", fps, "-i", str(src), "-c:v", "libx264",
               "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]  # fmt: skip
        subprocess.run(cmd, check=True)
        print(f"[render] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    if args.gif:
        dst = args.out / "composite.gif"
        vf = "fps=25,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(args.out / "composite.mp4"), "-vf", vf, str(dst)], check=True)
        print(f"[render] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def _ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", type=Path, help="Recording npz (scripts/fly_raptor.py or sim_to_sim.py record-l2f)")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="first recording index")
    p.add_argument("--steps", type=int, default=None, help="recording steps to render (default: all alive ones)")
    p.add_argument("--stride", type=int, default=2, help="recording steps per video frame (2 -> 50 fps from 100 Hz)")
    p.add_argument("--views", default=",".join(VIEWS), help=f"comma list from {', '.join(VIEWS)}")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--ground-depth", type=float, default=1.05, help="visual ground this far below the L2F origin (m)")
    p.add_argument("--path-style", choices=["curves", "dots"], default="curves")
    p.add_argument("--line-width", type=float, default=0.006)
    p.add_argument("--line-emissive", type=float, default=0.6)
    p.add_argument("--dome-intensity", type=float, default=400.0)
    p.add_argument("--sun-intensity", type=float, default=1500.0)
    p.add_argument("--renderer", default="PathTracing", help="PathTracing (default) or RealTimePathTracing (fast, ghosts moving objects)")
    p.add_argument("--spp", type=int, default=16, help="path-tracing samples per pixel per frame")
    p.add_argument("--subframes", type=int, default=1, help="RTX subframes per captured frame")
    p.add_argument("--anti-aliasing", type=int, default=3, help="0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA (SimulationApp)")
    p.add_argument("--no-denoiser", action="store_true")
    p.add_argument("--setting", action="append", default=[], metavar="/carb/path=VALUE", help="extra carb setting, repeatable")
    p.add_argument("--warmup", type=int, default=60, help="app updates before the first capture")
    p.add_argument("--usd", default=None, help="Crazyflie USD (default: <Isaac asset root>/" + CRAZYFLIE_SUBPATH + ")")
    p.add_argument("--keyframes", type=int, default=5)
    p.add_argument("--crf", type=int, default=20, help="H.264 CRF of the per-view videos")
    p.add_argument("--composite-crf", type=int, default=23)
    p.add_argument("--ffmpeg", default=None)
    p.add_argument("--gif", action="store_true", help="also write composite.gif (960 px, 25 fps)")
    p.add_argument("--skip-render", action="store_true", help="composite/encode existing frames only (no Kit)")
    args = p.parse_args()
    unknown = [v for v in args.views.split(",") if v not in VIEWS]
    if unknown:
        p.error(f"unknown views {unknown}")
    return args


def main() -> None:
    args = parse_args()
    views = [VIEWS[v] for v in args.views.split(",")]
    clip = load_clip(args)
    print(f"[render] {args.recording} env {args.env}: {len(clip.idx)} frames at {clip.fps:g} fps, views {[v.name for v in views]}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.skip_render:
        finish(args, clip, views)
    else:
        render(args, clip, views)


if __name__ == "__main__":
    main()
