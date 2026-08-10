"""Show random wrist_init samples with cap and dg5fs in Isaac Sim GUI.

Usage:
    conda activate task1
    python task1/tools/view_wrist_init_gui.py --num 10 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"", "none", "off", "-1"}:
        return None
    return float(value)


parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=10, help="number of wrist samples")
parser.add_argument("--seed", type=int, default=0, help="torch random seed")
parser.add_argument("--cap-z", type=float, default=0.225, help="cap reference z in meters")
parser.add_argument("--sampler", choices=("mcp_reachable", "triangle", "wide"), default="mcp_reachable")
parser.add_argument("--z-min", type=float, default=0.040, help="minimum wrist z above cap origin in meters")
parser.add_argument("--z-max", type=float, default=0.095, help="maximum wrist z above cap origin in meters")
parser.add_argument("--r-min", type=float, default=0.045, help="minimum wrist xy radius from cap center in meters")
parser.add_argument("--r-max", type=float, default=0.075, help="maximum wrist xy radius from cap center in meters")
parser.add_argument("--palm-tol", type=float, default=25.0, help="max palm aim error in degrees")
parser.add_argument("--dorsal-min-z", type=float, default=0.05, help="minimum dorsal world z component")
parser.add_argument("--palm-down-max-deg", type=float, default=20.0, help="max angle from palm normal to world down; negative disables")
parser.add_argument("--require-zero-triangle", dest="require_zero_triangle", action="store_true", default=True)
parser.add_argument("--no-require-zero-triangle", dest="require_zero_triangle", action="store_false")
parser.add_argument("--reject-palm-collision", dest="reject_palm_collision", action="store_true", default=True)
parser.add_argument("--no-reject-palm-collision", dest="reject_palm_collision", action="store_false")
parser.add_argument("--palm-collision-margin", type=float, default=0.002, help="palm AABB collision margin in meters")
parser.add_argument("--reach-min-fingers", type=int, default=2, help="1 or 2 reachable fingers among index/middle")
parser.add_argument("--reach-margin", type=float, default=0.020, help="cap side contact margin in meters")
parser.add_argument("--reach-mcp-steps", type=int, default=17, help="MCP grid steps for reachability filter")
parser.add_argument("--require-thumb", dest="require_thumb", action="store_true", default=True)
parser.add_argument("--no-require-thumb", dest="require_thumb", action="store_false")
parser.add_argument("--thumb-reach-margin", type=float, default=0.025, help="loose thumb cap side contact margin in meters")
parser.add_argument(
    "--thumb-mcp-outside-margin",
    type=optional_float,
    default=None,
    help="extra radial clearance for thumb MCP outside cap radius; none/off disables",
)
parser.add_argument("--thumb-opposition-steps", type=int, default=7, help="thumb opposition/abduction grid steps")
parser.add_argument("--thumb-flex-steps", type=int, default=11, help="thumb flexion grid steps")
parser.add_argument("--thumb-curl-steps", type=int, default=3, help="thumb PIP/DIP sparse curl grid steps")
parser.add_argument("--max-tries", type=int, default=3000, help="sampler retry budget per wrist pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

import torch  # noqa: E402

if str(args.device).startswith("cuda") and not torch.cuda.is_available():
    print(f"[device] {args.device} unavailable for torch; falling back to cpu")
    args.device = "cpu"

sim_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, Vt  # noqa: E402

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


ROOT = Path("/home/leegyuwon/Documents/task1")
sys.path.insert(0, str(ROOT / "tools"))
from wrist_init import (  # noqa: E402
    CAP_HEIGHT,
    CAP_RADIUS,
    PALM_LOCAL,
    _ray_hits_cap_cylinder,
    cap_center_in_zero_triangle_xy_mask,
    cap_palm_collision_mask,
    mcp_reach_contact_mask,
    sample_wrist_init,
    sample_wrist_init_mcp_reachable,
    thumb_mcp_outside_cap_mask,
    thumb_reach_contact_mask,
)


def quat_wxyz_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    r = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    r[:, 0, 0], r[:, 0, 1], r[:, 0, 2] = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
    r[:, 1, 0], r[:, 1, 1], r[:, 1, 2] = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    r[:, 2, 0], r[:, 2, 1], r[:, 2, 2] = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    return r


def place_mesh(stage, path: str, mesh_file: Path, pos, color):
    mesh_data = trimesh.load(str(mesh_file), process=False)
    if isinstance(mesh_data, trimesh.Scene):
        mesh_data = trimesh.util.concatenate(list(mesh_data.geometry.values()))
    verts = np.asarray(mesh_data.vertices, dtype=np.float32)
    faces = np.asarray(mesh_data.faces, dtype=np.int32)

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*pos))
    return mesh


def place_backdrop(stage, cap_z: float):
    floor_z = cap_z - 0.012
    verts = np.array(
        [
            [-0.34, -0.34, floor_z],
            [0.34, -0.34, floor_z],
            [0.34, 0.34, floor_z],
            [-0.34, 0.34, floor_z],
            [-0.34, 0.20, floor_z],
            [0.34, 0.20, floor_z],
            [0.34, 0.20, floor_z + 0.34],
            [-0.34, 0.20, floor_z + 0.34],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, "/World/backdrop")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateFaceVertexCountsAttr([4, 4])
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.035, 0.04, 0.045)])
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
sim.set_camera_view(eye=(0.45, -0.55, 0.50), target=(0.0, 0.0, args.cap_z + 0.03))

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr(1200.0)
place_backdrop(stage, args.cap_z)

generator = torch.Generator(device="cpu").manual_seed(args.seed)
cap = torch.zeros(args.num, 3, dtype=torch.float32)
cap[:, 2] = args.cap_z
palm_down_max_deg = None if args.palm_down_max_deg < 0 else args.palm_down_max_deg
if args.sampler == "mcp_reachable":
    pos, quat, ok = sample_wrist_init_mcp_reachable(
        cap,
        z_range=(args.z_min, args.z_max),
        r_range=(args.r_min, args.r_max),
        palm_tol_deg=args.palm_tol,
        dorsal_min_z=args.dorsal_min_z,
        palm_down_max_deg=palm_down_max_deg,
        contact_margin=args.reach_margin,
        min_reach_fingers=args.reach_min_fingers,
        require_thumb=args.require_thumb,
        thumb_contact_margin=args.thumb_reach_margin,
        thumb_mcp_outside_margin=args.thumb_mcp_outside_margin,
        require_cap_center_in_zero_triangle=args.require_zero_triangle,
        reject_palm_collision=args.reject_palm_collision,
        palm_collision_margin=args.palm_collision_margin,
        thumb_opposition_steps=args.thumb_opposition_steps,
        thumb_flex_steps=args.thumb_flex_steps,
        thumb_curl_steps=args.thumb_curl_steps,
        grid_steps=args.reach_mcp_steps,
        max_tries=args.max_tries,
        generator=generator,
    )
else:
    pos, quat, ok = sample_wrist_init(
        cap,
        z_range=(args.z_min, args.z_max),
        r_range=(args.r_min, args.r_max),
        palm_tol_deg=args.palm_tol,
        dorsal_min_z=args.dorsal_min_z,
        palm_down_max_deg=palm_down_max_deg,
        thumb_mcp_outside_margin=args.thumb_mcp_outside_margin,
        require_cap_center_in_zero_triangle=args.require_zero_triangle,
        reject_palm_collision=args.reject_palm_collision,
        palm_collision_margin=args.palm_collision_margin,
        max_tries=args.max_tries,
        generator=generator,
    )
if not bool(ok.all()):
    raise RuntimeError(f"sampling failed: {int(ok.sum())}/{args.num}")

r = quat_wxyz_to_rot(quat)
palm_w = torch.einsum("bij,j->bi", r, PALM_LOCAL)
palm_down_angle = torch.rad2deg(torch.acos((-palm_w[:, 2]).clamp(-1.0, 1.0)))
zero_triangle_ok = cap_center_in_zero_triangle_xy_mask(pos, quat, cap)
palm_collision = cap_palm_collision_mask(pos, quat, cap, palm_margin=args.palm_collision_margin)
cap_hit = _ray_hits_cap_cylinder(pos, palm_w, cap, CAP_RADIUS, CAP_HEIGHT)
reach_ok, finger_touch, min_dist = mcp_reach_contact_mask(
    pos,
    quat,
    cap,
    contact_margin=args.reach_margin,
    min_reach_fingers=args.reach_min_fingers,
    grid_steps=args.reach_mcp_steps,
)
thumb_ok, thumb_dist = thumb_reach_contact_mask(
    pos,
    quat,
    cap,
    contact_margin=args.thumb_reach_margin,
    opposition_steps=args.thumb_opposition_steps,
    flex_steps=args.thumb_flex_steps,
    curl_steps=args.thumb_curl_steps,
)
raw_thumb_mcp_ok, thumb_mcp_radius = thumb_mcp_outside_cap_mask(
    pos,
    quat,
    cap,
    outside_margin=0.0 if args.thumb_mcp_outside_margin is None else args.thumb_mcp_outside_margin,
)
thumb_mcp_ok = torch.ones_like(raw_thumb_mcp_ok) if args.thumb_mcp_outside_margin is None else raw_thumb_mcp_ok

place_mesh(
    stage,
    "/World/cap",
    ROOT / "assets/tumbler/meshes/cap_visual.obj",
    (0.0, 0.0, args.cap_z),
    (0.1, 0.7, 0.9),
)

hand_usd = str(ROOT / "assets/usd/dg5fs_right.usd")
for i, (p, q, p_ang, tri_ok, palm_col, hit, reach, touch, dist, th_ok, th_dist, th_mcp_ok, th_mcp_r) in enumerate(
    zip(
        pos.cpu().numpy(),
        quat.cpu().numpy(),
        palm_down_angle.cpu().numpy(),
        zero_triangle_ok.cpu().numpy(),
        palm_collision.cpu().numpy(),
        cap_hit.cpu().numpy(),
        reach_ok.cpu().numpy(),
        finger_touch.cpu().numpy(),
        min_dist.cpu().numpy(),
        thumb_ok.cpu().numpy(),
        thumb_dist.cpu().numpy(),
        thumb_mcp_ok.cpu().numpy(),
        thumb_mcp_radius.cpu().numpy(),
    )
):
    holder = UsdGeom.Xform.Define(stage, f"/World/dg5fs_{i:02d}")
    xf = UsdGeom.Xformable(holder)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    stage.DefinePrim(f"/World/dg5fs_{i:02d}/ref").GetReferences().AddReference(hand_usd)

    thumb_mcp_status = "off" if args.thumb_mcp_outside_margin is None else str(bool(th_mcp_ok))
    print(
        f"{i:02d}: pos={np.round(p, 4)} quat_wxyz={np.round(q, 4)} "
        f"palm_down_angle={p_ang:.1f}deg zero_triangle_xy={bool(tri_ok)} "
        f"palm_collision={bool(palm_col)} cap_hit={bool(hit)} mcp_reach={bool(reach)} "
        f"touch(index,middle)={touch.astype(bool).tolist()} min_dist_mm={np.round(dist * 1000, 1)} "
        f"thumb_reach={bool(th_ok)} thumb_min_dist_mm={th_dist * 1000:.1f} "
        f"thumb_mcp_outside={thumb_mcp_status} thumb_mcp_radius_mm={th_mcp_r * 1000:.1f}"
    )

thumb_mcp_margin_str = "off" if args.thumb_mcp_outside_margin is None else f"{args.thumb_mcp_outside_margin:.3f}m"
print(
    f"\nGUI ready: sampler={args.sampler}, require_thumb={args.require_thumb}, "
    f"r={args.r_min:.3f}~{args.r_max:.3f}m, z={args.z_min:.3f}~{args.z_max:.3f}m, "
    f"palm_down_max={palm_down_max_deg}, thumb_margin={args.thumb_reach_margin:.3f}m, "
    f"zero_triangle_xy={args.require_zero_triangle}, thumb_mcp_outside_margin={thumb_mcp_margin_str}, "
    f"reject_palm_collision={args.reject_palm_collision}:{args.palm_collision_margin:.3f}m, "
    "cyan cap and dg5fs samples."
)
print("Close the Isaac Sim window to exit.\n")

sim.reset()
while sim_app.is_running():
    sim.step()

sim_app.close()
