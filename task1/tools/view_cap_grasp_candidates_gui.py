"""View learned cap grasp wrist pose candidates in Isaac Sim GUI.

Usage:
    python task1/tools/view_cap_grasp_candidates_gui.py --csv task1/logs/cap_grasp_success_pose_candidates_*.csv --num 10
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="", help="candidate/success CSV path; latest success candidate is used if empty")
parser.add_argument("--num", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, Vt  # noqa: E402

from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


ROOT = Path("/home/leegyuwon/Documents/task1")


def latest_csv() -> Path:
    patterns = [
        str(ROOT / "logs/cap_grasp_success_pose_candidates_*.csv"),
        str(ROOT / "logs/cap_grasp_success_*.csv"),
    ]
    files = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob.glob(pattern))
        if files:
            break
    if not files:
        raise FileNotFoundError("No cap grasp success CSV found under task1/logs")
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_rows(path: Path, n: int):
    with open(path, newline="") as fobj:
        rows = list(csv.DictReader(fobj))
    if "success_count" in rows[0]:
        rows.sort(key=lambda r: (int(float(r["success_count"])), int(float(r["latest_train_env_frames"]))), reverse=True)
    return rows[:n]


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
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
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


def f(row, key):
    return float(row[key])


csv_path = Path(args.csv) if args.csv else latest_csv()
if not csv_path.is_absolute():
    csv_path = Path("/home/leegyuwon/Documents") / csv_path
rows = read_rows(csv_path, args.num)
if not rows:
    raise RuntimeError(f"No rows in {csv_path}")

if str(args.device).startswith("cuda"):
    import torch  # noqa: E402

    if not torch.cuda.is_available():
        print(f"[device] {args.device} unavailable for torch; falling back to cpu")
        args.device = "cpu"

sim_app = AppLauncher(args).app
sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))

cap_pos = (f(rows[0], "cap_pos_x"), f(rows[0], "cap_pos_y"), f(rows[0], "cap_pos_z"))
sim.set_camera_view(eye=(0.45, -0.55, cap_pos[2] + 0.28), target=(cap_pos[0], cap_pos[1], cap_pos[2] + 0.03))

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr(1200.0)
place_backdrop(stage, cap_pos[2])
place_mesh(stage, "/World/cap", ROOT / "assets/tumbler/meshes/cap_visual.obj", cap_pos, (0.1, 0.7, 0.9))

hand_usd = str(ROOT / "assets/usd/dg5fs_right.usd")
for i, row in enumerate(rows):
    p = (f(row, "wrist_pos_x"), f(row, "wrist_pos_y"), f(row, "wrist_pos_z"))
    q = (
        f(row, "wrist_quat_wxyz_w"),
        f(row, "wrist_quat_wxyz_x"),
        f(row, "wrist_quat_wxyz_y"),
        f(row, "wrist_quat_wxyz_z"),
    )
    holder = UsdGeom.Xform.Define(stage, f"/World/dg5fs_{i:02d}")
    xf = UsdGeom.Xformable(holder)
    xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
    stage.DefinePrim(f"/World/dg5fs_{i:02d}/ref").GetReferences().AddReference(hand_usd)
    print(f"{i:02d}: pos={np.round(p, 4)} quat_wxyz={np.round(q, 4)}")

print(f"\nLoaded {len(rows)} wrist candidates from {csv_path}")
print("Close the Isaac Sim window to exit.\n")

sim.reset()
while sim_app.is_running():
    sim.step()

sim_app.close()
