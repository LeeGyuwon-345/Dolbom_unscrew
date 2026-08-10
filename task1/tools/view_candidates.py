"""unscrew 후보 오브젝트를 한 장면에 나란히 띄워 눈으로 비교한다.

    A열 (x=0)     factory M16 볼트/너트  - 실제 나사산, 물리 authoring 완료
    B열 (x=+0.4)  Office 물병 5종        - 통짜 메시, 뚜껑 없음
    C열 (x=-0.4)  OakInk 병 + 뚜껑       - 뚜껑 분리됨, 나사산 없음

사용법:
    conda activate env_isaaclab   # task1 클론이 끝나면 task1
    python task1/tools/view_candidates.py
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-hand", action="store_true", help="dg5fs 손을 빼고 오브젝트만 본다")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

sim_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, Vt  # noqa: E402

PREVIEW = "/home/leegyuwon/Documents/task1/assets/preview"
OAKINK = "/home/leegyuwon/Documents/ManipTrans/data/OakInk-v2/object_preview/align_ds"

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr(1200.0)


def place_reference(path: str, usd_file: str, pos, scale=1.0):
    """외부 USD를 참조로 붙이고 위치/스케일을 준다.

    참조된 USD가 이미 xformOpOrder 를 갖고 있으면 같은 prim 에 translate 를 더
    추가할 수 없다. 그래서 홀더 Xform 을 하나 두고 그 자식에 참조를 건다.
    """
    holder = UsdGeom.Xform.Define(stage, path)
    holder.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if scale != 1.0:
        holder.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    stage.DefinePrim(f"{path}/ref").GetReferences().AddReference(usd_file)
    return holder


def place_mesh(path: str, ply_file: str, pos, color):
    """OakInk .ply 를 UsdGeom.Mesh 로 직접 굽는다 (USD 변환 단계 생략)."""
    m = trimesh.load(ply_file, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    verts = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int32)

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*pos))
    size = verts.max(0) - verts.min(0)
    return size


print("\n" + "=" * 70)
print("  unscrew 후보 오브젝트  (단위: m)")
print("=" * 70)

# ---- A열: factory M16 볼트/너트 --------------------------------------------
# 너트를 볼트 나사산 위에 올려둔다. bolt 높이 0.035, base 0.010.
place_reference("/World/A_bolt_m16", f"{PREVIEW}/factory_bolt_m16.usd", (0.0, 0.0, 0.0))
place_reference("/World/A_nut_m16", f"{PREVIEW}/factory_nut_m16.usd", (0.0, 0.0, 0.020))
print("A  factory_bolt_m16   0.024 x 0.028 x 0.035   나사산 O, thread_pitch 0.002 m/rev")
print("A  factory_nut_m16    0.024 x 0.028 x 0.013   나사산 O")

# ---- B열: Office 물병 -------------------------------------------------------
for i, name in enumerate(["SM_BottleA", "SM_BottleB", "SM_BottleC", "SM_BottleD", "SM_BottleF"]):
    f = f"{PREVIEW}/{name}.usd"
    if os.path.exists(f):
        place_reference(f"/World/B_{name}", f, (0.4, i * 0.12 - 0.24, 0.0))
print("B  SM_BottleA~F       ~0.06 지름 x ~0.23 높이   통짜 메시, 뚜껑 없음")

# ---- C열: OakInk 병 + 뚜껑 --------------------------------------------------
# 00010 = green striped bottle cap, 00011 = 같은 병의 body.
# 원본 좌표계 그대로 두면 캡/바디가 제자리에 맞물린다.
for oid, color, label in [
    ("O02@0015@00010", (0.2, 0.8, 0.3), "cap"),
    ("O02@0015@00011", (0.8, 0.8, 0.8), "body"),
]:
    ply = f"{OAKINK}/{oid}/scan.ply"
    if os.path.exists(ply):
        size = place_mesh(f"/World/C_{oid.replace('@', '_')}", ply, (-0.4, 0.0, 0.0), color)
        print(f"C  {oid} {label:4s}  {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f}   뚜껑 분리 O, 나사산 X")
    else:
        print(f"C  {oid} -- .ply 없음: {ply}")

# ---- 크기 기준자: dg5fs 손 ---------------------------------------------------
# 24mm 너트 / 60mm 병 중 어느 쪽이 이 손으로 잡을 만한 크기인지가 선택의 핵심이라
# 손을 같이 띄운다. URDF -> USD 변환 결과는 캐시되므로 두 번째 실행부터는 즉시 로드된다.
if not args.no_hand:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

    urdf = "/home/leegyuwon/Documents/task1/assets/dg5fs_hand/dg5fs_right.urdf"
    conv = UrdfConverter(
        UrdfConverterCfg(
            asset_path=urdf,
            usd_dir="/home/leegyuwon/Documents/task1/assets/usd",
            usd_file_name="dg5fs_right.usd",
            fix_base=True,
            merge_fixed_joints=False,
            collider_type="convex_hull",
            self_collision=False,
            # ManipTrans 가 dg5fs 20 DOF 에 쓰던 값 (stiffness 500 / damping 30).
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=500.0, damping=30.0),
            ),
        )
    )
    place_reference("/World/D_dg5fs", conv.usd_path, (0.0, 0.35, 0.10))
    print(f"D  dg5fs_right       20 DOF / 26 body                        -> {conv.usd_path}")

print("=" * 70)
print("  좌: OakInk 병뚜껑   중앙: factory M16   우: Office 물병   뒤: dg5fs 손")
print("  창을 닫으면 종료됩니다.")
print("=" * 70 + "\n")

if args.headless:
    sim_app.close()
else:
    while sim_app.is_running():
        sim_app.update()
    sim_app.close()
