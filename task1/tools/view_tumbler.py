"""텀블러를 task1(Isaac Sim) 환경에서 GUI 로 열고, 나사 구속을 직접 확인한다.

URDF 의 <mimic> 태그는 변환에서 무시된다 (cap_lift 가 독립 DOF 로 남는 것으로 확인됨).
그래서 변환 후 USD 에 PhysxMimicJointAPI 를 직접 적용한다.

PhysX 구속식:  jointPosition + gearing * referenceJointPosition + offset = 0
  - jointPosition          = cap_lift 의 z (m)
  - referenceJointPosition = cap_spin 의 각도 (**degree**, 라디안 아님)
  - 원하는 관계            = z = (pitch/360) * theta_deg
  => gearing = -(pitch / 360)

조작:
    A / D    뚜껑 회전 (A=푸는 방향, D=조이는 방향)
    R        리셋
    ESC      종료

사용법:
    conda activate task1
    python task1/tools/view_tumbler.py
    python task1/tools/view_tumbler.py --headless      # 수치 검증만
"""

import argparse
import math
import time

_T0 = time.time()

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pitch", type=float, default=0.010, help="나사 pitch (m/rev)")
ap.add_argument("--spin-vel", type=float, default=4.0, help="회전 속도 (rad/s)")
ap.add_argument("--steps", type=int, default=1200, help="headless 검증 스텝 수")
ap.add_argument("--rebuild", action="store_true", help="URDF->USD 재변환 강제")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

sim_app = AppLauncher(args).app

import carb  # noqa: E402
import torch  # noqa: E402
from pxr import PhysxSchema, Usd  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

URDF = "/home/leegyuwon/Documents/task1/assets/tumbler/tumbler.urdf"
USD_DIR = "/home/leegyuwon/Documents/task1/assets/usd"
EXPECTED = args.pitch / (2 * math.pi)  # m/rad, 검증 기준

# force_usd_conversion=False 가 중요하다. True 로 두면 매 실행마다 볼록분해를 다시 돌려
# 기동에만 2분이 걸린다. 캐시되면 20초대.
conv = UrdfConverter(
    UrdfConverterCfg(
        asset_path=URDF,
        usd_dir=USD_DIR,
        usd_file_name="tumbler.usd",
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        collider_type="convex_decomposition",
        self_collision=False,
        force_usd_conversion=args.rebuild,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=5.0),
        ),
    )
)
print(f"\n[convert] {conv.usd_path}   ({time.time() - _T0:.1f}s)")


def apply_mimic(usd_path: str, pitch: float) -> bool:
    """cap_lift 가 cap_spin 을 나사 비율로 따라가도록 PhysxMimicJointAPI 를 붙인다."""
    stage = Usd.Stage.Open(usd_path)
    lift = stage.GetPrimAtPath("/tumbler/joints/cap_lift")
    spin = stage.GetPrimAtPath("/tumbler/joints/cap_spin")
    if not lift or not spin:
        print("!! cap_lift / cap_spin 프림을 못 찾음")
        return False

    # 단일 DOF 조인트라 instance name 토큰과 referenceJointAxis 는 실질적으로 무시된다
    # (스키마 문서: "for joint types with a single degree of freedom ... this attribute
    #  will be ignored since the axis is defined implicitly").
    api = PhysxSchema.PhysxMimicJointAPI.Apply(lift, "rotZ")
    api.CreateGearingAttr().Set(-pitch / 360.0)  # m/deg
    api.CreateOffsetAttr().Set(0.0)
    api.CreateReferenceJointRel().SetTargets([spin.GetPath()])
    stage.GetRootLayer().Save()
    print(f"[mimic] gearing = {-pitch / 360.0:.6e} m/deg  (pitch {pitch * 1000:.0f}mm/rev)")
    return True


apply_mimic(conv.usd_path, args.pitch)

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim.set_camera_view(eye=(0.45, 0.45, 0.42), target=(0.0, 0.0, 0.24))
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))

tumbler = Articulation(
    ArticulationCfg(
        prim_path="/World/Tumbler",
        spawn=sim_utils.UsdFileCfg(
            usd_path=conv.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["cap_spin"], stiffness=0.0, damping=5.0)},
    )
)

sim.reset()
_T_READY = time.time()

jn = tumbler.data.joint_names
bn = tumbler.data.body_names
i_spin = jn.index("cap_spin")
i_cap = bn.index("cap")
print(f"[joints] {jn}")
print(f"[timing] 기동+씬 {_T_READY - _T0:.1f}s")
print(f"\n이론 dz/dθ = {EXPECTED:.6e} m/rad\n")

state = {"vel": 0.0, "quit": False}
if not args.headless:
    import omni.appwindow  # noqa: E402

    def on_key(event, *_):
        n = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if n == "A":
                state["vel"] = args.spin_vel
            elif n == "D":
                state["vel"] = -args.spin_vel
            elif n == "R":
                tumbler.write_joint_state_to_sim(
                    torch.zeros_like(tumbler.data.joint_pos), torch.zeros_like(tumbler.data.joint_vel)
                )
            elif n == "ESCAPE":
                state["quit"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE and n in ("A", "D"):
            state["vel"] = 0.0
        return True

    kb = omni.appwindow.get_default_app_window().get_keyboard()
    sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(kb, on_key)
    print("=" * 62 + "\n  A/D 뚜껑 회전   R 리셋   ESC 종료\n" + "=" * 62 + "\n")
else:
    state["vel"] = args.spin_vel

target = torch.zeros(1, 1, device=sim.device)
prev_ang = prev_z = None
step = 0
_T_LOOP = time.time()

while sim_app.is_running() and not state["quit"]:
    if args.headless and step >= args.steps:
        break
    target[0, 0] = state["vel"]
    tumbler.set_joint_velocity_target(target, joint_ids=[i_spin])
    tumbler.write_data_to_sim()
    sim.step()
    tumbler.update(sim.get_physics_dt())

    step += 1
    if step % 120 == 0:
        ang = tumbler.data.joint_pos[0, i_spin].item()
        z = tumbler.data.body_pos_w[0, i_cap, 2].item()
        if prev_ang is not None and abs(ang - prev_ang) > 1e-6:
            ratio = (z - prev_z) / (ang - prev_ang)
            ok = "OK" if abs(ratio - EXPECTED) < 0.25 * EXPECTED else "불일치"
            print(f"  θ={ang:+8.2f} rad ({ang / (2 * math.pi):+5.2f}바퀴)  z={z:.5f}  dz/dθ={ratio:+.3e}  [{ok}]")
        prev_ang, prev_z = ang, z

_loop = time.time() - _T_LOOP
print(f"\n[timing] 기동 {_T_READY - _T0:.1f}s | {step} 스텝 {_loop:.1f}s = {step / max(_loop, 1e-9):.0f} step/s")
sim_app.close()
