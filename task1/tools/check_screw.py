"""tumbler.urdf 의 mimic 조인트가 진짜 나사 구속으로 동작하는지 검증한다.

확인할 것 두 가지:
  1. cap_spin 을 강제로 돌렸을 때 cap_lift 가 pitch/2pi 배율로 따라오는가
  2. env 1개 기준 step/s (SDF 나사산 방식의 89 step/s 와 비교)

사용법:
    python task1/tools/check_screw.py --headless
    python task1/tools/check_screw.py            # GUI 로 눈으로도 확인
"""

import argparse
import time

_T0 = time.time()

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--spin-vel", type=float, default=2.0, help="cap_spin 목표 각속도 (rad/s)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

sim_app = AppLauncher(args).app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

URDF = "/home/leegyuwon/Documents/task1/assets/tumbler/tumbler.urdf"
PITCH = 0.004
EXPECTED = PITCH / (2 * math.pi)  # m/rad

# convert_mimic_joints_to_normal_joints=False 여야 mimic 이 PhysxMimicJointAPI 로 남는다.
# True 로 두면 그냥 독립 prismatic 이 되어 나사 구속이 사라진다.
conv = UrdfConverter(
    UrdfConverterCfg(
        asset_path=URDF,
        usd_dir="/home/leegyuwon/Documents/task1/assets/usd",
        usd_file_name="tumbler.usd",
        fix_base=True,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        collider_type="convex_decomposition",
        self_collision=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=10.0, damping=1.0),
        ),
    )
)
print(f"\n[convert] {conv.usd_path}")

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim.set_camera_view(eye=(0.45, 0.45, 0.40), target=(0.0, 0.0, 0.24))
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
        # cap_spin 을 속도 제어로 돌려서 cap_lift 가 따라오는지 본다.
        actuators={"spin": ImplicitActuatorCfg(joint_names_expr=["cap_spin"], stiffness=0.0, damping=5.0)},
    )
)

sim.reset()
_T_READY = time.time()

names = tumbler.data.joint_names
print(f"\n[joints] {names}")
if "cap_spin" not in names:
    print("!! cap_spin 이 없다 — URDF 변환이 조인트를 삼켰다")
i_spin = names.index("cap_spin")
i_lift = names.index("cap_lift") if "cap_lift" in names else None
print(
    f"[mimic] cap_lift 가 독립 DOF 로 남아있는가: {'예 (mimic 미적용 의심)' if i_lift is not None else '아니오 (mimic 으로 흡수됨 — 정상)'}"
)
print(f"[timing] 기동+씬: {_T_READY - _T0:.1f}s")
print(f"\n이론 배율 dz/dθ = {EXPECTED:.6e} m/rad  (pitch {PITCH * 1000:.0f}mm)\n")

# 뚜껑 링크의 월드 z 를 직접 읽어 회전 대비 상승량을 본다.
body_names = tumbler.data.body_names
i_cap = body_names.index("cap") if "cap" in body_names else -1
print(f"[bodies] {body_names}  -> cap index {i_cap}\n")

target = torch.full((1, 1), args.spin_vel, device=sim.device)
prev_ang, prev_z, step = None, None, 0
_T_LOOP = time.time()

while sim_app.is_running() and step < args.steps:
    tumbler.set_joint_velocity_target(target, joint_ids=[i_spin])
    tumbler.write_data_to_sim()
    sim.step()
    tumbler.update(sim.get_physics_dt())

    step += 1
    if step % 120 == 0:
        ang = tumbler.data.joint_pos[0, i_spin].item()
        z = tumbler.data.body_pos_w[0, i_cap, 2].item() if i_cap >= 0 else float("nan")
        if prev_ang is not None and abs(ang - prev_ang) > 1e-6:
            ratio = (z - prev_z) / (ang - prev_ang)
            verdict = "OK" if abs(ratio - EXPECTED) < 0.25 * EXPECTED else "불일치"
            print(f"  θ={ang:+7.2f} rad  z={z:.5f} m   dz/dθ={ratio:+.3e}  [{verdict}]")
        prev_ang, prev_z = ang, z

_loop = time.time() - _T_LOOP
print(
    f"\n[timing] 기동 {_T_READY - _T0:.1f}s | {step} 스텝 {_loop:.1f}s = {step / max(_loop, 1e-9):.0f} step/s"
    f"   (SDF 나사산 방식은 89 step/s 였다)"
)
sim_app.close()
