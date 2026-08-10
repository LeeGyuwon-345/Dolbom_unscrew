"""텀블러 URDF 를 Isaac Gym GUI 로 빠르게 확인한다. (Python 3.8 / maniptrans 환경)

Isaac Sim 은 기동에만 20~120초가 걸리는데, 형상·크기·조인트 배치만 보고 싶을 때는
Isaac Gym 이 몇 초 만에 뜬다. 학습은 Isaac Sim 에서 하고, 눈으로 보는 건 여기서 한다.

주의: Isaac Gym 에는 mimic/기어 조인트가 없다. 그래서 나사 구속을 물리로 걸 수 없고,
여기서는 cap_lift = (pitch/2pi) * cap_spin 을 매 스텝 직접 써넣어 '보여주기만' 한다.
진짜 구속(힘이 양방향으로 전달되는)은 Isaac Sim 의 PhysxMimicJointAPI 쪽에서만 성립한다.

조작:
    A / D    뚜껑 회전 (A=푸는 방향, D=조이는 방향)
    R        리셋
    ESC      종료

사용법:
    conda activate maniptrans
    python task1/tools/view_tumbler_gym.py
"""

from __future__ import print_function

import math
import os

from isaacgym import gymapi

ASSET_ROOT = "/home/leegyuwon/Documents/task1/assets/tumbler"
URDF_FILE = "tumbler.urdf"
PITCH = 0.004
MULT = PITCH / (2 * math.pi)  # m/rad
SPIN_STEP = 0.03  # 키 한 번당 회전량 (rad)

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise RuntimeError("create_sim 실패")

plane = gymapi.PlaneParams()
plane.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane)

opts = gymapi.AssetOptions()
opts.fix_base_link = True
opts.collapse_fixed_joints = False
opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
opts.vhacd_enabled = False  # 형상 확인용이므로 볼록분해 생략 (로딩이 훨씬 빠르다)

print("[load] %s/%s" % (ASSET_ROOT, URDF_FILE))
asset = gym.load_asset(sim, ASSET_ROOT, URDF_FILE, opts)
if asset is None:
    raise RuntimeError("URDF 로드 실패")

n_dof = gym.get_asset_dof_count(asset)
dof_names = gym.get_asset_dof_names(asset)
body_names = gym.get_asset_rigid_body_names(asset)
print("[asset] DOF %d %s" % (n_dof, dof_names))
print("[asset] body %s" % (body_names,))

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
actor = gym.create_actor(env, asset, pose, "tumbler", 0, 0)

# 두 DOF 모두 위치 제어로 잡아둔다 (mimic 이 없으니 직접 써넣어야 한다).
props = gym.get_actor_dof_properties(env, actor)
for i in range(n_dof):
    props["driveMode"][i] = gymapi.DOF_MODE_POS
    props["stiffness"][i] = 1e4
    props["damping"][i] = 1e2
gym.set_actor_dof_properties(env, actor, props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise RuntimeError("viewer 생성 실패 (DISPLAY 확인)")
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.45, 0.45, 0.42), gymapi.Vec3(0, 0, 0.24))

for key, action in [
    (gymapi.KEY_A, "spin_open"),
    (gymapi.KEY_D, "spin_close"),
    (gymapi.KEY_R, "reset"),
    (gymapi.KEY_ESCAPE, "quit"),
]:
    gym.subscribe_viewer_keyboard_event(viewer, key, action)

i_spin = dof_names.index("cap_spin")
i_lift = dof_names.index("cap_lift") if "cap_lift" in dof_names else None

print("\n" + "=" * 62)
print("  A/D 뚜껑 회전   R 리셋   ESC 종료")
print("  pitch %.0f mm -> 1회전당 %.1f mm 상승 (배율 %.3e m/rad)" % (PITCH * 1000, PITCH * 1000, MULT))
print("  * Isaac Gym 에는 나사 구속이 없어 운동학으로 흉내낸 것이다")
print("=" * 62 + "\n")

angle = 0.0
targets = [0.0] * n_dof
quit_now = False

while not gym.query_viewer_has_closed(viewer) and not quit_now:
    for ev in gym.query_viewer_action_events(viewer):
        if ev.value == 0:
            continue
        if ev.action == "spin_open":
            angle += SPIN_STEP
        elif ev.action == "spin_close":
            angle -= SPIN_STEP
        elif ev.action == "reset":
            angle = 0.0
        elif ev.action == "quit":
            quit_now = True

    # 나사 관계를 손으로 강제한다: 올라간 높이 = 배율 * 회전각, 0~max 로 제한.
    lift = min(max(MULT * angle, 0.0), 0.020)
    targets[i_spin] = angle
    if i_lift is not None:
        targets[i_lift] = lift

    gym.set_actor_dof_position_targets(env, actor, targets)

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
print("[exit] 회전 %.2f rad = %.2f 바퀴,  상승 %.1f mm" % (angle, angle / (2 * math.pi), lift * 1000))
