"""RB5+Allegro 전체를 전관절 0 자세로 Isaac Gym 에 띄운다.

CSV(관절 명령, rad)를 실기에 넣기 전에 영점 자세가 실물 로봇의 영점과 같은
모습인지 눈으로 대조하는 용도다. 베이스 고정, 중력 없음, 매 프레임 관절을
0 으로 되쓰므로 화면의 자세가 곧 "모든 인코더 0" 이다.

    SPACE   DOF 이름·현재각 출력 (CSV 열 순서와 동일)
    ESC     종료

사용법:
    python view_zero_pose.py
    python view_zero_pose.py --q 0,0,1.57,0,0,0   # 팔 6축만 별도 지정(rad)
"""

import os
import sys

import numpy as np
from isaacgym import gymapi

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
URDF = os.path.basename(os.environ.get("ROBOT_URDF", "rb5_allegro.urdf"))

q_arm = None
if "--q" in sys.argv:
    q_arm = [float(v) for v in sys.argv[sys.argv.index("--q") + 1].split(",")]
    assert len(q_arm) == 6, "--q 는 팔 6축(rad) 콤마 구분"

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 60.0
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0)
sp.physx.solver_type = 1
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

opts = gymapi.AssetOptions()
opts.fix_base_link = True
opts.disable_gravity = True
opts.collapse_fixed_joints = False
asset = gym.load_asset(sim, URDF_DIR, URDF, opts)
n_dof = gym.get_asset_dof_count(asset)
dof_names = [gym.get_asset_dof_name(asset, i) for i in range(n_dof)]

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
actor = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro", 0, 0)

# 위치 드라이브로 잡아두면 떨림 없이 정지 자세가 유지된다.
props = gym.get_actor_dof_properties(env, actor)
props["driveMode"][:] = gymapi.DOF_MODE_POS
props["stiffness"][:] = 400.0
props["damping"][:] = 40.0
gym.set_actor_dof_properties(env, actor, props)

# Allegro 메시는 URDF 에 재질이 없어 뷰어에서 검게 나온다. 링크별로 직접
# 칠한다 -- 손가락마다 색을 달리해 CSV 인덱스 대조(0~3 검지 / 4~7 중지 /
# 8~11 약지 / 12~15 엄지)에 쓴다.
FINGER_COLORS = {
    "link_0": (0.9, 0.3, 0.3),   # 검지 빨강
    "link_1": (0.9, 0.3, 0.3),
    "link_2": (0.9, 0.3, 0.3),
    "link_3": (0.9, 0.3, 0.3),
    "link_4": (0.3, 0.8, 0.3),   # 중지 초록
    "link_5": (0.3, 0.8, 0.3),
    "link_6": (0.3, 0.8, 0.3),
    "link_7": (0.3, 0.8, 0.3),
    "link_8": (0.3, 0.5, 0.95),  # 약지 파랑
    "link_9": (0.3, 0.5, 0.95),
    "link_10": (0.3, 0.5, 0.95),
    "link_11": (0.3, 0.5, 0.95),
    "link_12": (0.95, 0.8, 0.2), # 엄지 노랑
    "link_13": (0.95, 0.8, 0.2),
    "link_14": (0.95, 0.8, 0.2),
    "link_15": (0.95, 0.8, 0.2),
}
n_body = gym.get_actor_rigid_body_count(env, actor)
for i in range(n_body):
    name = gym.get_actor_rigid_body_names(env, actor)[i]
    if name.startswith("link_") and "." in name:
        key = "link_" + name.split("_")[1].split(".")[0]
        c = FINGER_COLORS.get(key)
    elif name == "base_link":
        c = (0.35, 0.35, 0.4)    # 손바닥 짙은 회색
    elif name.startswith("link"):
        c = (0.75, 0.75, 0.78)   # 팔 밝은 회색
    else:
        c = (0.6, 0.6, 0.65)
    if c:
        gym.set_rigid_body_color(env, actor, i, gymapi.MESH_VISUAL, gymapi.Vec3(*c))

q0 = np.zeros(n_dof, dtype=np.float32)
if q_arm is not None:
    q0[:6] = q_arm
state = np.zeros(n_dof, dtype=gymapi.DofState.dtype)
state["pos"] = q0
gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env, actor, q0)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.2, 1.2, 1.2), gymapi.Vec3(0, 0, 0.5))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "dump")

print(f"DOF {n_dof}개, 전부 0 (CSV 열 순서 = 아래 순서)")
for i, n in enumerate(dof_names):
    tag = f"arm_cmd_j{i+1}" if i < 6 else f"hand_cmd_{i-6}"
    print(f"  {i:2d} {tag:12s} {n}")

while not gym.query_viewer_has_closed(viewer):
    for ev in gym.query_viewer_action_events(viewer):
        if ev.action == "dump" and ev.value > 0:
            st = gym.get_actor_dof_states(env, actor, gymapi.STATE_POS)
            for i, n in enumerate(dof_names):
                print(f"  {i:2d} {n:14s} {st['pos'][i]:+.4f} rad")
    gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
