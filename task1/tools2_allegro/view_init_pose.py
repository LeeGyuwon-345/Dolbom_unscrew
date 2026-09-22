"""학습 초기자세(arm_grasp_init JSON)를 텀블러와 함께 Isaac Gym 에 띄운다.

view_zero_pose 의 초기자세판. 22관절 dof_pos 를 그대로 물리고 텀블러를
TUMBLER_POS 에 세워, 새 fingertip 이 캡과 어떤 간격/접촉으로 시작하는지
본다. 매 프레임 자세를 되쓰는 고정 모드라 떨림 없이 관찰만 한다.

    SPACE   관절각 출력
    ESC     종료

사용법:
    ROBOT_URDF=.../rb5_allegro_vmount_siltip2.urdf \
    python view_init_pose.py [poses/arm_grasp_init_closed_vmount.json]
"""

import json
import os
import sys

import numpy as np
from isaacgym import gymapi

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
URDF = os.path.basename(os.environ.get("ROBOT_URDF", "rb5_allegro_vmount_siltip2.urdf"))
POSE = (
    sys.argv[1]
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json")
    else "/home/leegyuwon/Documents/task1/tools2_allegro/poses/arm_grasp_init_closed_vmount.json"
)
TUMBLER_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"
TUMBLER = os.path.basename(
    os.environ.get("TUMBLER_URDF", f"{TUMBLER_DIR}/tumbler_cyl_r44_fric02.urdf")
)
TPOS = [float(v) for v in os.environ.get("TUMBLER_POS", "0.4,0,0").split(",")]

pose = json.load(open(POSE))
q_init = np.array(pose["dof_pos"], dtype=np.float32)
print(f"pose: {POSE}\n  note: {pose.get('note', '-')}\n  dof {len(q_init)}")

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
assert n_dof == len(q_init), f"URDF dof {n_dof} != pose dof {len(q_init)}"

topts = gymapi.AssetOptions()
topts.fix_base_link = True
topts.disable_gravity = True
tum = gym.load_asset(sim, TUMBLER_DIR, TUMBLER, topts)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
actor = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro", 0, 0)
tt = gymapi.Transform()
tt.p = gymapi.Vec3(*TPOS)
gym.create_actor(env, tum, tt, "tumbler", 0, 0)

props = gym.get_actor_dof_properties(env, actor)
props["driveMode"][:] = gymapi.DOF_MODE_POS
props["stiffness"][:] = 400.0
props["damping"][:] = 40.0
gym.set_actor_dof_properties(env, actor, props)

FINGER_COLORS = {
    "link_0": (0.9, 0.3, 0.3), "link_1": (0.9, 0.3, 0.3),
    "link_2": (0.9, 0.3, 0.3), "link_3": (0.9, 0.3, 0.3),
    "link_4": (0.3, 0.8, 0.3), "link_5": (0.3, 0.8, 0.3),
    "link_6": (0.3, 0.8, 0.3), "link_7": (0.3, 0.8, 0.3),
    "link_8": (0.3, 0.5, 0.95), "link_9": (0.3, 0.5, 0.95),
    "link_10": (0.3, 0.5, 0.95), "link_11": (0.3, 0.5, 0.95),
    "link_12": (0.95, 0.8, 0.2), "link_13": (0.95, 0.8, 0.2),
    "link_14": (0.95, 0.8, 0.2), "link_15": (0.95, 0.8, 0.2),
}
n_body = gym.get_actor_rigid_body_count(env, actor)
body_names = gym.get_actor_rigid_body_names(env, actor)
for i in range(n_body):
    name = body_names[i]
    if name.startswith("link_") and "." in name:
        key = "link_" + name.split("_")[1].split(".")[0]
        c = FINGER_COLORS.get(key)
    elif name == "base_link":
        c = (0.35, 0.35, 0.4)
    elif name.startswith("link"):
        c = (0.75, 0.75, 0.78)
    else:
        c = (0.6, 0.6, 0.65)
    if c:
        gym.set_rigid_body_color(env, actor, i, gymapi.MESH_VISUAL, gymapi.Vec3(*c))

state = np.zeros(n_dof, dtype=gymapi.DofState.dtype)
state["pos"] = q_init
gym.set_actor_dof_states(env, actor, state, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env, actor, q_init)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(
    viewer, None, gymapi.Vec3(TPOS[0] + 0.5, 0.5, 0.7), gymapi.Vec3(TPOS[0], 0, 0.3)
)
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "dump")

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
