"""원격 Isaac Lab 정책의 궤적 CSV 를 로컬 IsaacGym 에서 kinematic 재생.

CSV = step, q0..q21 (IsaacGym 순서: 팔6 + 손16[J0-15 원순서]), cap_spin, cap_lift.
로봇 22관절 + 텀블러(cap_spin/cap_lift)를 프레임마다 세팅해 60fps 재생.
    SPACE 일시정지, R 처음부터, ESC 종료
사용: python lab_replay_view.py <traj.csv>
"""
import csv
import os
import sys

import numpy as np
from isaacgym import gymapi

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
URDF = "rb5_allegro_vmount_siltip2.urdf"
TUMBLER_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"
TUMBLER = "tumbler_cyl_r44_fric02.urdf"
TPOS = [0.4, 0.0, 0.0]

rows = list(csv.DictReader(open(sys.argv[1])))
frames = []
for r in rows:
    q22 = [float(r[f"q{i}"]) for i in range(22)]
    frames.append((q22, float(r["cap_spin"]), float(r["cap_lift"])))
print(f"프레임 {len(frames)}개 로드")

gym = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 1 / 60; sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0, 0, 0)
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
o = gymapi.AssetOptions(); o.fix_base_link = True; o.disable_gravity = True; o.collapse_fixed_joints = False
robot = gym.load_asset(sim, URDF_DIR, URDF, o)
tum = gym.load_asset(sim, TUMBLER_DIR, TUMBLER, o)
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
ra = gym.create_actor(env, robot, gymapi.Transform(), "robot", 0, 1)
tt = gymapi.Transform(); tt.p = gymapi.Vec3(*TPOS)
ta = gym.create_actor(env, tum, tt, "tumbler", 0, 1)
n_r = gym.get_actor_dof_count(env, ra)
tn = list(gym.get_actor_dof_names(env, ta))
i_spin, i_lift = tn.index("cap_spin"), tn.index("cap_lift")
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(TPOS[0] + 0.55, 0.55, 0.65), gymapi.Vec3(TPOS[0], 0, 0.3))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "pause")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "rewind")
k = 0; paused = False
rs = np.zeros(n_r, dtype=gymapi.DofState.dtype)
ts = np.zeros(len(tn), dtype=gymapi.DofState.dtype)
while not gym.query_viewer_has_closed(viewer):
    for ev in gym.query_viewer_action_events(viewer):
        if ev.action == "pause" and ev.value > 0: paused = not paused
        if ev.action == "rewind" and ev.value > 0: k = 0
    q22, spin, lift = frames[min(k, len(frames) - 1)]
    rs["pos"] = np.array(q22, dtype=np.float32)
    gym.set_actor_dof_states(env, ra, rs, gymapi.STATE_ALL)
    ts["pos"][i_spin] = spin; ts["pos"][i_lift] = lift
    gym.set_actor_dof_states(env, ta, ts, gymapi.STATE_ALL)
    if not paused: k = min(k + 1, len(frames) - 1)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
