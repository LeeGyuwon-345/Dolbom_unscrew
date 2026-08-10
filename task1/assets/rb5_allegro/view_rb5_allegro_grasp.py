"""RB5-850E + Allegro (tcp -y 10mm 고정 마운트) 결합 URDF 뷰어.

    SPACE   관절 이름/현재각/한계 출력
    J       관절 사인 스윕 토글 (축 방향 확인용, ±20도)
    ESC     종료
"""
import math
from isaacgym import gymapi, gymtorch
import torch

HERE = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"

gym = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 1 / 60; sp.substeps = 2
sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.physx.solver_type = 1; sp.physx.use_gpu = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane)

ao = gymapi.AssetOptions()
ao.fix_base_link = True
ao.default_dof_drive_mode = gymapi.DOF_MODE_POS
ao.collapse_fixed_joints = False
import os as _os0
asset = gym.load_asset(sim, HERE, _os0.environ.get("ROBOT_URDF", "rb5_allegro.urdf"), ao)
to = gymapi.AssetOptions(); to.fix_base_link = True
tum = gym.load_asset(sim, "/home/leegyuwon/Documents/task1/assets/tumbler", "tumbler_cyl.urdf", to)
nd = gym.get_asset_dof_count(asset)
names = gym.get_asset_dof_names(asset)
print("관절:", names)

env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
import sys as _sys
_sys.path.insert(0, "/home/leegyuwon/Documents/task1/tools")
from screw_coupling import ScrewCoupling
import os as _os1
_tp = [float(v) for v in _os1.environ.get("TUMBLER_POS", "0.6,0,0").split(",")]
tf = gymapi.Transform(); tf.p = gymapi.Vec3(*_tp)   # 텀블러 위치 (TUMBLER_POS)
tum_act = gym.create_actor(env, tum, tf, "tumbler", 0, 2)
sc = ScrewCoupling.from_urdf("/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl.urdf")
sc.bind(gym, env, tum_act, device="cpu", num_envs=1)
act = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro", 0, 1)
pr = gym.get_actor_dof_properties(env, act)
for k in range(nd):
    pr["driveMode"][k] = gymapi.DOF_MODE_POS
    if names[k].startswith("joint_"):
        pr["stiffness"][k] = 300.0; pr["damping"][k] = 18.0   # allegro
    else:
        pr["stiffness"][k] = 20000.0; pr["damping"][k] = 400.0   # RB5 (자세 유지용 고강성)
        pr["effort"][k] = 300.0   # URDF effort=10 은 placeholder -- 자중도 못 버팀
gym.set_actor_dof_properties(env, act, pr)
for _b in range(gym.get_actor_rigid_body_count(env, act)):
    gym.set_rigid_body_color(env, act, _b, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.9, 0.9))
lims = [(float(pr["lower"][k]), float(pr["upper"][k])) for k in range(nd)]

gym.prepare_sim(sim)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.9, -1.5, 1.0), gymapi.Vec3(0.4, 0, 0.4))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "info")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_J, "sweep")

ds = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
tgt = torch.zeros(nd)
import json as _json
import os as _os
# z10b = 처짐 보상(+10mm) + 원판 IK 가지 (팔 세션 최종 검증판). 구판 비교는
# ARM_IK_JSON=arm_ik_grasp.json 으로.
_ik = _json.load(open(_os.environ.get("ARM_IK_JSON",
    "/home/leegyuwon/Documents/task1/assets/rb5_allegro/arm_ik_grasp_z10b.json")))
for _n, _v in zip(_ik["names"], _ik["q"]):
    tgt[names.index(_n)] = _v            # RB5: IK 로 푼 파지 접근 자세
# 기본 = 닫힌 원파지(웜스타트 검증판). 폄판은 FINGER_POSE_JSON=...open01.json
_gp = _json.load(open(_os.environ.get("FINGER_POSE_JSON",
    "/home/leegyuwon/Documents/task1/tools2_allegro/poses/grasp_al_r50.json")))
_order = ["joint_0.0","joint_1.0","joint_2.0","joint_3.0","joint_12.0","joint_13.0",
          "joint_14.0","joint_15.0","joint_4.0","joint_5.0","joint_6.0","joint_7.0",
          "joint_8.0","joint_9.0","joint_10.0","joint_11.0"]
for _n, _v in zip(_order, _gp["dof_pos"]):
    tgt[names.index(_n)] = _v            # allegro: 파지 초기 관절각
# 시작 상태도 목표 자세로 (드라이브가 끌고 가는 과도 동작 생략)
import isaacgym.gymtorch as _gt
_ds = _gt.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1,2)
_off = gym.get_actor_dof_index(env, act, 0, gymapi.DOMAIN_SIM)   # 텀블러가 앞 6칸
_ds[_off:_off+nd,0] = tgt; _ds[:,1] = 0.0
gym.set_dof_state_tensor(sim, _gt.unwrap_tensor(_ds))
sweep = False
t = 0.0

print("SPACE 관절 정보, J 스윕 토글, ESC 종료", flush=True)
while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value == 0:
            continue
        if e.action == "info":
            gym.refresh_dof_state_tensor(sim)
            print("\n 관절      현재각도    한계도")
            for k in range(nd):
                print(f"  {names[k]:9s} {math.degrees(float(ds[_off+k,0])):+8.1f}  "
                      f"[{math.degrees(lims[k][0]):+.0f}, {math.degrees(lims[k][1]):+.0f}]")
        elif e.action == "sweep":
            sweep = not sweep
            print("  스윕", "켬" if sweep else "끔 (0 복귀)", flush=True)
            if not sweep:
                for _n, _v in zip(_ik["names"], _ik["q"]):
                    tgt[names.index(_n)] = _v
                for _n, _v in zip(_order, _gp["dof_pos"]):
                    tgt[names.index(_n)] = _v
    if sweep:
        t += 1 / 60
        for k in range(nd):
            tgt[k] = math.radians(20) * math.sin(2 * math.pi * 0.15 * t + k * 0.9)
    _full = torch.zeros(_ds.shape[0]); _full[_off:_off+nd] = tgt
    gym.refresh_dof_state_tensor(sim)
    _ft = _full.unsqueeze(0)
    sc.apply(_ds[:, 0].unsqueeze(0), _ft)     # 나사산: lift 가 spin 을 따라감
    _full = _ft[0]
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(_full))
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
