"""RB5-850E + 실물 90도 마운트(Body1.STL) + Allegro 결합 URDF 뷰어.

    관절각은 전부 0 고정 (position target 0, 한계 밖이면 한계값에 클램프)
    SPACE   관절 이름/현재각/한계 출력
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
ao.disable_gravity = True          # 0도 고정 확인용이라 중력 영향 제거
asset = gym.load_asset(sim, HERE, "rb5_allegro_realmount.urdf", ao)
nd = gym.get_asset_dof_count(asset)
names = gym.get_asset_dof_names(asset)
print("관절:", names)

env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
act = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro_realmount", 0, 1)
pr = gym.get_actor_dof_properties(env, act)
for k in range(nd):
    pr["driveMode"][k] = gymapi.DOF_MODE_POS
    if names[k].startswith("joint_"):
        pr["stiffness"][k] = 300.0; pr["damping"][k] = 18.0   # allegro
    else:
        pr["stiffness"][k] = 400.0; pr["damping"][k] = 40.0   # RB5
gym.set_actor_dof_properties(env, act, pr)

# 마운트 링크만 색을 달리해서 구분
mount_idx = gym.find_actor_rigid_body_index(env, act, "mount", gymapi.DOMAIN_ACTOR)
for _b in range(gym.get_actor_rigid_body_count(env, act)):
    c = gymapi.Vec3(0.9, 0.55, 0.2) if _b == mount_idx else gymapi.Vec3(0.9, 0.9, 0.9)
    gym.set_rigid_body_color(env, act, _b, gymapi.MESH_VISUAL, c)
lims = [(float(pr["lower"][k]), float(pr["upper"][k])) for k in range(nd)]

gym.prepare_sim(sim)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.2, -1.0, 1.2), gymapi.Vec3(0.0, 0, 0.6))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "info")

# 관절 상태/타깃 모두 0 고정
ds = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
ds[:, 0] = 0.0; ds[:, 1] = 0.0
gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(ds.contiguous()))
tgt = torch.zeros(nd)

print("SPACE 관절 정보, ESC 종료", flush=True)
while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value == 0:
            continue
        if e.action == "info":
            gym.refresh_dof_state_tensor(sim)
            print("\n 관절      현재각도    한계도")
            for k in range(nd):
                print(f"  {names[k]:9s} {math.degrees(float(ds[k,0])):+8.1f}  "
                      f"[{math.degrees(lims[k][0]):+.0f}, {math.degrees(lims[k][1]):+.0f}]")
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(tgt))
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
