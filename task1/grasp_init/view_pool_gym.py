"""노이즈 풀(make_init_pool.py)에서 뽑은 자세 여러 개를 고정 상태로 나란히 본다.

view_pose_gym.py 의 다인승 판. 정책도 물리도 돌리지 않고, 매 프레임 관절과
손목을 지정값으로 되써서 각 환경의 손을 풀 샘플 자세에 붙잡아 둔다.

    SPACE   새 샘플 6개 다시 뽑기
    ESC     종료

사용법:
    python view_pool_gym.py                                  # pool_m05.pt, 6개
    python view_pool_gym.py --pool ../tools2/poses/pool_m05.pt --n 6
"""

import argparse
import math
import os
import sys

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAPZ = 0.225
CAP = CapShape()

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default=os.path.join(HERE, "../tools2/poses/pool_m05.pt"))
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--overlap", action="store_true",
                help="손 N개를 같은 캡 위에 겹쳐 배치 (노이즈 퍼짐을 한눈에)")
args = ap.parse_args()

pool = torch.load(args.pool, map_location="cpu")
NP = pool["dof_pos"].shape[0]
print(f"풀: {os.path.basename(args.pool)}  {NP}개  ({pool.get('note','')})", flush=True)

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1 / 60.0
sp.substeps = 1
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0)
sp.use_gpu_pipeline = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

ho = gymapi.AssetOptions()
ho.fix_base_link = True
ho.disable_gravity = True
ho.default_dof_drive_mode = gymapi.DOF_MODE_POS
hand = gym.load_asset(sim, "/home/leegyuwon/Documents/task1/assets/dg5fs_hand", "dg5fs_right.urdf", ho)

N = args.n
envs, acts = [], []
per_row = 1 if args.overlap else 3
shared_env = None
for i in range(N):
    if args.overlap:
        # 전부 한 환경에 겹친다. 모든 shape 의 filter 를 1 로 두므로
        # (아래) 같은 그룹이어도 손끼리 부딪히지 않는다.
        if shared_env is None:
            shared_env = gym.create_env(sim, gymapi.Vec3(-0.35, -0.35, 0), gymapi.Vec3(0.35, 0.35, 1), 1)
        env = shared_env
    else:
        env = gym.create_env(sim, gymapi.Vec3(-0.35, -0.35, 0), gymapi.Vec3(0.35, 0.35, 1), per_row)
    # 충돌 그룹을 환경마다 다르게 -> 서로 안 부딪힘. 자기충돌도 전부 끔.
    act = gym.create_actor(env, hand, gymapi.Transform(), f"hand{i}", i + 1, 0)
    pr = gym.get_actor_dof_properties(env, act)
    for k in range(len(pr)):
        pr["driveMode"][k] = gymapi.DOF_MODE_POS
        pr["stiffness"][k] = 500.0
        pr["damping"][k] = 50.0
    gym.set_actor_dof_properties(env, act, pr)
    sh = gym.get_actor_rigid_shape_properties(env, act)
    for s in sh:
        s.filter = 1
    gym.set_actor_rigid_shape_properties(env, act, sh)
    envs.append(env)
    acts.append(act)

gym.prepare_sim(sim)
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if args.overlap:
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, -0.5, 0.55), gymapi.Vec3(0, 0, CAPZ + 0.02))
else:
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.1, -1.3, 1.0), gymapi.Vec3(0.35, 0.35, CAPZ))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "resample")

root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)
dof_st = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
nd = pool["dof_pos"].shape[1]
hand_sim_idx = [gym.get_actor_index(envs[i], acts[i], gymapi.DOMAIN_SIM) for i in range(N)]
origins = [gym.get_env_origin(envs[i]) for i in range(N)]
idx32 = torch.tensor(hand_sim_idx, dtype=torch.int32)

root_pin = torch.zeros(N, 13)
q_pin = torch.zeros(N, nd)


def resample():
    sel = torch.randint(NP, (N,))
    print("샘플:", sel.tolist(), flush=True)
    for i in range(N):
        j = int(sel[i])
        q_pin[i] = pool["dof_pos"][j]
        o = origins[i]
        root_pin[i, 0] = o.x + float(pool["wrist_rel_pos"][j, 0])
        root_pin[i, 1] = o.y + float(pool["wrist_rel_pos"][j, 1])
        root_pin[i, 2] = o.z + CAPZ + float(pool["wrist_rel_pos"][j, 2])
        root_pin[i, 3:7] = pool["wrist_quat"][j]
        root_pin[i, 7:] = 0.0


def pin():
    gym.refresh_actor_root_state_tensor(sim)
    for i in range(N):
        root[hand_sim_idx[i]] = root_pin[i]
    gym.set_actor_root_state_tensor_indexed(
        sim, gymtorch.unwrap_tensor(root), gymtorch.unwrap_tensor(idx32), N
    )
    dof_st[:, 0] = q_pin.reshape(-1)
    dof_st[:, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_st))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(q_pin.clone()))


# 노란 와이어프레임 원기둥 = 캡 (환경 로컬 좌표)
CYL_LO = CAPZ + CAP_FRAME_Z0
CYL_HI = CYL_LO + CAP.height
_ring = []
_NSEG = 48
for k in range(_NSEG):
    a0 = 2 * math.pi * k / _NSEG
    a1 = 2 * math.pi * (k + 1) / _NSEG
    for z in (CYL_LO, CYL_HI):
        _ring += [CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), z,
                  CAP.radius * math.cos(a1), CAP.radius * math.sin(a1), z]
    if k % 8 == 0:
        _ring += [CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), CYL_LO,
                  CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), CYL_HI]
CYL_V = np.array(_ring, dtype=np.float32).reshape(-1, 3)
CYL_C = np.tile(np.array([1.0, 0.85, 0.1], dtype=np.float32), (len(CYL_V) // 2, 1))

resample()
print("\nSPACE 새 샘플 6개, ESC 종료", flush=True)

while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value > 0 and e.action == "resample":
            resample()
    pin()
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.clear_lines(viewer)
    seen = set()
    for i in range(N):
        if id(envs[i]) in seen:
            continue
        seen.add(id(envs[i]))
        gym.add_lines(viewer, envs[i], len(CYL_V) // 2, CYL_V, CYL_C)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
