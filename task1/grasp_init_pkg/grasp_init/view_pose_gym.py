"""최적화된 파지 자세를 Isaac Gym 에서 실제 텀블러와 함께 본다.

matplotlib 렌더(view_pose.py)는 최적화가 낸 수를 그대로 그리므로 자세 자체를
보기에는 정확하지만, 실제 충돌 형상 위에서 그 자세가 성립하는지는 알려주지
않는다. 여기서는 진짜 URDF 를 PhysX 에 올리고 자세를 그대로 물려서, 캡을
파고드는지 손끝이 정말 벽에 닿는지 접촉력으로 확인한다.

중력과 손목 자유도는 끈다 (fix_base_link). 자세를 보는 것이지 정책을 돌리는
것이 아니므로, 손이 제자리에 고정되어 있어야 읽을 수 있다.

    SPACE   손끝별 벽까지 거리 / 접촉력 / 방위각 출력
    R/F     손가락 전체를 조금 더 굽히거나 펴서 여유 확인
    K       고정 <-> 물리 전환
    ESC     종료

기본은 고정 모드다. 위치 드라이브에 맡기면 자세가 캡을 파고든 순간 PhysX 의
탈출력과 드라이브가 서로 밀며 손이 계속 떨린다 -- 자세를 읽을 수가 없다.
고정 모드는 매 프레임 관절과 손목을 지정값으로 되써서 손을 붙잡아 두고,
접촉력은 그대로 읽으므로 어디가 닿고 어디가 파고드는지는 볼 수 있다.
K 로 풀면 실제로 그 자세가 물리적으로 성립하는지 확인할 수 있다.

기본은 손과 텀블러가 충돌하지 않는다. 자세를 보는 것이 목적인데, 아직 캡을
파고드는 자세를 올리면 PhysX 의 탈출력과 위치 드라이브가 서로 밀어 손이 계속
떨린다. --collide 로 켜면 실제로 그 자세가 물리적으로 성립하는지 볼 수 있다.
충돌을 꺼도 손끝 좌표는 그대로이므로 벽까지 거리와 방위각은 정확하다.

사용법:
    python view_pose_gym.py                       # poses/ 에서 가장 최근 것
    python view_pose_gym.py poses/cyl_r50_h17.json
    python view_pose_gym.py --with-tumbler        # 실제 STL 도 같이
    python view_pose_gym.py --collide             # 충돌 켜기
"""


import os as _os
# 패키지 루트: 이 파일이 든 grasp_init/ 의 부모. 어디에 풀어도 동작한다.
_PKG = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import glob
import json
import math
import os
import sys

import numpy as np
from isaacgym import gymapi, gymtorch
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAPZ = 0.225                     # tumbler.urdf 의 캡 조인트 원점 z
FING = ("엄지", "검지", "중지", "약지", "새끼")
FR = 0.0081
CAP = CapShape()

FLAGS = {"--collide", "--with-tumbler"}
argv = [a for a in sys.argv[1:] if a not in FLAGS]
COLLIDE = "--collide" in sys.argv[1:]
# 기본은 텀블러를 띄우지 않는다. 최적화가 푼 대상은 원기둥이고, 실제 STL 을
# 같이 그리면 자세가 어느 면을 기준으로 나온 것인지 화면에서 구분되지 않는다.
# --with-tumbler 로 실물과 대조할 수 있다 (파지 벽 차이 0.24mm).
WITH_TUM = "--with-tumbler" in sys.argv[1:]
# 한 자세만 본다. 인자가 없으면 poses/ 에서 가장 최근 것.
if argv:
    POSE = argv[0] if os.path.isabs(argv[0]) else os.path.join(HERE, argv[0])
else:
    cands = sorted(glob.glob(os.path.join(HERE, "poses", "*.json")), key=os.path.getmtime)
    if not cands:
        raise SystemExit("poses/ 에 자세 파일이 없다. optimize_pose.py -o 로 먼저 만들 것")
    POSE = cands[-1]
print(f"자세: {os.path.basename(POSE)}", flush=True)
print("충돌 %s" % ("켬 (--collide)" if COLLIDE else "끔 -- 캡·자기충돌 모두 없음"), flush=True)
print("노란 와이어프레임 = 최적화가 실제로 푼 원기둥 (r %.0fmm h %.0fmm), 월드 z %.3f~%.3f"
      % (CAP.radius * 1000, CAP.height * 1000, CAPZ + CAP_FRAME_Z0,
         CAPZ + CAP_FRAME_Z0 + CAP.height), flush=True)
print("텀블러 %s" % ("표시" if WITH_TUM else "미표시 -- 원기둥만"), flush=True)

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1 / 60.0
sp.substeps = 2
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0)
sp.use_gpu_pipeline = False
sp.physx.solver_type = 1
sp.physx.num_position_iterations = 8
sp.physx.contact_offset = 0.005
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)

ho = gymapi.AssetOptions()
ho.fix_base_link = True
ho.disable_gravity = True
ho.default_dof_drive_mode = gymapi.DOF_MODE_POS
hand = gym.load_asset(sim, _PKG + "/assets/dg5fs_hand", "dg5fs_right.urdf", ho)
tum = None
if WITH_TUM:
    to = gymapi.AssetOptions()
    to.fix_base_link = True
    to.vhacd_enabled = True
    to.vhacd_params = gymapi.VhacdParams()
    to.vhacd_params.resolution = 200000
    tum = gym.load_asset(sim, _PKG + "/assets/tumbler",
                         "tumbler.urdf", to)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
if WITH_TUM:
    gym.create_actor(env, tum, gymapi.Transform(), "tumbler", 0, 1)
# 충돌 그룹이 다르면 두 액터는 서로 부딪히지 않는다. 손 안쪽 자기충돌은
# 그룹이 같으므로 그대로 살아 있다.
act = gym.create_actor(env, hand, gymapi.Transform(), "hand", 0 if COLLIDE else 1, 0)

pr = gym.get_actor_dof_properties(env, act)
for k in range(len(pr)):
    pr["driveMode"][k] = gymapi.DOF_MODE_POS
    pr["stiffness"][k] = 500.0
    pr["damping"][k] = 50.0
gym.set_actor_dof_properties(env, act, pr)
# 손바닥과 엄지 뿌리는 학습 쪽과 같이 자기충돌에서 뺀다.
bn0 = gym.get_actor_rigid_body_names(env, act)
sh = gym.get_actor_rigid_shape_properties(env, act)
spn = gym.get_actor_rigid_body_shape_indices(env, act)
# 충돌을 끈 모드에서는 손 자기충돌도 끈다. 자세를 보는 것이 목적인데,
# 최적화가 손가락끼리 겹친 자세를 내면 (실제로 중지-약지 링크가 0.11mm 까지
# 붙어 있다) PhysX 가 계속 밀어내 손이 떨린다.
for b, n in enumerate(bn0):
    if not COLLIDE or n in ("link_base", "link_1_1", "link_1_2"):
        for k in range(spn[b].start, spn[b].start + spn[b].count):
            sh[k].filter = 1
gym.set_actor_rigid_shape_properties(env, act, sh)
gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.32, -0.28, 0.40),
                          gymapi.Vec3(0, 0, CAPZ + 0.02))
for key, ev in ((gymapi.KEY_SPACE, "info"),
                (gymapi.KEY_R, "flex+"), (gymapi.KEY_F, "flex-"),
                (gymapi.KEY_K, "pin")):
    gym.subscribe_viewer_keyboard_event(viewer, key, ev)

rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(-1, 13)
cfx = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(-1, 3)
root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)
bn = gym.get_actor_rigid_body_names(env, act)
off = gym.get_actor_rigid_body_index(env, act, 0, gymapi.DOMAIN_ENV)
tip_i = [off + bn.index(f"link_{k+1}_tip") for k in range(5)]
hand_root = gym.get_actor_index(env, act, gymapi.DOMAIN_SIM)
nd = len(gym.get_actor_dof_names(env, act))
# 손 액터만 지정해서 쓴다. 텐서 전체를 밀어 넣으면 텀블러 행까지 같이
# 나가는데, refresh 전의 그 행은 0 이라 쿼터니언 (0,0,0,0) 이 되어 캡이
# 화면에서 사라진다.
hand_idx32 = torch.tensor([hand_root], dtype=torch.int32)


def push_root():
    gym.set_actor_root_state_tensor_indexed(
        sim, gymtorch.unwrap_tensor(root), gymtorch.unwrap_tensor(hand_idx32), 1
    )

# 최적화가 실제로 푼 형상은 이 원기둥이다. 화면에 실제 STL 만 있으면 자세가
# 어느 면을 기준으로 나온 것인지 알 수 없어, 노란 와이어프레임으로 겹쳐
# 그린다. 실제 캡 벽(50.2mm)과 원기둥(50.0mm)의 차이는 0.24mm 라 거의
# 겹쳐 보이는 것이 정상이다.
CYL_LO = CAPZ + CAP_FRAME_Z0
CYL_HI = CYL_LO + CAP.height
_ring = []
_N = 72
_ZS = [CYL_LO + CAP.height * t / 4 for t in range(5)]
for k in range(_N):
    a0 = 2 * math.pi * k / _N
    a1 = 2 * math.pi * (k + 1) / _N
    for z in _ZS:
        _ring += [CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), z,
                  CAP.radius * math.cos(a1), CAP.radius * math.sin(a1), z]
    if k % 6 == 0:      # 세로선 12개
        _ring += [CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), CYL_LO,
                  CAP.radius * math.cos(a0), CAP.radius * math.sin(a0), CYL_HI]
CYL_V = np.array(_ring, dtype=np.float32).reshape(-1, 3)
CYL_C = np.tile(np.array([1.0, 0.85, 0.1], dtype=np.float32), (len(CYL_V) // 2, 1))

flex = 0.0
pinned = True
tgt = torch.zeros(1, nd)
dof_st = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
root_pin = torch.zeros(13)


def load():
    """자세를 손목 루트와 DOF 목표에 물린다."""
    global tgt, flex
    d = json.load(open(POSE))
    q = torch.tensor(d["dof_pos"], dtype=torch.float32)
    # 저장은 캡 프레임 상대. 캡 링크 원점은 월드 z=CAPZ 에 있다.
    p = np.array(d["wrist_rel_pos"], dtype=np.float64) + np.array([0.0, 0.0, CAPZ])
    quat = np.array(d["wrist_quat"], dtype=np.float64)      # xyzw
    root_pin[:3] = torch.tensor(p, dtype=torch.float32)
    root_pin[3:7] = torch.tensor(quat, dtype=torch.float32)
    root_pin[7:13] = 0.0
    gym.refresh_actor_root_state_tensor(sim)
    root[hand_root] = root_pin
    push_root()
    dof_st[:nd, 0] = q
    dof_st[:nd, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_st))
    tgt = q.clone().unsqueeze(0)
    flex = 0.0
    print(f"\n=== {os.path.basename(POSE)}  {d.get('note','')} ===", flush=True)
    print(f"  손목 캡기준 {[round(v*1000,1) for v in d['wrist_rel_pos']]} mm", flush=True)


def report():
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    print("\n 손가락   벽까지mm   높이mm   방위각도    접촉력N"
          + ("" if COLLIDE else "  (충돌 꺼짐)"))
    for k in range(5):
        p = rb[tip_i[k], :3]
        r = float(torch.linalg.norm(p[:2]))
        z = float(p[2]) - CAPZ                     # 캡 프레임 높이
        gap = (r - CAP.radius - FR) * 1000
        f = float(torch.linalg.norm(cfx[tip_i[k]]))
        az = math.degrees(math.atan2(float(p[1]), float(p[0])))
        mark = ("  닿음" if f > 0.05 else "") if COLLIDE else ""
        print("  %-6s  %+7.2f  %7.1f  %+8.1f  %8.2f%s" % (FING[k], gap, z * 1000, az, f, mark))
    a = [math.degrees(math.atan2(float(rb[tip_i[k], 1]), float(rb[tip_i[k], 0]))) for k in (0, 2)]
    s = abs(a[0] - a[1])
    print("  엄지-중지 방위각 %.1f도 (기준 90)" % min(s, 360 - s))


load()
print("\nSPACE 수치, R/F 굽힘, K 고정/물리 전환, ESC 종료", flush=True)
print("현재: 고정 모드 (손이 지정 자세에 붙어 있음)", flush=True)

while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value == 0:
            continue
        if e.action == "info":
            report()
        elif e.action == "pin":
            pinned = not pinned
            print("  %s 모드" % ("고정" if pinned else "물리"), flush=True)
            if pinned:
                load()
        elif e.action in ("flex+", "flex-"):
            flex += 0.03 if e.action == "flex+" else -0.03
            print(f"  굽힘 오프셋 {math.degrees(flex):+.1f}도", flush=True)
    q = tgt.clone()
    if flex:
        # 굽힘 관절(각 손가락의 2,3,4번)만 움직인다. 1번은 벌림이라 제외.
        for f in range(5):
            for j in (1, 2, 3):
                q[0, 4 * f + j] += flex
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(q))
    if pinned:
        # 매 프레임 되쓴다. 관통이 있으면 PhysX 가 밀어내는데, 자세를 보는
        # 것이 목적이므로 그 변위를 그대로 취소한다.
        dof_st[:nd, 0] = q[0]
        dof_st[:nd, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_st))
        root[hand_root] = root_pin
        push_root()
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.clear_lines(viewer)
    gym.add_lines(viewer, env, len(CYL_V) // 2, CYL_V, CYL_C)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
