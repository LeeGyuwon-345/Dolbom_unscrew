"""추론 CSV(arm_cmd/hand_cmd)를 개방루프로 재생하는 Isaac Gym 뷰어.

실기 배포와 같은 방식 -- 정책·관측 없이 기록된 위치 목표를 60Hz 로
그대로 스트리밍한다. 텀블러는 ScrewCoupling 으로 나사 물리(저항·리프트
팔로워·해제)를 env 와 동일하게 재현하므로, 이 재생으로 캡이 돌고 풀리면
실기에서도 (물리 정합 한도 내에서) 같은 결과를 기대할 수 있다.

사용법:
    python replay_csv_gui.py <commands.csv> [--loop]
        CSV: step,unscrew_deg,arm_cmd_j1..6,hand_cmd_0..15,... (cmd 열만 사용)
"""

import csv
import os
import sys

import numpy as np
from isaacgym import gymapi
import torch

sys.path.insert(0, "/home/leegyuwon/Documents/task1/tools")
from screw_coupling import ScrewCoupling  # noqa: E402

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/leegyuwon/Documents/task1/logs/a42_ep2600_inference_commands.csv"
LOOP = "--loop" in sys.argv

ROBOT_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
ROBOT_URDF = os.path.basename(os.environ.get(
    "RB5_ALLEGRO_URDF",
    f"{ROBOT_DIR}/rb5_allegro_vmount.urdf"))
TUM_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"
TUM_URDF = os.path.basename(os.environ.get(
    "TUMBLER_URDF", f"{TUM_DIR}/tumbler_cyl_fric02_marker.urdf"))
TUM_POS = [float(v) for v in os.environ.get("TUMBLER_POS", "0.4,0,0").split(",")]
HOLD_FORCE = float(os.environ.get("CAP_SCREW_HOLD_FORCE", "2000"))
RESIST_NM = float(os.environ.get("RESIST_NM", "0.2"))

rows = list(csv.DictReader(open(CSV_PATH)))
arm_cols = [f"arm_cmd_j{i}" for i in range(1, 7)]
hand_cols = [f"hand_cmd_{i}" for i in range(16)]
cmds = np.array([[float(r[c]) for c in arm_cols + hand_cols] for r in rows],
                dtype=np.float32)
print(f"[replay] {CSV_PATH}: {len(cmds)}스텝, 저항 {RESIST_NM}Nm, loop={LOOP}")

gym = gymapi.acquire_gym()
sp = gymapi.SimParams()
sp.dt = 1.0 / 60.0
sp.substeps = 2
sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, -9.81)
sp.physx.solver_type = 1
sp.physx.num_position_iterations = 8
sp.physx.num_velocity_iterations = 1
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
_pp = gymapi.PlaneParams()
_pp.normal = gymapi.Vec3(0, 0, 1)  # z-up (기본값은 y-up이라 바닥이 벽이 된다)
gym.add_ground(sim, _pp)

ro = gymapi.AssetOptions()
ro.fix_base_link = True
ro.collapse_fixed_joints = False
ro.disable_gravity = False
robot_asset = gym.load_asset(sim, ROBOT_DIR, ROBOT_URDF, ro)
to_ = gymapi.AssetOptions()
to_.fix_base_link = True
tum_asset = gym.load_asset(sim, TUM_DIR, TUM_URDF, to_)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 2), 1)
robot = gym.create_actor(env, robot_asset, gymapi.Transform(), "robot", 0, 0)
tt = gymapi.Transform()
tt.p = gymapi.Vec3(*TUM_POS)
tum = gym.create_actor(env, tum_asset, tt, "tumbler", 0, 2)

# 로봇 드라이브: env 와 같은 위치 서보 성격 (팔 강성 4000/160, 손 400/40)
rp = gym.get_actor_dof_properties(env, robot)
rp["driveMode"][:] = gymapi.DOF_MODE_POS
rp["stiffness"][:6] = 4000.0
rp["damping"][:6] = 160.0
rp["stiffness"][6:] = 400.0
rp["damping"][6:] = 40.0
gym.set_actor_dof_properties(env, robot, rp)

# 마찰 스탬프: env 와 동일 (손 4.0 / 텀블러 6.0 -> 유효 ~5.0). 이거 없이는
# URDF 기본 마찰로 돌아 파지가 미끄러진다 (재생 49도 정체 실측).
HANDF = float(os.environ.get("HAND_FRICTION", "4.0"))
OBJF = float(os.environ.get("OBJ_FRICTION", "6.0"))
_rs = gym.get_actor_rigid_shape_properties(env, robot)
for _s in _rs:
    _s.friction = HANDF
gym.set_actor_rigid_shape_properties(env, robot, _rs)
_ts = gym.get_actor_rigid_shape_properties(env, tum)
for _s in _ts:
    _s.friction = OBJF
gym.set_actor_rigid_shape_properties(env, tum, _ts)

# 나사 결합: env 와 동일한 공용 모듈
tum_urdf_path = os.path.join(TUM_DIR, TUM_URDF)
os.environ.setdefault("CAP_SCREW_HOLD_FORCE", str(HOLD_FORCE))
screw = ScrewCoupling.from_urdf(tum_urdf_path)
screw.bind(gym, env, tum, device="cpu", num_envs=1)
screw.set_resistance(gym, [env], [tum], RESIST_NM)

n_rdof = gym.get_asset_dof_count(robot_asset)
n_tdof = gym.get_asset_dof_count(tum_asset)

# 초기 자세 = CSV 첫 행 명령
init = cmds[0]
rstate = np.zeros(n_rdof, dtype=gymapi.DofState.dtype)
rstate["pos"] = init
gym.set_actor_dof_states(env, robot, rstate, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env, robot, init)
tstate = np.zeros(n_tdof, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, tum, tstate, gymapi.STATE_ALL)

TEST_STEPS = 0
if "--test" in sys.argv:
    TEST_STEPS = int(sys.argv[sys.argv.index("--test") + 1])
viewer = None
if not TEST_STEPS:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(1.0, 0.8, 0.9), gymapi.Vec3(0.4, 0, 0.35))

step = 0
while (TEST_STEPS and step < TEST_STEPS) or (viewer and not gym.query_viewer_has_closed(viewer)):
    i = min(step, len(cmds) - 1)
    gym.set_actor_dof_position_targets(env, robot, cmds[i])

    # 나사 결합 적용 -- apply() 는 env 전체 DOF 텐서(DOMAIN_ENV 인덱스)
    # 기준이므로 로봇(22)+텀블러 폭으로 만들어 텀블러 슬라이스만 쓴다.
    rq = gym.get_actor_dof_states(env, robot, gymapi.STATE_POS)["pos"]
    tq = gym.get_actor_dof_states(env, tum, gymapi.STATE_POS)["pos"]
    q = torch.zeros(1, n_rdof + n_tdof)
    q[0, :n_rdof] = torch.from_numpy(rq.astype(np.float32))
    q[0, n_rdof:] = torch.from_numpy(tq.astype(np.float32))
    tgt = torch.zeros(1, n_rdof + n_tdof)
    screw.apply(q, tgt)
    screw.release_bound(gym, [env], [tum], q)
    gym.set_actor_dof_position_targets(env, tum, tgt[0, n_rdof:].numpy())

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    if viewer:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    if step % 60 == 0:
        ang = float(torch.rad2deg(screw.angle())[0])
        # 재생 q 를 CSV 실측 q(정답지)와 대조 -- 개방루프 이탈의 정량 지표
        _ref = np.array([float(rows[i][f"arm_q_j{k}"]) for k in range(1, 7)]
                        + [float(rows[i][f"hand_q_{k}"]) for k in range(16)])
        _err = np.abs(rq.astype(np.float64) - _ref) if 'rq' in dir() else np.zeros(22)
        print(f"[replay] step {i} 풀림 {ang:.0f}도 | q-이탈 최대 {np.max(_err):.4f} rad (팔 {np.max(_err[:6]):.4f})", flush=True)
    step += 1
    if step >= len(cmds):
        if LOOP:
            step = 0
            gym.set_actor_dof_states(env, robot, rstate, gymapi.STATE_ALL)
            gym.set_actor_dof_states(env, tum, tstate, gymapi.STATE_ALL)
            _z = torch.zeros(1, n_rdof + n_tdof, 2)
            screw.zero_dofs(_z)
            screw.reset(torch.zeros(1, n_rdof + n_tdof))
            screw.rebind_engaged(gym, [env], [tum], torch.tensor([0]))
        else:
            pass  # 마지막 자세 유지

if viewer:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
