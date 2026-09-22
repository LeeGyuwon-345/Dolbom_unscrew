"""팔+손 22관절 초기자세 노이즈 풀 생성 (CAP_GRASP_INIT_POOL 용).

rh_init amortized 네트(손 MLP + RB5 net)로 기준 자세를 뽑되, 캡 배치높이 z 를
지터하고 관절 가우시안 노이즈를 섞은 뒤, Isaac Gym 헤드리스 FK 로
  - 4지 팁 패드-캡벽 간격이 [-max_pen, +max_gap] 안
  - 팁 z 가 캡 밴드와 min_overlap 이상 겹침
  - 팁 아닌 손 링크 원점이 캡/몸체 원기둥 안에 있지 않음(총체적 관통)
  - 자기충돌 없음(접촉력 0)
을 통과한 샘플만 저장한다. 리셋 때 env 가 무작위 인덱싱만 한다(reference_pose.GraspInitializer).

사용:
  python make_init_pool_arm.py --n 20000 --z 0.202 --z-jitter 0.005 \
      --joint-std 0.04 --arm-std 0.01 --out poses/pool_r44_siltip2_x04.pt
"""
from isaacgym import gymapi  # torch 보다 먼저
import argparse, math, os, sys, time
import numpy as np
import torch

RH = "/home/leegyuwon/Documents/task1/rh_init"
sys.path.insert(0, RH)
import train_amortized as T          # noqa: E402
import train_rb5 as TR               # noqa: E402
import rb5_ik as K                   # noqa: E402
from rb5_mlp import RB5Net           # noqa: E402
from model import matrix_to_quat_xyzw, rotate_pose  # noqa: E402

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
ap = argparse.ArgumentParser()
ap.add_argument("--r", type=float, default=0.044); ap.add_argument("--h", type=float, default=0.016)
ap.add_argument("--z", type=float, default=0.202, help="캡 배치높이 기준(m, rh_init 입력)")
ap.add_argument("--z-jitter", type=float, default=0.005, help="z 균등 지터 반폭(m)")
ap.add_argument("--joint-std", type=float, default=0.04, help="손 16관절 노이즈 std(rad)")
ap.add_argument("--arm-std", type=float, default=0.01, help="팔 6관절 노이즈 std(rad)")
ap.add_argument("--n", type=int, default=20000, help="시도 샘플 수")
ap.add_argument("--envs", type=int, default=256)
ap.add_argument("--urdf", default="rb5_allegro_vmount_siltip2.urdf")
ap.add_argument("--tumbler-xy", default="0.4,0")
ap.add_argument("--cap-r", type=float, default=0.044)
ap.add_argument("--band", default="0.215,0.231")
ap.add_argument("--pad-fr", type=float, default=0.0105)
ap.add_argument("--tip-half-h", type=float, default=0.0063)
ap.add_argument("--max-pen-mm", type=float, default=5.0)
ap.add_argument("--max-gap-mm", type=float, default=6.0)
ap.add_argument("--min-overlap-mm", type=float, default=6.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
torch.manual_seed(args.seed)
CAPC = np.array([float(v) for v in args.tumbler_xy.split(",")]); CAPR = args.cap_r
BLO, BHI = [float(v) for v in args.band.split(",")]

# ---- 1. rh_init 기준 자세 (z 지터 배치 추론) ----
dev = "cpu"
R_M, T_M = TR._to(dev)
gmlp = TR._grasp(dev)
net = RB5Net().to(dev); net.load_state_dict(torch.load(TR.CKPT, map_location=dev)); net.eval()
fk_g, _, hlo, hhi = T.build(dev)
N = args.n
r = torch.full((N,), args.r); h = torch.full((N,), args.h)
z = args.z + (torch.rand(N) * 2 - 1) * args.z_jitter
with torch.no_grad():
    q_h, wp, R = gmlp(T._norm(r, h), r, h)          # (N,16) 손 -- r,h 고정이라 상수
    th, q_a = net(TR._norm3(r, h, z))                # (N,), (N,6) 팔
    wp_t, R_t = rotate_pose(wp, R, th)
    wrist_rel = wp_t + torch.tensor([0.0, 0.0, TR.CAP_FRAME_Z0])
    wrist_quat = matrix_to_quat_xyzw(R_t)
q_a = q_a + torch.randn_like(q_a) * args.arm_std
q_h = q_h + torch.randn_like(q_h) * args.joint_std
q_a = torch.max(torch.min(q_a, K._HI), K._LO)
q_h = torch.max(torch.min(q_h, hhi), hlo)
q22 = torch.cat([q_a, q_h], dim=-1).float()          # Isaac Gym 순서: 팔6 + 손16
print(f"[pool] 기준 z {args.z*1000:.0f}±{args.z_jitter*1000:.0f}mm, 손 std {args.joint_std} 팔 std {args.arm_std}, 시도 {N}", flush=True)

# ---- 2. Isaac Gym 헤드리스 FK 필터 ----
gym = gymapi.acquire_gym(); sp = gymapi.SimParams(); sp.dt = 1 / 60; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0); sp.physx.solver_type = 1
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
o = gymapi.AssetOptions(); o.fix_base_link = True; o.disable_gravity = True; o.collapse_fixed_joints = False
asset = gym.load_asset(sim, URDF_DIR, args.urdf, o)
assert gym.get_asset_dof_count(asset) == 22
E = args.envs; envs = []; actors = []
for i in range(E):
    e = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), int(math.sqrt(E)) + 1)
    a = gym.create_actor(e, asset, gymapi.Transform(), "r", i, 0)   # filter 0 = 자기충돌 켬
    envs.append(e); actors.append(a)
names = gym.get_actor_rigid_body_names(envs[0], actors[0])
tip_idx = [i for i, n in enumerate(names) if n.endswith("_tip")]
hand_idx = [i for i, n in enumerate(names) if n.startswith("link_") and not n.endswith("_tip")]
assert len(tip_idx) == 4, names

keep = np.zeros(N, dtype=bool); gaps = np.zeros((N, 4)); ovs = np.zeros((N, 4))
rej = {"pen": 0, "gap": 0, "band": 0, "gross": 0, "selfc": 0}
t0 = time.time()
for b0 in range(0, N, E):
    ids = list(range(b0, min(b0 + E, N)))
    for k, i in enumerate(ids):
        st = gym.get_actor_dof_states(envs[k], actors[k], gymapi.STATE_ALL)
        st["pos"] = q22[i].numpy(); st["vel"] = 0
        gym.set_actor_dof_states(envs[k], actors[k], st, gymapi.STATE_ALL)
    gym.simulate(sim); gym.fetch_results(sim, True)
    for k, i in enumerate(ids):
        bs = gym.get_actor_rigid_body_states(envs[k], actors[k], gymapi.STATE_POS)
        P = np.stack([np.array([p[0], p[1], p[2]]) for p in bs["pose"]["p"]])
        rr = np.hypot(P[:, 0] - CAPC[0], P[:, 1] - CAPC[1]); zz = P[:, 2]
        g = (rr[tip_idx] - args.pad_fr - CAPR) * 1000
        ov = (np.minimum(zz[tip_idx] + args.tip_half_h, BHI) - np.maximum(zz[tip_idx] - args.tip_half_h, BLO)) * 1000
        gaps[i] = g; ovs[i] = ov
        ok = True
        if (g < -args.max_pen_mm).any(): rej["pen"] += 1; ok = False
        elif (g > args.max_gap_mm).any(): rej["gap"] += 1; ok = False
        elif (ov < args.min_overlap_mm).any(): rej["band"] += 1; ok = False
        elif ((rr[hand_idx] < CAPR) & (zz[hand_idx] < BHI + 0.005)).any(): rej["gross"] += 1; ok = False
        else:
            cf = gym.get_env_rigid_contact_forces(envs[k])
            if len(cf) and np.linalg.norm(np.stack([np.array([f[0], f[1], f[2]]) for f in cf]), axis=1).max() > 1e-3:
                rej["selfc"] += 1; ok = False
        keep[i] = ok
    if (b0 // E) % 10 == 0:
        print(f"  {min(b0+E,N)}/{N}  합격 {keep[:b0+E].sum()}  ({time.time()-t0:.0f}s)", flush=True)

n_ok = int(keep.sum())
print(f"[pool] 합격 {n_ok}/{N} ({100*n_ok/N:.1f}%)  거절 {rej}")
gk = gaps[keep]; ok_ = ovs[keep]
print(f"  벽간격 mm: 검지 {gk[:,0].mean():+.1f}±{gk[:,0].std():.1f}  엄지 {gk[:,1].mean():+.1f}±{gk[:,1].std():.1f}  중지 {gk[:,2].mean():+.1f}±{gk[:,2].std():.1f}  약지 {gk[:,3].mean():+.1f}±{gk[:,3].std():.1f}")
print(f"  밴드겹침 mm: 최소 {ok_.min():.1f} 평균 {ok_.mean():.1f}")
j6 = q22[keep][:, 5].numpy() * 180 / math.pi
print(f"  J6: {j6.min():.1f}~{j6.max():.1f}°  (기준 {q22[:,5].mean()*180/math.pi:.1f}°)")
d = {"dof_pos": q22[keep].contiguous(), "wrist_rel_pos": wrist_rel[keep].float().contiguous(),
     "wrist_quat": wrist_quat[keep].float().contiguous(),
     "note": f"arm22 r{args.r*1000:.0f}h{args.h*1000:.0f} z{args.z*1000:.0f}±{args.z_jitter*1000:.0f}mm hand_std{args.joint_std} arm_std{args.arm_std} {args.urdf} accept{100*n_ok/N:.0f}%"}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(d, args.out)
print("저장:", args.out, d["dof_pos"].shape)
