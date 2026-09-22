"""초기자세 노이즈 풀(.pt)에서 K개를 뽑아 텀블러와 함께 나란히 띄운다.

    R       다시 뽑기
    SPACE   각 env 의 22관절 출력
    ESC     종료

사용법:
    ROBOT_URDF=rb5_allegro_vmount_siltip2.urdf \
    python view_init_pool.py poses/pool_r44_siltip2_x04.pt [K=4] [seed=0]
    HEADLESS_CHECK=1 을 주면 뷰어 없이 구성만 확인하고 종료.
"""
import os, sys, math
import numpy as np
from isaacgym import gymapi
import torch

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
URDF = os.path.basename(os.environ.get("ROBOT_URDF", "rb5_allegro_vmount_siltip2.urdf"))
TUMBLER_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"
TUMBLER = os.path.basename(os.environ.get("TUMBLER_URDF", f"{TUMBLER_DIR}/tumbler_cyl_r44_fric02.urdf"))
TPOS = [float(v) for v in os.environ.get("TUMBLER_POS", "0.4,0,0").split(",")]
POOL = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
HEADLESS = os.environ.get("HEADLESS_CHECK", "0") == "1"

d = torch.load(POOL, map_location="cpu")
pool = d["dof_pos"].numpy().astype(np.float32)
print(f"pool: {POOL}  {pool.shape[0]}개  ({d.get('note','-')})")
rng = np.random.default_rng(SEED)

gym = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 1.0 / 60.0; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0); sp.physx.solver_type = 1
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
o = gymapi.AssetOptions(); o.fix_base_link = True; o.disable_gravity = True; o.collapse_fixed_joints = False
asset = gym.load_asset(sim, URDF_DIR, URDF, o)
n_dof = gym.get_asset_dof_count(asset); assert n_dof == pool.shape[1], (n_dof, pool.shape)
dof_names = [gym.get_asset_dof_name(asset, i) for i in range(n_dof)]
to = gymapi.AssetOptions(); to.fix_base_link = True; to.disable_gravity = True
tum = gym.load_asset(sim, TUMBLER_DIR, TUMBLER, to)

FINGER_COLORS = {0: (0.9, 0.3, 0.3), 1: (0.3, 0.8, 0.3), 2: (0.3, 0.5, 0.95), 3: (0.95, 0.8, 0.2)}
SPACING = 1.4
envs, actors = [], []
for k in range(K):
    env = gym.create_env(sim, gymapi.Vec3(-0.7, -0.7, 0), gymapi.Vec3(0.7, 0.7, 2), K)
    actor = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro", k, 0)
    tt = gymapi.Transform(); tt.p = gymapi.Vec3(*TPOS)
    gym.create_actor(env, tum, tt, "tumbler", k, 0)
    names = gym.get_actor_rigid_body_names(env, actor)
    for i, name in enumerate(names):
        if name.startswith("link_") and "." in name:
            c = FINGER_COLORS[int(name.split("_")[1].split(".")[0]) // 4]
        elif name == "base_link": c = (0.35, 0.35, 0.4)
        elif name.startswith("link"): c = (0.75, 0.75, 0.78)
        else: c = (0.6, 0.6, 0.65)
        gym.set_rigid_body_color(env, actor, i, gymapi.MESH_VISUAL, gymapi.Vec3(*c))
    envs.append(env); actors.append(actor)

states = [np.zeros(n_dof, dtype=gymapi.DofState.dtype) for _ in range(K)]
picked = None
def resample():
    global picked
    picked = rng.choice(pool.shape[0], size=K, replace=False)
    print("뽑은 인덱스:", picked.tolist())
    for k, idx in enumerate(picked):
        states[k]["pos"] = pool[idx]
        gym.set_actor_dof_states(envs[k], actors[k], states[k], gymapi.STATE_ALL)
        print(f"  env{k} #{idx}: J6 {math.degrees(pool[idx][5]):+.1f}°  손16 {[round(float(v), 3) for v in pool[idx][6:]]}")
resample()

if HEADLESS:
    gym.simulate(sim); gym.fetch_results(sim, True); print("headless check OK"); sys.exit(0)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# env 가 K열로 나란히 놓임 (x 방향으로 SPACING 간격) -> 전체가 보이게 카메라
cx = TPOS[0] + SPACING * (K - 1) / 2
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(cx, 1.6 + 0.3 * K, 0.9), gymapi.Vec3(cx, 0, 0.3))
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "dump")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "resample")
while not gym.query_viewer_has_closed(viewer):
    for ev in gym.query_viewer_action_events(viewer):
        if ev.action == "resample" and ev.value > 0: resample()
        if ev.action == "dump" and ev.value > 0:
            for k in range(K):
                st = gym.get_actor_dof_states(envs[k], actors[k], gymapi.STATE_POS)
                print(f"env{k} #{picked[k]}:", [round(float(v), 4) for v in st["pos"]])
    for k in range(K):
        gym.set_actor_dof_states(envs[k], actors[k], states[k], gymapi.STATE_ALL)
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
