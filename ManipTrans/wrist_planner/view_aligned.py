"""정렬된 양손 데이터셋(prepare_dataset.py 출력 npz)을 GUI로 동기 재생.

play_retarget_bih.py와 달리 '공통 타임라인으로 정렬된' 배열을 그대로 재생하므로,
양손+두 객체가 실제 같은 시각으로 맞춰져 보인다. 손목 플래너 학습 전 정합성 확인용.

실행:
  cd /home/leegyuwon/Documents/ManipTrans
  PATH=<env>/bin:$PATH LD_LIBRARY_PATH=<env>/lib:$LD_LIBRARY_PATH \
  python wrist_planner/view_aligned.py --data_idx 97fc3@1

키보드: SPACE=일시정지  LEFT/RIGHT=프레임±  R=처음  ESC=종료
"""
import os
import argparse
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_ROOT = "maniptrans_envs"


def aa_to_quat_xyzw(aa):
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-8, aa / np.clip(ang, 1e-8, None), np.array([1.0, 0, 0]))
    s = np.sin(ang / 2)
    return np.concatenate([axis * s, np.cos(ang / 2)], axis=-1).astype(np.float32)


def rotmat_to_quat_xyzw(R):
    m = R; t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([(m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s, 0.25*s], np.float32)
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = np.sqrt(1.0+m[0,0]-m[1,1]-m[2,2]) * 2
        return np.array([0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s], np.float32)
    elif m[1,1] > m[2,2]:
        s = np.sqrt(1.0+m[1,1]-m[0,0]-m[2,2]) * 2
        return np.array([(m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s], np.float32)
    else:
        s = np.sqrt(1.0+m[2,2]-m[0,0]-m[1,1]) * 2
        return np.array([(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s, (m[1,0]-m[0,1])/s], np.float32)


def make_stl_visual_urdf(src_fs):
    with open(src_fs) as f:
        txt = f.read()
    txt = txt.replace('.dae"', '.STL"').replace('.DAE"', '.STL"')
    dst = src_fs.replace(".urdf", "_stlvis.urdf")
    with open(dst, "w") as f:
        f.write(txt)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="97fc3@1")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    npz = np.load(os.path.join(HERE, "datasets", f"{args.data_idx}.npz"), allow_pickle=True)
    N = len(npz["common_t"])
    cap_T, body_T = npz["cap_T"], npz["body_T"]
    obj_pose = {  # side별 (pos(N,3), quat(N,4))
        "cap": (cap_T[:, :3, 3], np.stack([rotmat_to_quat_xyzw(cap_T[t, :3, :3]) for t in range(N)])),
        "body": (body_T[:, :3, 3], np.stack([rotmat_to_quat_xyzw(body_T[t, :3, :3]) for t in range(N)])),
    }
    hand = {
        "right": dict(wpos=npz["rh_wpos"], wquat=aa_to_quat_xyzw(npz["rh_waa"]), dof=npz["rh_dof"], obj="cap"),
        "left":  dict(wpos=npz["lh_wpos"], wquat=aa_to_quat_xyzw(npz["lh_waa"]), dof=npz["lh_dof"], obj="body"),
    }
    obj_urdf = {"cap": str(npz["cap_urdf"]), "body": str(npz["body_urdf"])}
    print(f"[view] {args.data_idx}  정렬프레임={N}")

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 1/60.; sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0); sp.use_gpu_pipeline = False; sp.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    fcol = {"1": (0.5,0.5,0.55), "2": (1,1,0), "3": (0,0.8,0), "4": (1,0,0), "5": (0,0.3,1)}
    obj_col = {"cap": (0.9,0.55,0.15), "body": (0.2,0.6,0.9)}
    root_order = []; dof_off = {}; cur = 0

    for side in ("right", "left"):
        dh = DexHandFactory.create_hand("dg5fs", side)
        urdf_rel = os.path.relpath(make_stl_visual_urdf(os.path.join(ASSET_ROOT, dh.urdf_path)), ASSET_ROOT)
        ao = gymapi.AssetOptions(); ao.fix_base_link = False; ao.disable_gravity = True; ao.collapse_fixed_joints = False
        asset = gym.load_asset(sim, ASSET_ROOT, urdf_rel, ao)
        bodies = gym.get_asset_rigid_body_names(asset)
        actor = gym.create_actor(env, asset, gymapi.Transform(), f"hand_{side}", 0, 1)
        hand[side]["actor"] = actor
        for bi, bn in enumerate(bodies):
            for fx, c in fcol.items():
                if bn.startswith(f"link_{fx}_"):
                    gym.set_rigid_body_color(env, actor, bi, gymapi.MESH_VISUAL, gymapi.Vec3(*c))
        root_order.append((side, "hand")); dof_off[side] = cur; cur += len(gym.get_asset_dof_names(asset))
        ok = hand[side]["obj"]
        oao = gymapi.AssetOptions(); oao.fix_base_link = False; oao.disable_gravity = True
        oasset = gym.load_asset(sim, os.path.dirname(obj_urdf[ok]), os.path.basename(obj_urdf[ok]), oao)
        oactor = gym.create_actor(env, oasset, gymapi.Transform(), f"obj_{ok}", 0, 1)
        gym.set_rigid_body_color(env, oactor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*obj_col[ok]))
        hand[side]["oactor"] = oactor; root_order.append((side, "obj"))

    for ri, (side, kind) in enumerate(root_order):
        hand[side][f"root_{kind}"] = ri
    for side in ("right", "left"):
        dp = gym.get_actor_dof_properties(env, hand[side]["actor"])
        dp["driveMode"][:] = gymapi.DOF_MODE_POS; dp["stiffness"][:] = 1000.0; dp["damping"][:] = 50.0
        gym.set_actor_dof_properties(env, hand[side]["actor"], dp)

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, 0.5, 0.6),
                              gymapi.Vec3(*hand["right"]["wpos"][0]))
    for k, a in {gymapi.KEY_SPACE:"pause", gymapi.KEY_LEFT:"prev", gymapi.KEY_RIGHT:"next",
                 gymapi.KEY_R:"reset", gymapi.KEY_ESCAPE:"quit"}.items():
        gym.subscribe_viewer_keyboard_event(viewer, k, a)

    def set_frame(t):
        for side in ("right", "left"):
            d = hand[side]
            root[d["root_hand"], 0:3] = torch.tensor(d["wpos"][t])
            root[d["root_hand"], 3:7] = torch.tensor(d["wquat"][t])
            root[d["root_hand"], 7:13] = 0.0
            pos, quat = obj_pose[d["obj"]]
            root[d["root_obj"], 0:3] = torch.tensor(pos[t])
            root[d["root_obj"], 3:7] = torch.tensor(quat[t])
            root[d["root_obj"], 7:13] = 0.0
            off = dof_off[side]; nd = d["dof"].shape[1]
            dof_state[off:off+nd, 0] = torch.tensor(d["dof"][t]); dof_state[off:off+nd, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        for side in ("right", "left"):
            gym.set_actor_dof_position_targets(env, hand[side]["actor"], hand[side]["dof"][t].astype(np.float32))

    print("[색] 손: thumb=회 index=노랑 middle=초록 ring=빨강 pinky=파랑 | 뚜껑=주황 몸통=하늘")
    print("[재생] SPACE=정지 LEFT/RIGHT=프레임± R=처음 ESC=종료")
    t = 0; paused = False; step = max(1, int(60/max(1, args.fps))); fc = 0
    while not gym.query_viewer_has_closed(viewer):
        for e in gym.query_viewer_action_events(viewer):
            if e.value == 0: continue
            if e.action == "pause": paused = not paused
            elif e.action == "prev": t = (t-1) % N; paused = True
            elif e.action == "next": t = (t+1) % N; paused = True
            elif e.action == "reset": t = 0
            elif e.action == "quit": gym.destroy_viewer(viewer); gym.destroy_sim(sim); return
        set_frame(t)
        gym.simulate(sim); gym.fetch_results(sim, True); gym.refresh_actor_root_state_tensor(sim)
        gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
        fc += 1
        if not paused and fc % step == 0:
            t = (t+1) % N


if __name__ == "__main__":
    main()
