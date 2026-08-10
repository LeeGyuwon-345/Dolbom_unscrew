"""양손(bimanual) 리타겟팅 결과를 한 씬에 같이 재생.

- RH(오른손)+LH(왼손) dg5fs를 각자의 mano2dg5fs pkl로 동시에 프레임순 재생
- 각 손의 조작 객체(RH=뚜껑, LH=몸체)도 함께 표시 → 완성된 텀블러 형태
- 손가락 tip 오차(양손)를 콘솔에 출력

실행:
    cd /home/leegyuwon/Documents/ManipTrans
    PATH=<env>/bin:$PATH LD_LIBRARY_PATH=<env>/lib:$LD_LIBRARY_PATH \
    <env>/bin/python tools/play_retarget_bih.py --data_idx 29266@1 --dexhand dg5fs

키보드: SPACE=일시정지  LEFT/RIGHT=프레임±  R=처음으로  ESC=종료
"""
import os
import pickle
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch

from main.dataset.factory import ManipDataFactory
from main.dataset.mano2dexhand import pack_data
from main.dataset.transform import aa_to_rotmat
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

ASSET_ROOT = "maniptrans_envs"


def mujoco2gym_transf():
    M = np.eye(4)
    M[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    M[:3, 3] = np.array([0, 0, 0.4 + 0.015])  # table_surface_z
    return M.astype(np.float32)


def aa_to_quat_xyzw(aa):
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-8, aa / np.clip(ang, 1e-8, None), np.array([1.0, 0, 0]))
    s = np.sin(ang / 2)
    return np.concatenate([axis * s, np.cos(ang / 2)], axis=-1).astype(np.float32)


def rotmat_to_quat_xyzw(R):
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def make_stl_visual_urdf(src_fs):
    with open(src_fs) as f:
        txt = f.read()
    txt = txt.replace('.dae"', '.STL"').replace('.DAE"', '.STL"')
    dst = src_fs.replace(".urdf", "_stlvis.urdf")
    with open(dst, "w") as f:
        f.write(txt)
    return dst


def load_side(side, idx, dexhand_name, device, M):
    """한쪽 손의 타겟/리타겟/객체 데이터를 모두 로드해 dict로 반환."""
    dexhand = DexHandFactory.create_hand(dexhand_name, side)
    dtype = ManipDataFactory.dataset_type(idx)
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dtype, side=side, device=device,
        mujoco2gym_transf=torch.eye(4, device=device), dexhand=dexhand, verbose=False,
    )
    demo = pack_data([demo_d[idx]], dexhand)
    T = demo["mano_joints"].shape[0]
    target_mano = demo["mano_joints"].view(T, -1, 3).detach().cpu().numpy()
    target_wrist = demo["wrist_pos"].view(T, 3).detach().cpu().numpy()
    target_joints = np.concatenate([target_wrist[:, None], target_mano], axis=1)  # (T,26,3)
    target_joints = (M[:3, :3] @ target_joints.reshape(-1, 3).T).T + M[:3, 3]
    target_joints = target_joints.reshape(T, -1, 3)

    base = os.path.split(demo["data_path"][0])[-1]
    _stage = idx.split("@")[-1]
    dump = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{base.replace('.pkl', f'@{_stage}.pkl')}"
    if not os.path.isfile(dump):
        raise FileNotFoundError(f"[{side}] 리타겟 결과 없음: {dump}\n먼저 mano2dexhand.py --side {side} 로 생성하세요.")
    ret = pickle.load(open(dump, "rb"))
    opt_wrist_pos = ret["opt_wrist_pos"].astype(np.float32)
    opt_wrist_quat = aa_to_quat_xyzw(ret["opt_wrist_rot"])
    opt_dof = ret["opt_dof_pos"].astype(np.float32)
    Tr = opt_dof.shape[0]
    print(f"[{side}] {dump}  프레임={Tr}")

    obj_traj = demo["obj_trajectory"].view(T, 4, 4).detach().cpu().numpy()
    obj_traj_gym = np.einsum("ij,tjk->tik", M, obj_traj).astype(np.float32)
    obj_pos = obj_traj_gym[:, :3, 3]
    obj_quat = np.stack([rotmat_to_quat_xyzw(obj_traj_gym[t, :3, :3]) for t in range(T)])
    obj_urdf = demo["obj_urdf_path"][0]
    print(f"[{side}] 객체 {obj_urdf}")

    return dict(
        side=side, dexhand=dexhand, T=T, Tr=Tr,
        target_joints=target_joints,
        opt_wrist_pos=opt_wrist_pos, opt_wrist_quat=opt_wrist_quat, opt_dof=opt_dof,
        obj_pos=obj_pos, obj_quat=obj_quat, obj_urdf=obj_urdf,
    )


def main():
    p = gymutil.parse_arguments(
        description="Replay bimanual retargeting",
        headless=False,
        custom_parameters=[
            {"name": "--data_idx", "type": str, "default": "29266@1"},
            {"name": "--dexhand", "type": str, "default": "dg5fs"},
            {"name": "--fps", "type": int, "default": 30},
        ],
    )
    idx = p.data_idx
    device = "cuda:0"
    M = mujoco2gym_transf()

    S = {"right": load_side("right", idx, p.dexhand, device, M),
         "left": load_side("left", idx, p.dexhand, device, M)}
    Tr = min(S["right"]["Tr"], S["left"]["Tr"])
    print(f"[replay] 공통 프레임={Tr}")

    # ---- Isaac Gym ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0)
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)

    fcol = {"1": (0.5, 0.5, 0.55), "2": (1, 1, 0), "3": (0, 0.8, 0), "4": (1, 0, 0), "5": (0, 0.3, 1)}
    obj_colors = {"right": (0.9, 0.55, 0.15), "left": (0.2, 0.6, 0.9)}  # RH객체=주황, LH객체=하늘

    # 액터 생성 순서를 기록 → root/dof 텐서 인덱스 계산
    # 순서: rh_hand, rh_obj, lh_hand, lh_obj
    root_order = []   # 각 actor의 root index 순서대로 side/kind
    dof_offsets = {}  # side -> dof 시작 인덱스
    dof_cursor = 0
    for side in ("right", "left"):
        d = S[side]
        dexhand = d["dexhand"]
        src_fs = os.path.join(ASSET_ROOT, dexhand.urdf_path)
        urdf_fs = make_stl_visual_urdf(src_fs)
        urdf_rel = os.path.relpath(urdf_fs, ASSET_ROOT)
        ao = gymapi.AssetOptions()
        ao.fix_base_link = False
        ao.disable_gravity = True
        ao.collapse_fixed_joints = False
        asset = gym.load_asset(sim, ASSET_ROOT, urdf_rel, ao)
        asset_bodies = gym.get_asset_rigid_body_names(asset)
        name2asset = {n: i for i, n in enumerate(asset_bodies)}
        d["bn_to_asset"] = [name2asset[n] for n in dexhand.body_names]
        actor = gym.create_actor(env, asset, gymapi.Transform(), f"hand_{side}", 0, 1)
        d["actor"] = actor
        for bi, bn in enumerate(asset_bodies):
            for fx, c in fcol.items():
                if bn.startswith(f"link_{fx}_"):
                    gym.set_rigid_body_color(env, actor, bi, gymapi.MESH_VISUAL, gymapi.Vec3(*c))
        root_order.append((side, "hand"))
        dof_offsets[side] = dof_cursor
        dof_cursor += len(gym.get_asset_dof_names(asset))

        # 객체
        d["obj_actor"] = None
        try:
            oao = gymapi.AssetOptions()
            oao.fix_base_link = False
            oao.disable_gravity = True
            obj_asset = gym.load_asset(sim, os.path.dirname(d["obj_urdf"]), os.path.basename(d["obj_urdf"]), oao)
            oactor = gym.create_actor(env, obj_asset, gymapi.Transform(), f"obj_{side}", 0, 1)
            gym.set_rigid_body_color(env, oactor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*obj_colors[side]))
            d["obj_actor"] = oactor
            root_order.append((side, "obj"))
        except Exception as e:
            print(f"[{side}] 객체 로드 실패: {e}")

    # root index 매핑
    for ri, (side, kind) in enumerate(root_order):
        S[side][f"root_{kind}"] = ri

    # DOF 드라이브 = POS 홀드 (양손 모두)
    for side in ("right", "left"):
        actor = S[side]["actor"]
        dp = gym.get_actor_dof_properties(env, actor)
        dp["driveMode"][:] = gymapi.DOF_MODE_POS
        dp["stiffness"][:] = 1000.0
        dp["damping"][:] = 50.0
        gym.set_actor_dof_properties(env, actor, dp)

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))   # (num_actors,13)
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))     # (total_dof,2)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    look_tgt = S["right"]["target_joints"][0, 0]
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, 0.5, 0.6), gymapi.Vec3(*look_tgt))
    for k, a in {gymapi.KEY_SPACE: "pause", gymapi.KEY_LEFT: "prev",
                 gymapi.KEY_RIGHT: "next", gymapi.KEY_R: "reset", gymapi.KEY_ESCAPE: "quit"}.items():
        gym.subscribe_viewer_keyboard_event(viewer, k, a)

    red = gymutil.WireframeSphereGeometry(0.006, 6, 6, gymapi.Transform(), color=(1, 0, 0))
    tips = {"thumb": 5, "index": 10, "middle": 15, "ring": 20, "pinky": 25}
    err_acc = {side: {k: [] for k in tips} for side in ("right", "left")}

    def set_frame(t):
        for side in ("right", "left"):
            d = S[side]
            root[d["root_hand"], 0:3] = torch.tensor(d["opt_wrist_pos"][t])
            root[d["root_hand"], 3:7] = torch.tensor(d["opt_wrist_quat"][t])
            root[d["root_hand"], 7:13] = 0.0
            if d["obj_actor"] is not None:
                root[d["root_obj"], 0:3] = torch.tensor(d["obj_pos"][t])
                root[d["root_obj"], 3:7] = torch.tensor(d["obj_quat"][t])
                root[d["root_obj"], 7:13] = 0.0
            off = dof_offsets[side]
            nd = d["opt_dof"].shape[1]
            dof_state[off:off + nd, 0] = torch.tensor(d["opt_dof"][t])
            dof_state[off:off + nd, 1] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        for side in ("right", "left"):
            gym.set_actor_dof_position_targets(env, S[side]["actor"], S[side]["opt_dof"][t].astype(np.float32))

    print("\n[색] 손: thumb=회색 index=노랑 middle=초록 ring=빨강 pinky=파랑 | 객체: RH=주황 LH=하늘")
    print("[재생] SPACE=정지 LEFT/RIGHT=프레임± R=처음 ESC=종료")
    t = 0
    paused = False
    step_frames = max(1, int(60 / max(1, p.fps)))
    fcount = 0
    while not gym.query_viewer_has_closed(viewer):
        for e in gym.query_viewer_action_events(viewer):
            if e.value == 0:
                continue
            if e.action == "pause": paused = not paused
            elif e.action == "prev": t = (t - 1) % Tr; paused = True
            elif e.action == "next": t = (t + 1) % Tr; paused = True
            elif e.action == "reset": t = 0
            elif e.action == "quit": gym.destroy_viewer(viewer); gym.destroy_sim(sim); return

        set_frame(t)
        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)

        gym.clear_lines(viewer)
        for side in ("right", "left"):
            d = S[side]
            rb = gym.get_actor_rigid_body_states(env, d["actor"], gymapi.STATE_POS)["pose"]["p"]
            ach = np.stack([np.array([rb[i]["x"], rb[i]["y"], rb[i]["z"]]) for i in d["bn_to_asset"]])
            tj = d["target_joints"][min(t, d["T"] - 1)]
            bones = d["dexhand"].bone_links
            bverts = np.empty((len(bones) * 2, 3), dtype=np.float32)
            for bi_, (a_, b_) in enumerate(bones):
                bverts[2 * bi_] = tj[a_]; bverts[2 * bi_ + 1] = tj[b_]
            bcol = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (len(bones), 1))
            gym.add_lines(viewer, env, len(bones), bverts, bcol)
            for k in range(26):
                gymutil.draw_lines(red, gym, viewer, env, gymapi.Transform(p=gymapi.Vec3(*tj[k])))
            for name, bi in tips.items():
                err_acc[side][name].append(float(np.linalg.norm(ach[bi] - tj[bi])))

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        fcount += 1
        if not paused and fcount % step_frames == 0:
            if t == Tr - 1:
                for side in ("right", "left"):
                    print(f"  [{side} tip 오차 cm] " + "  ".join(
                        f"{k}={np.mean(v) * 100:.2f}" for k, v in err_acc[side].items() if v))
                    for v in err_acc[side].values(): v.clear()
            t = (t + 1) % Tr


if __name__ == "__main__":
    main()
