"""리타겟팅 결과를 GUI로 시간순 재생 + 품질 확인.

- 저장된 mano2<dexhand> pkl(opt_wrist/opt_dof)로 dg5fs 손을 프레임순 재생
- MANO 타겟 키포인트 = 빨강 구, 손 달성 키포인트 = 초록 구 오버레이
- 손가락별 tip 오차(빨강↔손끝)를 콘솔에 출력

실행:
    cd /home/leegyuwon/Documents/ManipTrans
    PATH=<env>/bin:$PATH LD_LIBRARY_PATH=<env>/lib:$LD_LIBRARY_PATH \
    <env>/bin/python tools/play_retarget.py --data_idx 72f05@0 --dexhand dg5fs --side right

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
    """mano2dexhand.Mano2Dexhand.__init__ 과 동일한 데이터->gym 변환."""
    M = np.eye(4)
    M[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
    M[:3, 3] = np.array([0, 0, 0.4 + 0.015])  # table_surface_z
    return M.astype(np.float32)


def aa_to_quat_xyzw(aa):  # (N,3) axis-angle -> (N,4) xyzw
    ang = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-8, aa / np.clip(ang, 1e-8, None), np.array([1.0, 0, 0]))
    s = np.sin(ang / 2)
    return np.concatenate([axis * s, np.cos(ang / 2)], axis=-1).astype(np.float32)


def rotmat_to_quat_xyzw(R):  # (3,3) -> (4,) xyzw
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
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


def main():
    p = gymutil.parse_arguments(
        description="Replay retargeting",
        headless=False,
        custom_parameters=[
            {"name": "--data_idx", "type": str, "default": "72f05@0"},
            {"name": "--dexhand", "type": str, "default": "dg5fs"},
            {"name": "--side", "type": str, "default": "right"},
            {"name": "--fps", "type": int, "default": 30},
        ],
    )
    idx = p.data_idx
    dexhand = DexHandFactory.create_hand(p.dexhand, p.side)
    device = "cuda:0"

    # ---- 타겟(MANO) 로드 ----
    dtype = ManipDataFactory.dataset_type(idx)
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dtype, side=p.side, device=device,
        mujoco2gym_transf=torch.eye(4, device=device), dexhand=dexhand, verbose=False,
    )
    demo = pack_data([demo_d[idx]], dexhand)
    T = demo["mano_joints"].shape[0]
    target_mano = demo["mano_joints"].view(T, -1, 3).detach().cpu().numpy()   # (T,25,3) body_names[1:] 순
    target_wrist = demo["wrist_pos"].view(T, 3).detach().cpu().numpy()
    target_joints = np.concatenate([target_wrist[:, None], target_mano], axis=1)  # (T,26,3) body_names 순
    # 데이터 프레임 -> gym 프레임 (리타겟팅과 동일 변환). 안 하면 타겟이 딴 곳에 뜸.
    M = mujoco2gym_transf()
    target_joints = (M[:3, :3] @ target_joints.reshape(-1, 3).T).T + M[:3, 3]
    target_joints = target_joints.reshape(T, -1, 3)
    target_wrist = target_joints[:, 0]

    # ---- 리타겟 결과 로드 ----
    base = os.path.split(demo["data_path"][0])[-1]
    _stage = idx.split("@")[-1]  # 두 자리 stage(@15 등)도 올바로 (idx[-1]은 마지막 글자라 버그)
    dump = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{base.replace('.pkl', f'@{_stage}.pkl')}"
    if not os.path.isfile(dump):
        raise FileNotFoundError(f"리타겟 결과 없음: {dump}\n먼저 mano2dexhand.py로 생성하세요.")
    ret = pickle.load(open(dump, "rb"))
    opt_wrist_pos = ret["opt_wrist_pos"].astype(np.float32)          # (T,3)
    opt_wrist_quat = aa_to_quat_xyzw(ret["opt_wrist_rot"])           # (T,4) xyzw
    opt_dof = ret["opt_dof_pos"].astype(np.float32)                  # (T,20) asset dof 순
    Tr = opt_dof.shape[0]
    print(f"[replay] {dump}\n  프레임={Tr}, target 프레임={T}")

    # ---- 조작 대상 객체(뚜껑) 궤적 ----
    obj_traj = demo["obj_trajectory"].view(T, 4, 4).detach().cpu().numpy()   # 데이터 프레임
    obj_traj_gym = np.einsum("ij,tjk->tik", M, obj_traj).astype(np.float32)  # gym 프레임
    obj_pos = obj_traj_gym[:, :3, 3]                                          # (T,3)
    obj_quat = np.stack([rotmat_to_quat_xyzw(obj_traj_gym[t, :3, :3]) for t in range(T)])  # (T,4)
    obj_urdf = demo["obj_urdf_path"][0]
    print(f"[객체] {obj_urdf}")

    # ---- Isaac Gym ----
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1.0 / 60.0
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0, 0, 0)
    sp.use_gpu_pipeline = False
    sp.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    # 바닥 없음: 순수 kinematic 재생이라 불필요하고, 손과 충돌해 스켈레톤에서 분리시킴

    src_fs = os.path.join(ASSET_ROOT, dexhand.urdf_path)
    urdf_fs = make_stl_visual_urdf(src_fs)
    urdf_rel = os.path.relpath(urdf_fs, ASSET_ROOT)
    ao = gymapi.AssetOptions()
    ao.fix_base_link = False
    ao.disable_gravity = True
    ao.collapse_fixed_joints = False
    asset = gym.load_asset(sim, ASSET_ROOT, urdf_rel, ao)
    asset_bodies = gym.get_asset_rigid_body_names(asset)
    # asset body -> dexhand.body_names 인덱스 매핑 (오차 계산용)
    name2asset = {n: i for i, n in enumerate(asset_bodies)}
    bn_to_asset = [name2asset[n] for n in dexhand.body_names]

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    actor = gym.create_actor(env, asset, gymapi.Transform(), "hand", 0, 1)

    # 손가락별 색상: thumb=회색 index=노랑 middle=초록 ring=빨강 pinky=파랑
    fcol = {"1": (0.5, 0.5, 0.55), "2": (1, 1, 0), "3": (0, 0.8, 0), "4": (1, 0, 0), "5": (0, 0.3, 1)}
    for bi, bn in enumerate(asset_bodies):
        for fx, c in fcol.items():
            if bn.startswith(f"link_{fx}_"):
                gym.set_rigid_body_color(env, actor, bi, gymapi.MESH_VISUAL, gymapi.Vec3(*c))
    print("[색] thumb=회색 index=노랑 middle=초록 ring=빨강 pinky=파랑")

    # ---- 조작 대상 객체(뚜껑) actor ----
    obj_actor = None
    try:
        oao = gymapi.AssetOptions()
        oao.fix_base_link = False
        oao.disable_gravity = True
        obj_asset = gym.load_asset(sim, os.path.dirname(obj_urdf), os.path.basename(obj_urdf), oao)
        # filter=1: 손(filter=1)과 충돌 안 함 (kinematic 재생)
        obj_actor = gym.create_actor(env, obj_asset, gymapi.Transform(), "obj", 0, 1)
        gym.set_rigid_body_color(env, obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.55, 0.15))
        print("[객체] 로드 성공 (주황색)")
    except Exception as e:
        print(f"[객체] 로드 실패, 객체 없이 재생: {e}")

    # DOF를 POS 드라이브로 단단히 고정 (free joint 드리프트 → base 튕김 방지)
    dp = gym.get_actor_dof_properties(env, actor)
    dp["driveMode"][:] = gymapi.DOF_MODE_POS
    dp["stiffness"][:] = 1000.0
    dp["damping"][:] = 50.0
    gym.set_actor_dof_properties(env, actor, dp)

    gym.prepare_sim(sim)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))       # (1,13)
    dof_state = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))          # (20,2)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.4, 0.4, 0.5), gymapi.Vec3(*target_wrist[0]))
    for k, a in {gymapi.KEY_SPACE: "pause", gymapi.KEY_LEFT: "prev",
                 gymapi.KEY_RIGHT: "next", gymapi.KEY_R: "reset", gymapi.KEY_ESCAPE: "quit"}.items():
        gym.subscribe_viewer_keyboard_event(viewer, k, a)

    red = gymutil.WireframeSphereGeometry(0.006, 6, 6, gymapi.Transform(), color=(1, 0, 0))
    green = gymutil.WireframeSphereGeometry(0.005, 6, 6, gymapi.Transform(), color=(0, 1, 0))
    tips = {"thumb": 5, "index": 10, "middle": 15, "ring": 20, "pinky": 25}
    err_acc = {k: [] for k in tips}

    def set_frame(t):
        # 손목 root 재고정 (pos + rot, 속도 0)
        root[0, 0:3] = torch.tensor(opt_wrist_pos[t])
        root[0, 3:7] = torch.tensor(opt_wrist_quat[t])
        root[0, 7:13] = 0.0
        # 객체(뚜껑) root = obj_trajectory[t]  (root 텐서 인덱스 1)
        if obj_actor is not None:
            root[1, 0:3] = torch.tensor(obj_pos[t])
            root[1, 3:7] = torch.tensor(obj_quat[t])
            root[1, 7:13] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        # DOF: state로 정확히 스냅 + POS 드라이브 target으로 홀드
        dof_state[:, 0] = torch.tensor(opt_dof[t])
        dof_state[:, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        gym.set_actor_dof_position_targets(env, actor, opt_dof[t].astype(np.float32))

    # 원본(사람) 손 스켈레톤: dexhand.bone_links 로 타겟 키포인트 연결 → 흰 선
    bones = dexhand.bone_links  # body_names 인덱스 쌍 리스트
    bone_colors = np.tile(np.array([1.0, 1.0, 1.0], dtype=np.float32), (len(bones), 1))
    print("[오버레이] 흰 선=원본 사람손 스켈레톤(타겟), 컬러 메시=dg5fs 리타겟 결과, 빨간 구=타겟 관절")

    print("\n[재생] SPACE=정지 LEFT/RIGHT=프레임± R=처음 ESC=종료")
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

        # 달성 키포인트 (asset order) -> body_names order
        rb = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)["pose"]["p"]
        ach = np.stack([np.array([rb[i]["x"], rb[i]["y"], rb[i]["z"]]) for i in bn_to_asset])  # (26,3)
        tj = target_joints[min(t, T - 1)]  # (26,3)

        gym.clear_lines(viewer)
        # 원본 사람손 스켈레톤 (타겟 관절을 bone_links로 연결)
        bverts = np.empty((len(bones) * 2, 3), dtype=np.float32)
        for bi_, (a_, b_) in enumerate(bones):
            bverts[2 * bi_] = tj[a_]
            bverts[2 * bi_ + 1] = tj[b_]
        gym.add_lines(viewer, env, len(bones), bverts, bone_colors)
        # 타겟 관절 = 빨간 구 (작게)
        for k in range(26):
            gymutil.draw_lines(red, gym, viewer, env, gymapi.Transform(p=gymapi.Vec3(*tj[k])))
        for name, bi in tips.items():
            err_acc[name].append(float(np.linalg.norm(ach[bi] - tj[bi])))

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        fcount += 1
        if not paused and fcount % step_frames == 0:
            if t == Tr - 1:  # 한 바퀴 끝: 오차 요약 출력
                print("  [손가락 tip 오차 cm] " + "  ".join(
                    f"{k}={np.mean(v) * 100:.2f}" for k, v in err_acc.items() if v))
                for v in err_acc.values(): v.clear()
            t = (t + 1) % Tr


if __name__ == "__main__":
    main()
