"""
dg5fs URDF 육안 확인용 Isaac Gym 뷰어 (키보드 수동 제어).

실행:
    cd /home/leegyuwon/Documents/ManipTrans
    LD_LIBRARY_PATH=/home/leegyuwon/Documents/miniconda3/envs/maniptrans/lib:$LD_LIBRARY_PATH \
    /home/leegyuwon/Documents/miniconda3/envs/maniptrans/bin/python tools/view_dg5fs.py
    # 왼손: --side left

[DOF 제어]
    RIGHT / LEFT : 다음 / 이전 DOF 선택
    UP    / DOWN : 선택 DOF 값 +/- (스텝 0.1 rad)
    PAGEUP/PAGEDN: 선택 DOF를 upper / lower 리밋으로 이동
    R           : 모든 DOF 0 리셋
    P           : 현재 DOF 값 출력

[BASE(palm) 회전 — 방향 맞추기]
    1 / 2 : world X축 -90 / +90
    3 / 4 : world Y축 -90 / +90
    5 / 6 : world Z축 -90 / +90 (Z=위, +Z회전 = 위에서 볼 때 CCW)
    0     : base 회전 identity 리셋
    B     : 현재 base 쿼터니언 + axis-angle 출력 (relative_rotation 반영용)

    ESC   : 종료
"""
import argparse
import os
import numpy as np
from isaacgym import gymapi, gymtorch
import torch

ASSET_ROOT = "maniptrans_envs"  # 실제 에셋은 maniptrans_envs/assets 아래


def make_stl_visual_urdf(src_fs):
    """Isaac Gym이 DAE(visual) 메시를 잘못 렌더(손가락이 palm과 분리돼 보임)하므로,
    visual을 STL(=collision, 동일 지오메트리)로 바꾼 뷰어용 URDF 생성.
    물리/학습은 원래 STL collision을 쓰므로 이는 뷰어 표시용일 뿐."""
    with open(src_fs, "r") as f:
        txt = f.read()
    txt = txt.replace('.dae"', '.STL"').replace('.DAE"', '.STL"')
    dst_fs = src_fs.replace(".urdf", "_stlvis.urdf")
    with open(dst_fs, "w") as f:
        f.write(txt)
    return dst_fs


def quat_mul(a, b):  # xyzw ⊗ xyzw
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def axis_angle_quat(axis, ang):
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    s = np.sin(ang / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(ang / 2.0)], dtype=np.float64)


def quat_to_axis_angle(q):
    x, y, z, w = q / np.linalg.norm(q)
    w = np.clip(w, -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = np.array([x, y, z]) / s if s > 1e-6 else np.array([1.0, 0, 0])
    return axis, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--step", type=float, default=0.1, help="DOF 조정 스텝(rad)")
    ap.add_argument("--relrot", action="store_true",
                    help="시작 시 dg5fs.py의 relative_rotation(Y±90°) 적용 (기본은 identity)")
    args = ap.parse_args()

    src_fs = os.path.join(ASSET_ROOT, f"assets/dg5fs_hand/dg5fs_{args.side}.urdf")
    if not os.path.isfile(src_fs):
        raise FileNotFoundError(f"URDF 없음: {src_fs}")
    # DAE 렌더 아티팩트 회피: STL visual 버전 생성해 로드
    urdf_fs = make_stl_visual_urdf(src_fs)
    urdf_rel = os.path.relpath(urdf_fs, ASSET_ROOT)
    print(f"[viewer URDF] {urdf_fs} (visual=STL, DAE 렌더 아티팩트 회피)")

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.use_gpu = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("create_sim 실패")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    # base를 실시간 회전하려면 fix_base_link=False + 매 프레임 root state 고정
    asset_opts = gymapi.AssetOptions()
    asset_opts.fix_base_link = False
    asset_opts.disable_gravity = True
    asset_opts.collapse_fixed_joints = False
    asset_opts.flip_visual_attachments = False
    asset_opts.armature = 0.001
    print(f"[load] {urdf_rel}")
    asset = gym.load_asset(sim, ASSET_ROOT, urdf_rel, asset_opts)

    n_dofs = gym.get_asset_dof_count(asset)
    if n_dofs == 0:
        raise RuntimeError(f"URDF 파싱 실패 (n_dofs=0): {urdf_rel}")
    n_bodies = gym.get_asset_rigid_body_count(asset)
    dof_names = gym.get_asset_dof_names(asset)
    body_names = gym.get_asset_rigid_body_names(asset)
    dof_props = gym.get_asset_dof_properties(asset)
    lowers = dof_props["lower"].copy()
    uppers = dof_props["upper"].copy()

    # URDF finger index -> 해부학 라벨 (형상 검증 확정): 1=thumb..5=pinky
    FINGER = {"1": "thumb", "2": "index", "3": "middle", "4": "ring", "5": "pinky"}
    ROLE = {"1": "abduction", "2": "MCP", "3": "PIP", "4": "DIP"}

    def dof_label(nm):  # "joint_1_2" -> "thumb-MCP"
        try:
            _, fx, jy = nm.split("_")
            return f"{FINGER.get(fx, fx)}-{ROLE.get(jy, jy)}"
        except Exception:
            return nm

    print(f"\n=== dg5fs_{args.side} ===  DOF={n_dofs}, Body={n_bodies}")
    print("finger 매핑: 1=thumb 2=index 3=middle 4=ring 5=pinky")
    print("\n[DOF 리밋]")
    for i, nm in enumerate(dof_names):
        print(f"  [{i:2d}] {nm:12s} {dof_label(nm):16s} lower={lowers[i]:+.4f} upper={uppers[i]:+.4f}")

    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(1000.0)
    dof_props["damping"].fill(50.0)

    # base 시작 회전
    if args.relrot:
        ang = -np.pi / 2 if args.side == "right" else np.pi / 2
        base_q = axis_angle_quat([0, 1, 0], ang)
        print(f"[base] 시작: relative_rotation Y {np.degrees(ang):+.0f}°")
    else:
        base_q = np.array([0.0, 0.0, 0.0, 1.0])
        print("[base] 시작: identity (native 프레임). 1~6 키로 회전해 방향 맞추기")

    base_pos = np.array([0.0, 0.0, 0.3], dtype=np.float32)

    env = gym.create_env(sim, gymapi.Vec3(-0.5, -0.5, 0), gymapi.Vec3(0.5, 0.5, 1), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*base_pos)
    pose.r = gymapi.Quat(*base_q)
    actor = gym.create_actor(env, asset, pose, "dg5fs", 0, 1)
    gym.set_actor_dof_properties(env, actor, dof_props)

    targets = np.zeros(n_dofs, dtype=np.float32)
    gym.set_actor_dof_position_targets(env, actor, targets)

    # root state tensor (base 고정용)
    gym.prepare_sim(sim)
    root_t = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))  # (1,13)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("create_viewer 실패 (headless?)")
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.4, 0.4, 0.5), gymapi.Vec3(0, 0, 0.3))

    keymap = {
        gymapi.KEY_LEFT: "prev", gymapi.KEY_RIGHT: "next",
        gymapi.KEY_UP: "inc", gymapi.KEY_DOWN: "dec",
        gymapi.KEY_PAGE_UP: "tomax", gymapi.KEY_PAGE_DOWN: "tomin",
        gymapi.KEY_R: "reset", gymapi.KEY_P: "print",
        gymapi.KEY_1: "rx-", gymapi.KEY_2: "rx+",
        gymapi.KEY_3: "ry-", gymapi.KEY_4: "ry+",
        gymapi.KEY_5: "rz-", gymapi.KEY_6: "rz+",
        gymapi.KEY_0: "rident", gymapi.KEY_B: "bquat",
        gymapi.KEY_ESCAPE: "quit",
    }
    for k, a in keymap.items():
        gym.subscribe_viewer_keyboard_event(viewer, k, a)

    sel = [0]
    HALF = np.pi / 2

    def show_sel():
        i = sel[0]
        print(f"  >> DOF [{i:2d}] {dof_names[i]:12s} ({dof_label(dof_names[i]):16s}) "
              f"값={targets[i]:+.3f} 범위=[{lowers[i]:+.3f},{uppers[i]:+.3f}]", flush=True)

    def show_base():
        axis, ang = quat_to_axis_angle(base_q)
        print(f"  >> BASE quat(xyzw)=[{base_q[0]:+.4f},{base_q[1]:+.4f},"
              f"{base_q[2]:+.4f},{base_q[3]:+.4f}]  axis={np.round(axis,3)} "
              f"angle={np.degrees(ang):+.1f}°", flush=True)

    def rot_base(axis, deg):
        nonlocal base_q
        dq = axis_angle_quat(axis, np.radians(deg))
        base_q = quat_mul(dq, base_q)  # world-frame 회전 (pre-multiply)
        base_q /= np.linalg.norm(base_q)
        show_base()

    print("\n[DOF] LEFT/RIGHT=선택 UP/DOWN=값± PAGEUP/DN=리밋 R=리셋 P=출력")
    print("[BASE] 1/2=X∓90 3/4=Y∓90 5/6=Z∓90(CCW) 0=리셋 B=쿼터니언출력  ESC=종료")
    show_sel(); show_base()

    running = True
    while running and not gym.query_viewer_has_closed(viewer):
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value == 0:
                continue
            a = evt.action
            i = sel[0]
            if a == "next": sel[0] = (i + 1) % n_dofs; show_sel()
            elif a == "prev": sel[0] = (i - 1) % n_dofs; show_sel()
            elif a == "inc": targets[i] = min(uppers[i], targets[i] + args.step); show_sel()
            elif a == "dec": targets[i] = max(lowers[i], targets[i] - args.step); show_sel()
            elif a == "tomax": targets[i] = uppers[i]; show_sel()
            elif a == "tomin": targets[i] = lowers[i]; show_sel()
            elif a == "reset": targets[:] = 0.0; print("  >> 전체 0 리셋"); show_sel()
            elif a == "print":
                print("  [DOF 값]", {dof_names[j]: round(float(targets[j]), 3) for j in range(n_dofs)})
            elif a == "rx-": rot_base([1, 0, 0], -90)
            elif a == "rx+": rot_base([1, 0, 0], +90)
            elif a == "ry-": rot_base([0, 1, 0], -90)
            elif a == "ry+": rot_base([0, 1, 0], +90)
            elif a == "rz-": rot_base([0, 0, 1], -90)
            elif a == "rz+": rot_base([0, 0, 1], +90)
            elif a == "rident":
                base_q = np.array([0.0, 0.0, 0.0, 1.0]); print("  >> base identity"); show_base()
            elif a == "bquat": show_base()
            elif a == "quit": running = False

        # base를 매 프레임 고정 (pos 고정, 회전 = base_q, 속도 0)
        root_t[0, 0:3] = torch.tensor(base_pos)
        root_t[0, 3:7] = torch.tensor(base_q, dtype=torch.float32)
        root_t[0, 7:13] = 0.0
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_t))

        gym.set_actor_dof_position_targets(env, actor, targets)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    print("종료.")
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
