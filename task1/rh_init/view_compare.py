"""두 자세를 한 화면에 나란히 비교 (RB5+Allegro+캡).

    python view_compare.py <poseA.json> <poseB.json>
    A = 파랑빛(래퍼/목표), B = 주황빛(MLP). allegro 손은 둘 다 흰색.
    두 로봇을 y 방향으로 벌려 배치(기본 0.9m, SEP 환경변수로 조절).
    SPACE 관절값 비교 출력, ESC 종료.
"""
import json
import math
import os
import sys

from isaacgym import gymapi, gymtorch
import torch

RB5_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
ALLEGRO_ORDER = ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
                 "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
                 "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
                 "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"]


def main():
    if len(sys.argv) < 3:
        sys.exit("사용: python view_compare.py <poseA.json> <poseB.json>")
    pa, pb = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    labels = (os.path.basename(sys.argv[1]), os.path.basename(sys.argv[2]))
    sep = float(os.environ.get("SEP", "0.9"))          # 두 로봇 y 간격(m)
    cap = pa.get("cap", [0.4, 0.0, 0.202])
    CAP_FRAME_Z0 = 0.013
    import re as _re
    note = pa.get("note", "")
    rm, hm = _re.search(r"r=([0-9.]+)", note), _re.search(r"h=([0-9.]+)", note)
    cap_r = float(rm.group(1)) / 1000.0 if rm else 0.044
    cap_h = float(hm.group(1)) / 1000.0 if hm else 0.016

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 1 / 60; sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0, 0, 0)
    sp.physx.solver_type = 1; sp.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    gym.add_ground(sim, gymapi.PlaneParams())

    ao = gymapi.AssetOptions()
    ao.fix_base_link = True
    ao.default_dof_drive_mode = gymapi.DOF_MODE_POS
    ao.collapse_fixed_joints = False
    urdf = os.environ.get("ROBOT_URDF", "rb5_allegro_vmount_siltip2.urdf")
    print(f"[view_compare] URDF = {urdf}")
    asset = gym.load_asset(sim, RB5_DIR, urdf, ao)
    nd = gym.get_asset_dof_count(asset)
    names = list(gym.get_asset_dof_names(asset))
    nbody = gym.get_asset_rigid_body_count(asset)
    arm_bodies = {i for i in range(nbody)
                  if not gym.get_asset_rigid_body_name(asset, i).startswith(("link_", "base_link"))}

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    acts, targets = [], []
    # A = 파랑빛 팔, B = 주황빛 팔. allegro(손)는 둘 다 흰색.
    arm_cols = [gymapi.Vec3(0.35, 0.55, 0.95), gymapi.Vec3(0.95, 0.55, 0.2)]
    # 두 로봇을 벌릴 때는 평행이동이 아니라 z축 회전으로 배치해야 자세가 유지된다.
    # (관절값은 베이스원점 기준으로 풀린 값 -- 베이스를 회전시키면 캡도 같이 회전한다.)
    yaws = [(k - 0.5) * 2.0 * math.asin(min(1.0, sep / 2.0 / max(cap[0], 1e-6))) for k in range(2)]
    for k, pose in enumerate((pa, pb)):
        tf = gymapi.Transform()
        tf.p = gymapi.Vec3(0.0, 0.0, 0.0)
        tf.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), yaws[k])
        act = gym.create_actor(env, asset, tf, f"robot{k}", k, 1)
        pr = gym.get_actor_dof_properties(env, act)
        for j in range(nd):
            pr["driveMode"][j] = gymapi.DOF_MODE_POS
            if names[j].startswith("joint_"):
                pr["stiffness"][j] = 300.0; pr["damping"][j] = 18.0
            else:
                pr["stiffness"][j] = 20000.0; pr["damping"][j] = 400.0; pr["effort"][j] = 300.0
        gym.set_actor_dof_properties(env, act, pr)
        for b in range(nbody):
            col = arm_cols[k] if b in arm_bodies else gymapi.Vec3(1.0, 1.0, 1.0)  # 손=흰색
            gym.set_rigid_body_color(env, act, b, gymapi.MESH_VISUAL, col)
        tgt = torch.zeros(nd)
        for n, v in zip(pose["rb5"]["names"], pose["rb5"]["q"]):
            tgt[names.index(n)] = v
        for n, v in zip(ALLEGRO_ORDER, pose["dof_pos"]):
            tgt[names.index(n)] = v
        acts.append(act); targets.append(tgt)

    gym.prepare_sim(sim)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    # 두 로봇의 중앙을, 캡 반대편에서 정면으로 본다(원근 왜곡 최소화).
    cam = [float(v) for v in os.environ.get("CAM", "1.9,0,0.55").split(",")]
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*cam),
                              gymapi.Vec3(0.15, 0.0, cap[2]))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "info")

    ds = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
    full = torch.zeros(ds.shape[0])
    offs = []
    for act, tgt in zip(acts, targets):
        off = gym.get_actor_dof_index(env, act, 0, gymapi.DOMAIN_SIM)
        ds[off:off + nd, 0] = tgt; full[off:off + nd] = tgt
        offs.append(off)
    ds[:, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(ds))

    # 각 로봇 앞의 캡(노랑) 와이어프레임 (실제 r,h)
    def cyl(cx, cy, z0, z1, rad, nr=5, nv=16, seg=40):
        v = []
        for t in range(nr):
            z = z0 + (z1 - z0) * t / (nr - 1)
            for s in range(seg):
                a0, a1 = 2 * math.pi * s / seg, 2 * math.pi * (s + 1) / seg
                v += [cx + rad * math.cos(a0), cy + rad * math.sin(a0), z,
                      cx + rad * math.cos(a1), cy + rad * math.sin(a1), z]
        for s in range(nv):
            a = 2 * math.pi * s / nv
            v += [cx + rad * math.cos(a), cy + rad * math.sin(a), z0,
                  cx + rad * math.cos(a), cy + rad * math.sin(a), z1]
        return v
    cz = cap[2] + CAP_FRAME_Z0
    lines, cols = [], []
    for k in range(2):
        # 캡도 베이스와 같은 yaw 로 회전 (로봇 기준 상대위치 유지)
        cx = cap[0] * math.cos(yaws[k]) - cap[1] * math.sin(yaws[k])
        cy = cap[0] * math.sin(yaws[k]) + cap[1] * math.cos(yaws[k])
        cv = cyl(cx, cy, cz, cz + cap_h, cap_r)
        bv = cyl(cx, cy, 0.0, cz, cap_r, nr=4)
        lines += cv + bv
        cols += [1.0, 0.85, 0.1] * (len(cv) // 6) + [0.5, 0.5, 0.55] * (len(bv) // 6)

    print(f"[A 파랑팔] {labels[0]}   [B 주황팔] {labels[1]}   (손=흰색)")
    print("SPACE 관절값 비교, ESC 종료", flush=True)
    while not gym.query_viewer_has_closed(viewer):
        for e in gym.query_viewer_action_events(viewer):
            if e.value and e.action == "info":
                gym.refresh_dof_state_tensor(sim)
                print(f"\n 관절        A({labels[0][:14]})   B({labels[1][:14]})    차")
                for j in range(nd):
                    a = math.degrees(float(ds[offs[0] + j, 0]))
                    b = math.degrees(float(ds[offs[1] + j, 0]))
                    print(f"  {names[j]:10s} {a:+8.1f}  {b:+8.1f}  {b - a:+7.1f}")
        # 정적 비교: 물리로 유지하지 않고 관절 상태를 매 프레임 강제 세팅(흔들림 0).
        ds[:, 0] = full; ds[:, 1] = 0.0
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(ds))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(full))
        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.clear_lines(viewer)
        gym.add_lines(viewer, env, len(lines) // 6, lines, cols)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
