"""RB5 + Allegro + 텀블러 통합 포즈 뷰어.

MLP export json(dof_pos 16 + rb5.q 6 + cap)을 받아, task1 팔 모드 인퍼런스와
동일한 지오메트리로 정적 렌더한다:
  - RB5 base = env 원점(0,0,0), 고정·직립
  - 텀블러 바닥 = cap[:2] + z0 (기본 0.6,0,0), 캡은 나사원점 +0.225
  - 22 관절 = RB5 6(IK) + allegro 16(파지) 를 json 그대로 세팅

    python view_full.py <mlp_export.json>
    SPACE 관절값 출력, ESC 종료
"""
import json
import math
import os
import sys

from isaacgym import gymapi, gymtorch
import torch

RB5_ALLEGRO_DIR = "/home/leegyuwon/Documents/task1/assets/rb5_allegro"
TUMBLER_URDF_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"

# 통합 URDF DOF 순서 (get_asset_dof_names 로 확인) = RB5 6 + allegro 16.
# allegro 부분은 Isaac Gym 순서(joint_0-3, 12-15, 4-7, 8-11) = maniptrans dof_pos 순서.
ALLEGRO_ORDER = ["joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
                 "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
                 "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
                 "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0"]


def main():
    if len(sys.argv) < 2:
        sys.exit("사용: python view_full.py <mlp_export.json>")
    pose = json.load(open(sys.argv[1]))
    if "rb5" not in pose:
        sys.exit("json 에 rb5 필드 없음 -- --no-rb5 로 뽑았거나 구판. RB5 포함 export 필요.")
    cap = pose.get("cap", [0.6, 0.0, 0.225])         # RB5 base 기준 캡 원점
    note = pose.get("note", "")
    # 캡 실제 크기(r,h): note "allegro cyl r=30 h=20 ..." 파싱 (뷰어 캡을 이 크기로 그림)
    import re as _re
    _rm = _re.search(r"r=([0-9.]+)", note); _hm = _re.search(r"h=([0-9.]+)", note)
    cap_r = float(_rm.group(1)) / 1000.0 if _rm else 0.050
    cap_h = float(_hm.group(1)) / 1000.0 if _hm else 0.030
    print(f"[view_full] {note}")
    print(f"  캡 크기: r={cap_r*1000:.0f}mm h={cap_h*1000:.0f}mm (와이어프레임으로 실제 크기 표시)")
    print(f"  캡 월드위치(RB5 base 기준): {cap}  방위각 {pose.get('azimuth_deg')}도")
    print(f"  RB5 IK 오차: 위치 {pose['rb5']['pos_err_mm']}mm 자세 {pose['rb5']['rot_err_deg']}도 "
          f"도달 {pose['rb5']['feasible']}")

    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.dt = 1 / 60; sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z; sp.gravity = gymapi.Vec3(0, 0, 0)  # 정적: 중력 off
    sp.physx.solver_type = 1; sp.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    plane = gymapi.PlaneParams(); plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    ao = gymapi.AssetOptions()
    ao.fix_base_link = True
    ao.default_dof_drive_mode = gymapi.DOF_MODE_POS
    ao.collapse_fixed_joints = False
    urdf = os.environ.get("ROBOT_URDF", "rb5_allegro_realmount.urdf")  # 실물 마운트(mount.STL, J6⊥팜)
    print(f"[view_full] URDF = {urdf}")
    asset = gym.load_asset(sim, RB5_ALLEGRO_DIR, urdf, ao)
    nd = gym.get_asset_dof_count(asset)
    names = list(gym.get_asset_dof_names(asset))

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    # 캡/텀블러는 실제 r,h 크기의 와이어프레임으로 그린다 (고정메시 관통 방지).
    act = gym.create_actor(env, asset, gymapi.Transform(), "rb5_allegro", 0, 1)

    pr = gym.get_actor_dof_properties(env, act)
    for k in range(nd):
        pr["driveMode"][k] = gymapi.DOF_MODE_POS
        if names[k].startswith("joint_"):
            pr["stiffness"][k] = 300.0; pr["damping"][k] = 18.0
        else:
            pr["stiffness"][k] = 20000.0; pr["damping"][k] = 400.0; pr["effort"][k] = 300.0
    gym.set_actor_dof_properties(env, act, pr)
    for b in range(gym.get_actor_rigid_body_count(env, act)):
        gym.set_rigid_body_color(env, act, b, gymapi.MESH_VISUAL, gymapi.Vec3(0.85, 0.85, 0.88))
    lims = [(float(pr["lower"][k]), float(pr["upper"][k])) for k in range(nd)]

    # 목표 관절: RB5(IK) + allegro(파지) 를 이름으로 매핑
    tgt = torch.zeros(nd)
    for n, v in zip(pose["rb5"]["names"], pose["rb5"]["q"]):
        tgt[names.index(n)] = v
    for n, v in zip(ALLEGRO_ORDER, pose["dof_pos"]):
        tgt[names.index(n)] = v

    gym.prepare_sim(sim)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(2.0, -1.6, 1.1),
                              gymapi.Vec3(cap[0], cap[1], cap[2]))
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "info")

    # 캡(노랑) + 텀블러 몸체(회색) 와이어프레임: 실제 r,h 크기로 그려 관통여부를 정확히 표시.
    # 파지 프레임 원점(cap_world)은 나사원점이고, 캡 몸통은 그보다 CAP_FRAME_Z0(13mm)
    # 위에 얹힌다(저장된 wrist_rel_pos 가 +CAP_FRAME_Z0 이므로 캡도 그만큼 올려야 정합).
    CAP_FRAME_Z0 = 0.013
    cx, cy = cap[0], cap[1]
    cz = cap[2] + CAP_FRAME_Z0                        # 캡 바닥(파지 프레임 z=0 위치)
    _SEG = 48

    def _cyl_lines(z0, z1, rad, nrings=5, nvert=16):
        v = []
        for t in range(nrings):
            z = z0 + (z1 - z0) * t / (nrings - 1)
            for s in range(_SEG):
                a0 = 2 * math.pi * s / _SEG; a1 = 2 * math.pi * (s + 1) / _SEG
                v += [cx + rad * math.cos(a0), cy + rad * math.sin(a0), z,
                      cx + rad * math.cos(a1), cy + rad * math.sin(a1), z]
        for s in range(nvert):
            a = 2 * math.pi * s / nvert
            v += [cx + rad * math.cos(a), cy + rad * math.sin(a), z0,
                  cx + rad * math.cos(a), cy + rad * math.sin(a), z1]
        return v

    cap_v = _cyl_lines(cz, cz + cap_h, cap_r)                     # 캡 (파지 대상)
    body_v = _cyl_lines(0.0, cz, cap_r, nrings=4)                 # 몸체 (받침)
    cap_c = [1.0, 0.85, 0.1] * (len(cap_v) // 6)                  # 노랑 (라인당 1색)
    body_c = [0.5, 0.5, 0.55] * (len(body_v) // 6)

    ds = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(-1, 2)
    off = gym.get_actor_dof_index(env, act, 0, gymapi.DOMAIN_SIM)  # 텀블러 앞자리 없음(고정)→로봇 DOF 시작
    ds[off:off + nd, 0] = tgt; ds[:, 1] = 0.0
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(ds))
    full = torch.zeros(ds.shape[0]); full[off:off + nd] = tgt

    # 최적화 접촉점 마커: 빨강=현재 모델 접촉점(팁원점+PALM_LOCAL·FR),
    #   초록=실제 pad 중심(로컬 PAD_REAL), 파랑선=팁원점→모델점(PALM 방향).
    import numpy as _np
    PALM = _np.array([float(v) for v in os.environ.get("PAD_AXIS", "1,0,0").split(",")])
    FR = float(os.environ.get("PAD_FR", "0.012"))
    PAD_REAL = _np.array([float(v) for v in os.environ.get("PAD_REAL", "0.002,0,-0.015").split(",")])
    TIP_BODIES = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]  # 엄지,검지,중지,약지
    tip_idx = [gym.find_actor_rigid_body_index(env, act, n, gymapi.DOMAIN_SIM) for n in TIP_BODIES]
    rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(-1, 13)

    def _qrot(q, v):
        x, y, z, w = q; xyz = _np.array([x, y, z])
        t = 2.0 * _np.cross(xyz, v)
        return _np.asarray(v) + w * t + _np.cross(xyz, t)

    def _cross(p, s=0.007):
        out = []
        for ax in _np.eye(3):
            out += list(_np.asarray(p) - ax * s) + list(_np.asarray(p) + ax * s)
        return out

    print("SPACE 관절값 출력, ESC 종료", flush=True)
    while not gym.query_viewer_has_closed(viewer):
        for e in gym.query_viewer_action_events(viewer):
            if e.value and e.action == "info":
                gym.refresh_dof_state_tensor(sim)
                print("\n 관절       현재각도    한계도")
                for k in range(nd):
                    print(f"  {names[k]:10s} {math.degrees(float(ds[off + k, 0])):+8.1f}  "
                          f"[{math.degrees(lims[k][0]):+.0f}, {math.degrees(lims[k][1]):+.0f}]")
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(full))
        gym.simulate(sim); gym.fetch_results(sim, True)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.step_graphics(sim)
        gym.clear_lines(viewer)
        gym.add_lines(viewer, env, len(cap_v) // 6, cap_v, cap_c)
        gym.add_lines(viewer, env, len(body_v) // 6, body_v, body_c)
        # 손끝 좌표축: 빨강=+x(현재 접촉방향 PALM), 초록=+y, 파랑=+z(팁 축).
        #   + 노랑점 = 현재 모델 접촉점(팁원점+x·FR).  SHOW_PAD_AXES=0 이면 끔.
        mv, mc = [], []
        AX = 0.03
        for ti in tip_idx:
            p = rb[ti, :3].cpu().numpy(); q = rb[ti, 3:7].cpu().numpy()
            xw = _qrot(q, [1, 0, 0]); yw = _qrot(q, [0, 1, 0]); zw = _qrot(q, [0, 0, 1])
            mv += list(p) + list(p + xw * AX); mc += [1.0, 0.1, 0.1]   # +x 빨강
            mv += list(p) + list(p + yw * AX); mc += [0.1, 1.0, 0.1]   # +y 초록
            mv += list(p) + list(p + zw * AX); mc += [0.2, 0.4, 1.0]   # +z 파랑
            now = p + _qrot(q, PALM * FR)
            mv += _cross(now, 0.005); mc += [1.0, 0.9, 0.1] * 3        # 모델 접촉점 노랑
        gym.add_lines(viewer, env, len(mv) // 6, mv, mc)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
