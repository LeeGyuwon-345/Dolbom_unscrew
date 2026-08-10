"""Drive the dg5fs joints by hand over the tumbler, and read out what touches.

Built to settle one question by inspection: can this hand reach the cap's side
wall at all? The automated sweep in scratchpad/feas.py says no -- closing every
finger from a good top-down wrist pose leaves the tips at radius 74-118 mm
against a 47-50 mm wall, with the only contacts landing on the body below the
cap. This lets you check that by hand before redesigning the task around it.

Keys (Isaac Gym viewer must have focus):

  finger select   1 2 3 4 5      thumb, index, middle, ring, pinky
  joint select    Q W E R        abduction, MCP, PIP, DIP
  joint move      ] / [          flex / extend the selected joint by 0.05 rad
  all fingers     O / P          open / close every flexion joint together
  wrist move      D/A  F/S  X/Z  +-x, +-y, +-z   (10 mm steps)
  wrist rotate    L/J  I/K  U/M  roll, pitch, yaw (5 deg steps)
  reset           BACKSPACE
  print state     SPACE          tip positions in cap frame + contact forces
  print joints    G              every joint angle, target vs actual

Bracket keys rather than the arrows: the viewer takes the arrows for its own
camera control, so they never reach us.

] and [ mean flex and extend, not plus and minus. The thumb MCP runs -2.670..0
where every other joint runs positive, so a raw "+0.05" on it just clamps at
its upper stop and nothing moves -- which reads as the joint being broken.

The readout is what matters: radius and height are given in the cap's frame, so
"radius 48, z 20" is on the wall inside the scored band, and the band itself is
printed for reference.
"""

import math
import sys

sys.path.insert(0, "/home/leegyuwon/Documents/task1/tools")

from isaacgym import gymapi, gymtorch  # noqa: E402  (must precede torch)
import torch  # noqa: E402

CAP_Z = 0.225          # cap bottom above the tumbler base
CAP_H = 0.030
BAND = (0.0143, 0.030)  # grippable cap-frame band, see cap_unscrew_*_terms
WALL_R = (0.0469, 0.0500)
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JOINTS = ["abduction", "MCP", "PIP", "DIP"]


def main():
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = 1 / 60.0
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.contact_offset = 0.005
    sp.physx.rest_offset = 0.0
    sp.physx.max_depenetration_velocity = 1.0
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane)

    ho = gymapi.AssetOptions()
    ho.fix_base_link = True
    ho.disable_gravity = True
    ho.default_dof_drive_mode = gymapi.DOF_MODE_POS
    hand = gym.load_asset(
        sim, "/home/leegyuwon/Documents/task1/assets/dg5fs_hand", "dg5fs_right.urdf", ho
    )
    to = gymapi.AssetOptions()
    to.fix_base_link = True
    to.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    to.vhacd_enabled = True
    to.vhacd_params = gymapi.VhacdParams()
    to.vhacd_params.resolution = 200000
    tum = gym.load_asset(
        sim, "/home/leegyuwon/Documents/task1/assets/tumbler", "tumbler.urdf", to
    )

    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    # filter 1: the screw joint is the thread, the meshes must not fight it
    gym.create_actor(env, tum, gymapi.Transform(), "tumbler", 0, 1)

    start = gymapi.Transform()
    start.p = gymapi.Vec3(0.0, 0.0, CAP_Z + CAP_H + 0.05)
    start.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi / 2)  # palm down
    h = gym.create_actor(env, hand, start, "hand", 0, 0)
    props = gym.get_actor_dof_properties(env, h)
    for i in range(len(props)):
        props["driveMode"][i] = gymapi.DOF_MODE_POS
        props["stiffness"][i] = 500.0
        # 100, not the 30 training uses: a hand parked against contact buzzes at
        # the lower value, and this tool is for reading steady positions.
        props["damping"][i] = 100.0
    gym.set_actor_dof_properties(env, h, props)

    # Same palm<->thumb shape filter the training env applies. Without it the
    # palm's convex hull swallows the thumb's proximal link and PhysX pushes
    # back with ~16 kN forever, pinning joint_1_1 and joint_1_2 -- the thumb
    # simply will not follow the keys.
    _names = gym.get_actor_rigid_body_names(env, h)
    _shapes = gym.get_actor_rigid_shape_properties(env, h)
    _spans = gym.get_actor_rigid_body_shape_indices(env, h)
    for _b, _n in enumerate(_names):
        if _n in ("link_base", "link_1_1", "link_1_2"):
            for _s in range(_spans[_b].start, _spans[_b].start + _spans[_b].count):
                _shapes[_s].filter = 1
    gym.set_actor_rigid_shape_properties(env, h, _shapes)

    lo_lim = [float(props["lower"][i]) for i in range(len(props))]
    up_lim = [float(props["upper"][i]) for i in range(len(props))]

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    gym.viewer_camera_look_at(
        viewer, None, gymapi.Vec3(0.35, 0.35, 0.40), gymapi.Vec3(0, 0, CAP_Z + 0.015)
    )
    binds = [
        ("1", "f0"), ("2", "f1"), ("3", "f2"), ("4", "f3"), ("5", "f4"),
        ("Q", "j0"), ("W", "j1"), ("E", "j2"), ("R", "j3"),
        ("RIGHT_BRACKET", "jp"), ("LEFT_BRACKET", "jm"),
        ("O", "open"), ("P", "close"),
        ("A", "wx-"), ("D", "wx+"), ("S", "wy-"), ("F", "wy+"),
        ("Z", "wz-"), ("X", "wz+"),
        ("J", "rr-"), ("L", "rr+"), ("I", "rp+"), ("K", "rp-"),
        ("U", "ry+"), ("M", "ry-"),
        ("BACKSPACE", "reset"), ("SPACE", "print"), ("G", "joints"),
    ]
    for key, act in binds:
        gym.subscribe_viewer_keyboard_event(viewer, getattr(gymapi, "KEY_" + key), act)

    gym.prepare_sim(sim)
    dofs = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(1, -1, 2)
    rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(1, -1, 13)
    cf = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(1, -1, 3)
    root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)

    dn = gym.get_actor_dof_names(env, h)
    bn = gym.get_actor_rigid_body_names(env, h)
    n_dof = gym.get_sim_dof_count(sim)
    # env-domain indices: the tumbler's DOFs come first in the sim tensor
    dof_env = [gym.get_actor_dof_index(env, h, i, gymapi.DOMAIN_ENV) for i in range(len(dn))]
    body_off = gym.get_actor_rigid_body_index(env, h, 0, gymapi.DOMAIN_ENV)
    tips = [body_off + bn.index(f"link_{f}_tip") for f in range(1, 6)]
    distal = [body_off + bn.index(f"link_{f}_4") for f in range(1, 6)]
    cap_i = gym.get_actor_rigid_body_index(
        env, 0, gym.get_actor_rigid_body_dict(env, 0)["cap"], gymapi.DOMAIN_ENV
    )
    hand_root = gym.find_actor_index(env, "hand", gymapi.DOMAIN_SIM)

    # joint_<finger>_<1..4>; index 0 is abduction, 3 is DIP
    jloc = [[dn.index(f"joint_{f}_{j}") for j in range(1, 5)] for f in range(1, 6)]
    jidx = [[dof_env[i] for i in row] for row in jloc]
    # Flexion is not always the positive direction: the thumb MCP runs
    # -2.670..0 while every other joint runs positive. Driving them all one way
    # pushed the thumb past its upper stop, where the position drive buzzes
    # against the limit. Direction is taken from whichever limit is further
    # from zero, and every target is clamped to the joint's own range.
    lim = {dof_env[i]: (lo_lim[i], up_lim[i]) for i in range(len(dn))}
    sign = {dof_env[i]: (1.0 if abs(up_lim[i]) >= abs(lo_lim[i]) else -1.0)
            for i in range(len(dn))}
    flex = [jidx[f][j] for f in range(5) for j in (1, 2, 3)]

    def set_dof(idx, val):
        a, b = lim[idx]
        tgt[0, idx] = max(a, min(b, val))

    print("sim DOF/env=%d  hand DOF=%d" % (n_dof, len(dn)), flush=True)
    for f in range(5):
        print("  %-7s" % FINGERS[f] + "  ".join(
            "%s: local %d -> env %d" % (JOINTS[j], jloc[f][j], jidx[f][j]) for j in range(4)), flush=True)
    tgt = torch.zeros(1, n_dof)
    w0 = torch.tensor([start.p.x, start.p.y, start.p.z], dtype=torch.float32)
    q0 = torch.tensor([start.r.x, start.r.y, start.r.z, start.r.w], dtype=torch.float32)
    sel_f, sel_j = 0, 1

    def qmul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return torch.tensor([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])

    def axis_q(ax, ang):
        s = math.sin(ang / 2)
        v = [0.0, 0.0, 0.0]
        v[ax] = s
        return torch.tensor(v + [math.cos(ang / 2)])

    def report():
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        # On SPACE rather than its own key: the viewer swallows some letters for
        # its own controls, and one key that always works beats two that might.
        print("\n  손가락  " + "".join(f"{j:>16}" for j in JOINTS), flush=True)
        for fi in range(5):
            row = "".join(
                f"{float(tgt[0, jidx[fi][ji]]):+.2f}/{float(dofs[0, jidx[fi][ji], 0]):+.2f}".rjust(16)
                for ji in range(4)
            )
            print(f"  {FINGERS[fi]:<8}{row}", flush=True)
        print("  (목표/실제, rad)", flush=True)
        cp = rb[0, cap_i, :3]
        print(f"\n밴드 z {BAND[0]*1e3:.1f}~{BAND[1]*1e3:.1f}mm, 벽 반경 {WALL_R[0]*1e3:.1f}~{WALL_R[1]*1e3:.1f}mm", flush=True)
        print("  손가락   반경(mm)  높이z(mm)  방위각(deg)  접촉력(N)  밴드안", flush=True)
        az, hit = [], 0
        for k in range(5):
            p = rb[0, tips[k], :3] - cp
            r = float(torch.linalg.norm(p[:2])) * 1e3
            z = float(p[2]) * 1e3
            a = math.degrees(math.atan2(float(p[1]), float(p[0])))
            f = float(torch.linalg.norm(cf[0, tips[k]]) + torch.linalg.norm(cf[0, distal[k]]))
            ok = (BAND[0] * 1e3 - 9 <= z <= BAND[1] * 1e3 + 9) and (33 <= r <= 71)
            hit += ok
            az.append(a)
            print(f"  {FINGERS[k]:<8} {r:8.1f}  {z:9.1f}  {a:+10.1f}  {f:8.2f}    {'O' if ok else 'X'}", flush=True)
        sep = [abs((az[0] - az[k] + 180) % 360 - 180) for k in range(1, 5)]
        print("  엄지 대비 방위각: " + "  ".join(f"{FINGERS[k+1]} {sep[k]:.0f}deg" for k in range(4)), flush=True)
        print(f"  밴드 안 손가락 {hit}/5   손목 {[round(float(v)*1e3) for v in root[hand_root, :3]]}mm", flush=True)

    print(__doc__)
    while not gym.query_viewer_has_closed(viewer):
      try:
        for ev in gym.query_viewer_action_events(viewer):
            if ev.value == 0:
                continue
            a = ev.action
            print(f"[key] {a}", flush=True)
            if a.startswith("f"):
                sel_f = int(a[1]); print(f"손가락 -> {FINGERS[sel_f]}", flush=True)
            elif a.startswith("j") and a[1].isdigit():
                sel_j = int(a[1])
                k = jidx[sel_f][sel_j]
                gym.refresh_dof_state_tensor(sim)
                print(f"관절 -> {FINGERS[sel_f]} {JOINTS[sel_j]}  "
                      f"목표 {float(tgt[0,k]):+.2f}  실제 {float(dofs[0,k,0]):+.2f}  "
                      f"(범위 {lim[k][0]:+.2f}~{lim[k][1]:+.2f})")
            elif a in ("jp", "jm"):
                k = jidx[sel_f][sel_j]
                d = 0.05 * sign[k] * (1.0 if a == "jp" else -1.0)
                set_dof(k, float(tgt[0, k]) + d)
                gym.refresh_dof_state_tensor(sim)
                # Target vs actual: if the target moves and the joint does not,
                # something is blocking it -- contact, or a drive too weak.
                print(f"  {FINGERS[sel_f]} {JOINTS[sel_j]}  목표 {float(tgt[0,k]):+.2f}  "
                      f"실제 {float(dofs[0,k,0]):+.2f}  "
                      f"(범위 {lim[k][0]:+.2f}~{lim[k][1]:+.2f}, 굽힘 {'+' if sign[k] > 0 else '-'})")
            elif a == "open":
                for j in flex: set_dof(j, float(tgt[0, j]) - 0.10 * sign[j])
            elif a == "close":
                for j in flex: set_dof(j, float(tgt[0, j]) + 0.10 * sign[j])
            elif a.startswith("w"):
                d = {"x": 0, "y": 1, "z": 2}[a[1]]
                w0[d] += 0.010 * (1 if a[2] == "+" else -1)
            elif a.startswith("r"):
                ax = {"r": 0, "p": 1, "y": 2}[a[1]]
                q0[:] = qmul(axis_q(ax, math.radians(5) * (1 if a[2] == "+" else -1)), q0)
            elif a == "reset":
                tgt[:] = 0
                w0[:] = torch.tensor([start.p.x, start.p.y, start.p.z])
                q0[:] = torch.tensor([start.r.x, start.r.y, start.r.z, start.r.w])
            elif a == "print":
                report()
            elif a == "joints":
                gym.refresh_dof_state_tensor(sim)
                print("\n  손가락    " + "".join(f"{j:>18}" for j in JOINTS), flush=True)
                for f in range(5):
                    row = ""
                    for j in range(4):
                        k = jidx[f][j]
                        # both, because a target the joint is not tracking is
                        # the case worth spotting
                        row += f"{float(tgt[0,k]):+.2f}/{float(dofs[0,k,0]):+.2f}".rjust(18)
                    print(f"  {FINGERS[f]:<9}{row}", flush=True)
                print("  (목표/실제, rad)", flush=True)
            if a.startswith(("w", "r")) or a == "reset":
                root[hand_root, :3] = w0
                root[hand_root, 3:7] = q0
                root[hand_root, 7:13] = 0
                idx = torch.tensor([hand_root], dtype=torch.int32)
                gym.set_actor_root_state_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(root), gymtorch.unwrap_tensor(idx), 1
                )
      except Exception as exc:      # keep the viewer alive so the cause is visible
        import traceback; traceback.print_exc()
      gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(tgt))
      gym.simulate(sim)
      gym.fetch_results(sim, True)
      gym.step_graphics(sim)
      gym.draw_viewer(viewer, sim, True)
      gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
