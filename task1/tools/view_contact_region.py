"""Static viewer for what counts as fingertip contact. No policy, no training.

Loads the hand and the tumbler, holds a pose you drive from the keyboard, and
draws the three tests the reward applies:

  blue rings   the scored band on the cap wall, at the collision mesh's radius
               (the visual mesh is up to 5.6mm narrower -- what you see is not
               what the physics uses)
  probe dots   the points the geometry tests are evaluated at, spaced along the
               tip link; the contact radius around each is finger_radius
  axis line    each fingertip's palmar direction, coloured by verdict:
               green accepted, yellow in band but turned too far, red out of band

Keys: A/D S/W Z/X move the wrist, Q/E yaw, R/F flex all fingers, SPACE prints
the numbers, ESC quits.
"""
import math, os, sys
from isaacgym import gymapi, gymtorch
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_unscrew_rl_terms import cap_wall_radius
import cap_unscrew_curriculum_terms as M

CFG = M.DEFAULT_CFG
MIN_COS = float(os.environ.get("PALM_FACING_COS", CFG.palm_facing_min_cos))
FR = float(os.environ.get("FINGER_RADIUS", CFG.finger_radius))
CAPZ = 0.225
FING = ["thumb", "index", "middle", "ring", "pinky"]

gym = gymapi.acquire_gym()
sp = gymapi.SimParams(); sp.dt = 1/60.; sp.substeps = 2; sp.up_axis = gymapi.UP_AXIS_Z
sp.gravity = gymapi.Vec3(0, 0, 0); sp.use_gpu_pipeline = False
sp.physx.solver_type = 1; sp.physx.num_position_iterations = 8
sp.physx.contact_offset = 0.005
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sp)
gym.add_ground(sim, gymapi.PlaneParams())

ho = gymapi.AssetOptions(); ho.fix_base_link = True; ho.disable_gravity = True
ho.default_dof_drive_mode = gymapi.DOF_MODE_POS
hand = gym.load_asset(sim, "/home/leegyuwon/Documents/task1/assets/dg5fs_hand", "dg5fs_right.urdf", ho)
to = gymapi.AssetOptions(); to.fix_base_link = True
to.vhacd_enabled = True; to.vhacd_params = gymapi.VhacdParams(); to.vhacd_params.resolution = 200000
tum = gym.load_asset(sim, "/home/leegyuwon/Documents/task1/assets/tumbler", "tumbler.urdf", to)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
gym.create_actor(env, tum, gymapi.Transform(), "tumbler", 0, 1)
w0 = np.array([0.0, 0.0, CAPZ + CFG.cap_height + 0.05]); yaw0 = 0.0
t = gymapi.Transform(); t.p = gymapi.Vec3(*w0)
t.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), math.pi / 2)
act = gym.create_actor(env, hand, t, "hand", 0, 0)
pr = gym.get_actor_dof_properties(env, act)
for k in range(len(pr)):
    pr["driveMode"][k] = gymapi.DOF_MODE_POS; pr["stiffness"][k] = 500.; pr["damping"][k] = 50.
gym.set_actor_dof_properties(env, act, pr)
bn0 = gym.get_actor_rigid_body_names(env, act); sh = gym.get_actor_rigid_shape_properties(env, act)
spn = gym.get_actor_rigid_body_shape_indices(env, act)
for b, n in enumerate(bn0):
    if n in ("link_base", "link_1_1", "link_1_2"):
        for k in range(spn[b].start, spn[b].start + spn[b].count): sh[k].filter = 1
gym.set_actor_rigid_shape_properties(env, act, sh)
gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.35, -0.30, 0.42), gymapi.Vec3(0, 0, CAPZ + 0.02))
for key, ev in ((gymapi.KEY_A,"wx-"),(gymapi.KEY_D,"wx+"),(gymapi.KEY_S,"wy-"),(gymapi.KEY_W,"wy+"),
                (gymapi.KEY_Z,"wz-"),(gymapi.KEY_X,"wz+"),(gymapi.KEY_Q,"yaw-"),(gymapi.KEY_E,"yaw+"),
                (gymapi.KEY_R,"flex+"),(gymapi.KEY_F,"flex-"),(gymapi.KEY_SPACE,"info")):
    gym.subscribe_viewer_keyboard_event(viewer, key, ev)

dn = gym.get_actor_dof_names(env, act); nd = len(dn)
lim = {n: (float(pr["lower"][i]), float(pr["upper"][i])) for i, n in enumerate(dn)}
rb = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(-1, 13)
cfx = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(-1, 3)
root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).view(-1, 13)
bn = gym.get_actor_rigid_body_names(env, act)
hand_root = gym.get_actor_index(env, act, gymapi.DOMAIN_SIM)
tip_i = [bn.index("link_%d_tip" % (k+1)) for k in range(5)]
dis_i = [bn.index("link_%d_4" % (k+1)) for k in range(5)]
off = gym.get_actor_rigid_body_index(env, act, 0, gymapi.DOMAIN_ENV)
cap_i = gym.get_actor_rigid_body_index(env, 0, gym.get_actor_rigid_body_dict(env, 0)["cap"], gymapi.DOMAIN_ENV)

def qrot(q, v):
    xyz = q[:3]
    tt = 2.0 * torch.cross(xyz.expand_as(v), v, dim=-1)
    return v + q[3] * tt + torch.cross(xyz.expand_as(v), tt, dim=-1)

flex = 0.0; tgt = torch.zeros(1, nd)
print("A/D S/W Z/X 손목 이동, Q/E yaw, R/F 손가락, SPACE 수치, ESC 종료", flush=True)
print("PALM_FACING_COS=%.2f  FINGER_RADIUS=%.1fmm\n" % (MIN_COS, FR*1e3), flush=True)

while not gym.query_viewer_has_closed(viewer):
    for e in gym.query_viewer_action_events(viewer):
        if e.value == 0: continue
        a = e.action
        if a == "wx-": w0[0] -= 0.005
        elif a == "wx+": w0[0] += 0.005
        elif a == "wy-": w0[1] -= 0.005
        elif a == "wy+": w0[1] += 0.005
        elif a == "wz-": w0[2] -= 0.005
        elif a == "wz+": w0[2] += 0.005
        elif a == "yaw-": yaw0 -= math.radians(10)
        elif a == "yaw+": yaw0 += math.radians(10)
        elif a in ("flex+", "flex-"):
            flex = max(0.0, min(1.3, flex + (0.1 if a == "flex+" else -0.1)))
            for i, n in enumerate(dn):
                if n[-1] in "234":
                    lo, up = lim[n]
                    tgt[0, i] = flex * (up if abs(up) >= abs(lo) else lo) / max(abs(up), abs(lo), 1e-6)
        elif a == "info":
            gym.refresh_rigid_body_state_tensor(sim)
            cp = rb[cap_i, :3]
            print("\n  손목 (%.0f, %.0f, %.0f)mm  yaw %.0f도  오므림 %.1f"
                  % (w0[0]*1e3, w0[1]*1e3, w0[2]*1e3, math.degrees(yaw0), flex), flush=True)
            print("  %-8s %8s %9s %8s %8s %8s" % ("손가락","높이mm","반경mm","cos","접촉력","판정"), flush=True)
            for k in range(5):
                p = rb[off+tip_i[k], :3] - cp
                z = float(p[2]); r = float(torch.linalg.norm(p[:2]))
                u = p[:2] / max(r, 1e-6)
                palm = qrot(rb[off+dis_i[k], 3:7], torch.tensor([[0., 1., 0.]]))[0]
                cosv = float((palm[:2] * -u).sum())
                inb = (CFG.band_lo - FR) <= z <= (CFG.cap_height + FR)
                fm = float(torch.linalg.norm(cfx[off+tip_i[k]]) + torch.linalg.norm(cfx[off+dis_i[k]]))
                okf = fm > CFG.contact_force_threshold
                v = ("인정" if (okf and inb and cosv >= MIN_COS)
                     else ("안닿음" if (inb and cosv >= MIN_COS)
                           else ("각도부족" if inb else "밴드밖")))
                print("  %-8s %8.1f %9.1f %8.2f %7.2fN %8s"
                      % (FING[k], z*1e3, r*1e3, cosv, fm, v), flush=True)
    st = root[hand_root].clone()
    st[:3] = torch.tensor(w0, dtype=torch.float32)
    qb = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), math.pi/2)
    qy = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), yaw0)
    q = qy * qb
    st[3:7] = torch.tensor([q.x, q.y, q.z, q.w]); st[7:] = 0
    root[hand_root] = st
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(tgt))
    gym.simulate(sim); gym.fetch_results(sim, True)
    gym.refresh_rigid_body_state_tensor(sim); gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    gym.clear_lines(viewer)
    cp = rb[cap_i, :3]
    rings = []
    for zz in (CFG.band_lo, CFG.cap_height):
        rw = float(cap_wall_radius(torch.tensor([zz]), CFG))
        rings.append((zz, rw, [0.2, 0.6, 1.0]))            # 벽 (collision)
        rings.append((zz, rw + FR, [0.55, 0.35, 0.9]))     # 손끝 중심 목표
    for zz, rr, ringcol in rings:
        seg, col = [], []
        for i in range(48):
            a0, a1 = 2*math.pi*i/48, 2*math.pi*(i+1)/48
            seg.append([float(cp[0])+rr*math.cos(a0), float(cp[1])+rr*math.sin(a0), float(cp[2])+zz,
                        float(cp[0])+rr*math.cos(a1), float(cp[1])+rr*math.sin(a1), float(cp[2])+zz])
            col.append(ringcol)
        gym.add_lines(viewer, env, len(seg), np.array(seg, np.float32), np.array(col, np.float32))
    seg, col = [], []
    for k in range(5):
        tp = rb[off+tip_i[k], :3]; dp = rb[off+dis_i[k], :3]
        p = tp - cp; z = float(p[2]); r = float(torch.linalg.norm(p[:2]))
        u = p[:2] / max(r, 1e-6)
        palm = qrot(rb[off+dis_i[k], 3:7], torch.tensor([[0., 1., 0.]]))[0]
        cosv = float((palm[:2] * -u).sum())
        inb = (CFG.band_lo - FR) <= z <= (CFG.cap_height + FR)
        # Contact force decides green. Position and angle alone only mean the
        # finger is in a place that *would* count.
        fm = float(torch.linalg.norm(cfx[off+tip_i[k]]) + torch.linalg.norm(cfx[off+dis_i[k]]))
        okf = fm > CFG.contact_force_threshold
        if okf and inb and cosv >= MIN_COS:   c = [0.1,1.0,0.1]
        elif inb and cosv >= MIN_COS:         c = [0.1,0.7,1.0]
        elif inb:                             c = [1.0,0.9,0.1]
        else:                                 c = [1.0,0.15,0.15]
        # probe points along the tip segment, the geometry the reward tests
        axis = tp - dp; axis = axis / torch.linalg.norm(axis).clamp_min(1e-6)
        for j in range(4):
            q0 = tp + axis * (0.018 * j / 3.0)
            for d in ((FR,0,0),(0,FR,0),(0,0,FR)):
                seg.append([float(q0[0])-d[0], float(q0[1])-d[1], float(q0[2])-d[2],
                            float(q0[0])+d[0], float(q0[1])+d[1], float(q0[2])+d[2]])
                col.append(c)
        seg.append([float(tp[0]), float(tp[1]), float(tp[2]),
                    float(tp[0]+palm[0]*0.03), float(tp[1]+palm[1]*0.03), float(tp[2]+palm[2]*0.03)])
        col.append(c)
    gym.add_lines(viewer, env, len(seg), np.array(seg, np.float32), np.array(col, np.float32))
    gym.step_graphics(sim); gym.draw_viewer(viewer, sim, True); gym.sync_frame_time(sim)
gym.destroy_viewer(viewer); gym.destroy_sim(sim)
