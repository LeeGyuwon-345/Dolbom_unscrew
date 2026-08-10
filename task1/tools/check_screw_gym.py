"""Isaac Gym counterpart of check_screw.py: does the DOF-drive coupling hold?

check_screw.py verifies the URDF <mimic> tag under IsaacLab. Isaac Gym ignores
that tag, so this verifies the replacement in screw_coupling.py instead, and
checks the thing teleporting the cap's root state cannot give us: that turning
the cap produces a real axial reaction force on whatever holds it.

    python task1/tools/check_screw_gym.py
    python task1/tools/check_screw_gym.py --stiffness 20000 --spin-vel 4.0
"""

import argparse
import math
import sys
import time

sys.path.insert(0, "/home/leegyuwon/Documents/task1/tools")

from isaacgym import gymapi, gymtorch  # noqa: E402  (must precede torch)
import torch  # noqa: E402

from screw_coupling import ScrewCoupling  # noqa: E402

URDF_DIR = "/home/leegyuwon/Documents/task1/assets/tumbler"
URDF = "tumbler.urdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--spin-vel", type=float, default=2.0, help="cap_spin 목표 각속도 (rad/s)")
    ap.add_argument("--stiffness", type=float, default=200_000.0)
    ap.add_argument("--damping", type=float, default=1_000.0)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--load", type=float, default=20.0, help="해제 후 캡을 눌러보는 하중 (N)")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    gym = gymapi.acquire_gym()
    sp = gymapi.SimParams()
    sp.dt = args.dt
    sp.substeps = 2
    sp.up_axis = gymapi.UP_AXIS_Z
    sp.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sp.use_gpu_pipeline = False
    sp.physx.solver_type = 1
    sp.physx.num_position_iterations = 8
    sp.physx.num_velocity_iterations = 1
    sp.physx.contact_offset = 0.005
    sp.physx.rest_offset = 0.0
    # The whole reason for this file: depenetration at the PhysX default of
    # 1000 m/s ejects light objects. Match what training now uses.
    sp.physx.max_depenetration_velocity = 1.0
    # graphics_device -1 means 'no graphics'; a viewer then fails with
    # "Sim->Graphics is nullptr". Only ask for a graphics device when drawing.
    sim = gym.create_sim(0, 0 if args.gui else -1, gymapi.SIM_PHYSX, sp)

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane)

    ao = gymapi.AssetOptions()
    ao.fix_base_link = True
    ao.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    ao.collapse_fixed_joints = False
    asset = gym.load_asset(sim, URDF_DIR, URDF, ao)

    coupling = ScrewCoupling.from_urdf(
        f"{URDF_DIR}/{URDF}", stiffness=args.stiffness, damping=args.damping
    )
    print(f"[mimic] URDF 에서 읽은 구속: {coupling.describe()}")

    envs, actors = [], []
    for i in range(args.envs):
        env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), int(math.sqrt(args.envs)) or 1)
        # collision_filter=1 puts cap and body in the same filter group so they
        # do NOT collide with each other. They are modelled as overlapping by
        # 4.34 mm (the thread engagement), so leaving self-collision on makes
        # the contact solver fight the prismatic joint: measured cap_lift
        # pushed to 14.2 mm against an 8000 N/m drive, and cap_spin driven
        # backwards. The joint *is* the thread; the meshes must not also be.
        actor = gym.create_actor(env, asset, gymapi.Transform(), f"tumbler{i}", i, 1)
        envs.append(env)
        actors.append(actor)

    # Bind on every env; source DOF gets a velocity drive here purely so the
    # test can turn the cap without a hand in the scene.
    names = gym.get_actor_dof_names(envs[0], actors[0])
    for env, actor in zip(envs, actors):
        coupling.bind(gym, env, actor, num_envs=args.envs)
        props = gym.get_actor_dof_properties(env, actor)
        si = names.index(coupling.specs["cap_lift"].source)
        props["driveMode"][si] = gymapi.DOF_MODE_VEL
        props["stiffness"][si] = 0.0
        props["damping"][si] = 50.0
        props["effort"][si] = 50.0
        gym.set_actor_dof_properties(env, actor, props)
        # Must precede prepare_sim, or acquire_dof_force_tensor reports zeros
        # and the reaction force -- the whole point of the joint drive -- is
        # invisible.
        gym.enable_actor_dof_force_sensors(env, actor)

    viewer = None
    if args.gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0.5, 0.5, 0.45), gymapi.Vec3(0, 0, 0.24))

    gym.prepare_sim(sim)
    n_dof = gym.get_asset_dof_count(asset)
    dof = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(args.envs, n_dof, 2)
    forces = gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim)).view(args.envs, n_dof)

    spec = coupling.specs["cap_lift"]
    i_spin = names.index(spec.source)
    i_lift = names.index(spec.follower)
    expected = spec.multiplier
    print(f"[setup] DOF {names}  이론 배율 dz/dtheta = {expected:.6e} m/rad "
          f"(pitch {spec.pitch * 1000:.1f}mm)")
    print(f"[setup] cap_lift 위치구동 stiffness={args.stiffness} damping={args.damping}, "
          f"cap_spin 속도구동 {args.spin_vel} rad/s\n")

    pos_target = torch.zeros(args.envs, n_dof)
    effort = torch.zeros(args.envs, n_dof)
    vel_target = torch.zeros(args.envs, n_dof)
    vel_target[:, i_spin] = args.spin_vel

    # Half the run turns the cap freely; then a downward load is hung on the cap
    # to show the drive actually carries it. A root-state teleport would hold
    # position too, but it would do so without any force appearing anywhere --
    # here the load has to show up in the joint.
    load_from = args.steps // 2
    load_n = -abs(args.load)  # N pressing the cap back down its screw axis
    body_force = torch.zeros(args.envs, gym.get_asset_rigid_body_count(asset), 3)
    i_cap_body = gym.get_asset_rigid_body_dict(asset)["cap"]

    print("  step  누적각(rad)   cap_lift      목표      오차   lift 구동력   외부하중")
    worst_free = 0.0
    worst_load = 0.0
    samples = []
    for step in range(args.steps):
        gym.refresh_dof_state_tensor(sim)
        coupling.apply(dof[:, :, 0], pos_target)
        coupling.release_bound(gym, envs, actors, dof[:, :, 0])
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_target))
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(vel_target))
        loaded = step >= load_from
        if loaded:
            body_force[:, i_cap_body, 2] = load_n
            gym.apply_rigid_body_force_tensors(
                sim, gymtorch.unwrap_tensor(body_force), None, gymapi.ENV_SPACE
            )
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)   # else 900 steps flash past in half a second

        th = float(coupling._unwrapped[0, 0])
        z = float(dof[0, i_lift, 0])
        want = float(pos_target[0, i_lift])
        err = abs(z - want)
        # Only meaningful while the thread is still engaged and off its stops.
        engaged = not bool(coupling.released[0])
        if engaged and coupling.lower[0] + 1e-6 < want < float(coupling.thread_upper[0]) - 1e-6:
            samples.append((th, z))
            if loaded:
                worst_load = max(worst_load, err)
            else:
                worst_free = max(worst_free, err)
        if (step + 1) % 100 == 0:
            print(f"  {step + 1:4d}  {th:+8.3f}  {z * 1000:8.3f}mm  {want * 1000:8.3f}mm  "
                  f"{err * 1000:6.3f}mm  {float(forces[0, i_lift]):+8.2f} N   "
                  f"{'-20 N' if loaded else '  없음'}")

    # least-squares slope over the unsaturated samples
    n = len(samples)
    sx = sum(a for a, _ in samples); sy = sum(b for _, b in samples)
    sxx = sum(a * a for a, _ in samples); sxy = sum(a * b for a, b in samples)
    ratio = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    dev = abs(ratio - expected) / expected * 100

    print()
    print(f"[결과] dz/dtheta = {ratio:.6e}  (이론 {expected:.6e}, 편차 {dev:.2f}%)  "
          f"n={n} 샘플, 포화 구간 제외")
    print(f"[결과] 추종 오차  무부하 {worst_free * 1000:.3f} mm / 20N 하중 {worst_load * 1000:.3f} mm")
    rel = bool(coupling.released[0])
    print(f"[결과] 분리       {'예' if rel else '아니오'}  "
          f"(누적 {float(coupling._unwrapped[0, 0]) * 57.2958:.1f} deg / "
          f"기준 {float(coupling.engage_angle[0]) * 57.2958:.1f} deg)")
    if rel:
        print(f"       해제 후 cap_lift = {float(dof[0, i_lift, 0]) * 1000:.1f} mm "
              f"-- 이 테스트엔 손이 없으므로 떨어지는 게 정상")
    ok = dev <= 5.0 and worst_load < 0.002 and rel
    print(f"[결과] {'OK - 나사 구속 성립, 하중도 지지' if ok else 'FAIL'}")
    print(f"[timing] {time.time() - t0:.1f}s")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
