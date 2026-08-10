"""factory M16 볼트-너트를 GUI 에 띄우고 키보드로 직접 돌려본다.

나사산은 조인트가 아니라 SDF 메시 접촉으로만 구현돼 있다 (볼트/너트 USD 어디에도
둘을 잇는 조인트가 없다). 그래서 "정말 나사처럼 도는가"는 눈으로 확인하는 수밖에 없다.

조작:
    Q / E    나사축(Z) 토크   Q=푸는 방향(CCW)  E=조이는 방향(CW)
    W / S    Z축 힘           W=위로  S=아래로
    A / D    X축 힘           옆으로 밀어 걸림 확인
    G        중력 on/off
    R        너트 위치 리셋
    ESC      종료

마우스로 직접 잡아 끌려면: 재생 중 Ctrl + 좌클릭 드래그.

사용법:
    conda activate task1
    python task1/tools/play_nutbolt.py            # loose (기본)
    python task1/tools/play_nutbolt.py --tight    # 공차 작은 버전
"""

import argparse
import time

_T0 = time.time()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--tight", action="store_true", help="loose 대신 tight 나사산 사용")
# 너트 관성이 I_zz ~ 2e-6 kg·m^2 로 매우 작다. 0.02 N·m 를 걸면 각가속도가 1e4 rad/s^2
# 수준이라 나사산을 타지 못하고 튕겨 나간다. 기본값을 두 자릿수 낮춰 잡는다.
parser.add_argument("--torque", type=float, default=3e-4, help="Q/E 토크 크기 (N·m)")
parser.add_argument("--force", type=float, default=0.05, help="W/S/A/D 힘 크기 (N)")
parser.add_argument("--steps", type=int, default=600, help="headless 스모크 테스트 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
sim_app = app_launcher.app

import carb  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

PREVIEW = "/home/leegyuwon/Documents/task1/assets/preview"

# 볼트는 z=0.000~0.035 (머리 0.010 + 나사부 0.025), 너트는 z=0.010~0.023 을 차지하도록
# authoring 돼 있다. 즉 둘 다 원점에 놓으면 너트가 나사부 밑동에 완전히 물린 상태가 된다.
# 위쪽에서 시작하면 조금만 헛돌아도 그대로 이탈하므로 밑동에서 시작한다.
# 오른나사이므로 +Z 토크(CCW, Q키)를 걸면 풀리면서 위로 올라와야 한다.
NUT_START_Z = 0.0

# 이론적인 나사 커플링 비: 1 rad 돌 때 축방향으로 pitch/2pi 만큼 이동해야 한다.
THREAD_PITCH = 0.002
DZ_PER_RAD = THREAD_PITCH / (2 * 3.14159265)

# factory 태스크가 쓰는 값 그대로. 나사산 접촉은 solver 반복수에 매우 민감해서
# 192 라는 큰 값을 쓴다 (기본값은 4~16 수준).
RIGID = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=5.0,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=3666.0,
    enable_gyroscopic_forces=True,
    solver_position_iteration_count=192,
    solver_velocity_iteration_count=1,
    max_contact_impulse=1e32,
)
COLLISION = sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)

if args.tight:
    # tight 는 Props/Factory 하위에 폴더째로 들어 있다 (instanceable_meshes.usd 를 참조).
    bolt_usd = f"{PREVIEW}/factory_bolt_m16_tight/factory_bolt_m16_tight.usd"
    nut_usd = f"{PREVIEW}/factory_nut_m16_tight/factory_nut_m16_tight.usd"
else:
    bolt_usd = f"{PREVIEW}/factory_bolt_m16.usd"
    nut_usd = f"{PREVIEW}/factory_nut_m16.usd"
print(f"\n[play_nutbolt] 나사산 변종: {'tight' if args.tight else 'loose'}")

sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
sim.set_camera_view(eye=(0.12, 0.12, 0.10), target=(0.0, 0.0, 0.02))

sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0, 0, -0.05))
sim_utils.DomeLightCfg(intensity=1500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=1500.0))

bolt = Articulation(
    ArticulationCfg(
        prim_path="/World/Bolt",
        spawn=sim_utils.UsdFileCfg(
            usd_path=bolt_usd,
            activate_contact_sensors=True,
            rigid_props=RIGID,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=COLLISION,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={}, joint_vel={}),
        actuators={},
    )
)
nut = Articulation(
    ArticulationCfg(
        prim_path="/World/Nut",
        spawn=sim_utils.UsdFileCfg(
            usd_path=nut_usd,
            activate_contact_sensors=True,
            # 너트만 각속도를 제한한다. 자유 강체라 토크가 조금만 세도 발산하는데,
            # 나사산이 잡아주기 전에 튕겨 나가면 아무것도 관찰할 수 없다.
            rigid_props=RIGID.replace(angular_damping=0.5, max_angular_velocity=20.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=COLLISION,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, NUT_START_Z), joint_pos={}, joint_vel={}),
        actuators={},
    )
)

sim.reset()
nut_home = nut.data.root_state_w.clone()
_T_READY = time.time()
print(f"\n[timing] 앱 기동 + 씬 구성: {_T_READY - _T0:.1f}s")

held: set[str] = set()
state = {"gravity": True, "quit": False}


def on_key(event, *_):
    name = event.input.name
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        held.add(name)
        if name == "R":
            nut.write_root_state_to_sim(nut_home.clone())
            print("[reset] 너트를 시작 위치로 되돌림")
        elif name == "G":
            state["gravity"] = not state["gravity"]
            print(f"[gravity] {'ON' if state['gravity'] else 'OFF'}")
        elif name == "ESCAPE":
            state["quit"] = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held.discard(name)
    return True


# headless 에는 appwindow 가 없다. 그때는 키보드 대신 일정 토크를 계속 걸어
# "돌리면 정말 축을 따라 움직이는가"를 수치로만 확인한다 (스모크 테스트용).
if args.headless:
    print(f"\n[headless] 풀림 방향 토크 {args.torque} N·m 를 {args.steps} 스텝 인가하며 z 변화를 본다\n")
else:
    import omni.appwindow  # noqa: E402

    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, on_key)
    print(
        "\n"
        + "=" * 62
        + "\n  Q/E 나사축 토크(풀기/조이기)   W/S 위아래 힘   A/D 옆으로 힘\n"
        + "  G 중력토글   R 리셋   ESC 종료   |   Ctrl+좌드래그 = 마우스로 잡기\n"
        + "=" * 62
        + "\n"
    )

device = sim.device
forces = torch.zeros(1, 1, 3, device=device)
torques = torch.zeros(1, 1, 3, device=device)
step = 0
prev_z = nut.data.root_state_w[0, 2].item()

while sim_app.is_running() and not state["quit"]:
    forces.zero_()
    torques.zero_()

    if args.headless:
        held = {"Q"} if step < args.steps else set()
        if step >= args.steps:
            break

    if "Q" in held:
        torques[0, 0, 2] += args.torque
    if "E" in held:
        torques[0, 0, 2] -= args.torque
    if "W" in held:
        forces[0, 0, 2] += args.force
    if "S" in held:
        forces[0, 0, 2] -= args.force
    if "A" in held:
        forces[0, 0, 0] -= args.force
    if "D" in held:
        forces[0, 0, 0] += args.force
    if not state["gravity"]:
        # 중력 상쇄 (질량 0.03 kg)
        forces[0, 0, 2] += 0.03 * 9.81

    nut.set_external_force_and_torque(forces, torques, is_global=True)
    nut.write_data_to_sim()

    sim.step()
    bolt.update(sim.get_physics_dt())
    nut.update(sim.get_physics_dt())

    # 나사가 "회전한 만큼 축을 따라 움직이는지"를 수치로 본다.
    # dz/dtheta 가 pitch/2pi (=3.18e-4 m/rad) 에 가까우면 나사산이 제대로 물린 것이고,
    # 0 에 가까우면 헛도는 것, 훨씬 크면 이탈한 것이다.
    step += 1
    if step % 120 == 0:
        p = nut.data.root_state_w[0]
        z, wz = p[2].item(), p[12].item()
        ratio = (z - prev_z) / (wz * sim.get_physics_dt() * 120) if abs(wz) > 1e-3 else float("nan")
        print(
            f"  z = {z:+.4f} m   각속도 = {wz:+7.2f} rad/s   "
            f"dz/dθ = {ratio:+.2e} (이론 {DZ_PER_RAD:.2e})"
        )
        prev_z = z

_T_END = time.time()
_loop = _T_END - _T_READY
print(
    f"\n[timing] 기동 {_T_READY - _T0:.1f}s | 시뮬 {step} 스텝 {_loop:.1f}s "
    f"= {step / max(_loop, 1e-9):.0f} step/s (env 1개, SDF 충돌, solver 192회)"
)

sim_app.close()
