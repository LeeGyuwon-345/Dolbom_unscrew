"""텀블러 몸통/뚜껑 STL 을 나사 구속이 걸린 URDF 한 벌로 만든다.

나사산 형상은 쓰지 않는다. 대신 뚜껑에 revolute(축 회전) + prismatic(축 병진) 을 주고
prismatic 이 revolute 를 mimic 하게 해서 "1회전 = pitch 만큼 상승"을 구속으로 강제한다.
URDF 의 <mimic multiplier="..."> 가 Isaac Sim 에서 PhysxMimicJointAPI 로 매핑된다.

원본 STL 좌표계:
    단위 mm, 나사축 = Y, 축 중심 (X,Z) = (52.5436, 50.50), 몸통 밑면 Y=0
출력 좌표계:
    단위 m, 나사축 = Z (IsaacLab 관례), 축 중심 = 원점, 몸통 밑면 z=0

사용법:
    python task1/tools/make_screw_asset.py
    python task1/tools/make_screw_asset.py --pitch 0.003 --friction 3.0
"""

import argparse
import os

import numpy as np
import trimesh

ASSETS = "/home/leegyuwon/Documents/task1/assets"
OUT_DIR = f"{ASSETS}/tumbler"

# 원본 STL 에서 측정한 값 (tools 로 계측: 반경 히스토그램 + 단면 프로파일).
# 이건 cap.STL 의 축이다. body.STL 의 축은 (50.78, 55.28) 로 다르다 -- 원본 모델에서
# 두 부품이 동축이 아니다. 예전에는 이 값 하나를 두 메시에 똑같이 적용해서 몸통이
# 캡 대비 (1.67, 4.78) mm 편심된 자산이 나왔고, 캡이 몸통 칼라에 7.3 도 기울어 얹혔다.
# 이제는 각 메시의 축을 따로 재서 각자 원점에 맞춘다 (--no-refit 으로 옛 동작 복원).
AXIS_CX_MM = 52.5436
AXIS_CZ_MM = 50.50
MM = 0.001


def fit_screw_axis(m: trimesh.Trimesh, r_max_mm: float = 70.0, n_slices: int = 25):
    """원본 좌표계(mm, 나사축 = Y)에서 원통 축 (x, z) 를 최소자승으로 찾는다.

    나사축에 수직인 단면마다 원을 맞추고, 잔차가 큰 단면은 버린다. 몸통은 손잡이가
    붙어 있어서 반경 r_max_mm 밖의 점을 먼저 쳐내야 원 피팅이 손잡이로 끌려가지 않는다.
    """
    lo, hi = m.bounds[0][1], m.bounds[1][1]
    fits = []
    for y in np.linspace(lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo), n_slices):
        sec = m.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
        if sec is None:
            continue
        p = np.array(sec.vertices)[:, [0, 2]]
        p = p[np.linalg.norm(p - p.mean(axis=0), axis=1) < r_max_mm]
        if len(p) < 10:
            continue
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(axis=1), rcond=None)
        cx, cz, c = sol
        r = np.sqrt(max(c + cx * cx + cz * cz, 0.0))
        res = np.abs(np.linalg.norm(p - [cx, cz], axis=1) - r).mean()
        fits.append((res, cx, cz))
    if not fits:
        raise RuntimeError("축 피팅 실패: 유효한 단면이 없다")
    # keep the cleanest half; slices that cut a handle or a thread relief are noise
    fits.sort()
    keep = fits[: max(1, len(fits) // 2)]
    cx = float(np.mean([f[1] for f in keep]))
    cz = float(np.mean([f[2] for f in keep]))
    spread = float(np.std([f[1] for f in keep]) + np.std([f[2] for f in keep]))
    return cx, cz, spread, len(keep)


def main():
    ap = argparse.ArgumentParser()
    # 사양: 200도 회전에 10mm 상승하고 거기서 풀린다.
    #   pitch = 10mm / (200/360 rev) = 18 mm/rev
    # spin_max = max_lift / (pitch/2pi) 이므로 회전 한계와 승강 한계가 정확히 180도에서
    # 함께 걸린다. 한 바퀴(360도)를 쓰던 이전 사양보다 풀림 행정이 짧아, 손목 회전
    # 한 번으로 닿을 수 있는 범위 안에 들어온다.
    ap.add_argument("--pitch", type=float, default=0.020, help="나사 pitch (m/rev)")
    ap.add_argument("--max-lift", type=float, default=0.010, help="나사가 물고 있는 상승량 (m)")
    # 나사가 끝난 뒤 뚜껑을 들고 갈 수 있어야 "분리"다. prismatic 은 런타임에 없앨 수
    # 없으므로, 나사 구간(max_lift) 너머로 자유 행정을 열어두고 구속만 해제한다.
    ap.add_argument("--free-travel", type=float, default=0.120,
                    help="나사 풀린 뒤 뚜껑이 더 올라갈 수 있는 거리 (m)")
    # 원본 STL 은 캡이 몸통에 4.34mm 만 걸친 상태로 배치돼 있다. 더 깊이 앉히려면 내린다.
    ap.add_argument("--cap-offset-z", type=float, default=-0.010, help="캡 초기 z 오프셋 (m)")
    # Tightening is blocked at the joint. The cap starts fully screwed on, so
    # there is nowhere to tighten to -- and left open (the old -2*pi), it became
    # a local optimum: measured at 200 epochs the policy drove the cap to -47 deg
    # on average, eating the reverse penalty the whole way, because turning it
    # closed presses the cap into the body and makes contact easy to hold. The
    # reverse penalty could not stop it: that term is on angular *speed*, so
    # tightening slowly costs almost nothing.
    #
    # A small negative slack is left so the limit is not exactly at the start
    # position, which the solver handles better.
    ap.add_argument("--spin-margin", type=float, default=0.02,
                    help="조이는 방향 여유각 (rad). 0 이면 시작 위치가 곧 한계")
    # 상한을 나사 끝과 정확히 같게 두면 PhysX 가 하드 스톱에 수렴하지 못해 각도가
    # 177~178 도에서 멈추고, 180 도에 걸린 해제 판정이 영영 성립하지 않는다.
    # 사양(180 도에 풀림)은 그대로 두고 조인트에만 여유를 준다.
    ap.add_argument("--spin-headroom-deg", type=float, default=2.0,
                    help="나사 끝을 넘어선 회전 여유각 (deg)")
    ap.add_argument("--friction", type=float, default=2.0, help="정/동마찰 계수")
    ap.add_argument("--collision-faces", type=int, default=3000, help="collision 메시 목표 면수")
    ap.add_argument("--no-refit", action="store_true",
                    help="축을 재지 않고 AXIS_CX/CZ_MM 하나를 두 메시에 적용 (옛 동작)")
    ap.add_argument("--visual-faces", type=int, default=12000, help="visual 메시 목표 면수")
    args = ap.parse_args()

    os.makedirs(f"{OUT_DIR}/meshes", exist_ok=True)

    # 원본 STL 은 정점이 분리돼 있어 그대로 두면 watertight 가 아니다.
    # process=True 로 중복 정점을 병합해야 convex decomposition 이 제대로 나온다.
    body = trimesh.load(f"{ASSETS}/body.STL", process=True)
    cap = trimesh.load(f"{ASSETS}/cap.STL", process=True)

    # mm -> m, 축 중심을 원점으로, 그리고 Y축(나사축)을 Z축으로 세운다.
    # Rx(+90°): (x,y,z) -> (x,-z,y)  즉 원본 Y 가 새 Z 가 된다.
    # 축은 메시마다 따로 잰다: 원본에서 두 부품이 동축이 아니라, 하나의 값을 공유하면
    # 그 차이가 그대로 조립 편심이 된다. 실물 텀블러는 나사산으로 동축이 되므로,
    # 각자를 자기 축에 맞추는 쪽이 실제에 가깝다.
    axes = {}
    for name, m in (("body", body), ("cap", cap)):
        if args.no_refit:
            axes[name] = (AXIS_CX_MM, AXIS_CZ_MM, 0.0, 0)
        else:
            axes[name] = fit_screw_axis(m)
        cx, cz, spread, n = axes[name]
        note = "(고정값)" if args.no_refit else f"(단면 {n}개, 산포 {spread:.3f}mm)"
        print(f"[axis] {name:5s} 축 (x={cx:.3f}, z={cz:.3f}) mm  {note}")
    d_ax_x = axes["cap"][0] - axes["body"][0]
    d_ax_z = axes["cap"][1] - axes["body"][1]
    print(f"[axis] 두 부품 축 차이 ({d_ax_x:+.3f}, {d_ax_z:+.3f}) mm "
          f"-> {'무시(옛 동작)' if args.no_refit else '각자 원점에 정렬'}")

    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float) * MM
    for name, m in (("body", body), ("cap", cap)):
        cx, cz, _, _ = axes[name]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array([-cx, cz, 0.0]) * MM  # y 부호는 회전과 함께 상쇄됨
        m.apply_transform(T)

    # 몸통 밑면이 z=0 에 오도록 두 메시를 함께 평행이동 (상대 위치는 유지해야 한다).
    dz = -body.bounds[0][2]
    for m in (body, cap):
        m.apply_translation([0, 0, dz])

    # 캡을 원하는 만큼 더 앉힌다. 조인트 원점(캡 밑면)도 같이 내려가므로
    # 나사 운동의 기준점이 자동으로 따라온다.
    if args.cap_offset_z:
        cap.apply_translation([0, 0, args.cap_offset_z])
        print(f"[cap] 초기 z 를 {args.cap_offset_z * 1000:+.1f} mm 이동")

    print("=" * 68)
    for name, m in [("body", body), ("cap", cap)]:
        lo, hi = m.bounds
        print(
            f"{name:5s} z {lo[2]:+.4f}~{hi[2]:+.4f} m   지름 {2 * max(abs(lo[:2]).max(), abs(hi[:2]).max()):.4f} m"
            f"   watertight={m.is_watertight}  faces={len(m.faces)}"
        )
    print("=" * 68)

    # 뚜껑을 제자리에서 회전/승강시키려면 조인트 원점이 축 위에 있어야 한다.
    # 축은 이미 원점이므로 조인트 원점 z 만 잡으면 된다 (뚜껑 밑면).
    cap_bottom_z = cap.bounds[0][2]

    # 뚜껑 메시를 조인트 원점 기준 로컬 좌표로 옮긴다.
    cap_local = cap.copy()
    cap_local.apply_translation([0, 0, -cap_bottom_z])

    meshes = {}
    for name, m in [("body", body), ("cap", cap_local)]:
        vis = _decimate(m, args.visual_faces)
        col = _decimate(m, args.collision_faces)
        vis.export(f"{OUT_DIR}/meshes/{name}_visual.obj")
        col.export(f"{OUT_DIR}/meshes/{name}_collision.obj")
        meshes[name] = (len(vis.faces), len(col.faces))
        print(f"{name:5s} visual {len(vis.faces):6d}면   collision {len(col.faces):6d}면")

    urdf = _urdf(args, cap_bottom_z, body, cap_local)
    path = f"{OUT_DIR}/tumbler.urdf"
    with open(path, "w") as f:
        f.write(urdf)

    # Body-only variant: the static support that replaces pedestal.urdf in the
    # grasp env. Emitted here rather than hand-edited so it can never drift out
    # of sync with the meshes this script writes.
    body_path = f"{OUT_DIR}/tumbler_body.urdf"
    with open(body_path, "w") as f:
        f.write(_body_urdf(args, body))

    mult = args.pitch / (2 * np.pi)
    print("=" * 68)
    print(f"pitch          {args.pitch * 1000:.1f} mm/rev")
    print(f"mimic 배율     {mult:.6e} m/rad   (= pitch / 2pi)")
    print(f"나사 물림 구간  {args.max_lift * 1000:.1f} mm  →  {args.max_lift / args.pitch * 360:.1f} 도 회전이면 풀림")
    print(f"회전 상한      {args.max_lift / args.pitch * 360 + args.spin_headroom_deg:.1f} 도 (여유 {args.spin_headroom_deg:.1f} 도)")
    print(f"자유 행정      {args.free_travel * 1000:.1f} mm  (풀린 뒤 들어내는 구간)")
    print(f"조임 여유      {args.spin_margin:.3f} rad ({np.degrees(args.spin_margin):.1f} deg) - 그 이상은 조인트가 막음")
    print(f"마찰계수       {args.friction}")
    print(f"조인트 원점 z  {cap_bottom_z:.4f} m (뚜껑 밑면)")
    print(f"\n출력: {path}")
    print(f"출력: {body_path}")
    print("=" * 68)


def _decimate(m: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    """면수를 target 근처로 줄인다. 이미 작으면 그대로 둔다.

    open3d 로 하고, 반드시 mm 스케일에서 돌린다. trimesh 의
    simplify_quadric_decimation (내부적으로 fast_simplification) 은 오차 허용치가
    절대 단위라, m 단위 메시에 걸면 1000 배 관대해져서 형상이 무너진다. 실제로
    반경 50.00mm / 높이 30.00mm 인 캡이 반경 57.01mm / 높이 27.89mm 에 축에서
    4.1mm 편심된 물체로 나왔고 (요청 3000 면 대신 4298 면), 그 왜곡된 메시가
    그대로 VHACD 를 거쳐 물리 캡이 됐다. open3d 는 같은 자리에서 50.04mm /
    30.01mm 를 준다.
    """
    if len(m.faces) <= target:
        return m.copy()
    s = m.copy()
    s.apply_scale(1.0 / MM)  # -> mm
    try:
        import open3d as o3d

        o = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(s.vertices)),
            o3d.utility.Vector3iVector(np.asarray(s.faces)),
        )
        d = o.simplify_quadric_decimation(target_number_of_triangles=target)
        out = trimesh.Trimesh(np.asarray(d.vertices), np.asarray(d.triangles), process=True)
    except Exception as e:
        print(f"  [warn] open3d 데시메이션 실패({e}) — trimesh 로 대체 (정확도 낮음)")
        try:
            out = s.simplify_quadric_decimation(face_count=target)
        except Exception as e2:
            print(f"  [warn] 데시메이션 실패({e2}) — 원본 유지")
            return m.copy()
    out.apply_scale(MM)  # -> m
    _check_decimation(m, out)
    return out


def _check_decimation(src: trimesh.Trimesh, out: trimesh.Trimesh, tol_mm: float = 0.5):
    """데시메이션이 형상을 얼마나 바꿨는지 재서, 조용히 넘어가지 않게 한다.

    이 검사가 없어서 위의 스케일 버그가 몇 주를 갔다. 반경/높이/축중심은 보상
    항이 캡을 원통으로 근사할 때 그대로 쓰는 값이라, 여기서 어긋나면 보상이
    물리와 다른 물체를 채점한다.
    """
    def prof(m):
        p = m.sample(200000)
        lo, hi = m.bounds
        return (np.hypot(p[:, 0], p[:, 1]).max() / MM, (hi[2] - lo[2]) / MM,
                (lo[0] + hi[0]) / 2 / MM, (lo[1] + hi[1]) / 2 / MM)

    a, b = prof(src), prof(out)
    d = [abs(x - y) for x, y in zip(a, b)]
    tag = "" if max(d) <= tol_mm else "  <-- 허용치 초과"
    print(f"       최대r {a[0]:.2f}->{b[0]:.2f}  높이 {a[1]:.2f}->{b[1]:.2f}  "
          f"중심 ({a[2]:+.2f},{a[3]:+.2f})->({b[2]:+.2f},{b[3]:+.2f}) mm{tag}")


def _urdf(args, cap_bottom_z: float, body: trimesh.Trimesh, cap: trimesh.Trimesh) -> str:
    """몸통(고정) -> revolute -> 더미 -> prismatic(mimic) -> 뚜껑 체인."""
    mult = args.pitch / (2 * np.pi)
    # 완전히 풀리는 회전각. mimic 은 유한 limit 을 요구한다.
    # 두 limit 이 어긋나면(회전은 더 되는데 승강은 막힘) mimic 구속과 prismatic limit 이
    # 서로 싸운다. 정확히 일치시켜 한 바퀴에서 둘 다 끝나게 한다.
    spin_end = args.max_lift / mult                       # 나사가 풀리는 각 (사양)
    spin_max = spin_end + np.radians(args.spin_headroom_deg)
    spin_margin = args.spin_margin
    # 실측 부피에 플라스틱 밀도를 곱하는 대신, 텀블러다운 값을 직접 준다.
    body_mass, cap_mass = 0.30, 0.05
    bi = _inertia(body, body_mass)
    ci = _inertia(cap, cap_mass)
    f = args.friction

    return f"""<?xml version="1.0"?>
<!-- 자동 생성: tools/make_screw_asset.py
     나사산 형상 없이 mimic 조인트로 나사 운동을 구속한다.
     cap_lift(prismatic) 가 cap_spin(revolute) 을 배율 {mult:.6e} m/rad 로 따라간다. -->
<robot name="tumbler">
  <link name="body">
    <visual>
      <geometry><mesh filename="meshes/body_visual.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/body_collision.obj"/></geometry>
    </collision>
    <inertial>
      <mass value="{body_mass}"/>
      <origin xyz="{bi[0]:.6f} {bi[1]:.6f} {bi[2]:.6f}"/>
      <inertia ixx="{bi[3]:.8f}" iyy="{bi[4]:.8f}" izz="{bi[5]:.8f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- 회전 자유도. 뚜껑을 푸는 방향이 +.
       상한({np.degrees(spin_max):.0f}도)은 나사 끝({np.degrees(spin_end):.0f}도)보다 {args.spin_headroom_deg:.0f}도 여유를 둔다. 둘을 같게 두면
       PhysX 가 하드 스톱에 정확히 수렴하지 않아 각도가 177~178 에서
       멈추고, 거기 걸린 해제 판정이 영영 성립하지 않는다.
       type 은 반드시 revolute + 유한 limit 이어야 한다. continuous 로 두면 PhysX 가
       "needs a finite limit set to be used by the mimic joint feature" 로 거부한다. -->
  <link name="cap_spinner">
    <inertial>
      <mass value="1e-4"/>
      <inertia ixx="1e-8" iyy="1e-8" izz="1e-8" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="cap_spin" type="revolute">
    <parent link="body"/>
    <child link="cap_spinner"/>
    <origin xyz="0 0 {cap_bottom_z:.6f}"/>
    <axis xyz="0 0 1"/>
    <limit lower="{-spin_margin:.6f}" upper="{spin_max:.6f}" effort="50.0" velocity="20.0"/>
    <dynamics damping="0.001" friction="0.0"/>
  </joint>

  <!-- 축방향 자유도. 나사 구간(0~{args.max_lift} m)에서는 회전을 따라가고,
       그 뒤 {args.free_travel} m 는 뚜껑을 들어내는 자유 행정이다.
       Isaac Gym 은 mimic 을 무시하므로 tools/screw_coupling.py 가 대신 구동한다. -->
  <joint name="cap_lift" type="prismatic">
    <parent link="cap_spinner"/>
    <child link="cap"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="{args.max_lift + args.free_travel}" effort="100.0" velocity="5.0"/>
    <mimic joint="cap_spin" multiplier="{mult:.8e}" offset="0.0"/>
  </joint>

  <link name="cap">
    <visual>
      <geometry><mesh filename="meshes/cap_visual.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/cap_collision.obj"/></geometry>
    </collision>
    <inertial>
      <mass value="{cap_mass}"/>
      <origin xyz="{ci[0]:.6f} {ci[1]:.6f} {ci[2]:.6f}"/>
      <inertia ixx="{ci[3]:.8f}" iyy="{ci[4]:.8f}" izz="{ci[5]:.8f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- 매끈한 원통이라 그대로는 미끄럽다. 나사산 대신 마찰로 그립을 만든다. -->
  <gazebo reference="cap"><mu1>{f}</mu1><mu2>{f}</mu2></gazebo>
  <gazebo reference="body"><mu1>{f}</mu1><mu2>{f}</mu2></gazebo>
</robot>
"""


def _body_urdf(args, body: trimesh.Trimesh) -> str:
    """몸통만 있는 정적 지지체. cap_grasp 학습에서 pedestal.urdf 를 대체한다."""
    bi = _inertia(body, 0.30)
    f = args.friction
    lo, hi = body.bounds
    return f"""<?xml version="1.0"?>
<!-- 자동 생성: tools/make_screw_asset.py (tumbler.urdf 와 같은 메시/좌표계)
     나사 구속 없는 정적 지지체. 캡은 별도 자산으로 이 위에 놓인다.
     주의: 축대칭이 아니다. 손잡이가 한쪽으로 뻗어 있고(x {hi[0] * 1000:.0f}mm 까지),
     상단 칼라가 캡 옆면 아래쪽을 가린다. 손목 prior 는 이 형상을 모른다. -->
<robot name="tumbler_body">
  <link name="body">
    <visual>
      <geometry><mesh filename="meshes/body_visual.obj"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="meshes/body_collision.obj"/></geometry>
    </collision>
    <inertial>
      <mass value="0.30"/>
      <origin xyz="{bi[0]:.6f} {bi[1]:.6f} {bi[2]:.6f}"/>
      <inertia ixx="{bi[3]:.8f}" iyy="{bi[4]:.8f}" izz="{bi[5]:.8f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <gazebo reference="body"><mu1>{f}</mu1><mu2>{f}</mu2></gazebo>
</robot>
"""


def _inertia(m: trimesh.Trimesh, mass: float):
    """질량을 주어진 값으로 재조정한 관성 주대각 + 무게중심."""
    com = m.center_mass if m.is_watertight else m.centroid
    I = m.moment_inertia * (mass / max(m.mass, 1e-12)) if m.is_watertight else np.eye(3) * 1e-4
    return (com[0], com[1], com[2], I[0, 0], I[1, 1], I[2, 2])


if __name__ == "__main__":
    main()
