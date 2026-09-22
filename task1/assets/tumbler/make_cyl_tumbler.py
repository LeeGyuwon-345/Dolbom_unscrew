"""원기둥 근사 텀블러 생성기.

실물 STL 대신 캡(r,h)·몸체(R,H) 원기둥 두 개로 자산을 만든다. 조인트 사슬
(cap_spin + cap_lift mimic + free6 여분 4자유도)과 질량·관성·마찰은
tumbler_free6.urdf 와 동일해서, 학습·inference 어느 쪽에도 그대로 꽂힌다.
나사산은 현행대로 조인트(mimic + screw_coupling)가 담당한다.

기본값은 실물 STL 실측 근사다:
    캡   r=50.2mm  h=30mm   (cap_collision.obj 바깥벽/전체 높이)
    몸체 R=50.0mm  H=239.4mm (상부 벽 반경, 손잡이 제거)

사용:
    python make_cyl_tumbler.py                       # 기본값 -> tumbler_cyl.urdf
    python make_cyl_tumbler.py --r 0.045 --h 0.02 --R 0.04 --H 0.20 --out tumbler_cyl_s.urdf
"""

import argparse
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAPZ = 0.225   # cap_spin 조인트 원점 z. 기존 자산과 동일해야 obs(캡 위치)가 안 흔들린다.


def write_cyl_obj(path: str, radius: float, height: float, segments: int) -> None:
    """z=0..height 원기둥. convex 라 VHACD 없이 그대로 충돌에 쓸 수 있다."""
    vs, fs = [], []
    for z in (0.0, height):
        for k in range(segments):
            a = 2 * math.pi * k / segments
            vs.append((radius * math.cos(a), radius * math.sin(a), z))
    b0, b1 = 0, segments          # 아래/위 링 시작 인덱스 (0-기준)
    vs.append((0.0, 0.0, 0.0))    # 아래 중심
    vs.append((0.0, 0.0, height))  # 위 중심
    c0, c1 = 2 * segments, 2 * segments + 1
    for k in range(segments):
        k2 = (k + 1) % segments
        # 옆면 (바깥쪽 감김)
        fs.append((b0 + k, b0 + k2, b1 + k2))
        fs.append((b0 + k, b1 + k2, b1 + k))
        # 아래 뚜껑(법선 -z), 위 뚜껑(법선 +z)
        fs.append((c0, b0 + k2, b0 + k))
        fs.append((c1, b1 + k, b1 + k2))
    with open(path, "w") as f:
        f.write("# cylinder r=%.4f h=%.4f seg=%d\n" % (radius, height, segments))
        for v in vs:
            f.write("v %.6f %.6f %.6f\n" % v)
        for a, b, c in fs:
            f.write("f %d %d %d\n" % (a + 1, b + 1, c + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=0.0502, help="캡 반지름 m")
    ap.add_argument("--h", type=float, default=0.030, help="캡 높이 m")
    ap.add_argument("--R", type=float, default=0.050, help="몸체 반지름 m")
    ap.add_argument("--H", type=float, default=0.2394, help="몸체 높이 m")
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--spin-friction", type=float, default=0.0,
                    help="cap_spin 쿨롱 마찰 토크 Nm. 실물 나사 저항 근사 (0.1~0.5 권장)")
    ap.add_argument("--capz", type=float, default=CAPZ,
                    help="cap_spin 조인트 원점 z (m). 몸체 높이가 바뀌면 캡 안착 "
                         "높이도 같이 옮겨야 한다. 기본 0.225 = 기존 자산.")
    ap.add_argument("--out", default="tumbler_cyl.urdf")
    args = ap.parse_args()

    tag = args.out.replace(".urdf", "")
    cap_mesh = f"meshes/{tag}_cap.obj"
    body_mesh = f"meshes/{tag}_body.obj"
    write_cyl_obj(os.path.join(HERE, cap_mesh), args.r, args.h, args.segments)
    write_cyl_obj(os.path.join(HERE, body_mesh), args.R, args.H, args.segments)

    # 조인트 사슬은 tumbler_free6.urdf 를 그대로 가져오고 mesh 참조만 바꾼다.
    src = open(os.path.join(HERE, "tumbler_free6.urdf")).read()
    # 캡 관성은 균질 원기둥 공식으로 재계산 (기존 값은 실물 STL 형상 기준).
    m_cap = 0.05
    izz = 0.5 * m_cap * args.r ** 2
    ixx = m_cap * (3 * args.r ** 2 + args.h ** 2) / 12.0
    src = src.replace(
        '''      <mass value="0.05"/>
      <origin xyz="-0.000048 0.000284 0.015506"/>
      <inertia ixx="0.00004334" iyy="0.00004411" izz="0.00008291" ixy="0" ixz="0" iyz="0"/>''',
        '''      <mass value="%.3f"/>
      <origin xyz="0 0 %.6f"/>
      <inertia ixx="%.8f" iyy="%.8f" izz="%.8f" ixy="0" ixz="0" iyz="0"/>'''
        % (m_cap, args.h / 2.0, ixx, ixx, izz),
    )
    if args.spin_friction > 0:
        src = src.replace('<dynamics damping="0.001" friction="0.0"/>',
                          '<dynamics damping="0.001" friction="%.3f"/>' % args.spin_friction)
    if abs(args.capz - CAPZ) > 1e-9:
        assert '<origin xyz="0 0 0.225000"/>' in src, "free6 템플릿의 cap_spin 원점을 찾지 못함"
        src = src.replace('<origin xyz="0 0 0.225000"/>',
                          '<origin xyz="0 0 %.6f"/>' % args.capz)
    out = src.replace("meshes/body_visual.obj", body_mesh)
    out = out.replace("meshes/body_collision.obj", body_mesh)
    out = out.replace("meshes/cap_visual.obj", cap_mesh)
    out = out.replace("meshes/cap_collision.obj", cap_mesh)
    header = (
        "<!-- make_cyl_tumbler.py 생성: cap r=%.1f/h=%.1fmm, body R=%.1f/H=%.1fmm, %d각 -->\n"
        % (args.r * 1000, args.h * 1000, args.R * 1000, args.H * 1000, args.segments)
        + ("<!-- spin friction %.3f Nm -->\n" % args.spin_friction if args.spin_friction > 0 else "")
    )
    out = out.replace("<robot name=\"tumbler\">", header + "<robot name=\"tumbler\">", 1)
    with open(os.path.join(HERE, args.out), "w") as f:
        f.write(out)
    print(f"저장: {args.out} + {cap_mesh} + {body_mesh}")
    print(f"캡 원점 z={args.capz}m, 캡 벽 z 범위 {args.capz*1000:.0f}~{(args.capz+args.h)*1000:.0f}mm")


if __name__ == "__main__":
    main()
