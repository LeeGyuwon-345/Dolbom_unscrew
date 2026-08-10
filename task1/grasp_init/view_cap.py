"""파라메트릭 캡을 실제 캡 메시와 겹쳐 본다.

파지 자세를 이 원통 위에서 최적화할 것이므로, 원통이 실물을 어디까지
대신하는지 보고 시작하는 편이 낫다. 특히 아래쪽 전이 구간(스커트 47.3mm ->
벽 50.0mm)은 원통이 표현하지 못한다.

왼쪽은 반경-높이 단면, 오른쪽은 3D. 단면 쪽이 오차를 읽기에는 정확하다 --
3D 렌더는 원통 두 개가 겹쳐 보일 뿐이다.

trimesh 의 창 뷰어는 이 환경에서 못 쓴다 (pyglet 2.x 는 trimesh.viewer.windowed
가 요구하는 1.x API 가 아니다). matplotlib 으로 그린다.

사용법:
    python view_cap.py                      # 창으로 보기
    python view_cap.py --png out.png        # 파일로 저장
    python view_cap.py -r 0.045 -H 0.020
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402

REAL = "/home/leegyuwon/Documents/task1/assets/tumbler/meshes/cap_collision.obj"


def real_profile(z_edges):
    """실제 메시의 높이별 반경 (p99.5). 없으면 None."""
    if not os.path.exists(REAL):
        return None, None
    import trimesh

    m = trimesh.load(REAL, force="mesh")
    p = m.sample(600000)
    z = p[:, 2] - m.bounds[0][2]
    r = np.hypot(p[:, 0], p[:, 1])
    zc, rc = [], []
    for a, b in zip(z_edges[:-1], z_edges[1:]):
        s = (z >= a) & (z < b)
        if s.sum() < 200:
            continue
        zc.append((a + b) / 2)
        rc.append(np.percentile(r[s], 99.5))
    return np.array(zc), np.array(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--radius", type=float, default=0.050)
    ap.add_argument("-H", "--height", type=float, default=0.017)
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    import matplotlib
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap = CapShape(args.radius, args.height)
    # 원기둥은 자기 좌표(0~h). 실제 캡 프레임에 얹어 비교한다.
    z_lo, z_hi = CAP_FRAME_Z0, CAP_FRAME_Z0 + cap.height
    zc, rc = real_profile(np.arange(0.0, 0.0301, 0.0015))

    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    if zc is not None:
        ax.plot(zc * 1000, rc * 1000, "-o", ms=3, color="0.35", label="실제 collision 메시")
        band = (zc >= z_lo) & (zc <= z_hi)
        err = (cap.radius - rc[band]) * 1000
        ax.set_title(f"단면   파지 구간 오차  최대 {np.abs(err).max():.2f}mm  "
                     f"평균 {np.abs(err).mean():.2f}mm")
    ax.plot([z_lo * 1000, z_hi * 1000], [cap.radius * 1000] * 2,
            lw=3, color="#3c82f0", label=f"파라메트릭 r={cap.radius*1000:.0f}mm")
    ax.axvline(z_lo * 1000, color="#e03c3c", ls="--", lw=1.2, label="밴드 하단")
    ax.axvline(z_hi * 1000, color="#3cc83c", ls="--", lw=1.2, label="밴드 상단")
    ax.set_xlabel("캡 프레임 높이 (mm)")
    ax.set_ylabel("반경 (mm)")
    ax.set_ylim(44, 54)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    th = np.linspace(0, 2 * np.pi, 80)
    if zc is not None:
        Z, T = np.meshgrid(zc, th)
        R = np.interp(Z, zc, rc)
        ax3.plot_surface(R * np.cos(T) * 1000, R * np.sin(T) * 1000, Z * 1000,
                         color="0.6", alpha=0.35, linewidth=0, shade=False)
    zp = np.array([z_lo, z_hi])
    Z2, T2 = np.meshgrid(zp, th)
    R2 = np.full_like(Z2, cap.radius)
    ax3.plot_surface(R2 * np.cos(T2) * 1000, R2 * np.sin(T2) * 1000, Z2 * 1000,
                     color="#3c82f0", alpha=0.55, linewidth=0, shade=False)
    for z, c in ((z_lo, "#e03c3c"), (z_hi, "#3cc83c")):
        ax3.plot(cap.radius * np.cos(th) * 1000, cap.radius * np.sin(th) * 1000,
                 np.full_like(th, z * 1000), color=c, lw=2)
    ax3.set_xlabel("x (mm)")
    ax3.set_ylabel("y (mm)")
    ax3.set_zlabel("z (mm)")
    ax3.set_title("회색=실제  파랑=파라메트릭")
    ax3.set_box_aspect((1, 1, 0.5))

    fig.tight_layout()
    print(f"파라메트릭  r {cap.radius*1000:.1f}mm  h {cap.height*1000:.1f}mm  "
          f"z {z_lo*1000:.1f}~{z_hi*1000:.1f}mm")
    if zc is not None:
        band = (zc >= z_lo) & (zc <= z_hi)
        print("\n 높이mm   실제      파라메트릭   차이mm")
        for z, r in zip(zc[band], rc[band]):
            print("  %5.1f   %6.2f      %6.2f     %+5.2f"
                  % (z * 1000, r * 1000, cap.radius * 1000, (cap.radius - r) * 1000))
    if args.png:
        fig.savefig(args.png, dpi=130)
        print(f"\n저장: {args.png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
