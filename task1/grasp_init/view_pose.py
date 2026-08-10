"""최적화된 파지 자세를 원기둥과 함께 그린다.

정책을 태우지 않는다. optimize_pose 가 낸 26차원을 URDF FK 로 그대로 펴서
링크 메시를 놓을 뿐이라, 화면에 보이는 것이 최적화가 실제로 만든 자세다.
Isaac Gym 으로 보면 리셋 직후 정책이 곧바로 손을 움직여 버려서 자세 자체를
확인하기 어렵다.

    회색      손 링크 메시
    파랑      캡 원기둥 (r x h)
    초록 점   손끝. 벽에 닿아야 한다
    빨강 선   손끝 손바닥 방향

사용법:
    python view_pose.py poses/cyl_r50_h17.json
    python view_pose.py poses/cyl_r50_h17.json --png out.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402
from maniptrans_dof_order import DOF_NAMES  # noqa: E402
from optimize_pose import FINGER_RADIUS, PALM_LOCAL, TIP_LINKS, URDF  # noqa: E402

MESH_DIR = "/home/leegyuwon/Documents/task1/assets/dg5fs_hand/meshes/dg5fs_right"
LINKS = ["link_base"] + [f"link_{f}_{k}" for f in range(1, 6)
                         for k in ["1", "2", "3", "4", "tip"]]


def link_transforms(q: np.ndarray, device="cpu"):
    """모든 링크의 손목 기준 4x4. pytorch_kinematics 의 전체 체인 FK."""
    import pytorch_kinematics as pk

    chain = pk.build_chain_from_urdf(open(URDF, "rb").read()).to(device=device)
    names = chain.get_joint_parameter_names()
    th = {n: float(q[DOF_NAMES.index(n)]) for n in names}
    ret = chain.forward_kinematics(th)
    return {k: v.get_matrix()[0].numpy() for k, v in ret.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose")
    ap.add_argument("-r", "--radius", type=float, default=0.050)
    ap.add_argument("-H", "--height", type=float, default=0.017)
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    import matplotlib
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.pose if os.path.isabs(args.pose) else os.path.join(os.path.dirname(__file__), args.pose)
    d = json.load(open(path))
    q = np.array(d["dof_pos"], dtype=np.float64)
    # 저장된 손목 위치는 실제 캡 프레임 기준. 원기둥 좌표로 되돌린다.
    wp = np.array(d["wrist_rel_pos"]) - np.array([0.0, 0.0, CAP_FRAME_Z0])
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(d["wrist_quat"]).as_matrix()

    cap = CapShape(args.radius, args.height)
    tf = link_transforms(q)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    th = np.linspace(0, 2 * np.pi, 72)
    for z in (0.0, cap.height):
        ax.plot(cap.radius * np.cos(th) * 1000, cap.radius * np.sin(th) * 1000,
                np.full_like(th, z * 1000), color="#3c82f0", lw=2)
    Z, T = np.meshgrid(np.array([0.0, cap.height]), th)
    Rm = np.full_like(Z, cap.radius)
    ax.plot_surface(Rm * np.cos(T) * 1000, Rm * np.sin(T) * 1000, Z * 1000,
                    color="#3c82f0", alpha=0.30, linewidth=0, shade=False)

    for name in LINKS:
        f = os.path.join(MESH_DIR, f"{name}.STL")
        if not os.path.exists(f) or name not in tf:
            continue
        m = trimesh.load(f, force="mesh")
        if len(m.faces) > 400:
            import open3d as o3d
            o = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(m.vertices)),
                o3d.utility.Vector3iVector(np.asarray(m.faces)))
            s = o.simplify_quadric_decimation(400)
            m = trimesh.Trimesh(np.asarray(s.vertices), np.asarray(s.triangles), process=False)
        v = (tf[name][:3, :3] @ m.vertices.T).T + tf[name][:3, 3]   # 손목 기준
        v = (R @ v.T).T + wp                                        # 원기둥 기준
        ax.plot_trisurf(v[:, 0] * 1000, v[:, 1] * 1000, v[:, 2] * 1000,
                        triangles=m.faces, color="0.65", alpha=0.85,
                        linewidth=0, shade=True)

    print(" 손가락   벽까지mm   높이mm   방위각도")
    names = ("엄지", "검지", "중지", "약지", "새끼")
    az = []
    for k, link in enumerate(TIP_LINKS):
        p = R @ tf[link][:3, 3] + wp
        n = R @ (tf[link][:3, :3] @ PALM_LOCAL.numpy())
        gap = np.hypot(p[0], p[1]) - cap.radius - FINGER_RADIUS
        a = math.degrees(math.atan2(p[1], p[0]))
        az.append(a)
        print("  %-6s  %+7.2f  %7.1f  %+8.1f" % (names[k], gap * 1000, p[2] * 1000, a))
        ax.scatter(*[p[i] * 1000 for i in range(3)], color="#20b020", s=45)
        e = p + n * 0.020
        ax.plot(*[[p[i] * 1000, e[i] * 1000] for i in range(3)], color="#e02020", lw=1.6)
    sep = abs(az[0] - az[2])
    sep = min(sep, 360 - sep)
    print(f"\n엄지-중지 방위각 {sep:.1f}도")

    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_title(f"{os.path.basename(path)}   r={cap.radius*1000:.0f}mm h={cap.height*1000:.0f}mm")
    lim = 140
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-60, 120)
    ax.set_box_aspect((1, 1, 0.65))
    fig.tight_layout()
    if args.png:
        fig.savefig(args.png, dpi=130)
        print(f"저장: {args.png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
