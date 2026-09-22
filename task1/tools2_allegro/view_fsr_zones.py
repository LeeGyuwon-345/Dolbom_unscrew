"""fingertip_v2 팜쪽 3면 × 위아래 2분할 = 6개 FSR 직사각형 패드 뷰어.

실물 FSR 부착 계획(팜쪽 3개 면에 각 2개, 사각형 패드)을 메시 위에서 확인.
각 면의 평면 좌표계에서 면 경계를 실측해 마진을 두고 직사각형을 얹는다.
마우스 드래그 회전, 휠 줌, ESC 종료. 실행 시 패드별 크기·중심(STL mm)을
표로 출력한다 — 시뮬 접촉 판정 존 매핑과 실물 부착 도면의 기준.

사용법:
    python view_fsr_zones.py            # 창 띄우기
    python view_fsr_zones.py --dump     # 표만 출력 (창 없음)
"""

import sys

import numpy as np
import trimesh

STL = "/home/leegyuwon/Documents/task1/assets/rb5_allegro/allegro_meshes/fingertip_sil_v2.stl"
Z_SPLIT = 15.0     # 패드 z 0~30 절반 (STL: z=0 쪽이 손끝)
MARGIN = 1.2       # 면 가장자리·분할선에서 패드까지 여백 (mm)
LIFT = 0.15        # 면 위로 띄우는 높이 (z-fighting 방지, mm)

FACES = [  # (이름, 법선 판별: (nx>0.3 공통) ny 조건)
    ("중앙", lambda ny: np.abs(ny) < 0.3),
    ("좌경사", lambda ny: ny >= 0.3),
    ("우경사", lambda ny: ny <= -0.3),
]
COLORS = [  # 원위(손끝쪽), 근위(링크쪽) 순
    ([230, 60, 60], [255, 160, 60]),      # 중앙: 빨강/주황
    ([60, 200, 60], [160, 230, 90]),      # 좌경사: 초록/연두
    ([70, 110, 230], [120, 200, 240]),    # 우경사: 파랑/하늘
]
BODY_COLOR = [150, 150, 150]

m = trimesh.load(STL)
n = m.face_normals
c = m.triangles_center
a = m.area_faces

geoms = []      # (정점(3k,3), 색(3,))
rows = []
# 면적 필터: 패드 면은 대형 삼각형(~195mm²)뿐이고, 내부 구멍·홈의
# 미세 면(<5mm²)이 법선 방향만으로는 같은 그룹에 섞여 평균을 오염시킨다.
palm = (n[:, 0] > 0.3) & (c[:, 2] <= 30.0) & (a > 20.0)
for fi, (fname, cond) in enumerate(FACES):
    sel = palm & cond(n[:, 1])
    w = a[sel]
    fn = (n[sel] * w[:, None]).sum(0) / w.sum()
    fn /= np.linalg.norm(fn)
    # 면 평면 좌표계: v=z축(길이방향), u=fn×v (면 내 폭방향)
    v_ax = np.array([0.0, 0.0, 1.0])
    u_ax = np.cross(v_ax, fn)
    u_ax /= np.linalg.norm(u_ax)
    verts = m.vertices[m.faces[sel]].reshape(-1, 3)
    origin = verts.mean(0)
    uu = (verts - origin) @ u_ax
    u_lo, u_hi = uu.min() + MARGIN, uu.max() - MARGIN
    for hi, half in enumerate(("원위", "근위")):
        z_lo = (0.0 if hi == 0 else Z_SPLIT) + MARGIN
        z_hi = (Z_SPLIT if hi == 0 else 30.0) - MARGIN
        # 직사각형 네 모서리 (면 평면 위 + LIFT)
        corners = []
        for u_, z_ in ((u_lo, z_lo), (u_hi, z_lo), (u_hi, z_hi), (u_lo, z_hi)):
            p = origin + u_ax * u_ + fn * LIFT
            # z(길이방향)는 STL z 절대값으로 고정
            p = p + v_ax * (z_ - p[2])
            # 면이 평면이므로 x,y는 해당 z에서의 면 위 점으로 보정 불필요
            corners.append(p)
        corners = np.array(corners)
        tri = np.array([corners[0], corners[1], corners[2],
                        corners[0], corners[2], corners[3]])
        geoms.append((tri, COLORS[fi][hi]))
        ctr = corners.mean(0)
        rows.append((f"{fname}-{half}", u_hi - u_lo, z_hi - z_lo, ctr, fn))

print(f"{'패드':<8} {'폭mm':>6} {'길이mm':>6} {'중심 x':>7} {'y':>6} {'z':>6}  법선")
for name, wdt, hgt, ctr, fn in rows:
    print(f"{name:<8} {wdt:6.1f} {hgt:6.1f} {ctr[0]:7.1f} {ctr[1]:6.1f} {ctr[2]:6.1f}  "
          f"[{fn[0]:.2f} {fn[1]:.2f} {fn[2]:.2f}]")

if "--dump" not in sys.argv:
    import open3d as o3d

    out = []
    # 몸통 (회색)
    tri = m.vertices[m.faces].reshape(-1, 3)
    body = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(tri),
        o3d.utility.Vector3iVector(np.arange(len(tri)).reshape(-1, 3)),
    )
    body.paint_uniform_color(np.array(BODY_COLOR) / 255.0)
    body.compute_vertex_normals()
    out.append(body)
    # FSR 패드 (양면 렌더링 위해 앞뒤 삼각형 모두)
    for tri, col in geoms:
        tri2 = np.vstack([tri, tri[::-1]])
        g = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(tri2),
            o3d.utility.Vector3iVector(np.arange(len(tri2)).reshape(-1, 3)),
        )
        g.paint_uniform_color(np.array(col) / 255.0)
        g.compute_vertex_normals()
        out.append(g)
    o3d.visualization.draw_geometries(
        out, window_name="fingertip_v2 FSR 6패드 (마우스 회전, ESC 종료)",
        width=1100, height=800,
    )
