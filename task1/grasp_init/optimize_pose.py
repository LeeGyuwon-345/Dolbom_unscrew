"""반지름 r, 높이 h 인 원기둥을 잡는 26차원 자세를 찾는 뼈대.

26 = 손가락 관절 20 + 손목 6 (위치 3, 축각 3). 손목은 원기둥 좌표 기준이므로
캡이 어디에 놓이든 그대로 옮겨 쓸 수 있다.

pytorch_kinematics 로 URDF FK 를 미분 가능하게 돌리고, 목적함수를 Adam 으로
내린다. 여러 초기값을 배치로 동시에 굴려 가장 좋은 것을 고른다 -- 손이 캡을
반대편에서 잡거나 손가락이 엉킨 국소해가 많다.

목적함수는 비어 있다. objective() 를 채우면 harness 가 그대로 굴린다.

쓸 수 있는 재료:
    to_cap_frame(q, wp, wr, fk)  손끝 위치/손바닥 방향/링크 점을 원기둥 좌표로
    cap.nearest(p)               옆면 최근접점 / 바깥 법선 / 부호거리
    cap.radius, cap.height
    fk.tips(q)                   손끝 5개 (손목 기준)
    fk.all_link_transforms(q)    전체 링크 4x4 (손목 기준)
    fk.link_points(n)            링크 메시 표면점 (링크 로컬)
    lo, hi                       관절 가동범위
    FINGER_RADIUS                손끝 구 반경 8.1mm

이전 목적함수에서 실제로 겪은 것 -- 다시 쓸 때 참고:

  집계를 평균으로 하면 소수의 심한 위반이 묻힌다. 관통 항을 점 전체의 제곱
  평균으로 뒀더니 손가락 마디가 캡을 20mm 파고든 자세가 pen=0.000003 으로
  보고됐다. 3000점 중 300점이 깊이 들어가도 평균은 작다.

  표면점 샘플이 성기면 접근을 놓친다. 링크당 40점으로는 자기충돌 0.58mm 를
  0 으로 봤고, 같은 자세를 600점으로 재면 잡혔다.

  힌지(clamp_min(0))로 만든 부등식 항은 기준을 넘는 순간 0 이 되므로 최적화가
  정확히 경계에 앉는다. 엄지-중지 90.9도, cos 0.80 처럼. 리셋에서 노이즈를
  얹으면 바로 기준 아래로 떨어진다.

  손끝만 구속하면 손바닥과 중간 마디가 캡을 통과한다. 손끝 다섯 개가 모두
  벽에 닿은 채 손이 캡 위를 가로지르는 해가 나왔다.

  반경 오차와 축 오차를 합친 거리 하나로 접촉을 보면, 테두리 위에 떠 있는
  손끝도 "닿았다" 가 된다.

사용법:
    python optimize_pose.py                          # r=50mm h=17mm
    python optimize_pose.py -r 0.045 -H 0.020 -n 64
    python optimize_pose.py -o poses/cyl_50_17.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402

URDF = "/home/leegyuwon/Documents/task1/assets/dg5fs_hand/dg5fs_right.urdf"
MESH_DIR = "/home/leegyuwon/Documents/task1/assets/dg5fs_hand/meshes/dg5fs_right"
TIP_LINKS = ["link_1_tip", "link_2_tip", "link_3_tip", "link_4_tip", "link_5_tip"]
FINGERS = ("엄지", "검지", "중지", "약지", "새끼")
THUMB, INDEX, MIDDLE, RING, PINKY = 0, 1, 2, 3, 4
# 손끝 링크 *자기 프레임* 에서의 손바닥 축. 학습 쪽 dexhandmanip_sh 가 쓰는
# 값과 같아야 한다: 거기서는 말단 링크(link_X_4)에 로컬 +y 를 적용하고,
# DIP 를 굽혔을 때 손끝이 움직이는 방향을 재서 다섯 손가락 모두 [-0.48,+0.88,0]
# 즉 +y 로 나온 결과다.
#
# 여기를 (1,0,0) 으로 두었을 때 손끝이 뻗는 방향이 벽에 붙었다 -- +x 는 손목
# 프레임의 손바닥 축(wrist_init.PALM_LOCAL)이지 손끝 프레임의 것이 아니다.
# link_X_tip 은 link_X_4 에 rpy=0 0 0 으로 붙어 있어 회전이 같으므로, tip
# 프레임에 +y 를 써도 학습과 일치한다.
PALM_LOCAL = torch.tensor([0.0, 1.0, 0.0])
FINGER_RADIUS = 0.0081                       # link_X_tip.STL 실측
MIN_GAP_LINK = 0.004                         # 다른 손가락 링크와의 최소 간격
PEN_TOL = 0.0005                             # 관통 허용 오차. 패드가 벽에 정확히
                                             # 닿으면 표면점 일부가 수치적으로
                                             # 벽 위에 놓이므로, 0 으로 두면
                                             # pad 항과 서로 밀며 싸운다


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(...,3) 축각 -> (...,3,3). Rodrigues."""
    th = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s, c = torch.sin(th).unsqueeze(-1), torch.cos(th).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


class HandFK:
    """URDF FK. 손끝과 전체 링크를 손목 기준으로 준다. 미분 가능."""

    def __init__(self, device, dtype=torch.float32):
        import pytorch_kinematics as pk

        from maniptrans_dof_order import DOF_NAMES

        self.device, self.dtype = device, dtype
        self.chains = []
        for link in TIP_LINKS:
            ch = pk.build_serial_chain_from_urdf(open(URDF, "rb").read(), link)
            self.chains.append(ch.to(dtype=dtype, device=device))
        self.slices = [
            [DOF_NAMES.index(n) for n in ch.get_joint_parameter_names()] for ch in self.chains
        ]
        self.chain = pk.build_chain_from_urdf(open(URDF, "rb").read()).to(
            dtype=dtype, device=device
        )
        self._chain_names = self.chain.get_joint_parameter_names()
        self.all_links = ["link_base"] + [
            f"link_{f}_{k}" for f in range(1, 6) for k in ("1", "2", "3", "4", "tip")
        ]
        self.link_pts = {}
        self.capsules = self.build_capsules()

    def tips(self, q: torch.Tensor):
        """q (B,20) -> (손끝 위치 (B,5,3), 손바닥 방향 (B,5,3)), 손목 기준."""
        pos, dirs = [], []
        for ch, sl in zip(self.chains, self.slices):
            m = ch.forward_kinematics(q[:, sl]).get_matrix()
            pos.append(m[:, :3, 3])
            dirs.append(m[:, :3, :3] @ PALM_LOCAL.to(q.device, q.dtype))
        return torch.stack(pos, dim=1), torch.stack(dirs, dim=1)

    def all_link_transforms(self, q: torch.Tensor):
        """q (B,20) -> {링크명: (B,4,4)} 손목 기준."""
        th = {n: q[:, i] for i, n in enumerate(self._chain_names)}
        return {k: v.get_matrix() for k, v in self.chain.forward_kinematics(th).items()}

    def build_capsules(self, n_samples: int = 400):
        """링크마다 (로컬 선분 양끝 (2,3), 반경) -- 자기충돌용 캡슐 근사.

        표면점 최소거리는 두 링크가 교차해도 샘플이 교차 부위를 비껴가면 0 을
        보고한다. 실제로 20점 샘플이 중지-새끼 0.11mm 겹침을 놓쳤고, 40점도
        과거 0.58mm 를 놓쳤다. 선분은 링크의 뼈대 그 자체라 놓칠 부위가 없고,
        쌍당 거리 계산이 점 방식의 수천 개에서 25개로 준다.

        선분은 링크 원점 -> 자식 조인트 원점 (URDF), tip 은 메시의 +x 최대점.
        반경은 메시 표면점의 선분까지 최대거리 -- 보수적(뚱뚱한) 근사다.
        """
        import xml.etree.ElementTree as ET

        import numpy as np
        import trimesh

        root = ET.parse(URDF).getroot()
        child_origin = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            child_origin[j.find("parent").get("link")] = xyz
        caps = {}
        for name in self.all_links:
            if name == "link_base":
                continue
            f = os.path.join(MESH_DIR, f"{name}.STL")
            if not os.path.exists(f):
                continue
            m = trimesh.load(f, force="mesh")
            p1 = np.array(child_origin[name], dtype=np.float64) if name in child_origin                 else np.array([float(m.vertices[:, 0].max()), 0.0, 0.0])
            pts = np.asarray(m.sample(n_samples))
            L2 = float(p1 @ p1)
            t = np.clip(pts @ p1 / max(L2, 1e-12), 0.0, 1.0)
            r = float(np.linalg.norm(pts - t[:, None] * p1[None], axis=1).max())
            ends = torch.tensor(np.stack([np.zeros(3), p1]), dtype=self.dtype, device=self.device)
            caps[name] = (ends, r)
        return caps

    def link_points(self, n_per_link: int = 200):
        """링크 메시 표면점, 링크 로컬 좌표. 관통/자기충돌 검사용."""
        import trimesh

        out = {}
        for name in self.all_links:
            f = os.path.join(MESH_DIR, f"{name}.STL")
            if not os.path.exists(f):
                continue
            m = trimesh.load(f, force="mesh")
            out[name] = torch.tensor(
                m.sample(n_per_link), device=self.device, dtype=self.dtype
            )
        return out


def seg_seg_dist(p1, q1, p2, q2, eps=1e-9):
    """(...,3) 선분 [p1,q1] 과 [p2,q2] 사이 최소거리. Ericson 5.1.9 클램프판.

    구간 클램프 때문에 조각별로 매끄럽지만 어디서나 연속이라 Adam 에는 충분하다.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1)
    e = (d2 * d2).sum(-1)
    f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1)
    b = (d1 * d2).sum(-1)
    denom = (a * e - b * b).clamp_min(eps)
    s = ((b * f - c * e) / denom).clamp(0.0, 1.0)
    t = ((b * s + f) / e.clamp_min(eps)).clamp(0.0, 1.0)
    s = ((b * t - c) / a.clamp_min(eps)).clamp(0.0, 1.0)
    cp1 = p1 + s.unsqueeze(-1) * d1
    cp2 = p2 + t.unsqueeze(-1) * d2
    return (cp1 - cp2).norm(dim=-1)


def to_cap_frame(q, wp, wr, fk):
    """26차원 -> (손끝, 손바닥 방향, {링크: 점들}, {링크: 캡슐 끝점}) 원기둥 좌표."""
    R = axis_angle_to_matrix(wr)
    tp_l, td_l = fk.tips(q)
    tp = (R.unsqueeze(1) @ tp_l.unsqueeze(-1)).squeeze(-1) + wp.unsqueeze(1)
    td = (R.unsqueeze(1) @ td_l.unsqueeze(-1)).squeeze(-1)
    td = td / td.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    links, caps = {}, {}
    need_T = bool(fk.link_pts) or bool(fk.capsules)
    T = fk.all_link_transforms(q) if need_T else {}

    def xf(pl, M):
        p_l = (M[:, None, :3, :3] @ pl[None, :, :, None]).squeeze(-1) + M[:, None, :3, 3]
        return (R[:, None] @ p_l.unsqueeze(-1)).squeeze(-1) + wp[:, None]

    for name, pl in fk.link_pts.items():
        if name in T:
            links[name] = xf(pl, T[name])
    for name, (ends, _) in fk.capsules.items():
        if name in T:
            caps[name] = xf(ends, T[name])
    return tp, td, links, caps


# ---------------------------------------------------------------------------
# 목적함수 -- 비어 있다. 여기를 채운다.
# ---------------------------------------------------------------------------
def pad_points(tp: torch.Tensor, td: torch.Tensor) -> torch.Tensor:
    """손끝 원점에서 손바닥 방향으로 손끝반경만큼 나간 점 = 패드 면.

    손끝 중심과 벽 사이 거리를 재고 거기서 반경을 빼는 방식은 손끝을 구로
    보는 것이라 어느 면이 닿는지를 구분하지 못한다. 실제로 닿아야 하는 것은
    손바닥 쪽 패드이고, 손등으로 미는 자세는 캡을 돌릴 수 없다.
    """
    return tp + td * FINGER_RADIUS


def tip_cos_inward(tp: torch.Tensor, td: torch.Tensor) -> torch.Tensor:
    """(B,5) 손끝 법선과 캡 *안쪽* 법선의 코사인. +1 이면 축을 정면으로 본다.

    학습 쪽 cap_unscrew_curriculum_terms.palm_facing_mask 와 같은 규약:
        cos = palm . (-outward)
    여기가 어긋나면 최적화가 낸 자세를 학습 보상이 정반대로 채점한다.
    """
    u = tp[..., :2] / torch.linalg.norm(tp[..., :2], dim=-1, keepdim=True).clamp_min(1e-9)
    inward = torch.cat([-u, torch.zeros_like(tp[..., 2:3])], dim=-1)
    return (td * inward).sum(dim=-1)


def contact_fan_area(pad: torch.Tensor) -> torch.Tensor:
    """(B,) 엄지를 꼭짓점으로 한 삼각형 넓이의 합.

    (엄지,검지,중지) + (엄지,중지,약지) + (엄지,약지,새끼). 각 삼각형은 외적
    노름으로 계산한 3D 넓이라 접촉점들이 한 평면에 있을 필요가 없다 -- xy 로
    투영해 신발끈 공식을 쓰던 첫 버전은 점들이 평면에 있다고 가정하는 셈이었다.

    부채꼴 분할이므로 다섯 점이 실제로 한 평면 위 볼록 배치라면 오각형 넓이와
    정확히 같다. 정오각형 기준값을 그대로 쓸 수 있는 이유다.
    """
    t = pad[:, 0]
    area = None
    for a, b in ((1, 2), (2, 3), (3, 4)):
        u = pad[:, a] - t
        v = pad[:, b] - t
        tri = 0.5 * torch.linalg.cross(u, v, dim=-1).norm(dim=-1)
        area = tri if area is None else area + tri
    return area


def objective(q, wp, wr, fk, cap, lo, hi, detail=False):
    """(B,) 손실. 작을수록 좋다.

    q  (B,20)  손가락 관절각
    wp (B,3)   손목 위치, 원기둥 좌표
    wr (B,3)   손목 축각

    detail=True 면 (손실, {이름: (B,) 값}) 을 돌려준다.

    현재 항:
      pad     손끝 패드 면과 원기둥 옆면 사이 거리를 0 으로
      facing  손끝 법선이 캡 안쪽(축)을 향하게
      under   손의 어느 점도 원기둥 밑면보다 아래로 내려가지 않게
      area    엄지 포함 삼각형 3개의 넓이 합 최대화 (대향+펼침)
      pen     tip 을 뺀 모든 링크가 원기둥 고체 안으로 침투하지 않게
      selfc   다른 손가락 캡슐과 4mm 이상 (선분+반경 근사)

    가중치는 1mm 의 패드 오차와 cos 0.9 가 비슷한 값을 갖도록 맞췄다. pad 는
    m^2 이라 1mm = 1e-6 이고, facing 은 무차원 O(1) 이라 그냥 더하면 facing 이
    5000배 무겁다.
    """
    tp, td, links, caps = to_cap_frame(q, wp, wr, fk)
    pad = pad_points(tp, td)
    # 옆면까지 거리의 제곱. 옆면 위아래로 벗어난 점은 테두리로 투영되므로,
    # 테두리 바깥에 뜬 패드는 그만큼 먼 것으로 잡힌다 -- 반경만 맞추고 위에
    # 떠 있는 해가 0 을 받지 않는다.
    pad_err = cap.dist2(pad).mean(dim=-1)

    # 손끝 법선이 캡 안쪽을 향하게. 학습 쪽 palm_facing_mask 와 같은 규약으로
    # 안쪽 법선을 쓴다 -- 거기서는 cos = palm . (-outward) 이고 cos >= 0.8 이
    # "패드가 벽을 마주봄" 이다. 바깥 법선으로 재면 부호가 뒤집혀 손등으로
    # 미는 자세를 정답으로 만든다.
    #
    # 힌지가 아니라 (1-cos)^2 로 둔다. 임계값을 넘는 순간 0 이 되는 형태는
    # 최적화를 정확히 경계에 앉히고, 리셋 노이즈에 바로 무너진다.
    cos = tip_cos_inward(tp, td)
    facing = ((1.0 - cos) ** 2).mean(dim=-1)

    # 원기둥 밑면 아래는 텀블러 몸통이다. 손이 거기로 내려갈 수 없다.
    #
    # 최댓값으로 집계한다. 평균을 쓰면 소수의 깊은 침범이 묻힌다 -- 관통 항을
    # 제곱평균으로 뒀을 때 마디가 20mm 파고든 자세가 0.000003 으로 보고된 적이
    # 있다. 여기서는 가장 깊이 내려간 점 하나가 값을 정한다.
    under = torch.zeros_like(facing)
    if links:
        deep = [(-p[..., 2]).clamp_min(0.0).amax(dim=-1) for p in links.values()]
        under = (torch.stack(deep, dim=-1).amax(dim=-1)) ** 2

    # 접촉 다각형 넓이 최대화 = 대향. 다섯 패드가 벽 반경의 원에 정오각형으로
    # 앉는 넓이를 기준으로 잡고 모자란 만큼을 벌점으로 둔다. 손이 기구학적으로
    # 정오각형을 못 만들면 닿는 데까지 가고 멈춘다 -- 어느 손가락이 반대편으로
    # 갈지는 지정하지 않는다. 힌지가 아니라서 도달 가능한 한 계속 벌린다.
    area = contact_fan_area(pad)
    area_ref = 2.5 * cap.radius ** 2 * math.sin(math.radians(72.0))
    area_def = ((area_ref - area).clamp_min(0.0) / area_ref) ** 2

    # 원기둥 내부 침투. 캡 고체(반경<=r, z<=h) 안에 들어온 점의 깊이는
    # "가장 가까운 표면까지" = min(반경 여유, 윗면 여유) 다. 밖이면 어느 한쪽이
    # 0 이라 min 도 0. 집계는 최댓값 -- 평균은 20mm 관통을 0.000003 으로
    # 보고한 전례가 있다.
    pen = torch.zeros_like(facing)
    selfc = torch.zeros_like(facing)
    if links:
        # tip 링크는 침투 검사에서 뺀다. 패드가 벽에 닿는 링크라 표면점이
        # 경계에 걸리는 것이 정상이고, 넣어두면 pad 항과 서로 밀며 싸운다.
        # 마디(link_X_1~4)와 손바닥이 고체를 통과하는 것만 막으면 된다 --
        # 중지가 반대편으로 넘어가려고 14.7mm 뚫고 지나간 것이 그 경우다.
        allp = torch.cat(
            [p for n, p in links.items() if not n.endswith("_tip")], dim=1)
        rr = torch.linalg.norm(allp[..., :2], dim=-1)
        depth = torch.minimum((cap.radius - rr).clamp_min(0.0),
                              (cap.height - allp[..., 2]).clamp_min(0.0))
        pen = ((depth - PEN_TOL).clamp_min(0.0).amax(dim=-1)) ** 2

    # 자기충돌: 캡슐끼리. 손가락별 선분 5개와 반경을 모아, 쌍마다 25개 선분
    # 거리에서 반경 합을 빼고 최소를 취한다. 선분은 링크의 뼈대라 표면점처럼
    # 교차 부위를 비껴갈 수 없다 -- 20점 샘플이 중지-새끼 0.11mm 겹침을
    # selfc=0 으로 보고한 것이 이 교체의 이유다.
    if caps:
        by_f = {}
        for name, ends in caps.items():
            by_f.setdefault(name.split("_")[1], []).append((ends, fk.capsules[name][1]))
        packs = []
        for k in sorted(by_f):
            e = torch.stack([x[0] for x in by_f[k]], dim=1)          # (B,S,2,3)
            r = torch.tensor([x[1] for x in by_f[k]], device=q.device, dtype=q.dtype)
            packs.append((e, r))
        acc = None
        for a in range(len(packs)):
            for b in range(a + 1, len(packs)):
                ea, ra = packs[a]
                eb, rb = packs[b]
                d = seg_seg_dist(
                    ea[:, :, None, 0], ea[:, :, None, 1],
                    eb[:, None, :, 0], eb[:, None, :, 1])            # (B,Sa,Sb)
                surf = d - (ra[None, :, None] + rb[None, None, :])
                viol = ((MIN_GAP_LINK - surf.amin(dim=(-2, -1))).clamp_min(0.0)) ** 2
                acc = viol if acc is None else acc + viol
        selfc = acc

    total = (1e4 * pad_err + 1.0 * facing + 1e5 * under + 1.0 * area_def
             + 1e5 * pen + 1e5 * selfc)
    if detail:
        return total, {"pad": pad_err, "facing": facing, "under": under,
                       "area": area_def, "pen": pen, "selfc": selfc}
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--radius", type=float, default=0.050)
    ap.add_argument("-H", "--height", type=float, default=0.017)
    ap.add_argument("-n", "--restarts", type=int, default=96)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-pts", type=int, default=200, help="링크당 표면점 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    from maniptrans_dof_order import DOF_LIMITS, DOF_NAMES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cap = CapShape(args.radius, args.height)
    fk = HandFK(dev)
    fk.link_pts = fk.link_points(args.n_pts)
    lo = torch.tensor([DOF_LIMITS[n][0] for n in DOF_NAMES], device=dev)
    hi = torch.tensor([DOF_LIMITS[n][1] for n in DOF_NAMES], device=dev)

    B = args.restarts
    q = (lo + (hi - lo) * torch.rand(B, 20, device=dev)).requires_grad_(True)
    # 손목 방위각만 원 둘레에 고르게 뿌린다. 어느 쪽에서 잡느냐가 국소해를
    # 가르는 가장 큰 변수다.
    th = torch.rand(B, device=dev) * 2 * math.pi
    # 손목 26차원 전부 자유. 높이만 원기둥 밑면(z=0) 아래로 못 내려가게
    # 스텝마다 잘라낸다 -- 그 아래는 텀블러 몸통이다. 손의 나머지는 under
    # 항이 같은 경계를 지킨다.
    wp = torch.stack([
        (cap.radius + 0.09) * torch.cos(th),
        (cap.radius + 0.09) * torch.sin(th),
        torch.full((B,), cap.height + 0.035, device=dev),
    ], dim=-1).requires_grad_(True)
    wr = (torch.rand(B, 3, device=dev) - 0.5).requires_grad_(True)

    def wrist_pos():
        return wp

    optim = torch.optim.Adam([q, wp, wr], lr=args.lr)
    print(f"r {cap.radius*1000:.1f}mm  h {cap.height*1000:.1f}mm   "
          f"재시작 {B}개  {args.iters}회  링크당 {args.n_pts}점  {dev}")
    print("손목 26차원 전부 자유, z >= 0 (원기둥 밑면) 만 강제")
    for it in range(args.iters):
        optim.zero_grad()
        loss = objective(q, wrist_pos(), wr, fk, cap, lo, hi)
        if loss.requires_grad:
            loss.sum().backward()
            optim.step()
            with torch.no_grad():
                q.clamp_(lo, hi)
        if it % 300 == 0 or it == args.iters - 1:
            print(f"  {it:5d}  최소 {float(loss.min()):.6f}  중앙 {float(loss.median()):.6f}")
        if not loss.requires_grad:
            print("  목적함수가 비어 있다 -- objective() 를 채울 것")
            break

    with torch.no_grad():
        wp = wrist_pos()
        total, det = objective(q, wp, wr, fk, cap, lo, hi, detail=True)
        i = int(torch.argmin(total))
        print(f"\n최적 #{i}  총 {float(total[i]):.6f}")
        if det:
            print("  " + "   ".join(f"{k} {float(v[i]):.6f}" for k, v in det.items()))
        tp, td, _, _ = to_cap_frame(q, wp, wr, fk)
        A = float(contact_fan_area(pad_points(tp, td))[i]) * 1e6
        A_ref = 2.5 * cap.radius ** 2 * math.sin(math.radians(72.0)) * 1e6
        print(f"  접촉 다각형 넓이 {A:.0f}mm^2  (정오각형 기준 {A_ref:.0f}, {100*A/A_ref:.0f}%)")
        cos = tip_cos_inward(tp, td)[i]
        az = torch.rad2deg(torch.atan2(tp[i, :, 1], tp[i, :, 0]))
        pad = pad_points(tp, td)[i]
        dd = cap.dist2(pad).sqrt()
        print("\n  손가락   패드-벽mm   cos    방위각도   패드높이mm")
        for k in range(5):
            print("  %-6s  %+8.2f  %+6.2f  %+8.1f  %9.1f"
                  % (FINGERS[k], float(dd[k]) * 1000, float(cos[k]),
                     float(az[k]), float(pad[k, 2]) * 1000))

        if args.out:
            from scipy.spatial.transform import Rotation

            from reference_pose import GraspInitPose

            R = axis_angle_to_matrix(wr[i:i + 1])[0].cpu().numpy()
            pose = GraspInitPose(
                dof_pos=q[i].detach().cpu(),
                # 원기둥 좌표 -> 실제 캡 프레임: 밑면이 13mm 에 앉는다
                wrist_rel_pos=wp[i].detach().cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0]),
                wrist_quat=torch.tensor(Rotation.from_matrix(R).as_quat(), dtype=torch.float32),
                note=f"cyl r={cap.radius*1000:.0f} h={cap.height*1000:.0f} loss={float(total[i]):.6f}",
            )
            out = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(__file__), args.out)
            pose.save(out)
            print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
