"""반지름 r, 높이 h 인 원기둥을 잡는 allegro 22차원 자세를 찾는다.

optimize_pose.py(dg5fs) 의 allegro 이식판. 차이:
  * 손가락 16 + 손목 6 = 22차원 (dg5fs 는 20+6=26).
  * 4지(엄지/검지/중지/약지). contact_fan 은 엄지 꼭짓점 + 삼각형 2개.
  * URDF 충돌형상이 *박스*다(dg5fs 는 링크별 STL). sim 이 실제로 쓰는
    충돌이 박스이므로 박스를 파싱해 관통/자기충돌을 잰다. tip 만 메시.
  * 손끝 프레임 손바닥축 PALM_LOCAL = 로컬 +x (dg5fs 는 +y).
    학습 dexhandmanip_sh 의 CAP_PALM_LOCAL_AXIS allegro 기본값과 같다.
  * DOF 순서가 URDF 순차(0..15)가 아니라 ManipTrans 순서
    (index -> thumb -> middle -> ring). all_link_transforms 매핑에서 이걸
    DOF_NAMES.index 로 풀어야 FK 각도가 안 섞인다.

목적함수/집계 규율은 dg5fs 판 주석을 그대로 따른다(평균 대신 최댓값,
힌지 대신 제곱, 손끝만이 아니라 마디까지 구속 등).

사용법:
    python optimize_pose_allegro.py                       # r=50mm h=17mm
    python optimize_pose_allegro.py -r 0.045 -H 0.020 -n 64
    python optimize_pose_allegro.py -o poses/allegro_cyl_50_17.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from cap_shape import CAP_FRAME_Z0, CapShape  # noqa: E402

ALLEGRO_DIR = "/home/leegyuwon/Documents/ManipTrans/maniptrans_envs/assets/allegro_hand"
URDF = os.path.join(ALLEGRO_DIR, "allegro_hand_right.urdf")
TIP_MESH = os.path.join(ALLEGRO_DIR, "meshes/collision/link_tip.obj")

# 엄지를 0번(부채꼴 꼭짓점)으로. dg5fs 판의 contact_fan_area 규약과 맞춘다.
TIP_LINKS = ["link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"]
FINGERS = ("엄지", "검지", "중지", "약지")
THUMB, INDEX, MIDDLE, RING = 0, 1, 2, 3

# 손끝 링크 *자기 프레임* 의 손바닥 축. 학습 dexhandmanip_sh 가 distal 링크에
# 적용하는 값과 같아야 한다: 거기서 DIP 를 굽혔을 때 손끝이 움직이는 방향을
# 재서 allegro 는 로컬 +x 로 나왔다(CAP_PALM_LOCAL_AXIS allegro 기본 1,0,0).
# link_X.0_tip 은 link_X.0(distal) 에 회전 없이 붙으므로 tip 프레임 +x 로 써도
# 학습과 일치한다.
PALM_LOCAL = torch.tensor([1.0, 0.0, 0.0])
FINGER_RADIUS = 0.012                        # link_tip.obj bbox 24mm -> 반경 ~12mm
TIP_DOWN_W = 0.0                             # 팁 +z 를 바닥(-z)으로 (main 에서 설정)
MIN_GAP_LINK = 0.004                          # 다른 손가락 링크와의 최소 간격
PEN_TOL = 0.0005                              # 관통 허용 오차 (dg5fs 판과 동일 근거)

# 손가락(0엄지 1검지 2중지 3약지) -> q(DOF_NAMES 순서) 인덱스.
# DOF_NAMES = index(0-3) thumb(4-7) middle(8-11) ring(12-15) [Isaac Gym 순서].
FINGER_DOF = {0: [4, 5, 6, 7], 1: [0, 1, 2, 3], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15]}


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


def _sample_box_surface(center, size, n, device, dtype):
    """박스 표면점 (n,3), 링크 로컬. center/size 는 (3,). rpy=0 가정(allegro 충돌)."""
    h = torch.tensor(size, device=device, dtype=dtype) * 0.5
    c = torch.tensor(center, device=device, dtype=dtype)
    # 각 면을 넓이 비례로 뽑는다.
    areas = torch.tensor(
        [h[1] * h[2], h[1] * h[2], h[0] * h[2], h[0] * h[2], h[0] * h[1], h[0] * h[1]],
        device=device, dtype=dtype,
    )
    face = torch.multinomial(areas / areas.sum(), n, replacement=True)
    uv = torch.rand(n, 2, device=device, dtype=dtype) * 2 - 1  # [-1,1]
    p = torch.zeros(n, 3, device=device, dtype=dtype)
    # face 0/1: ±x, 2/3: ±y, 4/5: ±z
    for f in range(6):
        m = face == f
        if not m.any():
            continue
        ax = f // 2
        sign = -1.0 if (f % 2) else 1.0
        a1, a2 = [i for i in range(3) if i != ax]
        p[m, ax] = sign * h[ax]
        p[m, a1] = uv[m, 0] * h[a1]
        p[m, a2] = uv[m, 1] * h[a2]
    return p + c


class HandFK:
    """URDF FK. 손끝과 전체 링크를 손목 기준으로 준다. 미분 가능.

    dg5fs 판과 달리 충돌형상이 박스라, 링크별 (center,size) 를 URDF 에서 파싱해
    관통/자기충돌에 쓴다. tip 링크만 메시(link_tip.obj)를 구로 근사.
    """

    def __init__(self, device, dtype=torch.float32):
        import pytorch_kinematics as pk

        from maniptrans_dof_order_allegro import DOF_NAMES

        self.device, self.dtype = device, dtype
        self.DOF_NAMES = DOF_NAMES
        urdf_bytes = open(URDF, "rb").read()
        self.chains = []
        for link in TIP_LINKS:
            ch = pk.build_serial_chain_from_urdf(urdf_bytes, link)
            self.chains.append(ch.to(dtype=dtype, device=device))
        self.slices = [
            [DOF_NAMES.index(n) for n in ch.get_joint_parameter_names()] for ch in self.chains
        ]
        self.chain = pk.build_chain_from_urdf(urdf_bytes).to(dtype=dtype, device=device)
        self._chain_names = self.chain.get_joint_parameter_names()
        # q 는 DOF_NAMES 순서 -> chain 순서로 매핑 (allegro 는 둘이 다르다)
        self._q_idx_for_chain = [DOF_NAMES.index(n) for n in self._chain_names]

        self.all_links = ["base_link"] + [f"link_{i}.0" for i in range(16)] + list(TIP_LINKS)
        self.boxes, self.child_origin = self._parse_collision()
        self._tip_r = self._tip_radius()
        self.link_pts = {}
        self.capsules = self.build_capsules()

    def _parse_collision(self):
        """링크별 충돌 박스 (center,size), 조인트 자식원점 파싱."""
        import xml.etree.ElementTree as ET

        root = ET.parse(URDF).getroot()
        boxes, child_origin = {}, {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            child_origin[j.find("child").get("link")] = xyz
        for l in root.findall("link"):
            name = l.get("name")
            col = l.find("collision")
            if col is None:
                continue
            g = col.find("geometry")
            box = g.find("box") if g is not None else None
            if box is None:
                continue
            size = [float(v) for v in box.get("size").split()]
            o = col.find("origin")
            center = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
            boxes[name] = (center, size)
        return boxes, child_origin

    def _tip_radius(self):
        import trimesh

        try:
            m = trimesh.load(TIP_MESH, force="mesh")
            ext = m.bounds[1] - m.bounds[0]
            return float(max(ext[0], ext[1]) / 2)
        except Exception:
            return FINGER_RADIUS

    def tips(self, q: torch.Tensor):
        """q (B,16) -> (손끝 위치 (B,4,3), 손바닥 방향 (B,4,3)), 손목 기준."""
        pos, dirs = [], []
        for ch, sl in zip(self.chains, self.slices):
            m = ch.forward_kinematics(q[:, sl]).get_matrix()
            pos.append(m[:, :3, 3])
            dirs.append(m[:, :3, :3] @ PALM_LOCAL.to(q.device, q.dtype))
        return torch.stack(pos, dim=1), torch.stack(dirs, dim=1)

    def all_link_transforms(self, q: torch.Tensor):
        """q (B,16) -> {링크명: (B,4,4)} 손목 기준. q 는 DOF_NAMES 순서."""
        th = {n: q[:, self._q_idx_for_chain[i]] for i, n in enumerate(self._chain_names)}
        return {k: v.get_matrix() for k, v in self.chain.forward_kinematics(th).items()}

    def build_capsules(self, n_samples: int = 400):
        """링크마다 (로컬 선분 양끝 (2,3), 반경) -- 자기충돌용 캡슐 근사.

        finger link 는 박스 -> z 장축을 뼈대로, 반경은 단면 대각 절반(보수적).
        tip 은 구(선분 길이 0, 반경 = tip 메시 반경).
        """
        caps = {}
        for name in self.all_links:
            if name.endswith("_tip"):
                ends = torch.zeros(2, 3, device=self.device, dtype=self.dtype)
                caps[name] = (ends, self._tip_r)
                continue
            if name not in self.boxes:
                continue
            center, size = self.boxes[name]
            c = torch.tensor(center, device=self.device, dtype=self.dtype)
            hz = size[2] * 0.5
            ends = torch.stack([
                c + torch.tensor([0, 0, -hz], device=self.device, dtype=self.dtype),
                c + torch.tensor([0, 0, hz], device=self.device, dtype=self.dtype),
            ])
            r = 0.5 * float((size[0] ** 2 + size[1] ** 2) ** 0.5)  # 단면 대각 절반
            caps[name] = (ends, r)
        return caps

    def link_points(self, n_per_link: int = 200):
        """링크 표면점, 링크 로컬. 박스면 박스 표면, tip 이면 구 표면."""
        out = {}
        for name in self.all_links:
            if name.endswith("_tip"):
                # tip 구: 원점 중심 반경 _tip_r 구면점
                v = torch.randn(n_per_link, 3, device=self.device, dtype=self.dtype)
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * self._tip_r
                out[name] = v
                continue
            if name not in self.boxes:
                continue
            center, size = self.boxes[name]
            out[name] = _sample_box_surface(center, size, n_per_link, self.device, self.dtype)
        return out


def seg_seg_dist(p1, q1, p2, q2, eps=1e-9):
    """(...,3) 선분 [p1,q1] 과 [p2,q2] 사이 최소거리. Ericson 5.1.9 클램프판."""
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
    """22차원 -> (손끝, 손바닥 방향, {링크: 점들}, {링크: 캡슐 끝점}) 원기둥 좌표."""
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


def pad_points(tp: torch.Tensor, td: torch.Tensor) -> torch.Tensor:
    """손끝 원점에서 손바닥 방향으로 손끝반경만큼 나간 점 = 패드 면."""
    return tp + td * FINGER_RADIUS


def tip_cos_inward(tp: torch.Tensor, td: torch.Tensor) -> torch.Tensor:
    """(B,4) 손끝 법선과 캡 *안쪽* 법선의 코사인. 학습 palm_facing_mask 규약."""
    u = tp[..., :2] / torch.linalg.norm(tp[..., :2], dim=-1, keepdim=True).clamp_min(1e-9)
    inward = torch.cat([-u, torch.zeros_like(tp[..., 2:3])], dim=-1)
    return (td * inward).sum(dim=-1)


def contact_fan_area(pad: torch.Tensor) -> torch.Tensor:
    """(B,) 엄지를 꼭짓점으로 한 삼각형 넓이의 합. 4지: (엄지,검지,중지)+(엄지,중지,약지)."""
    t = pad[:, 0]
    area = None
    for a, b in ((1, 2), (2, 3)):
        u = pad[:, a] - t
        v = pad[:, b] - t
        tri = 0.5 * torch.linalg.cross(u, v, dim=-1).norm(dim=-1)
        area = tri if area is None else area + tri
    return area


def _wrap(x):
    """(-pi,pi] 로 감싼 각도."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def order_penalty(pad: torch.Tensor) -> torch.Tensor:
    """(B,) 검지·중지·약지 방위각 순서 위반 페널티. 중지가 검지~약지 사이(arc)에
    있어야 한다. 아니면 손가락이 교차(꼬임)한다. pad 인덱스 1=검지 2=중지 3=약지.

    검지 기준 상대각으로: 중지(a_m)가 [0, 약지(a_r)] 구간 밖이면 그 거리를 벌한다.
    구간 방향(부호)은 약지 위치로 자동 결정 -- CW/CCW 어느 쪽 파지든 성립.
    """
    az = torch.atan2(pad[..., 1], pad[..., 0])          # (B,4)
    a_m = _wrap(az[:, 2] - az[:, 1])                     # 중지 rel 검지
    a_r = _wrap(az[:, 3] - az[:, 1])                     # 약지 rel 검지
    lo = torch.minimum(torch.zeros_like(a_r), a_r)
    hi = torch.maximum(torch.zeros_like(a_r), a_r)
    outside = torch.relu(lo - a_m) + torch.relu(a_m - hi)
    return outside ** 2


def _fan_area_subset(pad_sub):
    """(B,) 부채꼴 넓이. pad_sub (B,K,3), 0번=엄지 꼭짓점. K>=3."""
    t = pad_sub[:, 0]
    area = None
    for i in range(1, pad_sub.shape[1] - 1):
        u = pad_sub[:, i] - t
        v = pad_sub[:, i + 1] - t
        tri = 0.5 * torch.linalg.cross(u, v, dim=-1).norm(dim=-1)
        area = tri if area is None else area + tri
    return area


def objective(q, wp, wr, fk, cap, lo, hi, detail=False, pad_cap=None, z_frac=None,
              active=(0, 1, 2, 3), palm_down_deg=None):
    """(B,) 손실. 작을수록 좋다. q(B,16) wp(B,3) wr(B,3).

    active = 벽에 밀착시킬 손가락 인덱스(0엄지 1검지 2중지 3약지). 작은 캡엔 수를
    줄여 핀치. 비활성 손끝은 벽 바깥으로 물러나 간섭하지 않게 한다(엄지는 항상 포함).
    pad_cap: 패드를 (r-press) 로 눌러 밀착. z_frac: 접촉 높이 목표(바닥기준 프랙션).
    """
    pc = pad_cap if pad_cap is not None else cap
    act = list(active)
    inact = [i for i in range(4) if i not in act]
    tp, td, links, caps = to_cap_frame(q, wp, wr, fk)
    pad = pad_points(tp, td)

    # 활성 손끝만 벽 밀착
    pad_err = pc.dist2(pad[:, act]).mean(dim=-1)

    # 접촉 높이 목표 (활성만)
    height_err = torch.zeros_like(pad_err)
    if z_frac is not None:
        z_target = z_frac * cap.height
        height_err = ((pad[:, act, 2] - z_target) ** 2).mean(dim=-1)

    cos = tip_cos_inward(tp, td)
    facing = ((1.0 - cos[:, act]) ** 2).mean(dim=-1)

    # 비활성 손가락: 관절을 중립(관절범위 중앙)으로. pen/selfc 가 충돌만 막고
    # 나머지는 결정적 중립 자세 -- 과소결정(임의 극단 자세) 방지.
    rest = torch.zeros_like(pad_err)
    if inact and lo is not None and hi is not None:
        q_mid = 0.5 * (lo + hi)
        idx = [d for f in inact for d in FINGER_DOF[f]]
        rest = ((q[:, idx] - q_mid[idx]) ** 2).mean(dim=-1)

    under = torch.zeros_like(facing)
    if links:
        deep = [(-p[..., 2]).clamp_min(0.0).amax(dim=-1) for p in links.values()]
        under = (torch.stack(deep, dim=-1).amax(dim=-1)) ** 2

    # 접촉 다각형 넓이: 활성 3개 이상일 때만 (엄지 꼭짓점 부채꼴). 정n각 근사 기준.
    area_def = torch.zeros_like(facing)
    if len(act) >= 3:
        area = _fan_area_subset(pad[:, act])
        n = len(act)
        area_ref = 0.5 * n * cap.radius ** 2 * math.sin(2 * math.pi / n)
        area_def = ((area_ref - area).clamp_min(0.0) / area_ref) ** 2

    pen = torch.zeros_like(facing)
    selfc = torch.zeros_like(facing)
    if links:
        # tip 은 벽에 닿는 링크라 침투검사에서 뺀다(패드가 경계에 걸리는 게 정상).
        allp = torch.cat([p for n, p in links.items() if not n.endswith("_tip")], dim=1)
        rr = torch.linalg.norm(allp[..., :2], dim=-1)
        depth = torch.minimum((cap.radius - rr).clamp_min(0.0),
                              (cap.height - allp[..., 2]).clamp_min(0.0))
        pen = ((depth - PEN_TOL).clamp_min(0.0).amax(dim=-1)) ** 2

    if caps:
        by_f = {}
        for name, ends in caps.items():
            # base_link 은 손가락 그룹에서 제외
            if name == "base_link":
                continue
            # allegro 링크명: link_<idx>.0[_tip] -> 손가락 = idx//4 (0 index,1 mid,2 ring,3 thumb)
            idx = int(name.split("_")[1].split(".")[0])
            grp = idx // 4
            by_f.setdefault(grp, []).append((ends, fk.capsules[name][1]))
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
        if acc is not None:
            selfc = acc

    # 검지-중지-약지 순서 유지 (셋 다 활성일 때만; 2개면 교차 불가)
    order = torch.zeros_like(facing)
    if 1 in act and 2 in act and 3 in act:
        order = order_penalty(pad)

    # 손등 ∥ 바닥: 손바닥(손목 +x)이 아래(-z)를 향하게. 학습 CAP_WRIST_PALM_MAX 규약.
    # 아래방향에서 palm_down_deg 초과한 만큼만 벌한다(그 안은 자유).
    palm_down = torch.zeros_like(pad_err)
    if palm_down_deg is not None:
        Rw = axis_angle_to_matrix(wr)
        cos_down = (-Rw[:, 2, 0]).clamp(-1.0, 1.0)     # 손바닥 x축의 -z 성분
        ang = torch.arccos(cos_down)
        palm_down = (ang - math.radians(palm_down_deg)).clamp_min(0.0) ** 2

    # tip +z(팁 축, 뷰어 파란선)를 월드 -z(바닥)로. 길쭉한 실리콘 팁을 캡 벽 따라
    # 세로로 눕혀 침투↓·면접촉↑. TIP_DOWN_W>0 일 때만.
    tip_down = torch.zeros_like(pad_err)
    if TIP_DOWN_W > 0:
        Rw = axis_angle_to_matrix(wr)
        Tl = fk.all_link_transforms(q)
        zloc = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
        accd = []
        for i in act:
            name = TIP_LINKS[i]
            if name not in Tl:
                continue
            zt_w = torch.einsum("bij,bj->bi", Rw, Tl[name][:, :3, :3] @ zloc)
            accd.append((1.0 + zt_w[:, 2]) ** 2)      # z=-1(바닥) 이면 0
        if accd:
            tip_down = torch.stack(accd, dim=-1).mean(dim=-1)

    total = (1e4 * pad_err + 1.0 * facing + 1e5 * under + 1.0 * area_def
             + 1e5 * pen + 1e5 * selfc + 10.0 * order + 1e4 * height_err
             + 10.0 * rest + 1e2 * palm_down + TIP_DOWN_W * tip_down)
    if detail:
        return total, {"pad": pad_err, "facing": facing, "under": under,
                       "area": area_def, "pen": pen, "selfc": selfc, "order": order,
                       "height": height_err, "rest": rest, "palmdn": palm_down,
                       "tipdown": tip_down}
    return total


def main():
    global FINGER_RADIUS, TIP_DOWN_W
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--radius", type=float, default=0.050)
    ap.add_argument("-H", "--height", type=float, default=0.017)
    ap.add_argument("-n", "--restarts", type=int, default=96)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-pts", type=int, default=200, help="링크당 표면점 수")
    ap.add_argument("--press", type=float, default=0.0015,
                    help="패드를 벽 안쪽으로 누르는 깊이(m). 클수록 밀착. 기본 1.5mm")
    ap.add_argument("--z-frac", type=float, default=None,
                    help="패드 접촉 높이 목표(바닥기준 프랙션). 예 0.6 = 위에서 40%%. 미지정시 자유")
    ap.add_argument("--fingers", default="thumb,index,middle,ring",
                    help="벽에 밀착시킬 손가락(콤마). 작은 캡엔 줄임. 예 thumb,index,middle")
    ap.add_argument("--palm-down-deg", type=float, default=20.0,
                    help="손등 ∥ 바닥: 손바닥이 아래방향에서 벗어남 허용각(도). 학습 40 규약. None으로 끔")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finger-radius", type=float, default=FINGER_RADIUS,
                    help="손끝 접촉 오프셋(m). 표준팁=0.012, 실리콘팁(+x실효)=0.0105")
    ap.add_argument("--tip-down-w", type=float, default=0.0,
                    help="팁 +z 를 바닥(-z)으로 향하게 하는 가중치(예 100). 0=끔")
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    FINGER_RADIUS = args.finger_radius          # 패드 접촉 FR 오버라이드(pad_points 가 읽음)
    TIP_DOWN_W = args.tip_down_w                 # 팁 축 바닥 지향 가중치
    _fmap = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "엄지": 0, "검지": 1, "중지": 2, "약지": 3}
    active = tuple(sorted({_fmap[s.strip()] for s in args.fingers.split(",") if s.strip()}))
    if 0 not in active:
        active = (0,) + active  # 엄지는 항상 포함(대향)

    from maniptrans_dof_order_allegro import DOF_LIMITS, DOF_NAMES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cap = CapShape(args.radius, args.height)
    # 패드 목표는 벽보다 press 만큼 안쪽 -> 밀착(press). pen/under 는 실제 cap 기준.
    pad_cap = CapShape(max(args.radius - args.press, 1e-3), args.height)
    fk = HandFK(dev)
    fk.link_pts = fk.link_points(args.n_pts)
    lo = torch.tensor([DOF_LIMITS[n][0] for n in DOF_NAMES], device=dev)
    hi = torch.tensor([DOF_LIMITS[n][1] for n in DOF_NAMES], device=dev)

    B = args.restarts
    q = (lo + (hi - lo) * torch.rand(B, 16, device=dev)).requires_grad_(True)
    th = torch.rand(B, device=dev) * 2 * math.pi
    wp = torch.stack([
        (cap.radius + 0.09) * torch.cos(th),
        (cap.radius + 0.09) * torch.sin(th),
        torch.full((B,), cap.height + 0.035, device=dev),
    ], dim=-1).requires_grad_(True)
    wr = (torch.rand(B, 3, device=dev) - 0.5).requires_grad_(True)

    optim = torch.optim.Adam([q, wp, wr], lr=args.lr)
    print(f"[allegro] r {cap.radius*1000:.1f}mm  h {cap.height*1000:.1f}mm   "
          f"재시작 {B}개  {args.iters}회  링크당 {args.n_pts}점  tip반경 {fk._tip_r*1000:.1f}mm  {dev}")
    print("손목 22차원 전부 자유, z >= 0 (원기둥 밑면) 만 강제")
    for it in range(args.iters):
        optim.zero_grad()
        loss = objective(q, wp, wr, fk, cap, lo, hi, pad_cap=pad_cap, z_frac=args.z_frac, active=active, palm_down_deg=args.palm_down_deg)
        if loss.requires_grad:
            loss.sum().backward()
            optim.step()
            with torch.no_grad():
                q.clamp_(lo, hi)
        if it % 300 == 0 or it == args.iters - 1:
            print(f"  {it:5d}  최소 {float(loss.min()):.6f}  중앙 {float(loss.median()):.6f}")

    with torch.no_grad():
        total, det = objective(q, wp, wr, fk, cap, lo, hi, detail=True, pad_cap=pad_cap, z_frac=args.z_frac, active=active, palm_down_deg=args.palm_down_deg)
        i = int(torch.argmin(total))
        print(f"\n최적 #{i}  총 {float(total[i]):.6f}")
        print("  " + "   ".join(f"{k} {float(v[i]):.6f}" for k, v in det.items()))
        tp, td, _, _ = to_cap_frame(q, wp, wr, fk)
        A = float(contact_fan_area(pad_points(tp, td))[i]) * 1e6
        A_ref = 2.0 * cap.radius ** 2 * 1e6
        print(f"  접촉 다각형 넓이 {A:.0f}mm^2  (정사각 기준 {A_ref:.0f}, {100*A/A_ref:.0f}%)")
        cos = tip_cos_inward(tp, td)[i]
        az = torch.rad2deg(torch.atan2(tp[i, :, 1], tp[i, :, 0]))
        pad = pad_points(tp, td)[i]
        dd = cap.dist2(pad).sqrt()
        print("\n  손가락   패드-벽mm   cos    방위각도   패드높이mm")
        for k in range(4):
            print("  %-6s  %+8.2f  %+6.2f  %+8.1f  %9.1f"
                  % (FINGERS[k], float(dd[k]) * 1000, float(cos[k]),
                     float(az[k]), float(pad[k, 2]) * 1000))

        if args.out:
            from scipy.spatial.transform import Rotation

            from reference_pose import GraspInitPose

            R = axis_angle_to_matrix(wr[i:i + 1])[0].cpu().numpy()
            pose = GraspInitPose(
                dof_pos=q[i].detach().cpu(),
                wrist_rel_pos=wp[i].detach().cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0]),
                wrist_quat=torch.tensor(Rotation.from_matrix(R).as_quat(), dtype=torch.float32),
                note=f"allegro cyl r={cap.radius*1000:.0f} h={cap.height*1000:.0f} loss={float(total[i]):.6f}",
            )
            out = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(__file__), args.out)
            pose.save(out)
            print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
