"""배치 objective: 샘플마다 다른 (r,h) 에 대해 파지 손실을 미분가능하게 준다.

grasp_init/optimize_pose_allegro 의 HandFK 와 기하 헬퍼(pad/facing/fan/자기충돌)를
재사용하되, cap 에 의존하는 항(pad_err, area_ref, pen)만 per-sample r,h 로 배치화한다.
회전은 6D->행렬로 만든 R 을 그대로 받는다(axis-angle 왕복 없음).

amortized 학습(train_amortized.py)의 손실로 쓴다 -- MLP 가 낸 자세에 이 objective 를
걸어 최소화하면, MLP 가 옵티마이저를 학습으로 흉내낸다.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/home/leegyuwon/Documents/task1/grasp_init")
import optimize_pose_allegro as OP  # noqa: E402

FR = OP.FINGER_RADIUS
PEN_TOL = OP.PEN_TOL
MIN_GAP = OP.MIN_GAP_LINK


def to_cap_frame_R(q, wp, R, fk):
    """q(B,16) wp(B,3) R(B,3,3) -> tp,td (B,4,3), links/caps {name:(B,·,3)} 원통좌표."""
    tp_l, td_l = fk.tips(q)                                   # (B,4,3) 손목프레임
    tp = torch.einsum("bij,bkj->bki", R, tp_l) + wp[:, None]
    td = torch.einsum("bij,bkj->bki", R, td_l)
    td = td / td.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    T = fk.all_link_transforms(q)                            # {name:(B,4,4)}
    links, caps = {}, {}
    for name, pl in fk.link_pts.items():
        if name not in T:
            continue
        M = T[name]
        p_l = torch.einsum("bij,nj->bni", M[:, :3, :3], pl) + M[:, None, :3, 3]
        links[name] = torch.einsum("bij,bnj->bni", R, p_l) + wp[:, None]
    for name, (ends, _) in fk.capsules.items():
        if name not in T:
            continue
        M = T[name]
        e = ends.to(q)
        p_l = torch.einsum("bij,nj->bni", M[:, :3, :3], e) + M[:, None, :3, 3]
        caps[name] = torch.einsum("bij,bnj->bni", R, p_l) + wp[:, None]
    return tp, td, links, caps, T


def active_mask_for_r(r):
    """(B,) r[m] -> (B,4) 활성 손가락 마스크. 작을수록 손가락 수↓ (핀치).
    엄지·검지 항상, 중지 r>=25mm, 약지 r>=40mm."""
    B = r.shape[0]
    m = torch.zeros(B, 4, dtype=torch.bool, device=r.device)
    m[:, 0] = True                 # 엄지
    m[:, 1] = True                 # 검지
    m[:, 2] = r >= 0.025           # 중지
    m[:, 3] = r >= 0.040           # 약지
    return m


def batched_objective(q, wp, R, r, h, fk, lo=None, hi=None, press=0.0007, z_frac=None,
                      palm_down_deg=None, finger_radius=None, tip_down_w=0.0,
                      flex_max_deg=None, flex_w=30.0, palm_gap_min=None, palm_gap_w=800.0,
                      wrist_tilt_ref=None, wrist_tilt_deg=17.0, wrist_tilt_w=1e2,
                      wrist_R_ref=None, wrist_align_w=1e2, mid_extra_press=0.0,
                      palm_abs_min=None, palm_abs_w=0.0, cap_top_z=0.231, cap_frame_z0=0.013,
                      oppose_min_deg=None, oppose_w=0.0, tipz_target=None, tipz_w=0.0,
                      tip_pad_back=0.0,
                      detail=False):
    """(B,) 손실. r,h (B,). r 에 따라 활성 손가락 수 자동(active_mask_for_r).
    활성만 밀착/facing/높이/순서/넓이, 비활성은 중립(rest). press/z_frac 지원.
    finger_radius: 접촉 FR(기본 OP.FINGER_RADIUS). tip_down_w>0: 팁+z 를 바닥(-z)으로."""
    B = q.shape[0]
    FRv = FR if finger_radius is None else finger_radius
    tp, td, links, caps, Tl = to_cap_frame_R(q, wp, R, fk)
    pad = tp + td * FRv                                      # (B,4,3)
    if tip_pad_back:
        # 실리콘 팁 평평한 pad 면 중심은 팁원점보다 팜쪽(-z 로컬)에 있다.
        zloc = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
        back = []
        for i in range(4):
            nm = OP.TIP_LINKS[i]
            zw = (torch.einsum("bij,bj->bi", R, Tl[nm][:, :3, :3] @ zloc) if nm in Tl
                  else torch.zeros_like(pad[:, i]))
            back.append(zw)
        pad = pad - torch.stack(back, dim=1) * tip_pad_back
    r_pad = (r - press).clamp_min(1e-3)[:, None].repeat(1, 4)  # (B,4) 손가락별
    if mid_extra_press:                                        # 중지(2) 추가 press -> 더 밀착
        r_pad[:, 2] = (r - press - mid_extra_press).clamp_min(1e-3)

    am = active_mask_for_r(r).float()                        # (B,4)
    cnt = am.sum(-1).clamp_min(1.0)

    # pad_err (활성 마스크): 패드를 (r-press) 벽에 -- 반경오차 + z밴드
    rad = pad[..., :2].norm(dim=-1)                          # (B,4)
    d_r = rad - r_pad
    d_z = pad[..., 2] - pad[..., 2].clamp(torch.zeros_like(h)[:, None], h[:, None])
    pf = d_r ** 2 + d_z ** 2
    pad_err = (pf * am).sum(-1) / cnt

    # facing (활성)
    u = pad[..., :2] / rad.clamp_min(1e-9)[..., None]
    inward = torch.cat([-u, torch.zeros_like(pad[..., 2:3])], dim=-1)
    cos = (td * inward).sum(-1)
    facing = (((1.0 - cos) ** 2) * am).sum(-1) / cnt

    # height (활성): 패드 z 를 z_frac*h 로
    height_err = torch.zeros(B, device=q.device)
    if z_frac is not None:
        zt = z_frac * h
        height_err = (((pad[..., 2] - zt[:, None]) ** 2) * am).sum(-1) / cnt

    # rest: 비활성 손가락 관절 -> 중립(관절범위 중앙)
    rest = torch.zeros(B, device=q.device)
    if lo is not None and hi is not None:
        q_mid = 0.5 * (lo + hi)
        dofmask = torch.zeros(B, 16, device=q.device)
        inact = (am < 0.5)                                   # (B,4)
        for f in range(4):
            for dd in OP.FINGER_DOF[f]:
                dofmask[:, dd] = inact[:, f].float()
        rest = (((q - q_mid) ** 2) * dofmask).sum(-1) / dofmask.sum(-1).clamp_min(1.0)

    # under: 밑면 아래 침범 (r,h 무관)
    under = torch.zeros(B, device=q.device)
    if links:
        deep = [(-p[..., 2]).clamp_min(0.0).amax(dim=-1) for p in links.values()]
        under = (torch.stack(deep, dim=-1).amax(dim=-1)) ** 2

    # area: 활성 4개(약지까지)일 때만 정사각 기준으로 대향/펼침 유도
    area = OP.contact_fan_area(pad)
    area_ref = 2.0 * r ** 2
    area_def = ((area_ref - area).clamp_min(0.0) / area_ref.clamp_min(1e-9)) ** 2
    area_def = area_def * (am[:, 2] * am[:, 3])              # 중지·약지 모두 활성일 때만

    # pen: tip 제외 링크가 원통 고체(반경<=r, z<=h) 침투 -- 최대깊이
    pen = torch.zeros(B, device=q.device)
    if links:
        allp = torch.cat([p for n, p in links.items() if not n.endswith("_tip")], dim=1)
        rr = allp[..., :2].norm(dim=-1)
        depth = torch.minimum((r[:, None] - rr).clamp_min(0.0),
                              (h[:, None] - allp[..., 2]).clamp_min(0.0))
        pen = ((depth - PEN_TOL).clamp_min(0.0).amax(dim=-1)) ** 2

    # selfc: 손가락 캡슐끼리 (r,h 무관). optimize_pose_allegro 와 동일 로직.
    selfc = torch.zeros(B, device=q.device)
    if caps:
        by_f = {}
        for name, ends in caps.items():
            if name == "base_link":
                continue
            idx = int(name.split("_")[1].split(".")[0])
            by_f.setdefault(idx // 4, []).append((ends, fk.capsules[name][1]))
        packs = []
        for k in sorted(by_f):
            e = torch.stack([x[0] for x in by_f[k]], dim=1)      # (B,S,2,3)
            rr = torch.tensor([x[1] for x in by_f[k]], device=q.device, dtype=q.dtype)
            packs.append((e, rr))
        acc = None
        for a in range(len(packs)):
            for b in range(a + 1, len(packs)):
                ea, ra = packs[a]
                eb, rb = packs[b]
                dseg = OP.seg_seg_dist(ea[:, :, None, 0], ea[:, :, None, 1],
                                       eb[:, None, :, 0], eb[:, None, :, 1])
                surf = dseg - (ra[None, :, None] + rb[None, None, :])
                viol = ((MIN_GAP - surf.amin(dim=(-2, -1))).clamp_min(0.0)) ** 2
                acc = viol if acc is None else acc + viol
        if acc is not None:
            selfc = acc

    # order: 검지-중지-약지 셋 다 활성일 때만 (중지·약지 활성)
    order = OP.order_penalty(pad) * (am[:, 2] * am[:, 3])

    # 손등 ∥ 바닥: 손바닥(손목 +x = R 첫 열)이 아래(-z) 향하게. R[:,2,0]=손바닥 x축의 z성분.
    palm_down = torch.zeros(B, device=q.device)
    if palm_down_deg is not None:
        import math as _m
        cos_down = (-R[:, 2, 0]).clamp(-1.0, 1.0)
        ang = torch.arccos(cos_down)
        palm_down = (ang - _m.radians(palm_down_deg)).clamp_min(0.0) ** 2

    # tip +z(팁 축)를 월드 -z(바닥)로. 길쭉한 실리콘 팁을 캡 벽 따라 세로로 눕혀 침투↓.
    tip_down = torch.zeros(B, device=q.device)
    if tip_down_w > 0:
        zloc = torch.tensor([0.0, 0.0, 1.0], device=q.device, dtype=q.dtype)
        accd = []
        for i in range(4):
            name = OP.TIP_LINKS[i]
            if name not in Tl:
                continue
            zt_w = torch.einsum("bij,bj->bi", R, Tl[name][:, :3, :3] @ zloc)
            accd.append(((1.0 + zt_w[:, 2]) ** 2) * am[:, i])
        if accd:
            tip_down = torch.stack(accd, dim=-1).sum(-1) / cnt

    # 굴곡합 상한: 활성 손가락 굴곡조인트(외전 J0 제외) 합 <= flex_max_deg.
    flex = torch.zeros(B, device=q.device)
    if flex_max_deg is not None:
        import math as _mf
        acc = torch.zeros(B, device=q.device)
        for f in range(4):
            fl = OP.FINGER_DOF[f][1:]                      # 각 손가락 굴곡 3개
            acc = acc + q[:, fl].sum(-1) * am[:, f]
        flex = torch.relu(acc - _mf.radians(flex_max_deg)) ** 2

    # palm-캡상단 간격 하한: 손목z - 캡top(h) >= palm_gap_min (손 폄+아래 도달).
    palm_gap = torch.zeros(B, device=q.device)
    if palm_gap_min is not None:
        gap = wp[:, 2] - h
        palm_gap = torch.relu(palm_gap_min - gap) ** 2

    # 손목 기울기: 팜 법선(손목+x=R[:,:,0]) 월드방향을 검증 자세 기준 ±wrist_tilt_deg 이내로.
    wrist_tilt = torch.zeros(B, device=q.device)
    if wrist_tilt_ref is not None:
        import math as _mw
        ref = torch.as_tensor(wrist_tilt_ref, device=q.device, dtype=q.dtype)
        ref = ref / ref.norm().clamp_min(1e-9)
        cosr = (R[:, :, 0] * ref).sum(-1)
        # arccos 는 |cos|->1 에서 기울기 무한대 -> relu' 0 과 곱해져 0*inf=NaN
        # (학습 중 간헐 폭사 원인). cos 영역에서 직접 벌점: cosr < cos(허용각) 일 때만.
        wrist_tilt = torch.relu(_mw.cos(_mw.radians(wrist_tilt_deg)) - cosr) ** 2

    # 손목 회전행렬 정렬: 정규(az0) 손목 R 를 검증 자세 R_ref 에. ||R - R_ref||_F^2.
    # 팜 법선(wtilt)으론 못 잡는 '법선 주위 회전'까지 맞춘다.
    wrist_align = torch.zeros(B, device=q.device)
    if wrist_R_ref is not None:
        Rref = torch.as_tensor(wrist_R_ref, device=q.device, dtype=q.dtype).reshape(3, 3)
        wrist_align = ((R - Rref) ** 2).sum(dim=(-2, -1))

    # === pose_gate 정합 항 (게이트와 동일 정의) =========================
    # 3 팜높이: palm(base_link) z - 캡윗면(231mm) >= 125mm. 월드 z = 캡원점z + CAP_FRAME_Z0 + wp_z.
    #   (파지 프레임 z=0 은 캡 바닥, 월드 캡윗면 = cap_top_z 로 고정 판정)
    palm_abs = torch.zeros(B, device=q.device)
    if palm_abs_min is not None and palm_abs_w > 0:
        palm_world_z = (cap_top_z - h) + wp[:, 2]   # 캡바닥 월드 = 캡윗면-h, 그 위 wp_z
        palm_abs = torch.relu(palm_abs_min - (palm_world_z - cap_top_z)) ** 2

    # 6 엄지대향: 엄지-중지 팁 방위각 간격 >= oppose_min_deg. arccos 금지 -> cos 영역 벌점.
    oppose = torch.zeros(B, device=q.device)
    if oppose_min_deg is not None and oppose_w > 0:
        import math as _mo
        u = pad[..., :2] / pad[..., :2].norm(dim=-1, keepdim=True).clamp_min(1e-9)  # (B,4,2)
        cos_tm = (u[:, 0] * u[:, 2]).sum(-1).clamp(-1.0, 1.0)      # 엄지(0)-중지(2)
        oppose = torch.relu(cos_tm - _mo.cos(_mo.radians(oppose_min_deg))) ** 2 * am[:, 2]

    # 1 팁z 밴드: 팁 중심 z 를 목표(밴드 중앙)로. 월드 z 기준.
    tipz = torch.zeros(B, device=q.device)
    if tipz_target is not None and tipz_w > 0:
        tip_world_z = (cap_top_z - h)[:, None] + tp[..., 2]
        tipz = (((tip_world_z - tipz_target) ** 2) * am).sum(-1) / cnt

    total = (1e5 * pad_err + 1.0 * facing + 1e5 * under + 1.0 * area_def
             + 1e5 * pen + 1e5 * selfc + 10.0 * order + 1e4 * height_err + 10.0 * rest
             + 1e2 * palm_down + tip_down_w * tip_down + flex_w * flex + palm_gap_w * palm_gap
             + wrist_tilt_w * wrist_tilt + wrist_align_w * wrist_align
             + palm_abs_w * palm_abs + oppose_w * oppose + tipz_w * tipz)
    if detail:
        return total, {"pad": pad_err, "facing": facing, "under": under, "area": area_def,
                       "pen": pen, "selfc": selfc, "order": order, "height": height_err,
                       "rest": rest, "palmdn": palm_down, "tipdown": tip_down,
                       "flex": flex, "palmgap": palm_gap, "wtilt": wrist_tilt,
                       "walign": wrist_align, "palmabs": palm_abs, "oppose": oppose,
                       "tipz": tipz}
    return total
