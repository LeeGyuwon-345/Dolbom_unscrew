"""베이스라인: (r,h) 격자에 옵티마이저 자세를 미리 풀어 저장 + 쌍선형 보간.

입력 2D·매끄러운 매핑이라, 격자 룩업+보간이 MLP 만큼 정확하고 더 안정적일 수
있다. 가장 단순한 기준선. amortized(train_amortized) 와 품질을 비교하는 용도로도.

★ 방위각 정규화: 옵티마이저는 (r,h)마다 랜덤 방위각을 내므로, 저장 전에 각 자세를
   손목 azimuth=0 으로 회전(_canon)해야 보간이 성립한다 (안 그러면 이웃 격자끼리
   방위각이 달라 손목 보간이 엉킨다).

    python lookup.py build --nr 9 --nh 6      # 격자 풀어 table.pt 저장 (느림)
    python lookup.py show 0.05 0.017          # 보간 자세 항별 확인
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/leegyuwon/Documents/task1/grasp_init")
import optimize_pose_allegro as OP  # noqa: E402
from maniptrans_dof_order_allegro import DOF_LIMITS, DOF_NAMES  # noqa: E402
from cap_shape import CapShape  # noqa: E402

from train_amortized import H_RANGE, R_RANGE  # 같은 범위 공유  # noqa: E402
from objective_batched import batched_objective  # noqa: E402
from model import matrix_to_quat_xyzw  # noqa: E402

TABLE = os.path.join(HERE, "lookup_table.pt")


def _canon(q, wp, R):
    """손목 azimuth=0 으로 회전(원통 z축). dof 그대로. 반환 (wp',R')."""
    az = math.atan2(float(wp[1]), float(wp[0]))
    c, s = math.cos(-az), math.sin(-az)
    Rz = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=wp.dtype)
    return Rz @ wp, Rz @ R


def _solve_one(r, h, dev, restarts=64, iters=1500, press=0.0007):
    """옵티마이저 1회 (grasp_init 재사용) -> 정규화된 (q, wp, R)."""
    from optimize_pose_allegro import (HandFK, axis_angle_to_matrix, objective,
                                       to_cap_frame)
    fk = HandFK(dev); fk.link_pts = fk.link_points(200)
    lo = torch.tensor([DOF_LIMITS[n][0] for n in DOF_NAMES], device=dev)
    hi = torch.tensor([DOF_LIMITS[n][1] for n in DOF_NAMES], device=dev)
    cap = CapShape(r, h); pad_cap = CapShape(max(r - press, 1e-3), h)
    B = restarts
    q = (lo + (hi - lo) * torch.rand(B, 16, device=dev)).requires_grad_(True)
    th = torch.rand(B, device=dev) * 2 * math.pi
    wp = torch.stack([(r + 0.09) * torch.cos(th), (r + 0.09) * torch.sin(th),
                      torch.full((B,), h + 0.035, device=dev)], -1).requires_grad_(True)
    wr = (torch.rand(B, 3, device=dev) - 0.5).requires_grad_(True)
    opt = torch.optim.Adam([q, wp, wr], lr=0.02)
    for _ in range(iters):
        opt.zero_grad()
        loss = objective(q, wp, wr, fk, cap, lo, hi, pad_cap=pad_cap)
        loss.sum().backward(); opt.step()
        with torch.no_grad():
            q.clamp_(lo, hi)
    with torch.no_grad():
        tot = objective(q, wp, wr, fk, cap, lo, hi, pad_cap=pad_cap)
        i = int(tot.argmin())
        R = axis_angle_to_matrix(wr[i:i + 1])[0]
        wpi, Ri = _canon(q[i].cpu(), wp[i].cpu(), R.cpu())
        return q[i].cpu(), wpi, Ri, float(tot[i])


def build(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rs = torch.linspace(R_RANGE[0], R_RANGE[1], args.nr)
    hs = torch.linspace(H_RANGE[0], H_RANGE[1], args.nh)
    Q = torch.zeros(args.nr, args.nh, 16)
    WP = torch.zeros(args.nr, args.nh, 3)
    RR = torch.zeros(args.nr, args.nh, 3, 3)
    for i, r in enumerate(rs):
        for j, h in enumerate(hs):
            q, wp, R, loss = _solve_one(float(r), float(h), dev,
                                        restarts=args.restarts, iters=args.iters)
            Q[i, j], WP[i, j], RR[i, j] = q, wp, R
            print(f"  r{float(r)*1000:.0f} h{float(h)*1000:.0f}  loss {loss:.4f}", flush=True)
    torch.save({"rs": rs, "hs": hs, "Q": Q, "WP": WP, "RR": RR}, TABLE)
    print("저장:", TABLE)


def lookup(r, h):
    """쌍선형 보간 -> (q(16), wp(3), R(3,3)). 회전은 보간 후 재정규화(SVD 대신 Gram-Schmidt)."""
    t = torch.load(TABLE)
    rs, hs = t["rs"], t["hs"]
    ir = torch.clamp(torch.searchsorted(rs, torch.tensor(r)) - 1, 0, len(rs) - 2)
    ih = torch.clamp(torch.searchsorted(hs, torch.tensor(h)) - 1, 0, len(hs) - 2)
    fr = ((r - rs[ir]) / (rs[ir + 1] - rs[ir])).clamp(0, 1)
    fh = ((h - hs[ih]) / (hs[ih + 1] - hs[ih])).clamp(0, 1)
    def bl(A):
        return ((A[ir, ih] * (1 - fr) + A[ir + 1, ih] * fr) * (1 - fh)
                + (A[ir, ih + 1] * (1 - fr) + A[ir + 1, ih + 1] * fr) * fh)
    q, wp, R = bl(t["Q"]), bl(t["WP"]), bl(t["RR"])
    from model import rot6d_to_matrix
    R = rot6d_to_matrix(torch.cat([R[:, 0], R[:, 1]])[None])[0]  # 재정규화
    return q, wp, R


def show(args):
    q, wp, R = lookup(args.rh[0], args.rh[1])
    print("dof:", [round(float(x), 3) for x in q])
    print("wrist_rel_pos(mm):", [round(float(x) * 1000, 1) for x in wp])
    print("quat xyzw:", [round(float(x), 3) for x in matrix_to_quat_xyzw(R)])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--nr", type=int, default=9)
    b.add_argument("--nh", type=int, default=6); b.add_argument("--restarts", type=int, default=64)
    b.add_argument("--iters", type=int, default=1500)
    s = sub.add_parser("show"); s.add_argument("rh", type=float, nargs=2)
    args = ap.parse_args()
    (build if args.cmd == "build" else show)(args)


if __name__ == "__main__":
    main()
