"""amortized RB5 IK 학습: (r,h,캡배치z) -> RB5 6관절+방위각 네트워크.

손 MLP(고정)가 낸 손목자세(캡프레임)를, RB5Net 이 낸 방위각으로 회전해 월드 손목을
만들고, 그 tcp 타겟을 RB5Net 의 q 가 FK 로 맞추도록(위치·자세) + 엘보우업 + base방향
을 미분가능 손실로 최소화한다. solve_rb5_ik 와 동일한 목적, 배치 학습판.

    python train_rb5.py --iters 8000
    python train_rb5.py --export 0.05 0.03 0.225   # (r,h,z) -> RB5 즉시 추론
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

import rb5_ik as K  # noqa: E402
import train_amortized as T  # noqa: E402 (손 MLP build + 범위)
from cap_shape import CAP_FRAME_Z0  # noqa: E402
from model import rotate_pose  # noqa: E402
from rb5_mlp import RB5Net  # noqa: E402

Z_RANGE = (0.170, 0.260)         # 캡 월드 배치높이(m) -- 내부(실제) 기준
# 텀블러 31mm 상승: 입력 z 는 기존 규약(예 0.202) 그대로 쓰고, 내부에서 +31mm 적용.
CAP_Z_OFFSET = 0.031
CAP_XY = (0.4, 0.0)              # 텀블러 수평위치(고정) -- RL 텀블러 x=0.4
CKPT = os.path.join(HERE, "rb5_mlp.pt")


def _norm3(r, h, z):
    rn = (r - T.R_RANGE[0]) / (T.R_RANGE[1] - T.R_RANGE[0])
    hn = (h - T.H_RANGE[0]) / (T.H_RANGE[1] - T.H_RANGE[0])
    zn = (z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
    return torch.stack([rn, hn, zn], dim=-1)


def _to(dev):
    """rb5 체인/마운트/한계를 dev 로 이동."""
    K._init()
    K._CHAIN = K._CHAIN.to(dtype=torch.float32, device=dev)
    K._LO = K._LO.to(dev); K._HI = K._HI.to(dev)
    return K._R_M.to(dev), K._T_M.to(dev)


def _loss(th, q, wp, R, z, R_M, T_M, base_w, dev, detail=False):
    """배치 IK 손실. wp,R = 손 MLP 손목(캡프레임, 방위각0). z (B,) 캡배치높이."""
    B = q.shape[0]
    wp_t, R_t = rotate_pose(wp, R, th)                       # 방위각 적용
    cap_w = torch.stack([torch.full_like(z, CAP_XY[0]),
                         torch.full_like(z, CAP_XY[1]), z], dim=-1)   # (B,3)
    p_bl = cap_w + wp_t + torch.tensor([0.0, 0.0, CAP_FRAME_Z0], device=dev)
    R_bl = R_t
    R_tcp = R_bl @ R_M.T                                     # (B,3,3)
    p_tcp = p_bl - torch.einsum("bij,j->bi", R_tcp, T_M)

    fk = K._CHAIN.forward_kinematics(q, end_only=False)
    m = fk["tcp"].get_matrix()                              # (B,4,4)
    pos = ((m[:, :3, 3] - p_tcp) ** 2).sum(-1)             # (B,)
    rot = ((m[:, :3, :3] - R_tcp) ** 2).sum((-1, -2))
    tcpz = m[:, 2, 3]
    l3z = fk["link3"].get_matrix()[:, 2, 3]
    elbow = torch.relu(tcpz + 0.05 - l3z) ** 2             # 엘보우업(link3>tcp)
    floor = sum(torch.relu(0.15 - fk[l].get_matrix()[:, 2, 3]) ** 2
                for l in ("link3", "link4", "link5"))
    xy = wp_t[:, :2]
    align = -xy[:, 0] / xy.norm(dim=-1).clamp_min(1e-9)     # base(−x) 정렬 (1=base쪽)
    basedir = (1.0 - align)
    total = pos * 4.0 + rot * 0.5 + elbow * 60.0 + floor * 50.0 + basedir * base_w
    if detail:
        return total, dict(pos=pos, rot=rot, elbow=elbow, floor=floor, base=basedir)
    return total


def _grasp(dev):
    fk_g, gmlp, lo, hi = T.build(dev)
    gmlp.load_state_dict(torch.load(T.CKPT, map_location=dev)); gmlp.eval()
    return gmlp


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R_M, T_M = _to(dev)
    gmlp = _grasp(dev)
    net = RB5Net().to(dev)
    if args.resume and os.path.exists(CKPT):
        net.load_state_dict(torch.load(CKPT, map_location=dev)); print("resumed", CKPT)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"RB5 amortized 학습  R{T.R_RANGE} H{T.H_RANGE} Z{Z_RANGE}  batch {args.batch}  {args.iters}회  {dev}")
    for it in range(args.iters):
        r = torch.rand(args.batch, device=dev) * (T.R_RANGE[1] - T.R_RANGE[0]) + T.R_RANGE[0]
        h = torch.rand(args.batch, device=dev) * (T.H_RANGE[1] - T.H_RANGE[0]) + T.H_RANGE[0]
        z = torch.rand(args.batch, device=dev) * (Z_RANGE[1] - Z_RANGE[0]) + Z_RANGE[0]
        with torch.no_grad():
            q_h, wp, R = gmlp(T._norm(r, h), r, h)          # 손 MLP 손목(방위각0)
        th, q = net(_norm3(r, h, z))
        loss = _loss(th, q, wp, R, z, R_M, T_M, args.base_w, dev).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            with torch.no_grad():
                _, det = _loss(th, q, wp, R, z, R_M, T_M, args.base_w, dev, detail=True)
                pos_mm = float(det["pos"].clamp_min(0).sqrt().mean()) * 1000
                print(f"  {it:6d}  loss {float(loss):.4f}  위치 {pos_mm:.1f}mm  "
                      + " ".join(f"{k} {float(v.mean()):.4f}" for k, v in det.items()), flush=True)
    torch.save(net.state_dict(), CKPT)
    print("저장:", CKPT)


def export(args):
    import json as _json

    from model import matrix_to_quat_xyzw, rotate_pose as _rp
    dev = "cpu"
    R_M, T_M = _to(dev)
    gmlp = _grasp(dev)
    net = RB5Net().to(dev); net.load_state_dict(torch.load(CKPT, map_location=dev)); net.eval()
    rr, hh, zz_in = args.export
    zz = zz_in + CAP_Z_OFFSET            # 입력(기존규약) -> 실제 캡 월드 z
    r = torch.tensor([rr]); h = torch.tensor([hh]); z = torch.tensor([zz])
    with torch.no_grad():
        q_h, wp, R = gmlp(T._norm(r, h), r, h)
        th, q = net(_norm3(r, h, z))
        _, det = _loss(th, q, wp, R, z, R_M, T_M, args.base_w, dev, detail=True)
        wp_t, R_t = _rp(wp, R, th)
        wrist_rel = wp_t[0] + torch.tensor([0.0, 0.0, CAP_FRAME_Z0])
        quat = matrix_to_quat_xyzw(R_t)[0]
        fk = K._CHAIN.forward_kinematics(q, end_only=False)
    pos_mm = round(float(det["pos"].clamp_min(0).sqrt()[0]) * 1000, 2)
    l3 = float(fk["link3"].get_matrix()[0, 2, 3])
    minz = min(float(fk[l].get_matrix()[0, 2, 3]) for l in ("link3", "link4", "link5"))
    print(f"(r={rr*1000:.0f} h={hh*1000:.0f} z입력={zz_in*1000:.0f}->실제{zz*1000:.0f}mm)  방위각 {math.degrees(float(th[0])):.0f}도"
          f"  위치오차 {pos_mm}mm  엘보우 link3 {l3*1000:.0f}mm")
    print("  RB5 q(도):", [round(math.degrees(float(v)), 1) for v in q[0]])
    if args.out:
        d = {
            "dof_pos": [float(v) for v in q_h[0]],
            "wrist_rel_pos": [float(v) for v in wrist_rel],
            "wrist_quat": [float(v) for v in quat],
            "note": f"allegro cyl r={rr*1000:.0f} h={hh*1000:.0f} (amortized net z={zz*1000:.0f})",
            "rb5": {"names": K._NAMES, "q": [float(v) for v in q[0]],
                    "q_deg": [round(math.degrees(float(v)), 2) for v in q[0]],
                    "pos_err_mm": pos_mm, "rot_err_deg": round(float(det["rot"][0]) ** 0.5 * 57.3, 2),
                    "feasible": bool(minz > 0.12)},
            "azimuth_deg": round(math.degrees(float(th[0])), 1),
            "cap": [CAP_XY[0], CAP_XY[1], float(zz)],
        }
        _json.dump(d, open(args.out, "w"), indent=1)
        print("저장:", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-w", type=float, default=0.02, help="base방향 정렬 가중치")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", type=float, nargs=3, default=None, metavar=("R", "H", "Z"))
    ap.add_argument("-o", "--out", default="", help="뷰어용 json 저장 경로")
    args = ap.parse_args()
    if args.export:
        export(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
