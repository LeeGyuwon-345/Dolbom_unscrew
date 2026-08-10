"""초기자세 노이즈 풀 사전 계산.

기준 자세에 노이즈(관절 std, 손목 위치/자세 std)를 섞은 샘플을 대량 생성하고,
FK 로 캡 침투·자기충돌을 검사해 합격한 것만 저장한다. 리셋 때는 이 풀에서
무작위 인덱싱만 하므로 런타임 FK 의존이 없다.

사용:
    python make_init_pool.py --pose ../tools2/poses/grasp_26d_m05.json \
        --out ../tools2/poses/pool_m05.pt \
        --n 100000 --joint-std 0.04 --wrist-pos-std 0.005 --wrist-rot-std 0.03 \
        --max-pen-mm 10.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from optimize_pose import HandFK, axis_angle_to_matrix, seg_seg_dist  # noqa: E402

R_CAP, H_CAP = 0.050, 0.017
FINGER_RADIUS = 0.008


def aa_to_quat_xyzw(aa: torch.Tensor) -> torch.Tensor:
    ang = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    axis = aa / ang
    half = 0.5 * ang
    return torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--joint-std", type=float, default=0.04)
    ap.add_argument("--wrist-pos-std", type=float, default=0.005)
    ap.add_argument("--wrist-rot-std", type=float, default=0.03)
    ap.add_argument("--max-pen-mm", type=float, default=10.0)
    ap.add_argument("--max-selfc-mm", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    fk = HandFK("cpu")
    base = json.load(open(args.pose))
    q0 = torch.tensor(base["dof_pos"], dtype=torch.float32)
    wp0 = torch.tensor(base["wrist_rel_pos"], dtype=torch.float32)
    x, y, z, w = base["wrist_quat"]
    ang = 2 * math.acos(max(-1.0, min(1.0, w)))
    nrm = math.sqrt(max(1e-12, 1 - w * w))
    wr0 = torch.tensor([x / nrm * ang, y / nrm * ang, z / nrm * ang], dtype=torch.float32)
    q0_quat = torch.tensor([x, y, z, w], dtype=torch.float32)

    cap_names = [n for n in fk.capsules if not n.endswith("_1")]

    acc_q, acc_wp, acc_wq = [], [], []
    tried = 0
    while sum(t.shape[0] for t in acc_q) < args.n:
        B = args.batch
        tried += B
        q = q0 + torch.randn(B, 20) * args.joint_std
        wp = wp0 + torch.randn(B, 3) * args.wrist_pos_std
        daa = torch.randn(B, 3) * args.wrist_rot_std
        wq = quat_mul_xyzw(aa_to_quat_xyzw(daa), q0_quat.unsqueeze(0).expand(B, -1))

        # 침투 검사는 저장되는 회전(wq) 그대로 사용해야 한다. axis-angle
        # 덧셈 근사로 검사하면 경계 샘플이 실제 자세에서 한도를 넘는다.
        qx, qy, qz, qw = wq.unbind(-1)
        Rm = torch.stack(
            [
                torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], -1),
                torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], -1),
                torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], -1),
            ],
            -2,
        )
        tp, td = fk.tips(q)
        tpw = (Rm.unsqueeze(1) @ tp.unsqueeze(-1)).squeeze(-1) + wp.unsqueeze(1)
        tdw = (Rm.unsqueeze(1) @ td.unsqueeze(-1)).squeeze(-1)
        pad = tpw + tdw / tdw.norm(dim=-1, keepdim=True).clamp_min(1e-9) * FINGER_RADIUS
        rad = pad[..., :2].norm(dim=-1)
        band = (pad[..., 2] > 0) & (pad[..., 2] < H_CAP + 0.015)
        pen = ((R_CAP - rad).clamp_min(0) * band.float()).amax(dim=1)

        T = fk.all_link_transforms(q)
        mind = torch.full((B,), 1e9)
        world = {}
        for name in cap_names:
            ends, _r = fk.capsules[name]
            M = T[name]
            e = (M[:, None, :3, :3] @ ends[None, :, :, None]).squeeze(-1) + M[:, None, :3, 3]
            world[name] = e
        for i in range(len(cap_names)):
            fi = cap_names[i].split("_")[1]
            for j in range(i + 1, len(cap_names)):
                if cap_names[j].split("_")[1] == fi:
                    continue
                e1, e2 = world[cap_names[i]], world[cap_names[j]]
                r1 = fk.capsules[cap_names[i]][1]
                r2 = fk.capsules[cap_names[j]][1]
                d = seg_seg_dist(e1[:, 0], e1[:, 1], e2[:, 0], e2[:, 1]) - (r1 + r2)
                mind = torch.minimum(mind, d)

        ok = (pen * 1000 <= args.max_pen_mm) & (mind * 1000 >= -args.max_selfc_mm)
        acc_q.append(q[ok])
        acc_wp.append(wp[ok])
        acc_wq.append(wq[ok])

    q = torch.cat(acc_q)[: args.n]
    wp = torch.cat(acc_wp)[: args.n]
    wq = torch.cat(acc_wq)[: args.n]
    note = (
        f"{args.pose.rsplit('/', 1)[-1]} + 노이즈(관절 {args.joint_std}rad, "
        f"손목 {args.wrist_pos_std * 1000:.0f}mm/{math.degrees(args.wrist_rot_std):.1f}°) "
        f"합격만: 침투<={args.max_pen_mm}mm, 자기충돌겹침<={args.max_selfc_mm}mm. "
        f"합격률 {args.n / tried * 100:.1f}%"
    )
    torch.save(
        {"dof_pos": q, "wrist_rel_pos": wp, "wrist_quat": wq, "note": note}, args.out
    )
    print(f"저장: {args.out}  {args.n}개  ({note})")


if __name__ == "__main__":
    main()
