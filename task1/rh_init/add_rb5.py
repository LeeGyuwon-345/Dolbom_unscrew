"""GraspInitPose json(최적화기/MLP 공용)에 RB5 팔 IK 를 추가한다.

파지는 캡 z축 방위각 자유 -> RB5 가 가장 잘 닿는 방위각 탐색(coarse) 후 refine.
저장된 wrist_rel_pos 는 이미 +CAP_FRAME_Z0 규약이므로 그대로 solve_rb5_ik 에 넘긴다
(train_amortized.export 와 동일 시맨틱: export 는 raw wp+CAP_FRAME_Z0 를 넘김).

    python add_rb5.py <pose.json> [--cap 0.6 0 0.225] [--n-az 8]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import matrix_to_quat_xyzw, rotate_pose  # noqa: E402
from rb5_ik import _quat_to_R, solve_rb5_ik  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose")
    ap.add_argument("--cap", type=float, nargs=3, default=[0.6, 0.0, 0.225])
    ap.add_argument("--n-az", type=int, default=8)
    ap.add_argument("--base-dir-w", type=float, default=15.0,
                    help="tip→손목 방향을 RB5 base 쪽으로 정렬 선호 가중치(mm 등가). 0=끔")
    a = ap.parse_args()

    d = json.load(open(a.pose))
    wrp = torch.tensor(d["wrist_rel_pos"], dtype=torch.float32).unsqueeze(0)  # +CAP_FRAME_Z0 포함
    R = _quat_to_R([float(v) for v in d["wrist_quat"]]).unsqueeze(0)
    cap = tuple(a.cap)
    # tip→손목 ≈ 손목의 수평 방위 방향(캡 밖 radial). 이를 base(원점) 방향과 정렬.
    tb = -torch.tensor(cap[:2]); tb = tb / tb.norm().clamp_min(1e-9)   # 캡→base 수평단위

    def _base_align_pen(wrp_t):
        xy = wrp_t[0, :2]
        align = float((xy / xy.norm().clamp_min(1e-9) * tb).sum())      # -1..1 (1=base쪽)
        return a.base_dir_w * (1.0 - align)                            # 0(정렬)~2w(반대)

    best = None
    for k in range(a.n_az):
        th = torch.tensor([2 * math.pi * k / a.n_az])
        wrp_t, R_t = rotate_pose(wrp, R, th)
        qt = matrix_to_quat_xyzw(R_t)[0]
        res = solve_rb5_ik(wrp_t[0], qt, cap_world=cap, trials=3, iters=150)  # coarse
        sc = (res["pos_err_mm"] + 5 * res["rot_err_deg"] + (0 if res["feasible"] else 1e4)
              + _base_align_pen(wrp_t))
        if best is None or sc < best[0]:
            best = (sc, float(th), wrp_t, R_t)
    _, th_best, wrp, R = best
    quat = matrix_to_quat_xyzw(R)
    res = solve_rb5_ik(wrp[0], quat[0], cap_world=cap, trials=16, iters=500)  # refine

    print(f"선택 방위각: {math.degrees(th_best):.0f}도")
    print("RB5 관절(도):", dict(zip(res["names"], res["q_deg"])))
    print(f"  IK 오차: 위치 {res['pos_err_mm']}mm 자세 {res['rot_err_deg']}도 "
          f"엘보우(J3)={res['q_deg'][2]}도 도달 {res['feasible']}")

    d["wrist_rel_pos"] = [float(v) for v in wrp[0]]
    d["wrist_quat"] = [float(v) for v in quat[0]]
    d["rb5"] = res
    d["azimuth_deg"] = round(math.degrees(th_best), 1)
    d["cap"] = list(cap)
    json.dump(d, open(a.pose, "w"), indent=1)
    print("저장:", a.pose)


if __name__ == "__main__":
    main()
