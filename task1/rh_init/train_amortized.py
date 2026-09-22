"""amortized optimization: (r,h) -> allegro 정규 파지자세 MLP 학습.

옵티마이저 데이터 없이, MLP 가 낸 자세에 배치 objective 를 걸어 최소화한다.
방위각 정규화는 model.GraspMLP 가 손목을 +x 축(azimuth 0)으로 강제해 자동 성립.

    python train_amortized.py --iters 20000
    python train_amortized.py --export 0.05 0.017 -o out.json   # 학습된 MLP로 자세 생성

r,h 범위는 아래 R_RANGE/H_RANGE 에 넣는다 (지금은 플레이스홀더).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/leegyuwon/Documents/task1/grasp_init")

import optimize_pose_allegro as OP  # noqa: E402
from maniptrans_dof_order_allegro import DOF_LIMITS, DOF_NAMES  # noqa: E402
from model import GraspMLP, matrix_to_quat_xyzw  # noqa: E402
from objective_batched import batched_objective  # noqa: E402

# ===========================================================================
#  r, h 범위 (m). 학습·정규화가 이 범위를 쓴다.
# ===========================================================================
R_RANGE = (0.015, 0.055)     # 반지름 15~55mm (작은 r 은 자동으로 손가락 수↓ 핀치)
H_RANGE = (0.014, 0.030)     # 높이 14~30mm (h=16 포함)
Z_FRAC = None                # 팁 높이는 TIPZ_TARGET(월드 229mm)로 직접 지정 -- 게이트 정합
PALM_DOWN_DEG = 20          # 손등 ∥ 바닥: 손바닥 아래방향 허용각(학습 규약)
FINGER_RADIUS = 0.0105       # 실리콘 팁 +x 실효 접촉반경(표준팁 0.012)
TIP_DOWN_W = 20.0            # 팁 +z 바닥 지향. 웜스타트(facing 좋은 모델)에서 fine-tune.
FLEX_MAX_DEG = 360.0         # 굴곡합 상한(4지 굴곡조인트 합) -- 과굴곡 방지, 손 폄
PALM_GAP_MIN = 0.125         # palm-캡상단 간격 하한(m) -- 손목 위로, 손가락 폄+아래 도달
PALM_TILT_REF = (-0.163, -0.128, -0.978)   # 검증 자세 팜 법선(월드) 기준
PALM_TILT_DEG = 17.0         # 팜 법선을 기준에서 ±17도 이내로 (손목 기울기 제약)
MID_EXTRA_PRESS = 0.0000
TIP_PAD_BACK = 0.012         # 팁 pad 면 중심이 팁원점보다 팜쪽 12mm -- 실접촉면 정합     # 중지 추가 press(m) -- +0.8mm 부양 -> 밀착
# === pose_gate.py 정합 (게이트와 동일 정의) ===
CAP_TOP_Z = 0.262            # 게이트 기준 캡 윗면 월드 z
PALM_ABS_MIN = 0.1425         # 3 팜높이 타깃(하한125 + 여유17mm) -- 실측 123 -> 상향
PALM_ABS_W = 60000.0         # 갭·팁z 강화와 균형 맞추려 증량 (8000 -> 60000)
OPPOSE_MIN_DEG = 152.0       # 6 엄지대향 >= 140deg (여유 5도)
OPPOSE_W = 3000.0
TIPZ_TARGET = 0.2395         # 팁z 목표 재하향 (검지 243 경계 -> 마진 5mm 확보)
TIPZ_W = 20000.0             # 팁z 229mm 강하게(검지 247 -> 밴드 내). 팜은 별도 유지.
# 손목 회전행렬 정렬 기준 R_ref (3x3, 정규 az0 프레임 = Rz(-135°)@R_world_검증).
# 배포 시 rotate_pose(θ=135°)로 월드 검증 손목 자세 복원. 팜 법선만으론 법선-주위 회전 못 잡음.
WRIST_R_REF = ((-0.127433, -0.956319, -0.263088),
               (-0.063515, -0.256838,  0.964365),
               (-0.989812,  0.139602, -0.028011))
CKPT = os.path.join(HERE, "grasp_mlp.pt")


def _norm(r, h):
    """(r,h) -> [0,1]^2 (R_RANGE/H_RANGE 기준)."""
    rn = (r - R_RANGE[0]) / (R_RANGE[1] - R_RANGE[0])
    hn = (h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])
    return torch.stack([rn, hn], dim=-1)


def build(dev):
    fk = OP.HandFK(dev)
    fk.link_pts = fk.link_points(200)
    lo = torch.tensor([DOF_LIMITS[n][0] for n in DOF_NAMES], device=dev)
    hi = torch.tensor([DOF_LIMITS[n][1] for n in DOF_NAMES], device=dev)
    mlp = GraspMLP(lo, hi).to(dev)
    return fk, mlp, lo, hi


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fk, mlp, lo, hi = build(dev)
    if args.resume and os.path.exists(CKPT):
        mlp.load_state_dict(torch.load(CKPT, map_location=dev))
        print("resumed", CKPT)
    opt = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    print(f"amortized 학습  R{R_RANGE} H{H_RANGE}  batch {args.batch}  {args.iters}회  {dev}")
    for it in range(args.iters):
        r = torch.rand(args.batch, device=dev) * (R_RANGE[1] - R_RANGE[0]) + R_RANGE[0]
        h = torch.rand(args.batch, device=dev) * (H_RANGE[1] - H_RANGE[0]) + H_RANGE[0]
        q, wp, R = mlp(_norm(r, h), r, h)
        loss = batched_objective(q, wp, R, r, h, fk, lo=lo, hi=hi, press=args.press, z_frac=Z_FRAC, palm_down_deg=PALM_DOWN_DEG, finger_radius=FINGER_RADIUS, tip_down_w=TIP_DOWN_W, flex_max_deg=FLEX_MAX_DEG, palm_gap_min=PALM_GAP_MIN, wrist_tilt_ref=PALM_TILT_REF, wrist_tilt_deg=PALM_TILT_DEG, wrist_R_ref=WRIST_R_REF, mid_extra_press=MID_EXTRA_PRESS, tip_pad_back=TIP_PAD_BACK, palm_abs_min=PALM_ABS_MIN, palm_abs_w=PALM_ABS_W, cap_top_z=CAP_TOP_Z, oppose_min_deg=OPPOSE_MIN_DEG, oppose_w=OPPOSE_W, tipz_target=TIPZ_TARGET, tipz_w=TIPZ_W).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == args.iters - 1:
            with torch.no_grad():
                _, det = batched_objective(q, wp, R, r, h, fk, lo=lo, hi=hi, press=args.press, z_frac=Z_FRAC, palm_down_deg=PALM_DOWN_DEG, finger_radius=FINGER_RADIUS, tip_down_w=TIP_DOWN_W, flex_max_deg=FLEX_MAX_DEG, palm_gap_min=PALM_GAP_MIN, wrist_tilt_ref=PALM_TILT_REF, wrist_tilt_deg=PALM_TILT_DEG, wrist_R_ref=WRIST_R_REF, mid_extra_press=MID_EXTRA_PRESS, tip_pad_back=TIP_PAD_BACK, palm_abs_min=PALM_ABS_MIN, palm_abs_w=PALM_ABS_W, cap_top_z=CAP_TOP_Z, oppose_min_deg=OPPOSE_MIN_DEG, oppose_w=OPPOSE_W, tipz_target=TIPZ_TARGET, tipz_w=TIPZ_W, detail=True)
                print(f"  {it:6d}  loss {float(loss):.5f}  "
                      + " ".join(f"{k} {float(v.mean()):.4f}" for k, v in det.items()), flush=True)
    torch.save(mlp.state_dict(), CKPT)
    print("저장:", CKPT)


def export(args):
    """학습된 MLP 로 (r,h) 자세 생성 -> GraspInitPose json (뷰어/학습에서 소비)."""
    from scipy.spatial.transform import Rotation  # noqa
    sys.path.insert(0, "/home/leegyuwon/Documents/task1/grasp_init")
    from cap_shape import CAP_FRAME_Z0
    from reference_pose import GraspInitPose

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fk, mlp, lo, hi = build(dev)
    mlp.load_state_dict(torch.load(CKPT, map_location=dev)); mlp.eval()
    r = torch.tensor([args.export[0]], device=dev)
    h = torch.tensor([args.export[1]], device=dev)
    with torch.no_grad():
        q, wp, R = mlp(_norm(r, h), r, h)
        _, det = batched_objective(q, wp, R, r, h, fk, lo=lo, hi=hi, press=args.press, z_frac=Z_FRAC, palm_down_deg=PALM_DOWN_DEG, finger_radius=FINGER_RADIUS, tip_down_w=TIP_DOWN_W, flex_max_deg=FLEX_MAX_DEG, palm_gap_min=PALM_GAP_MIN, wrist_tilt_ref=PALM_TILT_REF, wrist_tilt_deg=PALM_TILT_DEG, wrist_R_ref=WRIST_R_REF, mid_extra_press=MID_EXTRA_PRESS, tip_pad_back=TIP_PAD_BACK, palm_abs_min=PALM_ABS_MIN, palm_abs_w=PALM_ABS_W, cap_top_z=CAP_TOP_Z, oppose_min_deg=OPPOSE_MIN_DEG, oppose_w=OPPOSE_W, tipz_target=TIPZ_TARGET, tipz_w=TIPZ_W, detail=True)
    quat = matrix_to_quat_xyzw(R)[0]
    pose = GraspInitPose(
        dof_pos=q[0].cpu(),
        wrist_rel_pos=wp[0].cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0]),
        wrist_quat=quat.cpu(),
        note=f"allegro cyl r={args.export[0]*1000:.0f} h={args.export[1]*1000:.0f} "
             f"mlp loss={float(sum(v[0] for v in det.values())):.4f}",
    )
    out = args.out or os.path.join(HERE, f"mlp_r{int(args.export[0]*1000)}_h{int(args.export[1]*1000)}.json")
    print("항별:", {k: round(float(v[0]), 5) for k, v in det.items()})

    # RB5 팔 IK: 파지는 방위각 자유이므로, 캡 z축으로 회전하며 RB5가 가장 잘 닿는
    # 방위각을 찾고(coarse), 그 자세로 최종 IK(refine). 손가락 dof 는 불변, 손목만 회전.
    import math as _m

    from model import rotate_pose
    if not args.no_rb5:
        from rb5_ik import solve_rb5_ik
        tb = -torch.tensor(args.cap[:2]); tb = tb / tb.norm().clamp_min(1e-9)  # 캡→base 수평단위
        best = None
        for k in range(args.n_az):
            th = torch.tensor([2 * _m.pi * k / args.n_az], device=dev)
            wp_t, R_t = rotate_pose(wp, R, th)
            qt = matrix_to_quat_xyzw(R_t)[0].cpu()
            res = solve_rb5_ik(wp_t[0].cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0]), qt,
                               cap_world=tuple(args.cap), trials=3, iters=150)  # coarse
            xy = wp_t[0, :2].cpu()
            align = float((xy / xy.norm().clamp_min(1e-9) * tb).sum())         # tip→손목 base정렬
            sc = (res["pos_err_mm"] + 5 * res["rot_err_deg"] + (0 if res["feasible"] else 1e4)
                  + args.base_dir_w * (1.0 - align))
            if best is None or sc < best[0]:
                best = (sc, float(th), wp_t, R_t, qt)
        _, th_best, wp, R, _ = best
        quat = matrix_to_quat_xyzw(R)
        res = solve_rb5_ik(wp[0].cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0]), quat[0].cpu(),
                           cap_world=tuple(args.cap), trials=16, iters=500)  # refine
        print(f"선택 방위각: {_m.degrees(th_best):.0f}도")
        print("RB5 관절(도):", dict(zip(res["names"], res["q_deg"])))
        print(f"  IK 오차: 위치 {res['pos_err_mm']}mm 자세 {res['rot_err_deg']}도 도달가능 {res['feasible']}")
        pose.wrist_rel_pos = wp[0].cpu() + torch.tensor([0.0, 0.0, CAP_FRAME_Z0])
        pose.wrist_quat = quat[0].cpu()
        pose.note += f" az={_m.degrees(th_best):.0f}"

    pose.save(out)
    if not args.no_rb5:
        import json as _json
        d = _json.load(open(out)); d["rb5"] = res; d["azimuth_deg"] = round(_m.degrees(th_best), 1)
        d["cap"] = list(args.cap)   # 캡 원점의 RB5 base 기준 월드위치(통합 뷰어용)
        _json.dump(d, open(out, "w"), indent=1)
    print("저장:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--press", type=float, default=0.0032)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export", type=float, nargs=2, default=None, metavar=("R", "H"),
                    help="학습된 MLP로 (r,h)[m] 자세 생성")
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--no-rb5", action="store_true", help="RB5 팔 IK 생략(손 관절만)")
    ap.add_argument("--cap", type=float, nargs=3, default=[0.6, 0.0, 0.225],
                    help="캡 원점의 RB5 베이스 기준 월드위치(m). 텀블러 배치.")
    ap.add_argument("--n-az", type=int, default=8, help="RB5 방위각 탐색 개수(도달 최적)")
    ap.add_argument("--base-dir-w", type=float, default=15.0,
                    help="tip→손목 방향을 RB5 base 쪽으로 정렬 선호(mm 등가). 0=끔")
    args = ap.parse_args()
    if args.export:
        export(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
