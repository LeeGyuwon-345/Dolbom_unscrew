"""절차적(procedural) unscrew 목표 생성 — 데모 궤적 없이 뚜껑 목표궤적을 수식으로 생성.

뚜껑을 나사축(기본 world +Z, 병이 서있음) 기준 제자리 회전 + 축방향 lift.
  cap_T(t): R_axis(θ(t)) 를 뚜껑중심 피벗 기준 적용 + axis·lift(t)
  body_T(t): 고정(몸통 hold)
그 목표를 학습된 플래너에 넣어 양손 손목궤적 생성 → _proc.npz 저장(GUI 재생용).

이것이 trajectory-free의 핵심: OakInk 데모 대신 '각도·lift 스펙'만으로 목표를 만든다.

실행:
  python wrist_planner/procedural_goal.py --data_idx 97fc3@1 --angle 200 --lift_cm 2.0
GUI: python wrist_planner/view_aligned.py --data_idx 97fc3@1_proc
"""
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from infer import WristPlanner

HERE = os.path.dirname(os.path.abspath(__file__))


def resample(a, n):
    """(T,D) → (n,D) 선형 리샘플."""
    T = len(a)
    xs = np.linspace(0, T - 1, n)
    return np.stack([np.interp(xs, np.arange(T), a[:, d]) for d in range(a.shape[1])], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_idx", default="97fc3@1")
    ap.add_argument("--angle", type=float, default=200.0, help="총 회전각(deg)")
    ap.add_argument("--lift_cm", type=float, default=2.0, help="총 축방향 lift(cm)")
    ap.add_argument("--n_frames", type=int, default=200)
    ap.add_argument("--axis", choices=["world_z", "body_up"], default="world_z")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    idx = args.data_idx

    npz = np.load(os.path.join(HERE, "datasets", f"{idx}.npz"), allow_pickle=True)
    cap_T0 = npz["cap_T"][0].astype(np.float32)
    body_T0 = npz["body_T"][0].astype(np.float32)
    N = args.n_frames

    # 회전축
    if args.axis == "world_z":
        axis = np.array([0, 0, 1.0], np.float32)
    else:  # 몸통 로컬 up
        axis = body_T0[:3, :3] @ np.array([0, 0, 1.0], np.float32)
        axis = axis / np.linalg.norm(axis)
    pivot = cap_T0[:3, 3]                      # 피벗 = 뚜껑 중심
    cap_R0 = cap_T0[:3, :3]

    theta = np.radians(np.linspace(0, args.angle, N)).astype(np.float32)
    lift = np.linspace(0, args.lift_cm / 100.0, N).astype(np.float32)

    cap_T = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    for t in range(N):
        Rrot = Rotation.from_rotvec(axis * theta[t]).as_matrix().astype(np.float32)
        cap_T[t, :3, :3] = Rrot @ cap_R0
        cap_T[t, :3, 3] = pivot + Rrot @ (pivot - pivot) + axis * lift[t]   # 중심회전 → 위치는 lift만
    body_T = np.tile(body_T0, (N, 1, 1))

    print(f"[proc] axis={args.axis} 회전={args.angle}deg lift={args.lift_cm}cm N={N}")
    # 몸통대비 상대회전 확인
    rel = np.einsum("ij,tjk->tik", body_T0[:3, :3].T, cap_T[:, :3, :3])
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    print(f"[proc] 뚜껑-몸통 상대회전: {ang.min():.0f}~{ang.max():.0f}deg")

    # 플래너로 손목궤적 생성
    pl = WristPlanner.from_idx(idx, args.device)
    rh_p, rh_aa, lh_p, lh_aa = pl.generate(cap_T, body_T)
    print(f"[planner] RH 손목 이동={np.linalg.norm(rh_p.max(0)-rh_p.min(0))*100:.1f}cm  "
          f"LH 손목 이동={np.linalg.norm(lh_p.max(0)-lh_p.min(0))*100:.1f}cm")

    # 시각화용 손가락 dof는 데모값 리샘플(플래너는 손가락 안 냄 → placeholder)
    rh_dof = resample(npz["rh_dof"], N)
    lh_dof = resample(npz["lh_dof"], N)

    out = os.path.join(HERE, "datasets", f"{idx}_proc.npz")
    np.savez(out,
             common_t=np.arange(N, dtype=np.float32), cap_T=cap_T, body_T=body_T,
             rh_wpos=rh_p, rh_waa=rh_aa, rh_dof=rh_dof,
             lh_wpos=lh_p, lh_waa=lh_aa, lh_dof=lh_dof,
             cap_urdf=npz["cap_urdf"], body_urdf=npz["body_urdf"], idx=f"{idx}_proc",
             rh_range=npz["rh_range"], lh_range=npz["lh_range"])
    print(f"[저장] {out}")
    print(f"GUI: python wrist_planner/view_aligned.py --data_idx {idx}_proc")


if __name__ == "__main__":
    main()
