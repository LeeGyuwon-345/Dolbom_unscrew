"""돌고 있는 학습 로그에서 잘 잡힌 순간의 자세를 떠서 poses/ 에 저장한다.

손으로 관절각을 적는 것보다 이미 학습된 정책이 만든 파지를 쓰는 편이 낫다.
지금 정책들은 회전은 못 끝내도 파지 자체는 인정 접촉 3.4~3.9, 품질 0.94 까지
간다 -- 탐색을 건너뛰고 싶은 구간이 정확히 그 부분이다.

환경 안에서 부르는 것이 아니라, 환경이 덤프해 둔 스냅샷을 읽어 변환한다.
스냅샷은 dexhandmanip_sh 가 CAP_GRASP_DUMP_POSE=1 일 때 남긴다.

사용법:
    python capture_pose.py <스냅샷.pt> -o poses/four_finger.json --note "Turn ep1900"
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from reference_pose import GraspInitPose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot", help="환경이 덤프한 .pt (dof_pos, base_state, cap_pos, quality)")
    ap.add_argument("-o", "--out", default="poses/captured.json")
    ap.add_argument("--note", default="")
    # 여러 환경이 들어 있으면 품질이 가장 높은 것을 고른다. 평균을 내면 어느
    # 환경의 것도 아닌 자세가 나오고, 관절각 평균은 특히 위험하다.
    ap.add_argument("--min-quality", type=float, default=0.8)
    args = ap.parse_args()

    d = torch.load(args.snapshot, map_location="cpu")
    q, base, cap = d["dof_pos"], d["base_state"], d["cap_pos"]
    qual = d.get("quality")
    if qual is None:
        i = 0
        print("[capture] 품질 정보 없음 -- 0번 환경 사용")
    else:
        i = int(torch.argmax(qual))
        if float(qual[i]) < args.min_quality:
            raise SystemExit(
                f"가장 좋은 환경의 품질이 {float(qual[i]):.3f} 로 기준 {args.min_quality} 미만이다. "
                "더 학습된 체크포인트에서 다시 뜨는 편이 낫다."
            )
        print(f"[capture] {len(qual)}개 중 {i}번 선택  품질 {float(qual[i]):.3f}")

    pose = GraspInitPose(
        dof_pos=q[i],
        wrist_rel_pos=base[i, :3] - cap[i],
        wrist_quat=base[i, 3:7],
        note=args.note or f"{os.path.basename(args.snapshot)} env{i}",
    )
    out = args.out if os.path.isabs(args.out) else os.path.join(os.path.dirname(__file__), args.out)
    pose.save(out)
    print(f"[capture] 저장: {out}")
    print(f"  손목 상대위치 {[round(1000 * float(v), 1) for v in pose.wrist_rel_pos]} mm")
    print(f"  관절 {pose.dof_pos.numel()}개  범위 "
          f"{float(pose.dof_pos.min()):+.3f} ~ {float(pose.dof_pos.max()):+.3f} rad")


if __name__ == "__main__":
    main()
