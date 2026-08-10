"""리셋 시 손을 이미 잡은 자세로 놓는다.

환경이 import 하는 유일한 파일. 관측도 보상도 건드리지 않으므로 이 기능을
켜고 끄는 것만으로 기존 체크포인트를 그대로 이어 쓸 수 있다.

자세는 손가락 관절각과 손목의 *캡 기준 상대* 위치/자세로 저장한다. 월드
좌표로 저장하면 캡 위치가 바뀌거나 캡이 풀려 올라간 상태에서 못 쓴다.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GraspInitPose:
    """캡 기준으로 표현된 파지 한 벌."""

    dof_pos: torch.Tensor       # (n_hand_dofs,) 손가락 관절각
    wrist_rel_pos: torch.Tensor  # (3,) 캡 원점 기준 손목 위치
    wrist_quat: torch.Tensor     # (4,) xyzw, 월드 기준 손목 자세
    note: str = ""

    @classmethod
    def load(cls, path: str, device) -> "GraspInitPose":
        with open(path) as f:
            d = json.load(f)
        t = lambda k: torch.tensor(d[k], device=device, dtype=torch.float32)
        return cls(t("dof_pos"), t("wrist_rel_pos"), t("wrist_quat"), d.get("note", ""))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "dof_pos": self.dof_pos.tolist(),
                    "wrist_rel_pos": self.wrist_rel_pos.tolist(),
                    "wrist_quat": self.wrist_quat.tolist(),
                    "note": self.note,
                },
                f,
                indent=2,
            )


class GraspInitializer:
    """환경변수로 설정되고, 리셋 때 env_ids 의 일부에만 자세를 덮어쓴다.

    비율이 기본 0.5 인 이유는 README 에 적어둔 대로 -- 전부를 같은 자세에서
    출발시키면 그 자세 전용 정책이 된다. 나머지 절반은 기존 리셋 경로를 그대로
    타므로 초기 조건의 다양성이 유지된다.
    """

    def __init__(self, device, n_hand_dofs: int):
        self.device = device
        self.n_hand_dofs = n_hand_dofs
        path = os.environ.get("CAP_GRASP_INIT_POSE", "").strip()
        self.frac = float(os.environ.get("CAP_GRASP_INIT_FRAC", "0.5"))
        self.noise = float(os.environ.get("CAP_GRASP_INIT_NOISE", "0.05"))
        self.pose: Optional[GraspInitPose] = None
        if path:
            if not os.path.isabs(path):
                path = os.path.join(os.path.dirname(__file__), path)
            self.pose = GraspInitPose.load(path, device)
            n = self.pose.dof_pos.numel()
            if n != n_hand_dofs:
                raise ValueError(
                    f"파지 자세의 관절 수가 손과 다르다: {n} vs {n_hand_dofs} ({path})"
                )
            print(
                f"[grasp-init] {os.path.basename(path)} 로드  비율 {self.frac:.2f}  "
                f"노이즈 {math.degrees(self.noise):.1f}도  ({self.pose.note})",
                flush=True,
            )

    @property
    def enabled(self) -> bool:
        return self.pose is not None and self.frac > 0.0

    def pick(self, env_ids: torch.Tensor) -> torch.Tensor:
        """env_ids 중 이 자세로 시작시킬 것들."""
        if not self.enabled or env_ids.numel() == 0:
            return env_ids[:0]
        m = torch.rand(env_ids.numel(), device=env_ids.device) < self.frac
        return env_ids[m]

    def apply(self, ids: torch.Tensor, dof_pos: torch.Tensor, base_state: torch.Tensor,
              cap_pos: torch.Tensor) -> None:
        """dof_pos 와 base_state 를 제자리에서 덮어쓴다.

        ids 는 pick() 이 고른 것. dof_pos 는 (E, n_hand_dofs), base_state 는
        (E, 13), cap_pos 는 (E, 3) 으로 전체 환경 기준이며 ids 로 색인한다.
        속도는 0 으로 둔다 -- 잡은 자세에서 출발하는 것이지 움직이던 중이
        아니다.
        """
        if ids.numel() == 0:
            return
        p = self.pose
        q = p.dof_pos.unsqueeze(0).expand(ids.numel(), -1).clone()
        if self.noise > 0:
            q = q + torch.randn_like(q) * self.noise
        dof_pos[ids] = q
        base_state[ids, :3] = cap_pos[ids] + p.wrist_rel_pos.unsqueeze(0)
        base_state[ids, 3:7] = p.wrist_quat.unsqueeze(0)
        base_state[ids, 7:13] = 0.0
