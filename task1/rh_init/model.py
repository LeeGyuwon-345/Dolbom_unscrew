"""(r,h) -> allegro 초기 파지자세 MLP + 6D 회전 유틸 + 방위각 후처리.

출력은 *정규 방위각*(손목을 원통 +x 축 위, azimuth=0)으로 고정된 자세다.
원통 축대칭이라 파지는 방위각 자유도가 무한하다 -- 이걸 학습에서 없애려고
손목을 +x 로 강제(azimuth 0). 손가락 dof_pos 는 방위각 무관이라 그대로 쓴다.
배포 시 원하는 방위각 theta 만큼 원통 z 축으로 회전(rotate_pose)해서 놓는다.

자세 배열(23) = dof_pos(16) + wrist_rel_pos(3) + wrist_quat(4, xyzw).
저장/소비 규약은 grasp_init/reference_pose.GraspInitPose 와 같다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 6D 연속 회전 표현 (Zhou et al. 2019). quaternion 의 double-cover/정규화 문제 회피.
# ---------------------------------------------------------------------------
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3). 앞 3 = a1, 뒤 3 = a2 를 Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # 열이 basis


def matrix_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,4) xyzw. 저장용."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(*R.shape[:-2], 4, device=R.device, dtype=R.dtype)
    s = torch.sqrt(torch.clamp(t + 1.0, min=1e-8)) * 2
    q[..., 3] = 0.25 * s
    q[..., 0] = (m[..., 2, 1] - m[..., 1, 2]) / s
    q[..., 1] = (m[..., 0, 2] - m[..., 2, 0]) / s
    q[..., 2] = (m[..., 1, 0] - m[..., 0, 1]) / s
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ---------------------------------------------------------------------------
# MLP: (r,h) -> 정규 파지자세 원(raw)
# ---------------------------------------------------------------------------
class GraspMLP(nn.Module):
    """(r,h) 정규화 입력 -> dof_pos(16) + 손목(정규 방위각) + rot6d(6).

    손목은 +x 축 위로 강제: wrist_pos = (radial, 0, z). azimuth=0 정규화가
    이 파라미터화로 자동 성립한다. dof/radial/z 는 sigmoid 로 물리범위에 매핑.
    """

    def __init__(self, dof_lo: torch.Tensor, dof_hi: torch.Tensor, hidden: int = 128):
        super().__init__()
        self.register_buffer("dof_lo", dof_lo)
        self.register_buffer("dof_hi", dof_hi)
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 16 + 2 + 6),   # dof(16) + [radial,z](2) + rot6d(6)
        )

    def forward(self, rh_norm: torch.Tensor, r: torch.Tensor, h: torch.Tensor):
        """rh_norm (B,2) 정규화 입력. r,h (B,) 실제값(m) -- 손목 범위 스케일용.

        반환: q(B,16), wp(B,3), R(B,3,3).  (원통 좌표, 정규 방위각)
        """
        o = self.net(rh_norm)
        dof_raw, rad_z, d6 = o[:, :16], o[:, 16:18], o[:, 18:]
        q = self.dof_lo + (self.dof_hi - self.dof_lo) * torch.sigmoid(dof_raw)
        # 손목: 원통 밖(반경 r+2~15cm), 높이 h+0~8cm. azimuth=0 (y=0).
        radial = r + 0.02 + 0.13 * torch.sigmoid(rad_z[:, 0])
        wz = h + 0.25 * torch.sigmoid(rad_z[:, 1])   # 캡top 위 [0,250]mm (palm 125mm 간격 허용)
        wp = torch.stack([radial, torch.zeros_like(radial), wz], dim=-1)
        R = rot6d_to_matrix(d6)
        return q, wp, R


# ---------------------------------------------------------------------------
# 배포: 정규 자세를 원하는 방위각 theta 로 회전 (원통 z 축)
# ---------------------------------------------------------------------------
def rotate_pose(wp: torch.Tensor, R: torch.Tensor, theta: torch.Tensor):
    """정규(azimuth 0) 손목을 원통 z 축으로 theta 회전. dof 는 그대로.

    wp (B,3), R (B,3,3), theta (B,). 반환 (wp', R')."""
    c, s = torch.cos(theta), torch.sin(theta)
    Rz = torch.zeros(theta.shape[0], 3, 3, device=wp.device, dtype=wp.dtype)
    Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
    Rz[:, 1, 0], Rz[:, 1, 1] = s, c
    Rz[:, 2, 2] = 1.0
    return torch.einsum("bij,bj->bi", Rz, wp), Rz @ R
