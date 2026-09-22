"""amortized RB5 IK 네트워크: (r,h,캡배치z) -> RB5 6관절 + 방위각.

손 MLP(model.GraspMLP)와 같은 철학. 옵티마이저 데이터 없이, 네트워크가 낸
(방위각 th, 관절 q)에 solve_rb5_ik 와 동일한 미분가능 목적함수(위치·자세·엘보우업·
base방향·바닥)를 걸어 최소화한다. 학습 후 추론은 sub-ms.

입력 정규화: r[15,55] h[20,30] z[140,230]mm -> [0,1]^3.
출력: 방위각 th (-pi,pi), q(6) (관절범위 내 sigmoid).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import rb5_ik as K

# 방위각 제약: θ ∈ [center-half, center+half]. half=0 이면 center 로 고정.
AZ_CENTER = math.radians(135.0)   # 135° 고정
AZ_HALF = math.radians(0.0)       # 0 -> θ=AZ_CENTER 고정


class RB5Net(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        K._init()
        self.register_buffer("lo", K._LO.clone())
        self.register_buffer("hi", K._HI.clone())
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1 + 6),
        )

    def forward(self, x: torch.Tensor):
        """x (B,3) 정규화 입력 -> (th (B,), q (B,6))."""
        o = self.net(x)
        th = AZ_CENTER + AZ_HALF * torch.tanh(o[:, 0])            # 방위각 [125°,145°] 제약
        q = self.lo + (self.hi - self.lo) * torch.sigmoid(o[:, 1:])  # 관절범위 내
        return th, q
