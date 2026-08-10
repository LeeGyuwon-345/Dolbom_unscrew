"""파지 대상 캡 = 반지름 r, 높이 h 인 원기둥. 파라미터는 이 둘뿐이다.

파지 자세를 최적화하려면 손끝이 닿아야 할 면이 함수로 있어야 한다. 학습 쪽은
측정된 21개 노드 프로파일(cap_unscrew_rl_terms.CAP_PROFILE_R)을 쓰지만, 그건
특정 STL 에 묶여 있어서 "반지름 r, 높이 h 인 캡이면 이렇게 잡아라" 를 풀 수
없다.

좌표계는 원기둥 자신의 것이다 -- 밑면 중심이 원점, 축은 +z, z 는 0~h.
자세를 이 좌표로 풀어두면 캡이 어디에 어떤 높이로 놓이든 그대로 옮겨 쓸 수
있다. 실제 텀블러 캡에 얹을 때는 캡 프레임 z 로 13mm 더하면 된다.

기본값 r=50mm, h=17mm 는 cap.STL 실측이다. 축을 y 로 두고 재면 상단이 반경
50.00mm 인 완전한 원통이고 그 구간이 17mm 다. 실제 collision 메시와 대조하면
그 구간에서 최대 0.24mm, 평균 0.21mm 차이다 (view_cap.py).

    캡 프레임 0~13mm    r 47.3mm   나사 스커트. 몸통 입구(50.1mm) 안이라
                                   손이 닿지 않는다 -- 파지 대상 아님
    캡 프레임 13~30mm   r 50.0mm   파지 벽        <-- 이 원기둥
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# 이 원기둥의 밑면이 실제 텀블러 캡 프레임에서 앉는 높이. 형상 파라미터가
# 아니라 배치 상수라 dataclass 밖에 둔다.
CAP_FRAME_Z0 = 0.013


@dataclass(frozen=True)
class CapShape:
    radius: float = 0.050      # r (m)
    height: float = 0.017      # h (m)

    def surface_points(self, n_theta: int = 64, n_z: int = 8, device=None) -> torch.Tensor:
        """옆면 격자 (n_theta*n_z, 3)."""
        th = torch.linspace(0, 2 * math.pi, n_theta + 1, device=device)[:-1]
        z = torch.linspace(0.0, self.height, n_z, device=device)
        T, Z = torch.meshgrid(th, z, indexing="ij")
        return torch.stack(
            [self.radius * torch.cos(T), self.radius * torch.sin(T), Z], dim=-1
        ).reshape(-1, 3)

    def nearest(self, p: torch.Tensor):
        """(...,3) 점들에 대해 (옆면 최근접점, 바깥 법선, 부호거리).

        부호거리는 바깥이 +, 안이 -. 옆면 위아래로 벗어난 점은 z 를 구간에 잘라
        테두리로 투영한다 -- 최적화가 벽을 벗어난 손끝을 도로 끌어오려면 그
        바깥에서도 기울기가 있어야 한다.
        """
        xy = p[..., :2]
        r = torch.linalg.norm(xy, dim=-1).clamp_min(1e-9)
        u = xy / r.unsqueeze(-1)                        # 바깥 방향 단위벡터
        z = p[..., 2].clamp(0.0, self.height)
        near = torch.cat([u * self.radius, z.unsqueeze(-1)], dim=-1)
        d_r = r - self.radius
        d_z = p[..., 2] - z                             # 구간 밖으로 벗어난 만큼
        sd = torch.sign(d_r) * torch.sqrt(d_r ** 2 + d_z ** 2 + 1e-18)
        normal = torch.cat([u, torch.zeros_like(z).unsqueeze(-1)], dim=-1)
        return near, normal, sd

    def dist2(self, p: torch.Tensor) -> torch.Tensor:
        """(...,3) 점들에서 옆면까지 거리의 제곱. 부호 없음.

        nearest() 의 부호거리를 제곱해도 값은 같지만, 그 경로는 sqrt 를 거친다.
        sqrt 는 0 에서 미분이 발산하므로 하필 해에 도달할수록 기울기가 나빠진다.
        여기서는 제곱을 바로 만든다.
        """
        r = torch.linalg.norm(p[..., :2], dim=-1)
        d_r = r - self.radius
        d_z = p[..., 2] - p[..., 2].clamp(0.0, self.height)
        return d_r ** 2 + d_z ** 2

    def outward_normal_at_angle(self, theta: torch.Tensor) -> torch.Tensor:
        """방위각에서의 바깥 법선. 손바닥이 벽을 마주보는지 볼 때 쓴다."""
        return torch.stack(
            [torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1
        )

    def export_obj(self, path: str, n_theta: int = 96) -> None:
        """눈으로 확인하기 위한 옆면 메시. 위아래 뚜껑 없이 옆면만."""
        th = [2 * math.pi * i / n_theta for i in range(n_theta)]
        lines = [f"# cap cylinder r={self.radius:.4f} h={self.height:.4f}"]
        for z in (0.0, self.height):
            for t in th:
                lines.append(
                    f"v {self.radius * math.cos(t):.6f} {self.radius * math.sin(t):.6f} {z:.6f}"
                )
        for i in range(n_theta):
            j = (i + 1) % n_theta
            a, b = 0, n_theta
            lines.append(f"f {a + i + 1} {a + j + 1} {b + j + 1}")
            lines.append(f"f {a + i + 1} {b + j + 1} {b + i + 1}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")


DEFAULT_CAP = CapShape()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="캡 원기둥 생성/확인")
    ap.add_argument("-r", "--radius", type=float, default=0.050)
    ap.add_argument("-H", "--height", type=float, default=0.017)
    ap.add_argument("-o", "--obj", default="")
    args = ap.parse_args()

    cap = CapShape(args.radius, args.height)
    print(f"r {cap.radius * 1000:.1f}mm   h {cap.height * 1000:.1f}mm")
    print(f"둘레 {2 * math.pi * cap.radius * 1000:.1f}mm   "
          f"옆면적 {2 * math.pi * cap.radius * cap.height * 1e4:.1f}cm2")
    print(f"실제 캡에 얹을 때 캡 프레임 z {CAP_FRAME_Z0 * 1000:.0f}~"
          f"{(CAP_FRAME_Z0 + cap.height) * 1000:.0f}mm")
    probe = torch.tensor([[cap.radius + 0.008, 0.0, cap.height / 2],
                          [cap.radius - 0.005, 0.0, cap.height / 2],
                          [cap.radius + 0.008, 0.0, cap.height + 0.010]])
    _, _, sd = cap.nearest(probe)
    for p, d in zip(probe, sd):
        print(f"  점 ({p[0]*1000:.0f}, {p[1]*1000:.0f}, {p[2]*1000:.0f})mm  "
              f"부호거리 {float(d)*1000:+.2f}mm")
    if args.obj:
        cap.export_obj(args.obj)
        print(f"메시 저장: {args.obj}")
