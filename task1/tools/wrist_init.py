"""dg5fs 손목 초기 자세 샘플러.

기본 wrist_init 제약:
  1. wrist z 가 cap 원점보다 40~95 mm 높다
  2. 손바닥 법선(+X)이 world down(-Z) 기준 20도 cone 안에 있다
  3. zero-pose index tip, pinky tip, wrist 를 월드 XY 로 투영한 삼각형 안에
     cap 중심 XY 가 있다
  4. cap 원통과 palm(link_base) 근사 AABB 가 충돌하지 않는다

reachable 래퍼는 위 조건을 만족한 후보 중 index/middle MCP sweep 과 optional
thumb sweep 으로 cap 옆면에 닿을 수 있는 wrist pose 만 통과시킨다.

현재 기본 샘플링은 구성법이다:
  - wrist z 를 cap 보다 위에 둔다
  - 손바닥 법선(+X)을 world down(-Z) 기준 cone 안에서 뽑는다
  - 손바닥 축 둘레의 roll 을 뽑는다
  - zero-pose wrist/index tip/pinky tip XY 삼각형 내부점을 뽑고,
    그 점이 cap 중심이 되도록 wrist xy 를 역산한다
"""

from __future__ import annotations

import os

import numpy as np
import torch

# dg5fs_right.urdf FK 로 실측한 값 (tools 로 계산).
THUMB_TIP_LOCAL = torch.tensor([0.0279, 0.1184, 0.0298])
# thumb joint_1_2 origin at zero finger pose. This is the thumb MCP/flexion
# joint position used to reject wrist poses where the knuckle starts inside cap.
THUMB_MCP_LOCAL = torch.tensor([0.0279, 0.0180, 0.0298])
INDEX_TIP_LOCAL = torch.tensor([0.0143, 0.0280, 0.1926])
MIDDLE_TIP_LOCAL = torch.tensor([0.0143, 0.0050, 0.1976])
PINKY_TIP_LOCAL = torch.tensor([0.0130, -0.0425, 0.1566])

# index/middle MCP flexion(joint_X_2) limits from dg5fs_right.urdf.
INDEX_MCP_RANGE = (0.0, 2.199114857512855)
MIDDLE_MCP_RANGE = (0.0, 2.2689280275926285)
# thumb limits from dg5fs_right.urdf. joint_1_1 은 대향/벌림, joint_1_2 는
# thumb flexion 으로 보고, joint_1_3/_4 는 sparse curl grid 로 섞는다.
THUMB_OPPOSITION_RANGE = (-1.5707963267948966, 1.5707963267948966)
THUMB_MCP_RANGE = (-2.670353755551324, 0.0)
THUMB_CURL_RANGE = (-0.17453292519943295, 1.5707963267948966)
# 실제 손바닥 쪽은 로컬 +X. URDF zero pose 에서 middle/index/ring flexion(+)
# 을 주면 fingertip 이 +X 로 말려 들어간다.
PALM_LOCAL = torch.tensor([1.0, 0.0, 0.0])
DORSAL_LOCAL = -PALM_LOCAL

# task1/assets/tumbler/meshes/cap_visual.obj bounds:
#   xy radius max ~= 0.0514 m, z ~= 0.0000~0.0308 m.
# 캡 포인트클라우드(_cap_grasp_cap_points)가 이 치수를 쓴다. 텀블러 자산을
# 바꿀 때 env 변수로 함께 맞춘다 (미설정 = 기존 실측값 그대로).
CAP_RADIUS = float(os.environ.get("CAP_CLOUD_RADIUS", "0.052"))
CAP_HEIGHT = float(os.environ.get("CAP_CLOUD_HEIGHT", "0.031"))
CAP_TARGET_MARGIN = 0.002
# task1/assets/dg5fs_hand/meshes/dg5fs_right/link_base.STL bounds.
PALM_BASE_AABB_MIN = torch.tensor([-0.0192112, -0.0410000, 0.0])
PALM_BASE_AABB_MAX = torch.tensor([0.0256337, 0.0419720, 0.1026310])

_REACH_POINTS_CACHE: dict[tuple, torch.Tensor] = {}


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _rot_from_a_to_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a 를 b 로 보내는 최소회전 행렬 (batch). 둘 다 단위벡터."""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True).unsqueeze(-1)
    s = v.norm(dim=-1, keepdim=True).unsqueeze(-1)
    n = a.shape[0]
    K = torch.zeros(n, 3, 3, device=a.device, dtype=a.dtype)
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    I = torch.eye(3, device=a.device, dtype=a.dtype).expand(n, 3, 3)
    R = I + K + K @ K * ((1 - c) / (s * s).clamp_min(1e-12))
    # a 와 b 가 정반대인 경우는 위 식이 발산한다. 그 샘플만 180° 회전으로 대체.
    flip = (s.squeeze(-1).squeeze(-1) < 1e-7) & (c.squeeze(-1).squeeze(-1) < 0)
    if flip.any():
        R[flip] = -I[flip]
    return R


def _axis_angle_R(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """축-각 -> 회전행렬 (batch). axis 는 단위벡터."""
    n = axis.shape[0]
    K = torch.zeros(n, 3, 3, device=axis.device, dtype=axis.dtype)
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
    I = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(n, 3, 3)
    s, c = ang.view(-1, 1, 1).sin(), ang.view(-1, 1, 1).cos()
    return I + s * K + (1 - c) * (K @ K)


def _rpy_matrix_np(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _origin_T(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = torch.as_tensor(_rpy_matrix_np(rpy), device=device, dtype=dtype)
    T[:3, 3] = torch.as_tensor(xyz, device=device, dtype=dtype)
    return T


def _z_axis_T(angles: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
    a = angles * sign
    c, s = a.cos(), a.sin()
    T = torch.eye(4, device=angles.device, dtype=angles.dtype).expand(angles.numel(), 4, 4).clone()
    T[:, 0, 0] = c
    T[:, 0, 1] = -s
    T[:, 1, 0] = s
    T[:, 1, 1] = c
    return T


def _finger_mcp_grid_points_local(
    finger: int,
    mcp_values: torch.Tensor,
) -> torch.Tensor:
    """index/middle MCP grid 에서 proximal~tip link origin points."""
    device, dtype = mcp_values.device, mcp_values.dtype
    if finger == 2:
        mount_y, mount_z = 0.028, 0.08365
    elif finger == 3:
        mount_y, mount_z = 0.005, 0.08865
    else:
        raise ValueError(f"Only index/middle fingers are supported, got finger={finger}")

    T1 = _origin_T((0.0017, mount_y, mount_z), (3.1416, -1.5708, 0.0), device, dtype)
    T2 = _origin_T((0.02415, 0.0, 0.0126), (-1.5708, 0.0, 0.0), device, dtype)
    T3 = _origin_T((0.0334, 0.0, 0.0), (3.1416, 0.0, 0.0), device, dtype)
    T4 = _origin_T((0.0334, 0.0, 0.0), (0.0, 0.0, 0.0), device, dtype)
    Ttip = _origin_T((0.018, 0.0, 0.0), (0.0, 0.0, 0.0), device, dtype)

    k = mcp_values.numel()
    T = T1.expand(k, 4, 4) @ T2.expand(k, 4, 4) @ _z_axis_T(mcp_values, sign=-1.0)
    points = [T[:, :3, 3]]
    T = T @ T3.expand(k, 4, 4)
    points.append(T[:, :3, 3])
    T = T @ T4.expand(k, 4, 4)
    points.append(T[:, :3, 3])
    T = T @ Ttip.expand(k, 4, 4)
    points.append(T[:, :3, 3])
    return torch.stack(points, dim=1)


def _thumb_grid_points_local(
    opposition_values: torch.Tensor,
    flex_values: torch.Tensor,
    pip_values: torch.Tensor,
    dip_values: torch.Tensor,
) -> torch.Tensor:
    """thumb joint grid 에서 proximal~tip link origin points."""
    device, dtype = opposition_values.device, opposition_values.dtype
    grid = torch.cartesian_prod(opposition_values, flex_values, pip_values, dip_values)
    q1, q2, q3, q4 = grid.unbind(dim=-1)

    T1 = _origin_T((0.0017, 0.018, 0.0298), (1.5708, 0.0, 1.5708), device, dtype)
    T2 = _origin_T((0.0, 0.0, 0.0262), (-1.5708, 0.0, 0.0), device, dtype)
    T3 = _origin_T((0.0381, 0.0, 0.0), (1.5708, 0.0, 0.0), device, dtype)
    T4 = _origin_T((0.0334, 0.0, 0.0), (0.0, 0.0, 0.0), device, dtype)
    Ttip = _origin_T((0.0289, 0.0, 0.0), (0.0, 0.0, 0.0), device, dtype)

    k = q1.numel()
    T = T1.expand(k, 4, 4) @ _z_axis_T(q1)
    T = T @ T2.expand(k, 4, 4) @ _z_axis_T(q2)
    points = [T[:, :3, 3]]
    T = T @ T3.expand(k, 4, 4) @ _z_axis_T(q3)
    points.append(T[:, :3, 3])
    T = T @ T4.expand(k, 4, 4) @ _z_axis_T(q4)
    points.append(T[:, :3, 3])
    T = T @ Ttip.expand(k, 4, 4)
    points.append(T[:, :3, 3])
    return torch.stack(points, dim=1)


def _polyline_sample_points(points: torch.Tensor, segment_samples: int) -> torch.Tensor:
    """Polyline vertices (..., P, 3) -> sampled segment points (..., M, 3)."""
    if segment_samples <= 1:
        return points
    a = points[..., :-1, :]
    b = points[..., 1:, :]
    alpha = torch.linspace(0.0, 1.0, segment_samples, device=points.device, dtype=points.dtype)
    sampled = a[..., :, None, :] * (1.0 - alpha.view(-1, 1)) + b[..., :, None, :] * alpha.view(-1, 1)
    return sampled.reshape(*points.shape[:-2], -1, 3)


def mcp_reach_points_local(
    grid_steps: int = 17,
    segment_samples: int = 5,
    tip_only: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """MCP min-max grid 에서 index/middle skeleton 또는 fingertip points.

    Returns:
        (2, M, 3) tensor. finger order: index, middle.
    """
    device = torch.device("cpu") if device is None else device
    grid_steps = max(2, int(grid_steps))
    segment_samples = max(1, int(segment_samples))
    key = ("mcp", str(device), dtype, grid_steps, segment_samples, bool(tip_only))
    if key in _REACH_POINTS_CACHE:
        return _REACH_POINTS_CACHE[key]
    index_q = torch.linspace(INDEX_MCP_RANGE[0], INDEX_MCP_RANGE[1], grid_steps, device=device, dtype=dtype)
    middle_q = torch.linspace(MIDDLE_MCP_RANGE[0], MIDDLE_MCP_RANGE[1], grid_steps, device=device, dtype=dtype)
    index_grid = _finger_mcp_grid_points_local(2, index_q)
    middle_grid = _finger_mcp_grid_points_local(3, middle_q)
    if tip_only:
        index_points = index_grid[:, -1:, :]
        middle_points = middle_grid[:, -1:, :]
    else:
        index_points = _polyline_sample_points(index_grid, segment_samples)
        middle_points = _polyline_sample_points(middle_grid, segment_samples)
    reach_points = torch.stack([index_points.reshape(-1, 3), middle_points.reshape(-1, 3)], dim=0)
    _REACH_POINTS_CACHE[key] = reach_points
    return reach_points


def thumb_reach_points_local(
    opposition_steps: int = 7,
    flex_steps: int = 11,
    curl_steps: int = 3,
    segment_samples: int = 5,
    tip_only: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Loose thumb reachability grid sampled at thumb tip or along skeleton."""
    device = torch.device("cpu") if device is None else device
    opposition_steps = max(2, int(opposition_steps))
    flex_steps = max(2, int(flex_steps))
    curl_steps = max(1, int(curl_steps))
    segment_samples = max(1, int(segment_samples))
    key = ("thumb", str(device), dtype, opposition_steps, flex_steps, curl_steps, segment_samples, bool(tip_only))
    if key in _REACH_POINTS_CACHE:
        return _REACH_POINTS_CACHE[key]
    q1 = torch.linspace(*THUMB_OPPOSITION_RANGE, opposition_steps, device=device, dtype=dtype)
    q2 = torch.linspace(*THUMB_MCP_RANGE, flex_steps, device=device, dtype=dtype)
    q3 = torch.linspace(*THUMB_CURL_RANGE, curl_steps, device=device, dtype=dtype)
    q4 = torch.linspace(*THUMB_CURL_RANGE, curl_steps, device=device, dtype=dtype)
    points = _thumb_grid_points_local(q1, q2, q3, q4)
    if tip_only:
        reach_points = points[:, -1, :]
    else:
        reach_points = _polyline_sample_points(points, segment_samples).reshape(-1, 3)
    _REACH_POINTS_CACHE[key] = reach_points
    return reach_points


def _ray_hits_cap_cylinder(
    origin: torch.Tensor,
    direction: torch.Tensor,
    cap_pos: torch.Tensor,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
) -> torch.Tensor:
    """Ray(origin, direction) 이 cap 원통을 통과하는지 본다.

    cap_pos 는 cap link 원점, 즉 cap 밑면 중심이다.
    """
    d = _normalize(direction)
    o_xy = origin[:, :2] - cap_pos[:, :2]
    d_xy = d[:, :2]
    a = (d_xy * d_xy).sum(-1)
    b = 2.0 * (o_xy * d_xy).sum(-1)
    c = (o_xy * o_xy).sum(-1) - cap_radius**2
    z0 = cap_pos[:, 2]
    z1 = z0 + cap_height

    disc = b * b - 4.0 * a * c
    valid_side = (a > 1e-12) & (disc >= 0.0)
    sqrt_disc = disc.clamp_min(0.0).sqrt()
    denom = (2.0 * a).clamp_min(1e-12)
    t_side0 = (-b - sqrt_disc) / denom
    t_side1 = (-b + sqrt_disc) / denom
    z_side0 = origin[:, 2] + t_side0 * d[:, 2]
    z_side1 = origin[:, 2] + t_side1 * d[:, 2]
    side_hit = valid_side & (
        ((t_side0 >= 0.0) & (z_side0 >= z0) & (z_side0 <= z1))
        | ((t_side1 >= 0.0) & (z_side1 >= z0) & (z_side1 <= z1))
    )

    valid_plane = d[:, 2].abs() > 1e-12
    dz_safe = torch.where(valid_plane, d[:, 2], torch.ones_like(d[:, 2]))
    t_bot = (z0 - origin[:, 2]) / dz_safe
    t_top = (z1 - origin[:, 2]) / dz_safe
    p_bot = origin[:, :2] + t_bot[:, None] * d_xy - cap_pos[:, :2]
    p_top = origin[:, :2] + t_top[:, None] * d_xy - cap_pos[:, :2]
    cap_hit = valid_plane & (
        ((t_bot >= 0.0) & ((p_bot * p_bot).sum(-1) <= cap_radius**2))
        | ((t_top >= 0.0) & ((p_top * p_top).sum(-1) <= cap_radius**2))
    )

    return side_hit | cap_hit


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """회전행렬 -> wxyz 쿼터니언 (IsaacLab 관례)."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)
    s = torch.sqrt((t + 1.0).clamp_min(1e-12)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / s
    q[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / s
    q[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / s
    # t 가 작으면 위 식이 불안정하다. 가장 큰 대각성분 기준 분기로 대체.
    bad = t <= 0
    if bad.any():
        mb = m[bad]
        d = torch.stack([mb[:, 0, 0], mb[:, 1, 1], mb[:, 2, 2]], -1)
        i = d.argmax(-1)
        qb = torch.zeros(mb.shape[0], 4, device=m.device, dtype=m.dtype)
        for k in range(3):
            sel = i == k
            if not sel.any():
                continue
            a, b, c = k, (k + 1) % 3, (k + 2) % 3
            ms = mb[sel]
            ss = torch.sqrt((1.0 + ms[:, a, a] - ms[:, b, b] - ms[:, c, c]).clamp_min(1e-12)) * 2
            qb[sel, 0] = (ms[:, c, b] - ms[:, b, c]) / ss
            qb[sel, a + 1] = 0.25 * ss
            qb[sel, b + 1] = (ms[:, b, a] + ms[:, a, b]) / ss
            qb[sel, c + 1] = (ms[:, c, a] + ms[:, a, c]) / ss
        q[bad] = qb
    return _normalize(q)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """wxyz 쿼터니언 -> 회전행렬."""
    q = _normalize(q)
    w, x, y, z = q.unbind(-1)
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _cap_side_contact_from_local_groups(
    local_groups: torch.Tensor,
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    contact_margin: float,
    cap_radius: float,
    cap_height: float,
    side_z_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local point groups (F,M,3) 가 cap side solid 근방에 닿는지 판정.

    Returns:
        group_touch: (N,F) 그룹별 side 접촉 가능 여부
        min_dist:    (N,F) cylinder side solid 근사 최소거리
        azimuth:     (N,F) 최근접점의 cap 중심축 기준 방위각 (rad, [-pi,pi])
    """
    R = quat_to_mat(wrist_quat_wxyz)
    points_w = wrist_pos[:, None, None, :] + torch.einsum("bij,fmj->bfmi", R, local_groups)
    rel = points_w - cap_pos[:, None, None, :]
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    z = rel[..., 2]

    z_ok = (z >= side_z_margin - contact_margin) & (z <= cap_height - side_z_margin + contact_margin)
    radial_dist = (radial - cap_radius).abs()
    radial_ok = radial_dist <= contact_margin
    group_touch = (z_ok & radial_ok).any(dim=-1)

    z_out = torch.maximum(
        (side_z_margin - z).clamp_min(0.0),
        (z - (cap_height - side_z_margin)).clamp_min(0.0),
    )
    dist = torch.sqrt(radial_dist * radial_dist + z_out * z_out)
    best = dist.argmin(dim=-1, keepdim=True)
    min_dist = torch.gather(dist, -1, best).squeeze(-1)
    # 그룹별 대표 접촉점(=최근접점)의 방위각. 손가락들이 캡의 같은 쪽만 미는지,
    # 서로 마주보는지를 판정하는 데 쓴다.
    rel_xy = torch.gather(rel[..., :2], 2, best.unsqueeze(-1).expand(-1, -1, -1, 2)).squeeze(2)
    azimuth = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
    return group_touch, min_dist, azimuth


def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """라디안을 [-pi, pi] 로 감싼다."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def mcp_reach_contact_mask(
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    contact_margin: float = 0.008,
    min_reach_fingers: int = 1,
    grid_steps: int = 17,
    segment_samples: int = 5,
    tip_only: bool = False,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
    side_z_margin: float = 0.001,
    return_azimuth: bool = False,
):
    """index/middle MCP sweep 로 cap side reachability 를 빠르게 판정한다.

    기본은 예전처럼 skeleton 선분 샘플 중 어떤 점이 cap side 근방에 들어오면
    허용한다. tip_only=True 로 두면 fingertip 위치만 본다.

    Returns:
        reach_ok: (N,) min_reach_fingers 조건을 만족하는지
        finger_touch: (N,2) [index, middle] reachability
        min_dist: (N,2) finger 별 cylinder solid 근사 최소거리
        azimuth: (N,2) return_azimuth=True 일 때만. 최근접점의 방위각 (rad)
    """
    device, dtype = wrist_pos.device, wrist_pos.dtype
    reach_local = mcp_reach_points_local(grid_steps, segment_samples, tip_only, device, dtype)
    finger_touch, min_dist, azimuth = _cap_side_contact_from_local_groups(
        reach_local,
        wrist_pos,
        wrist_quat_wxyz,
        cap_pos,
        contact_margin,
        cap_radius,
        cap_height,
        side_z_margin,
    )
    reach_ok = finger_touch.to(torch.int32).sum(dim=-1) >= int(min_reach_fingers)
    if return_azimuth:
        return reach_ok, finger_touch, min_dist, azimuth
    return reach_ok, finger_touch, min_dist


def thumb_reach_contact_mask(
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    contact_margin: float = 0.020,
    opposition_steps: int = 7,
    flex_steps: int = 11,
    curl_steps: int = 3,
    segment_samples: int = 5,
    tip_only: bool = False,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
    side_z_margin: float = 0.001,
    return_azimuth: bool = False,
):
    """thumb joint grid 로 cap side reachability 를 느슨하게 판정한다.

    return_azimuth=True 이면 최근접 접촉점의 방위각(rad)도 함께 돌려준다.
    """
    device, dtype = wrist_pos.device, wrist_pos.dtype
    reach_local = thumb_reach_points_local(
        opposition_steps=opposition_steps,
        flex_steps=flex_steps,
        curl_steps=curl_steps,
        segment_samples=segment_samples,
        tip_only=tip_only,
        device=device,
        dtype=dtype,
    )[None, :, :]
    touch, min_dist, azimuth = _cap_side_contact_from_local_groups(
        reach_local,
        wrist_pos,
        wrist_quat_wxyz,
        cap_pos,
        contact_margin,
        cap_radius,
        cap_height,
        side_z_margin,
    )
    if return_azimuth:
        return touch[:, 0], min_dist[:, 0], azimuth[:, 0]
    return touch[:, 0], min_dist[:, 0]


def thumb_mcp_outside_cap_mask(
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    cap_radius: float = CAP_RADIUS,
    outside_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """thumb MCP(joint_1_2 origin) 이 cap 중심축 반지름 밖에 있는지 판정한다."""
    device, dtype = wrist_pos.device, wrist_pos.dtype
    R = quat_to_mat(wrist_quat_wxyz)
    thumb_mcp_local = THUMB_MCP_LOCAL.to(device=device, dtype=dtype)
    thumb_mcp_w = wrist_pos + torch.einsum("bij,j->bi", R, thumb_mcp_local)
    radial = torch.linalg.norm(thumb_mcp_w[:, :2] - cap_pos[:, :2], dim=-1)
    return radial >= (float(cap_radius) + float(outside_margin)), radial


def _cap_collision_points_local(
    cap_radius: float,
    cap_height: float,
    radial_steps: int,
    z_steps: int,
    phi_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cap cylinder volume/surface probe points in cap-local coordinates."""
    radial_steps = max(2, int(radial_steps))
    z_steps = max(2, int(z_steps))
    phi_steps = max(6, int(phi_steps))
    key = ("cap_collision", str(device), dtype, float(cap_radius), float(cap_height), radial_steps, z_steps, phi_steps)
    if key in _REACH_POINTS_CACHE:
        return _REACH_POINTS_CACHE[key]
    radii = torch.linspace(0.0, float(cap_radius), radial_steps, device=device, dtype=dtype)
    zs = torch.linspace(0.0, float(cap_height), z_steps, device=device, dtype=dtype)
    phi = torch.arange(phi_steps, device=device, dtype=dtype) * (2.0 * np.pi / phi_steps)
    rr, zz, pp = torch.meshgrid(radii, zs, phi, indexing="ij")
    points = torch.stack([rr * pp.cos(), rr * pp.sin(), zz], dim=-1).reshape(-1, 3)
    centers = torch.stack(
        [
            torch.zeros_like(zs),
            torch.zeros_like(zs),
            zs,
        ],
        dim=-1,
    )
    points = torch.cat([points, centers], dim=0)
    _REACH_POINTS_CACHE[key] = points
    return points


def _palm_aabb_probe_points_local(
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sparse probe points on the palm AABB for the inverse box-in-cylinder case."""
    key = ("palm_aabb_probe", str(device), dtype)
    if key in _REACH_POINTS_CACHE:
        return _REACH_POINTS_CACHE[key]
    lo = PALM_BASE_AABB_MIN.to(device=device, dtype=dtype)
    hi = PALM_BASE_AABB_MAX.to(device=device, dtype=dtype)
    xs = torch.linspace(lo[0], hi[0], 4, device=device, dtype=dtype)
    ys = torch.linspace(lo[1], hi[1], 5, device=device, dtype=dtype)
    zs = torch.linspace(lo[2], hi[2], 6, device=device, dtype=dtype)
    x, y, z = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    on_surface = (
        torch.isclose(points[:, 0], lo[0]) | torch.isclose(points[:, 0], hi[0])
        | torch.isclose(points[:, 1], lo[1]) | torch.isclose(points[:, 1], hi[1])
        | torch.isclose(points[:, 2], lo[2]) | torch.isclose(points[:, 2], hi[2])
    )
    points = points[on_surface]
    _REACH_POINTS_CACHE[key] = points
    return points


def cap_palm_collision_mask(
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
    palm_margin: float = 0.002,
    cap_radial_steps: int = 4,
    cap_z_steps: int = 5,
    cap_phi_steps: int = 16,
) -> torch.Tensor:
    """Approximate cap cylinder vs palm(link_base) collision.

    The palm is approximated by the link_base mesh AABB in wrist-local frame.
    The cap is upright with origin at its bottom center, matching cap-only init.
    """
    device, dtype = wrist_pos.device, wrist_pos.dtype
    R = quat_to_mat(wrist_quat_wxyz)
    Rt = R.transpose(1, 2)
    margin = float(palm_margin)

    cap_points = _cap_collision_points_local(
        cap_radius,
        cap_height,
        cap_radial_steps,
        cap_z_steps,
        cap_phi_steps,
        device,
        dtype,
    )
    cap_points_w = cap_pos[:, None, :] + cap_points[None, :, :]
    cap_points_in_palm = torch.einsum("bij,bpj->bpi", Rt, cap_points_w - wrist_pos[:, None, :])
    lo = PALM_BASE_AABB_MIN.to(device=device, dtype=dtype) - margin
    hi = PALM_BASE_AABB_MAX.to(device=device, dtype=dtype) + margin
    cap_probe_inside_palm = ((cap_points_in_palm >= lo) & (cap_points_in_palm <= hi)).all(dim=-1).any(dim=-1)

    palm_points = _palm_aabb_probe_points_local(device, dtype)
    palm_points_w = wrist_pos[:, None, :] + torch.einsum("bij,pj->bpi", R, palm_points)
    rel = palm_points_w - cap_pos[:, None, :]
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    z = rel[..., 2]
    palm_probe_inside_cap = (
        (radial <= float(cap_radius) + margin)
        & (z >= -margin)
        & (z <= float(cap_height) + margin)
    ).any(dim=-1)
    return cap_probe_inside_palm | palm_probe_inside_cap


def cap_center_in_zero_triangle_xy_mask(
    wrist_pos: torch.Tensor,
    wrist_quat_wxyz: torch.Tensor,
    cap_pos: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Zero joint pose 의 wrist-index tip-pinky tip XY 삼각형 안에 cap 중심이 있는지 판정."""
    device, dtype = wrist_pos.device, wrist_pos.dtype
    R = quat_to_mat(wrist_quat_wxyz)
    index_local = INDEX_TIP_LOCAL.to(device=device, dtype=dtype)
    pinky_local = PINKY_TIP_LOCAL.to(device=device, dtype=dtype)

    a = wrist_pos[:, :2]
    b = (wrist_pos + torch.einsum("bij,j->bi", R, index_local))[:, :2]
    c = (wrist_pos + torch.einsum("bij,j->bi", R, pinky_local))[:, :2]
    p = cap_pos[:, :2]

    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = (v0 * v0).sum(dim=-1)
    dot01 = (v0 * v1).sum(dim=-1)
    dot02 = (v0 * v2).sum(dim=-1)
    dot11 = (v1 * v1).sum(dim=-1)
    dot12 = (v1 * v2).sum(dim=-1)
    denom = dot00 * dot11 - dot01 * dot01
    inv_denom = 1.0 / denom.clamp_min(eps)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    valid = denom > eps
    return valid & (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps)


def sample_wrist_init(
    cap_pos: torch.Tensor,
    z_range: tuple[float, float] = (0.040, 0.095),
    r_range: tuple[float, float] = (0.045, 0.075),
    palm_tol_deg: float = 25.0,
    dorsal_min_z: float = 0.05,
    palm_down_max_deg: float | None = None,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
    cap_target_margin: float = CAP_TARGET_MARGIN,
    thumb_mcp_outside_margin: float | None = None,
    require_cap_center_in_zero_triangle: bool = True,
    reject_palm_collision: bool = True,
    palm_collision_margin: float = 0.002,
    tip_local: torch.Tensor | None = None,
    dorsal_local: torch.Tensor | None = None,
    max_tries: int = 20,
    generator: torch.Generator | None = None,
):
    """캡이 손바닥 아래에 오는 손목 위치/자세를 뽑는다.

    Args:
        cap_pos: (N,3) cap link 원점 월드 좌표. 현재 cap 밑면 중심.
        z_range: cap_pos 대비 wrist 높이차 범위 (m).
        r_range: cap 중심축 기준 wrist xy 평면거리 범위 (m).
                 triangle 구성법에서는 wrist xy 를 역산하므로 쓰지 않는다.
        palm_tol_deg: 예전 중심 조준 조건과의 호출 호환용. 현재는 쓰지 않는다.
        dorsal_min_z: dorsal.z 하한. 0 이면 요청 그대로("양수"), 값을 올리면
                      손등이 더 확실히 위를 보게 된다.
        palm_down_max_deg: 주어지면 palm 법선(+X)이 월드 아래 방향(-Z)에서
                           이 각도 이내인 후보만 통과시킨다.
        cap_radius: legacy palm-ray 검사에 쓰는 cap 원통 반지름 (m).
        cap_height: legacy palm-ray 검사에 쓰는 cap 원통 높이 (m).
        cap_target_margin: legacy ray target 을 살짝 안쪽에서 뽑는 margin (m).
        thumb_mcp_outside_margin: thumb MCP 가 cap 반지름보다 추가로 떨어져야 하는 거리.
                                  None 이면 이 조건을 끈다.
        require_cap_center_in_zero_triangle: joint 0 pose 에서 wrist-index tip-pinky tip
                                             XY 삼각형 안에 cap 중심이 있어야 하는지.
        reject_palm_collision: True 이면 cap 원통과 palm(link_base) 근사 AABB 충돌을 제외한다.
        palm_collision_margin: palm AABB/cap cylinder 충돌 판정 여유거리 (m).
    Returns:
        pos:  (N,3) 손목 월드 위치
        quat: (N,4) 손목 월드 자세 (wxyz)
        ok:   (N,) 제약을 모두 만족했는지
    """
    device, dtype = cap_pos.device, cap_pos.dtype
    n = cap_pos.shape[0]
    # tip_local 은 middle-tip 조준을 쓰던 예전 호출과의 호환용이다.
    _ = tip_local, palm_tol_deg
    palm = _normalize(PALM_LOCAL.to(device, dtype) if dorsal_local is None else -dorsal_local.to(device, dtype))
    thumb_mcp_local = THUMB_MCP_LOCAL.to(device=device, dtype=dtype)
    index_local = INDEX_TIP_LOCAL.to(device=device, dtype=dtype)
    pinky_local = PINKY_TIP_LOCAL.to(device=device, dtype=dtype)

    def rand(*shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, device=device, dtype=dtype, generator=generator) * (hi - lo) + lo

    pos = torch.zeros(n, 3, device=device, dtype=dtype)
    quat = torch.zeros(n, 4, device=device, dtype=dtype)
    ok = torch.zeros(n, dtype=torch.bool, device=device)

    todo = torch.arange(n, device=device)
    for _ in range(max_tries):
        if todo.numel() == 0:
            break
        m = todo.numel()

        if require_cap_center_in_zero_triangle and palm_down_max_deg is not None:
            dz = rand(m, lo=z_range[0], hi=z_range[1])
            max_theta = float(np.deg2rad(palm_down_max_deg))
            cos_theta = rand(m, lo=float(np.cos(max_theta)), hi=1.0)
            sin_theta = (1.0 - cos_theta * cos_theta).clamp_min(0.0).sqrt()
            phi = rand(m, lo=0.0, hi=2 * np.pi)
            palm_wanted = torch.stack(
                [sin_theta * phi.cos(), sin_theta * phi.sin(), -cos_theta],
                dim=-1,
            )

            R0 = _rot_from_a_to_b(palm.expand(m, 3), palm_wanted)
            psi = rand(m, lo=0.0, hi=2 * np.pi)
            R = torch.einsum("bij,bjk->bik", _axis_angle_R(palm_wanted, psi), R0)

            index_offset_xy = torch.einsum("bij,j->bi", R, index_local)[:, :2]
            pinky_offset_xy = torch.einsum("bij,j->bi", R, pinky_local)[:, :2]
            u = rand(m)
            v = rand(m)
            flip = (u + v) > 1.0
            u = torch.where(flip, 1.0 - u, u)
            v = torch.where(flip, 1.0 - v, v)
            cap_offset_xy = u[:, None] * index_offset_xy + v[:, None] * pinky_offset_xy

            p = torch.zeros((m, 3), device=device, dtype=dtype)
            p[:, :2] = cap_pos[todo, :2] - cap_offset_xy
            p[:, 2] = cap_pos[todo, 2] + dz

            palm_w = torch.einsum("bij,j->bi", R, palm)
            dorsal_w = -palm_w
            good = dorsal_w[:, 2] > dorsal_min_z
            good = good & cap_center_in_zero_triangle_xy_mask(p, mat_to_quat(R), cap_pos[todo])
            if thumb_mcp_outside_margin is not None:
                thumb_mcp_w = p + torch.einsum("bij,j->bi", R, thumb_mcp_local)
                thumb_mcp_radial = torch.linalg.norm(thumb_mcp_w[:, :2] - cap_pos[todo, :2], dim=-1)
                good = good & (thumb_mcp_radial >= float(cap_radius) + float(thumb_mcp_outside_margin))
            if reject_palm_collision:
                good = good & ~cap_palm_collision_mask(
                    p,
                    mat_to_quat(R),
                    cap_pos[todo],
                    cap_radius=cap_radius,
                    cap_height=cap_height,
                    palm_margin=palm_collision_margin,
                )

            idx = todo[good]
            pos[idx], quat[idx], ok[idx] = p[good], mat_to_quat(R[good]), True
            todo = todo[~good]
            continue

        # --- 제약 1, 2: 원기둥 껍질에서 위치 ---
        phi = rand(m, lo=0.0, hi=2 * np.pi)
        # 면적 균일하게 뽑으려면 r^2 을 균일하게.
        h = torch.sqrt(rand(m, lo=r_range[0] ** 2, hi=r_range[1] ** 2))
        dz = rand(m, lo=z_range[0], hi=z_range[1])
        p = cap_pos[todo] + torch.stack([h * phi.cos(), h * phi.sin(), dz], -1)

        # --- 손바닥 법선이 통과할 cap 내부 target point ---
        target_phi = rand(m, lo=0.0, hi=2 * np.pi)
        target_r = torch.sqrt(rand(m, lo=0.0, hi=max(cap_radius - cap_target_margin, 1e-4) ** 2))
        target_z_hi = torch.minimum(
            torch.full((m,), cap_height - cap_target_margin, device=device, dtype=dtype),
            dz - cap_target_margin,
        ).clamp_min(cap_target_margin)
        target_z = rand(m) * (target_z_hi - cap_target_margin) + cap_target_margin
        target = cap_pos[todo] + torch.stack(
            [target_r * target_phi.cos(), target_r * target_phi.sin(), target_z],
            -1,
        )
        a = _normalize(target - p)

        # --- 손바닥 법선을 a 에 맞추는 기준 회전 ---
        R0 = _rot_from_a_to_b(palm.expand(m, 3), a)

        # --- 손바닥 축 둘레 1 자유도는 완전히 자유. 캡을 감싸는 방향이 다양해진다 ---
        psi = rand(m, lo=0.0, hi=2 * np.pi)
        R = torch.einsum("bij,bjk->bik", _axis_angle_R(a, psi), R0)

        # --- 검증 ---
        palm_w = torch.einsum("bij,j->bi", R, palm)
        dorsal_w = -palm_w
        cap_hit = _ray_hits_cap_cylinder(p, palm_w, cap_pos[todo], cap_radius, cap_height)
        good = (dorsal_w[:, 2] > dorsal_min_z) & cap_hit
        if palm_down_max_deg is not None:
            down_cos = float(np.cos(np.deg2rad(palm_down_max_deg)))
            good = good & ((-palm_w[:, 2]) >= down_cos)
        if thumb_mcp_outside_margin is not None:
            thumb_mcp_w = p + torch.einsum("bij,j->bi", R, thumb_mcp_local)
            thumb_mcp_radial = torch.linalg.norm(thumb_mcp_w[:, :2] - cap_pos[todo, :2], dim=-1)
            good = good & (thumb_mcp_radial >= float(cap_radius) + float(thumb_mcp_outside_margin))
        if require_cap_center_in_zero_triangle:
            cand_quat = mat_to_quat(R)
            good = good & cap_center_in_zero_triangle_xy_mask(p, cand_quat, cap_pos[todo])
        if reject_palm_collision:
            cand_quat = mat_to_quat(R)
            good = good & ~cap_palm_collision_mask(
                p,
                cand_quat,
                cap_pos[todo],
                cap_radius=cap_radius,
                cap_height=cap_height,
                palm_margin=palm_collision_margin,
            )

        idx = todo[good]
        pos[idx], quat[idx], ok[idx] = p[good], mat_to_quat(R[good]), True
        todo = todo[~good]

    return pos, quat, ok


def sample_wrist_init_mcp_reachable(
    cap_pos: torch.Tensor,
    z_range: tuple[float, float] = (0.040, 0.095),
    r_range: tuple[float, float] = (0.045, 0.075),
    palm_tol_deg: float = 25.0,
    dorsal_min_z: float = 0.05,
    palm_down_max_deg: float | None = None,
    contact_margin: float = 0.020,
    min_reach_fingers: int = 1,
    require_thumb: bool = True,
    thumb_contact_margin: float = 0.025,
    thumb_mcp_outside_margin: float | None = None,
    thumb_opposition_steps: int = 7,
    thumb_flex_steps: int = 11,
    thumb_curl_steps: int = 3,
    grid_steps: int = 17,
    segment_samples: int = 5,
    cap_radius: float = CAP_RADIUS,
    cap_height: float = CAP_HEIGHT,
    cap_target_margin: float = CAP_TARGET_MARGIN,
    require_cap_center_in_zero_triangle: bool = True,
    reject_palm_collision: bool = True,
    palm_collision_margin: float = 0.002,
    opposition_min_deg: float | None = None,
    max_tries: int = 3000,
    generator: torch.Generator | None = None,
):
    """index/middle MCP sweep 에서 cap side 접촉 가능한 wrist pose 만 뽑는다.

    먼저 기존 palm-ray wrist 조건을 만족하는 후보를 만들고, 그 wrist pose 에서
    index/middle MCP 를 min~max 로 훑었을 때 finger skeleton point 가 cap side
    cylinder 에 닿을 수 있는 후보만 통과시킨다. require_thumb=True 이면
    thumb 도 더 널널한 margin 으로 side reach 가능해야 한다.
    thumb_mcp_outside_margin 이 None 이 아니면 thumb MCP(joint_1_2 origin) 는
    cap 반지름 밖에 있어야 한다.
    zero-pose wrist-index tip-pinky tip XY 삼각형 안에 cap 중심이 있어야 한다.
    reject_palm_collision=True 이면 cap 과 palm(link_base) 충돌 후보를 제외한다.

    opposition_min_deg 가 주어지면 "닿을 수 있는가" 뿐 아니라 "잡을 수 있는가"
    까지 본다. thumb 의 도달 접촉점과 index/middle 중 하나의 접촉점이 cap
    중심축 기준으로 이 각도 이상 벌어져야 통과한다. 이 조건이 없으면 세 손가락이
    모두 림의 같은 쪽을 미는 자세도 전부 통과하는데, 지름 104 mm 원반은 대향
    파지가 아니면 들리지 않는다 (Stage-0 실측 closure_err ~= 0.97).
    """
    device, dtype = cap_pos.device, cap_pos.dtype
    n = cap_pos.shape[0]
    pos = torch.zeros(n, 3, device=device, dtype=dtype)
    quat = torch.zeros(n, 4, device=device, dtype=dtype)
    ok = torch.zeros(n, dtype=torch.bool, device=device)

    todo = torch.arange(n, device=device)
    for _ in range(max_tries):
        if todo.numel() == 0:
            break
        cand_pos, cand_quat, cand_ok = sample_wrist_init(
            cap_pos[todo],
            z_range=z_range,
            r_range=r_range,
            palm_tol_deg=palm_tol_deg,
            dorsal_min_z=dorsal_min_z,
            palm_down_max_deg=palm_down_max_deg,
            cap_radius=cap_radius,
            cap_height=cap_height,
            cap_target_margin=cap_target_margin,
            thumb_mcp_outside_margin=None,
            require_cap_center_in_zero_triangle=require_cap_center_in_zero_triangle,
            reject_palm_collision=reject_palm_collision,
            palm_collision_margin=palm_collision_margin,
            max_tries=3,
            generator=generator,
        )
        reach_ok, finger_touch, _, finger_az = mcp_reach_contact_mask(
            cand_pos,
            cand_quat,
            cap_pos[todo],
            contact_margin=contact_margin,
            min_reach_fingers=min_reach_fingers,
            grid_steps=grid_steps,
            segment_samples=segment_samples,
            cap_radius=cap_radius,
            cap_height=cap_height,
            return_azimuth=True,
        )
        good = cand_ok & reach_ok
        if thumb_mcp_outside_margin is not None:
            thumb_mcp_ok, _ = thumb_mcp_outside_cap_mask(
                cand_pos,
                cand_quat,
                cap_pos[todo],
                cap_radius=cap_radius,
                outside_margin=thumb_mcp_outside_margin,
            )
            good = good & thumb_mcp_ok
        if require_thumb or opposition_min_deg is not None:
            thumb_ok, _, thumb_az = thumb_reach_contact_mask(
                cand_pos,
                cand_quat,
                cap_pos[todo],
                contact_margin=thumb_contact_margin,
                opposition_steps=thumb_opposition_steps,
                flex_steps=thumb_flex_steps,
                curl_steps=thumb_curl_steps,
                segment_samples=segment_samples,
                cap_radius=cap_radius,
                cap_height=cap_height,
                return_azimuth=True,
            )
            if require_thumb:
                good = good & thumb_ok
            if opposition_min_deg is not None:
                # thumb 접촉점과 index/middle 접촉점이 cap 축 기준으로 충분히
                # 벌어져야 대향 파지(force closure)가 가능하다.
                sep = _wrap_angle(finger_az - thumb_az[:, None]).abs()
                opposed = (sep >= float(np.deg2rad(opposition_min_deg))) & finger_touch
                good = good & thumb_ok & opposed.any(dim=-1)
        idx = todo[good]
        pos[idx], quat[idx], ok[idx] = cand_pos[good], cand_quat[good], True
        todo = todo[~good]

    return pos, quat, ok


if __name__ == "__main__":
    torch.manual_seed(0)
    N = 20000
    cap = torch.zeros(N, 3)
    cap[:, 2] = 0.225  # 현재 텀블러 캡 밑면 z

    print("cap-hit / dorsal_min_z 스윕:")
    for mz in (0.05, 0.10, 0.20, 0.30):
        pos, quat, ok = sample_wrist_init(cap, dorsal_min_z=mz)
        p, c = pos[ok], cap[ok]
        dz = p[:, 2] - c[:, 2]
        h = (p[:, :2] - c[:, :2]).norm(dim=-1)
        print(f"  min_z {mz:.2f}  성공률 {ok.float().mean()*100:5.1f}%   "
              f"dz {dz.min()*1000:5.1f}~{dz.max()*1000:5.1f}mm   h {h.min()*1000:5.1f}~{h.max()*1000:5.1f}mm")

    pos, quat, ok = sample_wrist_init(cap, dorsal_min_z=0.05)
    p, c = pos[ok], cap[ok]
    q = quat[ok]
    w, x, y, z = q.unbind(-1)
    R = torch.zeros(p.shape[0], 3, 3)
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = 1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = 2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x)
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = 2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)
    palm_w = torch.einsum("bij,j->bi", R, PALM_LOCAL)
    dorsal_w = -palm_w
    cap_hit = _ray_hits_cap_cylinder(p, palm_w, c)
    dz = p[:, 2] - c[:, 2]
    h = (p[:, :2] - c[:, :2]).norm(dim=-1)
    dist = (c - p).norm(dim=-1)
    tip_w = p + torch.einsum("bij,j->bi", R, MIDDLE_TIP_LOCAL)
    print("\n제약 확인 (cap-hit, dorsal_min_z 0.05, 쿼터니언 왕복 후):")
    print(f"  1. dz                  : {dz.min()*1000:6.1f} ~ {dz.max()*1000:6.1f} mm")
    print(f"  2. h                   : {h.min()*1000:6.1f} ~ {h.max()*1000:6.1f} mm")
    print(f"  3. dorsal.z > 0        : min {dorsal_w[:,2].min():+.4f}   {'OK' if dorsal_w[:,2].min()>0 else 'FAIL'}")
    print(f"  4. palm ray hits cap   : {cap_hit.float().mean()*100:5.1f}%   {'OK' if cap_hit.all() else 'FAIL'}")
    print(f"\n  참고  wrist-cap 거리  {dist.min()*1000:.1f}~{dist.max()*1000:.1f} mm   (middle tip 닿는 거리 198.2mm)")
    print(f"        middle tip z   {tip_w[:,2].min():.3f}~{tip_w[:,2].max():.3f} m   (캡 z=0.225)")
