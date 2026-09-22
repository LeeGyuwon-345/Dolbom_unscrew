"""Reward and success terms for the dg5fs cap-unscrew task.

Task
----
The cap is a link of the tumbler articulation, held on a screw joint: turning it
200 deg raises it 10 mm and it comes free. The policy drives the wrist (6) and
all finger joints (20). Success is unscrewing the cap all the way while still
holding it.

Why this replaces cap_grasp_rl_terms
------------------------------------
That module scored "the cap ended up 30 mm higher", which a screw-constrained
cap cannot do without being turned -- and which, on the pedestal, turned out to
be farmable three different ways. Measured over a 10k-epoch run there: reward
climbed 757 -> 1755 while success sat at ~1.8%, because a cap batted into the
air, scooped up tilted at 57 deg, or launched at 19 m/s all collected the same
full lift reward as a clean grasp.

The unscrew angle is a much harder quantity to fake. It only advances if the
hand applies a *couple* about the cap axis, which needs opposed contacts on the
cap wall; a slap or a one-sided push cannot produce sustained rotation against
the joint. Two further choices close the remaining gaps:

  * progress is scored on the *current* angle, not the rate. A rate term is
    farmable by turning forward fast and letting it slip back slowly, since the
    clamp makes the two directions asymmetric. Scoring the angle makes any
    oscillation net exactly zero and rewards holding what you gained.
  * progress is gated on an opposed grasp, so spinning the cap by batting it
    (the joint has almost no damping) pays nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class CapUnscrewConfig:
    # --- thread ----------------------------------------------------------
    # 200 deg of turn = 10 mm of rise = free of the thread. Must match the
    # asset: make_screw_asset.py --pitch 0.018 --max-lift 0.010.
    unscrew_target_rad: float = math.radians(200.0)
    # How far the cap must come up, measured on the cap_lift joint -- i.e.
    # relative to the tumbler body, not in world z. The joint is the right
    # datum twice over: it is zero at reset by construction, and it stays
    # meaningful even if the body itself ever moves.
    #
    # The thread only lifts the cap 10 mm (200 deg); past that is free stroke,
    # where the drive is off and nothing holds the cap up. So anything above
    # 10 mm can only be the hand carrying it -- that is what makes this a
    # separation test rather than a restatement of the turn angle.
    #
    # 12 mm = thread 10 mm + 2 mm carried. Note this does NOT fully clear the
    # body collar, which shrouds the cap's lower 14.3 mm: at 12 mm the cap is
    # still 2.3 mm inside it. Clearing outright would need >= 14.3 mm (see the
    # `separated` diagnostic, which reports exactly that).
    lift_success_height: float = 0.012
    # Slack for drive sag under load (measured 0.051 mm at 20 N).
    lift_tol: float = 0.0005
    # Cap-frame depth of the body collar, used only for the `separated` readout.
    collar_overlap: float = 0.0143
    # Fraction of the target that must be held, with the cap still gripped,
    # for hold_time before the episode counts as a success.
    hold_time: float = 0.3
    hold_decay_rate: float = 3.0

    # --- cap geometry (cap frame, origin = cap bottom face) --------------
    # Measured from assets/tumbler/meshes/cap_collision.obj: a truncated cone,
    # r 46.9 mm at the base widening to 50.0 mm by z = 17 mm, then straight.
    cap_radius_lo: float = 0.0469
    cap_radius_hi: float = 0.0500
    cap_radius_knee: float = 0.017
    cap_height: float = 0.0300
    finger_radius: float = 0.009

    # Cap-frame height where the grippable band starts: the tumbler's collar
    # shrouds everything below it. Default 14.3 mm = collar top (0.2393) minus
    # the seated cap bottom (0.2250), leaving a 15.7 mm band up to the rim.
    #
    # Held FIXED as the cap unscrews, rather than following the real exposure.
    # A band that widens with progress moves the contact gate under the policy:
    # the same finger placement flips from rejected to accepted with no change
    # in what the hand did, which is noise in the credit assignment for exactly
    # the term the whole task is scored on.
    band_lo: float = 0.0143
    # Slack on the exposed band so a finger right at the rim still counts.
    band_margin: float = 0.002
    radial_margin: float = 0.010
    # cap_wall_radius 프로파일 선택: 1 = 실물 STL 프로파일 테이블(기본, 구
    # 50mm 텀블러 실측), 0 = cap_radius_lo→hi 원뿔/원기둥 해석식. 원기둥
    # 근사 텀블러(r44 등)로 바꿀 때는 반드시 0 — 테이블은 구 형상 고정값이다.
    cap_profile: int = 1

    # --- contact ---------------------------------------------------------
    contact_force_threshold: float = 0.05
    min_contact_tips: int = 2
    grip_force_ref: float = 0.5
    # Two contacts N deg apart give closure_err cos(N/2); 0.75 admits >=82 deg.
    closure_err_max: float = 0.75

    # --- reward weights --------------------------------------------------
    # Unscrew dominates on purpose: it is the only term that cannot be earned
    # without a real grasp, so the shaping terms must not be able to out-earn it.
    unscrew_weight: float = 10.0
    hold_weight: float = 4.0
    grasp_contact_weight: float = 2.0
    closure_weight: float = 1.0
    grip_weight: float = 1.0
    thumb_contact_weight: float = 0.5
    two_tip_contact_weight: float = 0.5
    # The approach term is the only thing that pays before contact, so with a
    # policy-driven wrist it is what has to steer the hand to the cap at all.
    # At scale 0.02 it was worth +0.003/step against a -0.086/step leash -- a
    # 29:1 ratio pointing away from the task. Widened so it still reads at the
    # ~40 mm the fingers actually start out at, and weighted to match.
    near_weight: float = 2.0
    tip_dist_scale: float = 0.06
    # Approach reward fades once the hand grips, so hovering cannot compete.
    near_gate_on_grip: float = 0.5

    # --- penalties -------------------------------------------------------
    penetration_penalty_weight: float = 2.0
    penetration_ref: float = 0.02
    penetration_tol: float = 0.005
    # Tightening the cap is worse than doing nothing.
    reverse_penalty_weight: float = 2.0
    reverse_ref: float = 0.5  # rad/s at which the reverse penalty saturates
    action_penalty_weight: float = 0.01
    # The wrist is policy-driven now, so it can simply leave. Penalise drifting
    # away from the cap axis beyond the radius a grasp needs.
    wrist_leash_weight: float = 0.3
    wrist_leash_radius: float = 0.16
    # Palm must keep facing down. Free below palm_down_max_deg off straight
    # down, then ramping to full cost palm_down_ref_deg further on. The hand
    # reaches the cap wall by curling its fingers inward under a downward palm;
    # rolled past ~40 deg the fingers sweep across the cap's top face instead,
    # which is outside the scored band and cannot generate a turning couple.
    # The thumb has to get away from BOTH index and middle. reward_near scores
    # each finger's own distance independently, so five fingers piling onto the
    # nearest point of the wall is its exact optimum; nothing else asks them to
    # spread.
    #
    # Penalise only while both are close, so the cost clears as soon as one
    # finger breaks away to oppose the thumb -- which is the behaviour wanted.
    # A plain thumb-index distance term did not do this: it pushed them 78 mm
    # apart while opposition stayed at 0%, because two fingers can sit far
    # apart and still be on the same side of the cap.
    #
    # Threshold is a multiple of the cap radius: 1.6 * 50 mm = 80 mm, between
    # the 64 mm chord the 83-deg opposition gate implies and the 100 mm
    # diameter of a fully opposed grasp.
    thumb_pair_penalty_weight: float = 1.0
    thumb_pair_dist_factor: float = 1.6

    palm_down_penalty_weight: float = 1.0
    palm_down_max_deg: float = 40.0
    palm_down_ref_deg: float = 40.0


DEFAULT_CAP_UNSCREW_CFG = CapUnscrewConfig()


def _quat_rotate_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v - w * t + torch.cross(xyz, t, dim=-1)


def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _as_probe_points(p: torch.Tensor) -> torch.Tensor:
    """(N,F,3) or (N,F,P,3) -> (N,F,P,3)."""
    return p.unsqueeze(2) if p.dim() == 3 else p


def cap_local_probes(
    points_w: torch.Tensor, cap_pos_w: torch.Tensor, cap_quat_xyzw: torch.Tensor | None
) -> torch.Tensor:
    """Probe points in the cap frame (origin = cap bottom face centre)."""
    p = _as_probe_points(points_w)
    rel = p - cap_pos_w[:, None, None, :]
    if cap_quat_xyzw is not None:
        q = cap_quat_xyzw[:, None, None, :].expand(-1, rel.shape[1], rel.shape[2], -1)
        rel = _quat_rotate_inverse_xyzw(q, rel)
    return rel


# Cap wall radius against cap-frame height, read off cap_collision.obj at 1.5mm
# steps (99.5th percentile of the sampled radii at each level, so a stray vertex
# cannot set the wall).
#
# The cap is two stacked cylinders: a 47.4mm threaded skirt to z=13, then the
# 50.2mm grip wall from z=14 to the 30mm top. cap.STL measures 50.00mm flat over
# that whole span; the 0.2mm here is the collision mesh's own margin.
#
# An earlier version of this table read 55.5mm at z=24 and was wrong by up to
# 5.3mm across the scored band. It was measured honestly -- off the collision
# mesh that was actually being simulated -- but that mesh was itself broken:
# make_screw_asset.py decimated in metres with a decimator whose error tolerance
# is in absolute units, so the cap came out at radius 57.0mm, height 27.9mm, and
# 4.1mm off-axis. Fitting the reward to it moved the target 5mm outside the real
# wall, and since the touch test is |radial - r_wall| - finger_radius, a fingertip
# stopped 5mm short of the cap already scored as touching. The policy hovering
# just off the wall was doing exactly what it was paid to do. The decimator is
# fixed and checked now (_check_decimation), so both meshes match the STL.
CAP_PROFILE_Z = (
    0.0000, 0.0008, 0.0023, 0.0038, 0.0053, 0.0067, 0.0083, 0.0098, 0.0112,
    0.0128, 0.0142, 0.0158, 0.0173, 0.0188, 0.0203, 0.0217, 0.0232, 0.0248,
    0.0263, 0.0278, 0.0292, 0.0300,
)
CAP_PROFILE_R = (
    0.04708, 0.04708, 0.04742, 0.04736, 0.04739, 0.04742, 0.04742, 0.04738,
    0.04737, 0.04668, 0.05014, 0.05024, 0.05024, 0.05019, 0.05023, 0.05023,
    0.05023, 0.05023, 0.05023, 0.05023, 0.05018, 0.05018,
)
_PROFILE_CACHE: Dict[str, torch.Tensor] = {}


def cap_wall_radius(z: torch.Tensor, cfg: CapUnscrewConfig) -> torch.Tensor:
    """Cap wall radius at cap-frame height z, interpolated from the mesh profile.

    Clamped at both ends: below the cap it holds the base radius, above the rim
    it holds the top one, which is what the band's z-margin needs -- a probe just
    past the rim should read the rim's radius, not fall off to zero.

    cfg.cap_profile = 0 restores the old cone-to-knee form for comparison.
    """
    if not bool(getattr(cfg, "cap_profile", 1)):
        t = (z / max(cfg.cap_radius_knee, 1e-6)).clamp(0.0, 1.0)
        return cfg.cap_radius_lo + (cfg.cap_radius_hi - cfg.cap_radius_lo) * t

    key = f"{z.device}:{z.dtype}"
    knots = _PROFILE_CACHE.get(key)
    if knots is None:
        knots = (
            torch.tensor(CAP_PROFILE_Z, device=z.device, dtype=z.dtype),
            torch.tensor(CAP_PROFILE_R, device=z.device, dtype=z.dtype),
        )
        _PROFILE_CACHE[key] = knots
    zk, rk = knots
    zc = z.clamp(float(CAP_PROFILE_Z[0]), float(CAP_PROFILE_Z[-1]))
    i = torch.bucketize(zc, zk).clamp(1, zk.numel() - 1)
    z0, z1 = zk[i - 1], zk[i]
    r0, r1 = rk[i - 1], rk[i]
    return r0 + (r1 - r0) * ((zc - z0) / (z1 - z0).clamp_min(1e-9))


def band_bounds(cfg: CapUnscrewConfig) -> Tuple[float, float]:
    """Grippable cap-frame band (lo, hi). Constant -- see band_lo."""
    return float(min(cfg.band_lo, cfg.cap_height)), float(cfg.cap_height)


def wall_distance_and_contact(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor | None,
    cfg: CapUnscrewConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per finger: surface distance to the exposed wall, on-band mask, penetration.

    Distances are measured from the finger *surface* (finger_radius subtracted),
    so a finger resting on the wall reads 0 -- measuring from link origins makes
    0 unreachable and turns hovering into the optimum.
    """
    rel = cap_local_probes(points_w, cap_pos_w, cap_quat_xyzw)
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    z = rel[..., 2]
    r_wall = cap_wall_radius(z, cfg)
    r = cfg.finger_radius

    lo_v, hi_v = band_bounds(cfg)
    lo = torch.full_like(z, lo_v)
    hi = torch.full_like(z, hi_v)
    z_out = torch.maximum((lo - z).clamp_min(0.0), (z - hi).clamp_min(0.0))
    radial_out = (radial - r_wall).abs()
    dist = (torch.sqrt(radial_out * radial_out + z_out * z_out) - r).clamp_min(0.0)

    # band_lo_slack: 하한 슬랙 [m]. 0 이면 기존 동작(finger_radius). 캡 밑이
    # 몸통과 flush 인 텀블러에서 몸통 오인정을 막기 위해 좁힌다 (A73).
    _r_lo = float(getattr(cfg, "band_lo_slack", 0.0)) or r
    on_band = (z >= lo - _r_lo) & (z <= hi + r) & ((radial_out - r) <= cfg.radial_margin)

    inside = (radial < r_wall + r) & (z > -r) & (z < cfg.cap_height + r)
    depth = torch.where(
        inside,
        torch.minimum((r_wall + r - radial).clamp_min(0.0), (z + r).clamp_min(0.0)),
        torch.zeros_like(radial),
    )
    return dist.amin(dim=-1), on_band.any(dim=-1), depth.amax(dim=-1)


def contact_closure(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor | None,
    contact: torch.Tensor,
    cfg: CapUnscrewConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """How well the contacts oppose each other around the cap axis.

    Returns (closure_err, reward). 1/0 when fewer than two fingers are in
    contact. Contacts all on one side cancel to err 1: that is a push, not a
    grip, and it cannot generate the couple unscrewing needs.
    """
    rel = cap_local_probes(points_w, cap_pos_w, cap_quat_xyzw)
    radial_vec = rel[..., :2]
    n = torch.linalg.norm(radial_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    unit = (radial_vec / n).mean(dim=2)  # (N,F,2), average over probes
    unit = unit / torch.linalg.norm(unit, dim=-1, keepdim=True).clamp_min(1e-6)

    w = contact.to(dtype=unit.dtype)[..., None]
    cnt = contact.to(dtype=unit.dtype).sum(dim=-1)
    summed = (unit * w).sum(dim=1)
    err = torch.linalg.norm(summed, dim=-1) / cnt.clamp_min(1.0)
    enough = cnt >= 2
    err = torch.where(enough, err, torch.ones_like(err))
    return err, torch.where(enough, 1.0 - err, torch.zeros_like(err))


def tip_force_magnitude(f: torch.Tensor) -> torch.Tensor:
    """(N,F,3) or (N,F,K,3) -> (N,F). Sums magnitudes over K so that opposing
    contacts on different bodies of one finger do not cancel."""
    if f.dim() == 4:
        return torch.linalg.norm(f, dim=-1).sum(dim=-1)
    return torch.linalg.norm(f, dim=-1)


def cap_unscrew_reward_success(
    unscrew_angle: torch.Tensor,
    prev_unscrew_angle: torch.Tensor,
    cap_rise: torch.Tensor,
    hold_timer: torch.Tensor,
    dt: float,
    tip_pos_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor,
    wrist_pos_w: torch.Tensor,
    wrist_quat_xyzw: torch.Tensor | None = None,
    tip_forces_w: torch.Tensor | None = None,
    tip_contact: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    cfg: CapUnscrewConfig = DEFAULT_CAP_UNSCREW_CFG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Reward, success, updated hold timer, diagnostics.

    ``unscrew_angle`` is the cumulative (unwrapped) cap rotation in radians,
    which ScrewCoupling already tracks. Finger order is
    [thumb, index, middle, ring, pinky]; ``tip_pos_w`` samples the distal
    segment, ``tip_forces_w`` the net contact force of the bodies making it up.
    """
    if tip_forces_w is None and tip_contact is None:
        raise ValueError("provide tip_contact or tip_forces_w")
    if tip_forces_w is None:
        force_mag = torch.zeros_like(tip_contact, dtype=cap_pos_w.dtype)
    else:
        force_mag = tip_force_magnitude(tip_forces_w)
    if tip_contact is None:
        tip_contact = force_mag > cfg.contact_force_threshold
    raw_contact = tip_contact

    dist, on_band, depth = wall_distance_and_contact(tip_pos_w, cap_pos_w, cap_quat_xyzw, cfg)
    tip_contact = raw_contact & on_band

    contact_count = tip_contact.to(dtype=torch.int32).sum(dim=-1)
    thumb_contact = tip_contact[:, 0]
    enough = contact_count >= cfg.min_contact_tips
    closure_err, reward_closure = contact_closure(
        tip_pos_w, cap_pos_w, cap_quat_xyzw, tip_contact, cfg
    )
    opposed = closure_err <= cfg.closure_err_max
    grasp_contact = thumb_contact & enough & opposed
    grasp_f = grasp_contact.to(dtype=cap_pos_w.dtype)

    grip = (force_mag / cfg.grip_force_ref).clamp(0.0, 1.0) * tip_contact.to(dtype=force_mag.dtype)
    reward_grip = grip.mean(dim=-1)

    reward_near = torch.exp(-dist / cfg.tip_dist_scale).mean(dim=-1)
    reward_near = reward_near * (1.0 - cfg.near_gate_on_grip * reward_grip.clamp(0.0, 1.0))

    # --- progress --------------------------------------------------------
    frac = (unscrew_angle / max(cfg.unscrew_target_rad, 1e-6)).clamp(0.0, 1.0)
    reward_unscrew = frac * grasp_f
    unscrewed = frac >= 1.0
    lifted = cap_rise >= (cfg.lift_success_height - cfg.lift_tol)

    d_angle = unscrew_angle - prev_unscrew_angle
    reverse_rate = (-d_angle / max(dt, 1e-6)).clamp_min(0.0)
    reverse_penalty = (reverse_rate / max(cfg.reverse_ref, 1e-6)).clamp(0.0, 1.0)

    penetration_excess = (depth.amax(dim=-1) - cfg.penetration_tol).clamp_min(0.0)
    penetration_penalty = (penetration_excess / cfg.penetration_ref).clamp(0.0, 1.0)

    # Palm direction: wrist_init.PALM_LOCAL is +x in the wrist frame.
    if wrist_quat_xyzw is None:
        palm_down_deg = torch.zeros_like(frac)
        palm_penalty = torch.zeros_like(frac)
    else:
        palm_local = torch.zeros_like(wrist_pos_w)
        palm_local[:, 0] = 1.0
        palm_w = _quat_rotate_xyzw(wrist_quat_xyzw, palm_local)
        cos_down = (-palm_w[:, 2]).clamp(-1.0, 1.0)
        palm_down_deg = torch.rad2deg(torch.arccos(cos_down))
        palm_penalty = (
            (palm_down_deg - cfg.palm_down_max_deg) / max(cfg.palm_down_ref_deg, 1e-6)
        ).clamp(0.0, 1.0)

    tips = _as_probe_points(tip_pos_w)[:, :, -1, :]  # (N,5,3) tip ends
    thumb_index_dist = torch.linalg.norm(tips[:, 0] - tips[:, 1], dim=-1)
    thumb_middle_dist = torch.linalg.norm(tips[:, 0] - tips[:, 2], dim=-1)
    pair_thresh = max(cfg.thumb_pair_dist_factor * cfg.cap_radius_hi, 1e-6)
    p_index = ((pair_thresh - thumb_index_dist) / pair_thresh).clamp(0.0, 1.0)
    p_middle = ((pair_thresh - thumb_middle_dist) / pair_thresh).clamp(0.0, 1.0)
    # min = a soft AND: zero the moment either finger is clear of the thumb.
    thumb_pair_penalty = torch.minimum(p_index, p_middle)

    leash = torch.linalg.norm(wrist_pos_w[:, :2] - cap_pos_w[:, :2], dim=-1)
    leash_penalty = (leash - cfg.wrist_leash_radius).clamp_min(0.0) / max(cfg.wrist_leash_radius, 1e-6)
    leash_penalty = leash_penalty.clamp(0.0, 1.0)

    action_penalty = (
        torch.zeros_like(reward_near) if actions is None else actions.square().mean(dim=-1)
    )

    # --- success ---------------------------------------------------------
    # Fully unscrewed, up at the unscrewed height, and still gripped. A cap
    # that was turned all the way and then dropped back down the free stroke is
    # not a completed task.
    hold_frame = unscrewed & lifted & grasp_contact
    updated_hold = torch.where(
        hold_frame, hold_timer + dt, (hold_timer - cfg.hold_decay_rate * dt).clamp_min(0.0)
    )
    success = updated_hold >= cfg.hold_time
    reward_hold = (updated_hold / max(cfg.hold_time, 1e-6)).clamp(0.0, 1.0)

    reward = (
        cfg.unscrew_weight * reward_unscrew
        + cfg.hold_weight * reward_hold
        + cfg.grasp_contact_weight * grasp_f
        + cfg.closure_weight * reward_closure
        + cfg.grip_weight * reward_grip
        + cfg.thumb_contact_weight * thumb_contact.to(dtype=cap_pos_w.dtype)
        + cfg.two_tip_contact_weight * enough.to(dtype=cap_pos_w.dtype)
        + cfg.near_weight * reward_near
        - cfg.penetration_penalty_weight * penetration_penalty
        - cfg.reverse_penalty_weight * reverse_penalty
        - cfg.wrist_leash_weight * leash_penalty
        - cfg.palm_down_penalty_weight * palm_penalty
        - cfg.thumb_pair_penalty_weight * thumb_pair_penalty
        - cfg.action_penalty_weight * action_penalty
    )

    terms = {
        "unscrew_angle_deg": torch.rad2deg(unscrew_angle),
        "unscrew_frac": frac,
        "reward_unscrew": reward_unscrew,
        "unscrewed": unscrewed.to(dtype=cap_pos_w.dtype),
        "cap_rise": cap_rise,
        "lifted": lifted.to(dtype=cap_pos_w.dtype),
        "separated": (cap_rise >= cfg.collar_overlap).to(dtype=cap_pos_w.dtype),
        "reverse_penalty": reverse_penalty,
        "raw_contact_count": raw_contact.to(dtype=cap_pos_w.dtype).sum(dim=-1),
        "band_contact_count": on_band.to(dtype=cap_pos_w.dtype).sum(dim=-1),
        "contact_count": contact_count.to(dtype=cap_pos_w.dtype),
        "thumb_contact": thumb_contact.to(dtype=cap_pos_w.dtype),
        "grasp_contact": grasp_f,
        "closure_err": closure_err,
        "opposed": opposed.to(dtype=cap_pos_w.dtype),
        "reward_grip": reward_grip,
        "max_tip_force": force_mag.amax(dim=-1),
        "tip_surface_dist": dist.amin(dim=-1),
        "band_lo": torch.full_like(frac, band_bounds(cfg)[0]),
        "penetration_depth": depth.amax(dim=-1),
        "penetration_penalty": penetration_penalty,
        "wrist_cap_radius": leash,
        "thumb_index_dist": thumb_index_dist,
        "thumb_middle_dist": thumb_middle_dist,
        "thumb_pair_penalty": thumb_pair_penalty,
        "palm_down_deg": palm_down_deg,
        "palm_down_penalty": palm_penalty,
        "leash_penalty": leash_penalty,
        "hold_timer": updated_hold,
        "success": success.to(dtype=cap_pos_w.dtype),
    }
    return reward, success, updated_hold, terms
