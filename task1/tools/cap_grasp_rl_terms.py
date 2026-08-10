"""Default reward/success terms for cap-only dg5fs grasp RL.

This module is intentionally Isaac-agnostic: pass tensors from Isaac Gym,
Isaac Lab, or another vectorized env, and feed the returned reward/success
buffers back into the env.

Contact geometry
----------------
Contact probe points are expected to sample the *distal segment* of each
finger (link_X_4 origin -> link_X_tip origin -> tip mesh end), not just the
tip link origin. Every geometric term subtracts ``cfg.finger_radius`` so that
distances are measured from the finger *surface*: a probe point resting on the
cap wall has surface distance 0, which is physically reachable. Measuring from
link origins instead makes distance 0 unreachable and turns "hover without
touching" into the reward optimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from wrist_init import CAP_HEIGHT, CAP_RADIUS


@dataclass(frozen=True)
class CapGraspRLConfig:
    # Phase schedule.
    grasp_time: float = 1.0
    settle_time: float = 0.5
    hold_time: float = 2.0
    gravity_ramp_time: float = 0.0

    # Success thresholds.
    drop_threshold: float = 0.015
    cap_vel_threshold: float = 0.05
    contact_force_threshold: float = 0.05
    min_contact_tips: int = 2
    side_contact_margin: float = 0.010
    side_z_margin: float = 0.002
    # A grasp only counts if the contacts actually oppose each other. Two
    # contacts N degrees apart about the cap axis give closure_err
    # |u1+u2|/2 = cos(N/2), so 0.75 admits roughly >=82 deg of separation.
    # Without this gate the policy collects thumb+two-tip+grasp rewards from
    # contacts all on one side of the rim -- measured 0.8% of >=2-contact
    # frames were well opposed, yet Stage 0 reported 10% "success".
    closure_err_max: float = 0.75
    # Stage-0 curriculum: score sustained contact even while gravity is off.
    require_gravity_for_hold: bool = True

    # --- lift mode -------------------------------------------------------
    # The cap rests on a static pedestal under normal gravity and the wrist is
    # commanded upward; a grasp counts only if the cap comes up with the hand.
    # This replaces the pinned-cap formulation, where the pin absorbed an
    # unopposed one-sided push and then ejected the cap on release -- and where
    # pressing the cap against the pin earned ~5.0 reward/step for a behaviour
    # that cannot lift anything. Here physics does the judging: pressing a
    # supported cap simply loads the pedestal and lifts nothing.
    lift_mode: bool = False
    lift_success_height: float = 0.030
    lift_weight: float = 4.0
    # The lift reward must be gated on the grasp, or it pays full marks for any
    # way the cap ends up high. Measured ungated over 10k epochs: success sat
    # at ~1.8% from epoch 800 onward while reward climbed 757 -> 1755, and in
    # failing episodes this term alone paid 2.94/step -- 57% of them scooped
    # the cap up tilted 57 deg on 0.72 contacts, and 13% launched it over a
    # metre at up to 19 m/s. All three collected the same 4.0 a clean lift got.
    lift_requires_grasp: bool = True
    # Keeping the cap level (disc normal aligned with world up) is what a real
    # carry looks like; a cap that is batted upward tumbles instead.
    level_weight: float = 1.0
    # Success also requires the cap not to be tipped over beyond this cosine
    # (0.87 ~ 30 deg of tilt).
    level_cos_min: float = 0.87
    # Below the pedestal top by more than this, the cap has been knocked off and
    # is out of reach: the episode is unwinnable and no shaping should pay out.
    # Measured before this gate, 31.5% of knocked-off caps still collected the
    # levelness reward while lying on the table.
    fallen_threshold: float = 0.050
    # Hold timer decays instead of hard-resetting, so a single bad frame does
    # not throw away a nearly complete hold.
    hold_decay_rate: float = 3.0

    # Cap cylinder approximation. The cap is treated as a SOLID puck: no probe
    # point may end up inside the cylinder. The collision asset must match --
    # see CAP_GRASP_SOLID_CAP in dexhandmanip_sh, which loads the cap as a
    # single convex hull instead of a VHACD shell. Without that, VHACD
    # decomposes the lid mesh into a thin skirt with a hollow interior and
    # fingers slide inside it, which the penetration penalty then flags.
    cap_radius: float = CAP_RADIUS
    cap_height: float = CAP_HEIGHT
    # Distal phalanx / fingertip cross-section radius, measured from
    # meshes/dg5fs_right/link_X_4.STL (half-extent 8.0~9.8 mm).
    finger_radius: float = 0.009

    # Reward scales.
    tip_dist_scale: float = 0.02
    near_weight: float = 0.5
    # Fraction of the near reward that is switched off once the hand actually
    # grips, so approaching cannot out-earn grasping.
    near_gate_on_grip: float = 0.5
    # Cap is 0.05 kg (0.49 N). With ~2 opposing contacts and mu~1 a normal
    # force of ~0.25 N per finger already holds it, so saturate the grip
    # reward at 0.5 N rather than some arbitrarily large force.
    grip_force_ref: float = 0.5
    grip_weight: float = 1.5
    thumb_contact_weight: float = 1.0
    two_tip_contact_weight: float = 1.0
    grasp_contact_weight: float = 2.0
    hold_weight: float = 3.0
    # Force-closure proxy: contacts should be spread around the cap axis.
    closure_weight: float = 1.0
    penetration_penalty_weight: float = 2.0
    penetration_ref: float = 0.02
    # Contact needs a little interpenetration; only penalize beyond this.
    penetration_tol: float = 0.005

    # --- pedestal avoidance ---------------------------------------------
    # Fingers jammed into the support cannot grip the cap, and the side gate
    # only *ignores* them, which costs the policy nothing. This makes it cost
    # something. Positional rather than contact-gated on purpose: the policy
    # cannot observe the pedestal, so it needs a gradient that steers the
    # fingers clear *before* they collide, not a reaction after the fact.
    #
    # Keep this modest. Measured on the ep-5000 policy, only ~24% of wasted
    # contacts were on the pedestal; the other ~76% were off-cap contacts the
    # term does not see. And successful grasps hook *under* the cap rim once
    # it is airborne, which is why the depth is measured in world coordinates
    # against the fixed pedestal rather than "below the cap" in the cap frame
    # -- the latter cannot tell a jammed finger from a correct one.
    pedestal_penalty_weight: float = 0.0
    pedestal_ref: float = 0.020
    pedestal_radius: float = CAP_RADIUS
    # Dead zone, and it has to be here. Every successful grasp measured brushes
    # the pedestal (median 6.2 mm, max 7.7 mm) because holding a 31 mm disc
    # means wrapping down to its lower rim -- successes touch it *more* than
    # live failures do (median 3.7 mm). Only the deep tail (failures reach
    # 37 mm) is genuinely a jammed finger, so charge nothing below this.
    pedestal_tol: float = 0.010
    drop_penalty_weight: float = 2.0
    cap_vel_penalty_weight: float = 0.2
    action_penalty_weight: float = 0.01

    @property
    def gravity_on_time(self) -> float:
        return self.grasp_time + self.settle_time


DEFAULT_CAP_GRASP_CFG = CapGraspRLConfig()


def gravity_scale_from_progress(
    progress_buf: torch.Tensor,
    dt: float,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
) -> torch.Tensor:
    """Per-env gravity scale: 0 during grasp/settle, 1 during hold."""
    t = progress_buf.to(dtype=torch.float32) * dt
    if cfg.gravity_ramp_time <= 0.0:
        return (t >= cfg.gravity_on_time).to(dtype=torch.float32)
    return ((t - cfg.gravity_on_time) / cfg.gravity_ramp_time).clamp(0.0, 1.0)


def cap_manual_gravity_force(
    progress_buf: torch.Tensor,
    dt: float,
    cap_mass: float,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
    gravity_z: float = -9.81,
) -> torch.Tensor:
    """Force to apply when cap asset gravity is disabled for per-env scheduling."""
    scale = gravity_scale_from_progress(progress_buf, dt, cfg)
    force = torch.zeros(progress_buf.shape[0], 3, device=progress_buf.device, dtype=torch.float32)
    force[:, 2] = cap_mass * gravity_z * scale
    return force


def tip_force_magnitude(tip_forces_w: torch.Tensor) -> torch.Tensor:
    """Net contact force (N,F,3) or (N,F,K,3) -> per-finger magnitude (N,F).

    With K>1 the per-body magnitudes are summed rather than vector-summed, so
    that opposing contacts on different bodies of the same finger do not cancel.
    """
    if tip_forces_w.dim() == 4:
        return torch.linalg.norm(tip_forces_w, dim=-1).sum(dim=-1)
    return torch.linalg.norm(tip_forces_w, dim=-1)


def tip_contact_from_forces(
    tip_forces_w: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
) -> torch.Tensor:
    """Convert fingertip net contact forces to contact booleans."""
    return tip_force_magnitude(tip_forces_w) > cfg.contact_force_threshold


def _quat_rotate_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by inverse quaternion. q is xyzw."""
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.cross(q_xyz, t, dim=-1)


def _quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) by quaternion. q is xyzw."""
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def cap_level_cosine(cap_quat_xyzw: torch.Tensor | None, n: int, device, dtype) -> torch.Tensor:
    """(N,) cosine between the cap's disc normal and world up. 1 = level.

    A cap that is carried in a real grasp stays level; one that is knocked or
    flicked upward tumbles, so this separates lifting from batting it away.
    """
    if cap_quat_xyzw is None:
        return torch.ones(n, device=device, dtype=dtype)
    up = torch.zeros(cap_quat_xyzw.shape[0], 3, device=cap_quat_xyzw.device, dtype=cap_quat_xyzw.dtype)
    up[:, 2] = 1.0
    return _quat_rotate_xyzw(cap_quat_xyzw, up)[:, 2]


def _as_probe_points(points_w: torch.Tensor) -> torch.Tensor:
    """Accept (N,F,3) or (N,F,P,3) and always return (N,F,P,3)."""
    if points_w.dim() == 3:
        return points_w.unsqueeze(2)
    return points_w


def cap_local_probe_points(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Express probe points (N,F,P,3) in the cap frame (origin = bottom center)."""
    points_w = _as_probe_points(points_w)
    rel = points_w - cap_pos_w[:, None, None, :]
    if cap_quat_xyzw is not None:
        q = cap_quat_xyzw[:, None, None, :].expand(-1, rel.shape[1], rel.shape[2], -1)
        rel = _quat_rotate_inverse_xyzw(q, rel)
    return rel


def cap_cylinder_surface_distance(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
    cap_quat_xyzw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distance (N,F) from the finger *surface* to the cap cylinder side wall.

    Reduced over the P probe points of each finger with a min, so the closest
    part of the distal segment defines the distance.
    """
    rel = cap_local_probe_points(points_w, cap_pos_w, cap_quat_xyzw)
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    radial_dist = (radial - cfg.cap_radius).abs()
    z = rel[..., 2]
    z_out = torch.maximum(
        (cfg.side_z_margin - z).clamp_min(0.0),
        (z - (cfg.cap_height - cfg.side_z_margin)).clamp_min(0.0),
    )
    center_dist = torch.sqrt(radial_dist * radial_dist + z_out * z_out)
    return (center_dist - cfg.finger_radius).clamp_min(0.0).amin(dim=-1)


def cap_cylinder_side_contact_mask(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
    cap_quat_xyzw: torch.Tensor | None = None,
) -> torch.Tensor:
    """Whether any probe point of each finger (N,F) is near the cap side wall."""
    rel = cap_local_probe_points(points_w, cap_pos_w, cap_quat_xyzw)
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    z = rel[..., 2]
    r = cfg.finger_radius
    side_z_ok = (z >= cfg.side_z_margin - r) & (z <= cfg.cap_height - cfg.side_z_margin + r)
    side_radial_ok = ((radial - cfg.cap_radius).abs() - r) <= cfg.side_contact_margin
    return (side_z_ok & side_radial_ok).any(dim=-1)


def cap_cylinder_penetration_depth(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
    cap_quat_xyzw: torch.Tensor | None = None,
) -> torch.Tensor:
    """How deep (N,F) the finger surface tunnels into the solid cap cylinder.

    A finger resting on the wall reads ~0; only genuine mesh interpenetration
    grows this. Reduced over probe points with a max.
    """
    rel = cap_local_probe_points(points_w, cap_pos_w, cap_quat_xyzw)
    r = cfg.finger_radius
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    z = rel[..., 2]
    inside = (radial < cfg.cap_radius + r) & (z > -r) & (z < cfg.cap_height + r)
    radial_depth = (cfg.cap_radius + r - radial).clamp_min(0.0)
    z_depth = torch.minimum(z + r, cfg.cap_height + r - z).clamp_min(0.0)
    depth = torch.where(inside, torch.minimum(radial_depth, z_depth), torch.zeros_like(radial))
    return depth.amax(dim=-1)


def pedestal_penetration_depth(
    points_w: torch.Tensor,
    cap_initial_pos_w: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
) -> torch.Tensor:
    """How far (N,F) each finger reaches into the static pedestal.

    Measured in *world* coordinates against the pedestal, not in the cap frame:
    the pedestal never moves, while the cap tilts and lifts. Doing this in the
    cap frame conflates two opposite situations -- a finger jammed against the
    support before the lift, and a finger correctly hooked under the rim after
    it -- because both read as "below the cap". The pedestal top plane sits at
    the cap's initial bottom face, which is where ``cap_initial_pos_w`` is.

    Zero unless the finger surface is both under that plane and inside the
    pedestal's radius, so a finger hanging low out beyond the rim, or any
    finger at all once the cap has been lifted clear, costs nothing.
    """
    points_w = _as_probe_points(points_w)
    rel = points_w - cap_initial_pos_w[:, None, None, :]
    radial = torch.linalg.norm(rel[..., :2], dim=-1)
    r = cfg.finger_radius
    top_depth = (cfg.side_z_margin - r - rel[..., 2]).clamp_min(0.0)
    radial_depth = (cfg.pedestal_radius + r - radial).clamp_min(0.0)
    # min of the two so a finger just outside the rim, or just below the top
    # plane but off to the side, reads ~0 instead of a spurious deep value.
    depth = torch.minimum(top_depth, radial_depth)
    return depth.amax(dim=-1)


def cap_contact_closure(
    points_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    contact: torch.Tensor,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
    cap_quat_xyzw: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force-closure proxy from how contacts are spread around the cap axis.

    For each contacting finger take the outward radial unit vector of its
    closest probe point. If those vectors cancel, the grasp squeezes the cap
    from opposite sides; if they all point the same way, the hand is just
    pushing the cap sideways.

    Returns:
        closure_err: (N,) 0 when contacts are perfectly balanced, 1 when they
                     all sit on the same side. 1 when fewer than 2 contacts.
        reward:      (N,) (1 - closure_err), zeroed when fewer than 2 contacts.
    """
    rel = cap_local_probe_points(points_w, cap_pos_w, cap_quat_xyzw)
    radial_vec = rel[..., :2]
    radial = torch.linalg.norm(radial_vec, dim=-1)
    # Representative point per finger = the one closest to the side wall.
    closest = (radial - cfg.cap_radius).abs().argmin(dim=-1, keepdim=True)
    idx = closest.unsqueeze(-1).expand(-1, -1, -1, 2)
    rep = torch.gather(radial_vec, 2, idx).squeeze(2)
    rep = rep / torch.linalg.norm(rep, dim=-1, keepdim=True).clamp_min(1e-9)

    w = contact.to(dtype=rep.dtype)
    n = w.sum(dim=-1)
    resultant = torch.linalg.norm((w[..., None] * rep).sum(dim=1), dim=-1)
    closure_err = (resultant / n.clamp_min(1.0)).clamp(0.0, 1.0)
    enough = n >= 2
    closure_err = torch.where(enough, closure_err, torch.ones_like(closure_err))
    return closure_err, (1.0 - closure_err) * enough.to(dtype=rep.dtype)


def cap_grasp_reward_success(
    progress_buf: torch.Tensor,
    hold_timer: torch.Tensor,
    dt: float,
    tip_pos_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_initial_pos_w: torch.Tensor,
    cap_lin_vel_w: torch.Tensor,
    tip_contact: torch.Tensor | None = None,
    tip_forces_w: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    cap_quat_xyzw: torch.Tensor | None = None,
    cfg: CapGraspRLConfig = DEFAULT_CAP_GRASP_CFG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute reward, success, updated hold timer, and diagnostic terms.

    Finger order is expected to be [thumb, index, middle, ring, pinky].
    ``tip_pos_w`` is (N,5,3) or (N,5,P,3) probe points sampling the distal
    segment; ``tip_forces_w`` is (N,5,3) or (N,5,K,3) net contact forces of the
    bodies making up that segment.

    Success requires the hold phase, thumb contact, at least two fingertip
    contacts, cap not dropped, and low cap linear velocity for cfg.hold_time.
    """
    if tip_forces_w is None and tip_contact is None:
        raise ValueError("Provide either tip_contact or tip_forces_w.")
    if tip_forces_w is None:
        force_mag = torch.zeros_like(tip_contact, dtype=cap_pos_w.dtype)
    else:
        force_mag = tip_force_magnitude(tip_forces_w)
    if tip_contact is None:
        tip_contact = force_mag > cfg.contact_force_threshold

    gravity_scale = gravity_scale_from_progress(progress_buf, dt, cfg).to(device=cap_pos_w.device)
    gravity_active = gravity_scale >= 1.0

    tip_dist = cap_cylinder_surface_distance(tip_pos_w, cap_pos_w, cfg, cap_quat_xyzw)
    reward_near = torch.exp(-tip_dist / cfg.tip_dist_scale).mean(dim=-1)

    raw_tip_contact = tip_contact
    side_contact_mask = cap_cylinder_side_contact_mask(tip_pos_w, cap_pos_w, cfg, cap_quat_xyzw)
    tip_contact = raw_tip_contact & side_contact_mask

    penetration_depth = cap_cylinder_penetration_depth(tip_pos_w, cap_pos_w, cfg, cap_quat_xyzw)
    penetration_excess = (penetration_depth.amax(dim=-1) - cfg.penetration_tol).clamp_min(0.0)
    penetration_penalty = (penetration_excess / cfg.penetration_ref).clamp(0.0, 1.0)

    pedestal_depth = pedestal_penetration_depth(tip_pos_w, cap_initial_pos_w, cfg)
    # Mean over fingers, so the weight reads as "cost when every finger is
    # pedestal_ref deep into the pedestal" and one stray finger costs 1/5 of it.
    pedestal_excess = (pedestal_depth - cfg.pedestal_tol).clamp_min(0.0)
    pedestal_penalty = (pedestal_excess / max(cfg.pedestal_ref, 1e-6)).clamp(0.0, 1.0).mean(dim=-1)

    # Split the fingers that physically touched something but failed the cap
    # gate, so "stuck on the pedestal" is distinguishable from "resting on the
    # cap's top face" -- the two need opposite fixes.
    wasted_contact = raw_tip_contact & (~side_contact_mask)
    waste_below = wasted_contact & (pedestal_depth > 0.0)
    waste_other = wasted_contact & (pedestal_depth <= 0.0)

    contact_count = tip_contact.to(dtype=torch.int32).sum(dim=-1)
    thumb_contact = tip_contact[:, 0]
    enough_contacts = contact_count >= cfg.min_contact_tips

    # Continuous grip signal so the policy gets a gradient for squeezing
    # harder, instead of the flat 0/1 step the force threshold gives.
    grip = (force_mag / cfg.grip_force_ref).clamp(0.0, 1.0) * tip_contact.to(dtype=force_mag.dtype)
    reward_grip = grip.mean(dim=-1)

    closure_err, reward_closure = cap_contact_closure(
        tip_pos_w, cap_pos_w, tip_contact, cfg, cap_quat_xyzw
    )
    # Opposed contacts are what makes this a grasp rather than a push.
    opposed = closure_err <= cfg.closure_err_max
    grasp_contact = thumb_contact & enough_contacts & opposed

    # Once the hand actually grips, fade the approach reward out so hovering
    # near the cap can never out-earn grasping it.
    near_scale = 1.0 - cfg.near_gate_on_grip * reward_grip.clamp(0.0, 1.0)
    reward_near = reward_near * near_scale

    drop = (cap_initial_pos_w[:, 2] - cap_pos_w[:, 2]).clamp_min(0.0)
    not_dropped = drop <= cfg.drop_threshold
    cap_speed = torch.linalg.norm(cap_lin_vel_w, dim=-1)
    stable_cap = cap_speed <= cfg.cap_vel_threshold

    cap_rise = (cap_pos_w[:, 2] - cap_initial_pos_w[:, 2]).clamp_min(0.0)
    lifted = cap_rise >= cfg.lift_success_height
    reward_lift = (cap_rise / max(cfg.lift_success_height, 1e-6)).clamp(0.0, 1.0)

    cap_level_cos = cap_level_cosine(cap_quat_xyzw, cap_pos_w.shape[0], cap_pos_w.device, cap_pos_w.dtype)
    level_ok = cap_level_cos >= cfg.level_cos_min
    # A cap knocked onto the table is flat, hence "level", so the levelness
    # reward has to be gated on the cap still being in play or it pays out for
    # having failed.
    fallen = drop > cfg.fallen_threshold
    reward_level = cap_level_cos.clamp(0.0, 1.0) * (~fallen).to(dtype=cap_level_cos.dtype)

    if cfg.lift_requires_grasp:
        # Height alone says nothing about how the cap got there. Requiring an
        # opposed grasp kills the scoop and the launch outright, and scaling by
        # levelness keeps the term dense: a cap coming up straight in a proper
        # grip is worth full marks, one riding along at 57 deg is worth half.
        lift_quality = grasp_contact.to(dtype=reward_lift.dtype) * cap_level_cos.clamp(0.0, 1.0)
        reward_lift = reward_lift * lift_quality

    if cfg.lift_mode:
        # The cap sits on a pedestal, so it cannot "drop"; the only question is
        # whether it came up with the hand, gripped and still level.
        hold_frame = lifted & grasp_contact & level_ok
    else:
        hold_gate = gravity_active if cfg.require_gravity_for_hold else torch.ones_like(gravity_active)
        hold_frame = hold_gate & grasp_contact & not_dropped & stable_cap
    updated_hold_timer = torch.where(
        hold_frame,
        hold_timer + dt,
        (hold_timer - cfg.hold_decay_rate * dt).clamp_min(0.0),
    )
    success = updated_hold_timer >= cfg.hold_time

    reward_hold = (updated_hold_timer / cfg.hold_time).clamp(0.0, 1.0)
    drop_penalty = (drop / cfg.drop_threshold).clamp(0.0, 1.0)
    cap_vel_penalty = cap_speed
    if actions is None:
        action_penalty = torch.zeros_like(reward_near)
    else:
        action_penalty = actions.square().mean(dim=-1)

    reward = (
        cfg.near_weight * reward_near
        + cfg.grip_weight * reward_grip
        + cfg.thumb_contact_weight * thumb_contact.to(dtype=torch.float32)
        + cfg.two_tip_contact_weight * enough_contacts.to(dtype=torch.float32)
        + cfg.grasp_contact_weight * grasp_contact.to(dtype=torch.float32)
        + cfg.closure_weight * reward_closure
        + cfg.level_weight * reward_level
        + cfg.hold_weight * reward_hold
        - cfg.penetration_penalty_weight * penetration_penalty
        - cfg.pedestal_penalty_weight * pedestal_penalty
        - cfg.cap_vel_penalty_weight * cap_vel_penalty
        - cfg.action_penalty_weight * action_penalty
    )
    if cfg.lift_mode:
        # Dense signal for actually raising the cap. No drop penalty: the cap is
        # supported, so failing to grasp costs the lift reward rather than
        # ending the episode, which leaves room to retry within one rollout.
        reward = reward + cfg.lift_weight * reward_lift
    else:
        reward = reward - cfg.drop_penalty_weight * drop_penalty

    terms = {
        "gravity_scale": gravity_scale,
        "reward_near": reward_near,
        "tip_surface_dist": tip_dist.amin(dim=-1),
        "raw_contact_count": raw_tip_contact.to(dtype=torch.float32).sum(dim=-1),
        "side_contact_count": side_contact_mask.to(dtype=torch.float32).sum(dim=-1),
        "raw_thumb_contact": raw_tip_contact[:, 0].to(dtype=torch.float32),
        "thumb_contact": thumb_contact.to(dtype=torch.float32),
        "contact_count": contact_count.to(dtype=torch.float32),
        "grasp_contact": grasp_contact.to(dtype=torch.float32),
        "max_tip_force": force_mag.amax(dim=-1),
        "reward_grip": reward_grip,
        "closure_err": closure_err,
        "reward_closure": reward_closure,
        "opposed": opposed.to(dtype=torch.float32),
        "penetration_depth": penetration_depth.amax(dim=-1),
        "penetration_penalty": penetration_penalty,
        "pedestal_depth": pedestal_depth.amax(dim=-1),
        "pedestal_penalty": pedestal_penalty,
        "pedestal_contact_count": waste_below.to(dtype=torch.float32).sum(dim=-1),
        "waste_offcap_count": waste_other.to(dtype=torch.float32).sum(dim=-1),
        "not_dropped": not_dropped.to(dtype=torch.float32),
        "stable_cap": stable_cap.to(dtype=torch.float32),
        "hold_timer": updated_hold_timer,
        "success": success.to(dtype=torch.float32),
        "drop": drop,
        "cap_rise": cap_rise,
        "lifted": lifted.to(dtype=torch.float32),
        "reward_lift": reward_lift,
        "cap_level_cos": cap_level_cos,
        "cap_tilt_deg": torch.rad2deg(torch.arccos(cap_level_cos.clamp(-1.0, 1.0))),
        "level_ok": level_ok.to(dtype=torch.float32),
        "fallen": fallen.to(dtype=torch.float32),
        "cap_speed": cap_speed,
    }
    return reward, success, updated_hold_timer, terms
