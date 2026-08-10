"""Cap-unscrew reward, ANYmal-style: dense objective + curriculum on constraints.

An alternative to cap_unscrew_rl_terms, kept as a separate file so the two can
be trained and compared. The geometry is shared -- only the reward composition
differs.

What the other module does, and why it stalls
---------------------------------------------
There the objective is gated:

    reward_unscrew = 10.0 * unscrew_frac * grasp_contact       # binary gate
    grasp_contact  = thumb_contact & (>=2 contacts) & opposed  # opposed = 83 deg

Measured over 400 epochs: opposition never left 0%, so reward_unscrew was
*exactly zero for the entire run*. What the policy did learn was to park its
fingers on the cap wall and collect the approach term -- near +1.6 out of a
+1.9 total, contacts 2.4, rotation 0. It optimised the only term that ever paid.

Hwangbo et al. 2019 (Learning agile and dynamic motor skills for legged robots,
arXiv:1901.08652) hit the same shape of problem: "high penalty on them results
in a standing behavior. The main reason for the standing behavior is that such
a behavior is already a good local minimum when there is high penalty
associated with motion." Their fix:

    k_c,j+1 <- (k_c,j)^k_d        k0 = 0.3, k_d = 0.997

"All of cost terms are multiplied by this curriculum factor, except the cost
terms related to the objective ... This way, the robot first learns how to
achieve the objective and then how to respect various constraints."

Two changes are needed to port that here
----------------------------------------
1. Their objective cost (base velocity *error*) is dense -- always finite, always
   improvable. Ours was zero behind a binary gate, so shrinking the constraints
   would have left nothing at all to learn from. The gate is replaced with a
   continuous grasp quality, so a partial grasp pays partially and improves
   monotonically toward a real one:

       grasp_quality  = (1 - closure_err) * min(contacts / min_tips, 1)
       reward_unscrew = w * unscrew_frac * grasp_quality

   This keeps the property the gate was there for -- batting the cap around
   still pays almost nothing, because contacts on one side leave closure_err
   near 1 -- while giving the policy a gradient to climb.

2. The approach term counts as objective, not constraint. In ANYmal, velocity
   tracking is what makes the robot discover walking; here, approach is what
   makes the hand discover touching. Fading it early would remove the only
   thing that gets the fingers to the cap at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from cap_unscrew_rl_terms import (
    band_bounds,
    cap_local_probes,
    cap_wall_radius,
    contact_closure,
    tip_force_magnitude,
    wall_distance_and_contact,
    _as_probe_points,
    _quat_rotate_xyzw,
)


@dataclass(frozen=True)
class CapUnscrewCurriculumConfig:
    # --- curriculum -------------------------------------------------------
    # k0 "should be chosen to prevent the initial tendency to stand still",
    # tunable by watching the first ~100 iterations; k_d "such that the
    # curriculum factor almost reaches 1 (or ~0.9) at the end of training".
    # At k_d = 0.997, k_c goes 0.3 -> 0.9 in about 400 iterations.
    kc_initial: float = 0.4
    kc_advance: float = 0.997

    # --- thread -----------------------------------------------------------
    unscrew_target_rad: float = math.radians(200.0)
    # Success is this one number. It implies the rest: the thread only lifts
    # the cap 10.0 mm (200 deg x 2.865e-3 m/rad), and while engaged the drive
    # can be stretched at most 1.0 mm past its target (200 N effort limit over
    # 200000 N/m), so 11.0 mm is the ceiling with the cap still screwed on.
    # 16 mm therefore cannot be reached without having unscrewed all the way,
    # and cannot be *held* without the hand carrying a cap that nothing else
    # supports. The old three-part gate (200 deg AND 11.5 mm AND opposed grasp)
    # said the same thing three times, and when it read 0% there was no way to
    # tell which part was failing.
    lift_success_height: float = 0.016
    # Where the thread runs out and the drive is switched off. Between here and
    # lift_success_height nothing else in the reward changes -- unscrew_frac has
    # already saturated at 200 deg, hold does not start until the success height,
    # and grasp_quality is as easy to earn pressing down as lifting. Pressing is
    # in fact better, because it also keeps the fingers close for the approach
    # term. Observed exactly that: the hand pushes the released cap back down to
    # the floor. reward_lift fills the gap.
    lift_release_height: float = 0.010
    lift_tol: float = 0.0005
    # Turn-only mode. The task becomes "keep unscrewing"; success is reaching
    # unscrew_target_rad and nothing else, and lift/hold/release drop out of the
    # objective entirely.
    #
    # Two reasons to run it. The deployment one: turn, remain, released and
    # lift_frac are the observations a real hand cannot get -- cumulative
    # rotation needs unwrapped tracking that never drops a frame, and thread
    # release is not observable at all -- so a policy that never reads them is
    # the only one that transfers. The training one: with the payout
    # proportional to the current angle, parking pays. Measured over seven
    # checkpoints, reward climbed 2547 -> 3598 while success fell 12.97% -> 0%,
    # and 54% of envs sat in a 5-degree band at 170-175 collecting 73% of the
    # full unscrew reward for holding still. unscrew_norm_rad past the reachable
    # range removes the landmark that made 180 worth aiming at only in the last
    # few degrees.
    turn_only: bool = False
    unscrew_norm_rad: float = math.radians(360.0)
    # Pay for turning, not for having turned. The level-based payout is what
    # makes parking rational: sitting at 171 deg collects 73% of the full
    # unscrew reward every step, forever, and pushing the last 9 deg risks all
    # of it for a 5.3% raise. Normalising by a bigger angle does not help --
    # both forms are linear, so the 171->180 gain stays 5.3% either way.
    #
    # On the difference of consecutive angles the same 9 deg pays the same as
    # any other 9 deg, holding still pays exactly zero, and there is no
    # landmark anywhere. Signed, not clamped at zero: clamping would pay for
    # the forward half of an oscillation and charge nothing for the back half,
    # which is a worse exploit than the one being removed. Summed over an
    # episode this telescopes to (final - initial) angle, so the total is
    # path-independent and speed only matters up to the clamp.
    turn_progress: bool = False
    unscrew_rate_ref: float = 2.0        # rad/s, 이 속도에서 만점
    collar_overlap: float = 0.0143
    hold_time: float = 0.3
    hold_decay_rate: float = 3.0

    # --- cap geometry (shared with cap_unscrew_rl_terms) ------------------
    cap_radius_lo: float = 0.0469
    cap_radius_hi: float = 0.0500
    cap_radius_knee: float = 0.017
    cap_height: float = 0.0300
    # Measured off link_X_tip.STL, not guessed. At the old 0.009 the distance
    # term bottomed out 0.9mm before the tip could actually reach the wall.
    finger_radius: float = 0.0081
    band_lo: float = 0.0143
    band_margin: float = 0.002
    radial_margin: float = 0.010

    # --- contact ----------------------------------------------------------
    # Contact only counts when the finger's *palmar* face is the one touching.
    # Nothing in the reward distinguished the two faces before, so pushing the
    # cap round with the backs of the fingers scored exactly like gripping it --
    # and it is easier, needing only extension. Observed directly in the viewer.
    # It is also a dead end: fingers flex toward the palm only, so a dorsal
    # push can spin the cap but can never hold it through the 12 mm lift.
    #
    # Compared per finger, per step: the finger's own palmar axis rotated by its
    # own distal-link orientation, against the cap's outward normal at that
    # finger's nearest probe. No global direction enters.
    #
    # 0.8 = 37 deg. Was 0.5 (60 deg), which still passed a pad meeting the wall
    # edge-on: with the first working grasp in hand the thumb read cos 0.63, and
    # the viewer showed it touching on the side of the tip rather than the pad.
    # Side contact carries less friction, which matters for the 6mm the hand has
    # to carry the cap after the thread lets go.
    #
    # Earlier history: 0.0 = cos(90) -- "not the back of the finger"
    # and nothing more, so a pad grazing the wall edge-on counted the same as one
    # pressed flat against it. With opposition at 96-100% the grasp is real
    # enough to ask for more, and the measurement said it was needed: only 3.3
    # of 5 fingers passed at all and the mean cosine was 0.35, i.e. contacts
    # sitting ~70 deg off the surface normal. That is the thumb rolling onto its
    # side, which is what the viewer showed.
    palm_facing_min_cos: float = 0.8
    contact_force_threshold: float = 0.05
    # The hard precondition: thumb plus this many fingers total.
    min_contact_tips: int = 2
    # How many contacts count as a full grip, kept separate from the gate above.
    # While the two were the same number, contact_frac saturated on the second
    # contact and nothing in the objective could tell two fingers from five --
    # the policy held exactly two for a whole 30000-epoch run, which is what the
    # reward asked for. The gate stays at 2 so a partial grasp still earns;
    # contact_frac now keeps climbing.
    #
    # Back to 4. It was raised to 5 to give the idle index finger some value,
    # and that target turned out not to exist: across five runs the index never
    # exceeded 3%% contact and the best simultaneous count ever recorded was 3.65
    # of 5. Normalising by a number the hand cannot reach just scales every
    # grasp down -- a solid four-finger grip scored 0.8 instead of 1.0, and the
    # unscrew reward behind it lost a fifth of its value for no gain.
    #
    # The original note, on why it was raised: 4 was saturating it, the hand
    # settled on thumb,
    # middle, ring and pinky at 68-96%% and left the index at 0%% for four runs
    # running, with the wrist turned so it could not reach. Nothing was wrong
    # with the wrist -- the palm angle sat at 29-38 deg against a 40 deg clamp,
    # well clear -- the fifth finger simply had no value once contact_frac had
    # topped out at four.
    contact_ref: int = 4
    grip_force_ref: float = 0.5

    # --- objective (never scaled by kc) -----------------------------------
    unscrew_weight: float = 10.0
    hold_weight: float = 4.0
    near_weight: float = 2.0
    # Gated on grasp quality like unscrew is, so it cannot be earned by wedging
    # the cap upward without holding it.
    lift_weight: float = 4.0
    # Paid for having the thread all the way off, on top of what unscrew_frac
    # already gives. Without it the last few degrees are worth almost nothing:
    # measured at the stall, the hand sits at 176.5 deg with frac 0.981, and
    # closing to 180 adds 0.19/step -- against the risk of disturbing a grasp
    # that is currently earning 9.7. Everything that makes crossing worthwhile
    # (lift, hold) only becomes visible *after* the crossing, so it has to be
    # found by exploration. This puts a step at the boundary itself.
    #
    # Scaled by grasp_quality, so it cannot be collected by knocking the cap
    # loose and letting go -- the hand has to still be holding it.
    release_weight: float = 3.0
    tip_dist_scale: float = 0.06
    near_gate_on_grip: float = 0.5
    # Short-range closure, and the reason it is separate from near.
    #
    # near is a mean over five fingers at a 60 mm scale, which is right for
    # finding the cap from across the workspace and useless for the last few
    # millimetres: measured mid-run, the hand held a correctly opposed posture
    # (thumb 106 deg around from the index) hovering 3.7 mm off the wall, where
    # near already reads 1.88 of its 2.00 maximum. Closing the gap was worth
    # +0.12/step, against a penetration penalty for overshooting -- so it never
    # closed, and every gated term stayed at exactly 0.000 for 900 epochs.
    #
    # The gate itself is a step: one contact and two contacts pay the same
    # nothing, so there is no gradient pointing at it. This term is that
    # gradient. It is a product over the thumb and the nearest other finger, so
    # like the gate it cannot be earned by one finger alone, but unlike the gate
    # it rises smoothly as the pinch closes.
    pinch_weight: float = 2.0
    pinch_dist_scale: float = 0.008
    # Where pinch stops and force begins.
    #
    # dist subtracts finger_radius and clamps at zero, so it bottoms out while
    # the finger is still a fraction of a millimetre off the wall -- at which
    # point PhysX reports no force at all. near and pinch are both already
    # saturated there, and grip, the only term that pays for pressing, sits
    # behind grasp_ok. So between "almost touching" and "touching" the reward is
    # exactly flat, and the hand parks in the gap. Observed in the viewer, and in
    # the diagnostic as band_contact 2.58 against raw_contact 1.53 -- a finger
    # inside the band registering no force.
    #
    # This picks up there: same thumb-and-one-other structure as pinch, but on
    # contact force rather than distance, and ungated so it can be earned on the
    # way to a grasp rather than only after one.
    press_weight: float = 1.0
    # Grasp quality is a product, closure x count. It was a weighted sum for a
    # while, because as a product it had read 0.000 for a whole 200-epoch run --
    # contacts averaged 0.21 of the 2 wanted and closure_err sat at ~1, so both
    # factors were near zero and the product killed the objective it was meant
    # to feed. The sum was the wrong repair. contact_frac saturates at 1.0 on the
    # second contact, so its half alone cleared grasp_stable_min and closure
    # stopped mattering: measured at epoch 2000, quality sat at 0.515 with
    # closure_err ~0.97 and opposition at 0%, which is the minimum that passes
    # and exactly what the policy settled on -- thumb and fingers bunched on one
    # side of the cap, turning it by scraping.
    #
    # What actually made the product read zero was the approach, not the form:
    # nothing paid for closing the last few millimetres, so contacts never
    # happened at all. pinch_weight fixed that (accepted contacts 0.56 -> 2.98),
    # so the product now has something to multiply, and it restores the property
    # the sum threw away -- no opposition, no quality.
    # Turning only pays once a grasp has been *held*. Without this the policy
    # can start rotating the instant anything touches, and it did: observed
    # poking the cap round with a single fingertip. grasp_hold ramps from 0 to 1
    # over grasp_hold_time while grasp_quality stays above grasp_stable_min, and
    # decays when it drops -- so losing the grip mid-turn fades the reward out
    # and the policy has to re-establish it.
    #
    # With the product form this is the opposition requirement, written as a
    # threshold instead of a branch: quality = (1 - closure_err) * contact_frac,
    # so quality >= 0.25 with contact_frac saturated means closure_err <= 0.75 --
    # the same 83 deg the `opposed` diagnostic uses. Combined with the thumb and
    # count preconditions in grasp_ok, `stable` is exactly "thumb contact AND
    # >=2 contacts AND opposed".
    #
    # Written this way rather than as that literal conjunction because the
    # conjunction has no gradient: at closure_err 0.97, where the policy sits, a
    # hard test gives the same zero as at 0.80, and nothing points toward 0.75.
    # As a threshold on a continuous quality, the quality term itself (weight 2)
    # pays for every bit of closure gained on the way, while the large terms
    # behind grasp_hold stay locked until the conjunction is genuinely met.
    #
    # 0.125 = (min_contact_tips / contact_ref) * 0.25. Rescaled when contact_frac
    # stopped saturating at two contacts, so that the condition it encodes is
    # unchanged: two contacts give contact_frac 0.5, and 0.5 * (1 - err) >= 0.125
    # is still err <= 0.75, still 83 deg. Holding it at 0.25 would have quietly
    # demanded four contacts before anything unlocked.
    grasp_stable_min: float = 0.125
    grasp_hold_time: float = 0.2
    grasp_hold_decay: float = 3.0
    grasp_hold_weight: float = 2.0
    # Contact terms belong to the objective, not the constraints. Putting them
    # under kc was a misreading of the paper: ANYmal keeps the *task
    # achievement* measure (velocity error) at full weight and fades only the
    # regularisers. Contact here is task achievement -- you cannot turn what
    # you are not holding. Measured with it under kc=0.5, accepted contacts
    # fell 1.77 -> 0.21 against the gated baseline.
    grasp_quality_weight: float = 2.0
    grip_weight: float = 1.0

    # --- constraints (all scaled by kc): pure regularisers only -----------
    # 6.0, not 2.0. The policy had learned to hold the cap by pushing *into*
    # it: measured 14.4 mm of finger surface inside the cap wall. Stiffening the
    # solver does suppress it (32 iters / 8 substeps took the penalty from
    # -0.257 to -0.024) but it also deletes the grip -- accepted contacts fell
    # 1.89 -> 0.02 -- and costs 4.7x the wall clock, because the behaviour only
    # works *through* the penetration. PhysX rigid contacts are penalty-based
    # anyway, so some penetration always remains; there is nothing to switch
    # off. Cheaper to price it properly and let the policy find a grip that
    # holds on the surface.
    # 3.0, with the tolerance already tightened 5 mm -> 2 mm below. Raising
    # both at once put this at 49% of the whole constraint budget: at the
    # measured 11.4 mm depth it came to -2.82/step against an unscrew term
    # contributing +2.15, cancelling the objective outright. Suppressing
    # penetration hard is exactly how the grip disappears -- the solver sweep
    # showed contacts going 1.89 -> 0.02 when the physics did the same thing.
    penetration_penalty_weight: float = 3.0
    penetration_ref: float = 0.02
    # 2 mm of slack, not 5. Contact needs a little interpenetration to register
    # at all, but 5 mm was most of a finger radius (9 mm) -- a grip could sink
    # nearly halfway into the wall before anything objected.
    penetration_tol: float = 0.002
    # Two separate costs for turning the wrong way, because they catch
    # different things. reverse_* is on angular *speed* and only bites while
    # the cap is actually moving backwards -- tightening slowly slips past it,
    # which is exactly what happened: -47 deg mean at 200 epochs while the
    # speed-based term sat at -0.62. reverse_angle_* is on the accumulated
    # negative angle, so any tightening keeps costing until it is undone.
    # The joint now also blocks tightening past -0.02 rad (make_screw_asset
    # --spin-margin), so this is the second line rather than the only one.
    reverse_penalty_weight: float = 2.0
    reverse_ref: float = 0.5
    reverse_angle_penalty_weight: float = 2.0
    # Fraction of the unscrew target at which the accumulated-angle cost
    # saturates: 0.25 * 200 deg = 50 deg of tightening is already full cost.
    reverse_angle_ref_frac: float = 0.25
    action_penalty_weight: float = 0.01
    wrist_leash_weight: float = 0.3
    # 100 mm, twice the cap radius. At the old 160 mm the term never fired: the
    # wrist was measured orbiting at 115 mm, three cap radii out, for free --
    # and from there no finger can reach the cap wall.
    wrist_leash_radius: float = 0.10
    palm_down_penalty_weight: float = 0.0
    palm_down_max_deg: float = 40.0
    palm_down_ref_deg: float = 40.0
    # Separation measured as an angle about the cap axis, not a distance.
    # Distance does not say what the task needs: two fingers can sit 80 mm
    # apart and still be on the same side of the cap, spread up and down its
    # wall. Measured exactly that -- thumb-index 76.5 mm, thumb-middle 86.8 mm,
    # both past the old 1.6*r threshold, while closure_err stayed ~1 and
    # opposition sat at 0%. Confirmed in the viewer: unscrewing happens, but
    # thumb and index stay together.
    #
    # 90 deg matches the opposition gate: two contacts N deg apart give
    # closure_err cos(N/2), so the 0.75 gate is 83 deg.
    # 2.0, not 1.0: this is a constraint, so it is multiplied by kc (0.4 at the
    # start), and at weight 1.0 it came out around -0.12/step against a +1.42
    # approach term that is happy with the fingers bunched together.
    thumb_pair_penalty_weight: float = 2.0
    # How far the thumb is from where the rest of the hand is, measured on tip
    # POSITIONS -- no contact required. That is the whole point: closure_err is
    # pinned at 1.0 until a grasp exists, so as a penalty it is a constant early
    # on and constants do not steer anything. This one has a gradient from the
    # first step, while the hand is still deciding where to put the thumb, and
    # by the time contacts form the arrangement is already committed.
    #
    # Continuous to 180 deg rather than cut off at 90 like thumb_pair, and taken
    # against the mean of the other four rather than two named pairs. Weighted
    # heavily because four runs have now ended with every finger inside a 90 deg
    # arc while every other term read healthy.
    opposition_penalty_weight: float = 4.0
    # Below this the penalty ramps in; at or above it, nothing.
    opposition_min_sep_deg: float = 90.0
    thumb_pair_min_sep_deg: float = 90.0


DEFAULT_CFG = CapUnscrewCurriculumConfig()


def advance_kc(kc: float, cfg: CapUnscrewCurriculumConfig) -> float:
    """One curriculum step: k_c <- k_c ** k_d, monotonically approaching 1."""
    return float(min(1.0, kc ** cfg.kc_advance))


def palm_facing_mask(
    tip_pos_w: torch.Tensor,
    tip_palm_dir_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor | None,
    cfg: CapUnscrewCurriculumConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N,F) mask and cosine: is each finger's palmar face turned toward the cap?

    The cap's outward normal on its side wall is the radial direction, taken at
    whichever probe of that finger is nearest the wall -- the same point the
    contact distance uses.
    """
    rel = cap_local_probes(tip_pos_w, cap_pos_w, cap_quat_xyzw)      # (N,F,P,3)
    radial = torch.linalg.norm(rel[..., :2], dim=-1)                 # (N,F,P)
    r_wall = cap_wall_radius(rel[..., 2], cfg)
    near_i = (radial - r_wall).abs().argmin(dim=-1, keepdim=True)    # (N,F,1)
    pick = near_i[..., None].expand(-1, -1, -1, 3)
    rel_n = rel.gather(2, pick).squeeze(2)                           # (N,F,3) cap frame

    outward = torch.zeros_like(rel_n)
    outward[..., :2] = rel_n[..., :2]
    outward = outward / torch.linalg.norm(outward, dim=-1, keepdim=True).clamp_min(1e-6)
    if cap_quat_xyzw is not None:  # back to world
        q = cap_quat_xyzw[:, None, :].expand(-1, outward.shape[1], -1)
        outward = _quat_rotate_xyzw(q, outward)

    palm = tip_palm_dir_w / torch.linalg.norm(tip_palm_dir_w, dim=-1, keepdim=True).clamp_min(1e-6)
    cos = (palm * (-outward)).sum(dim=-1)     # palm pointing at the cap axis
    return cos >= cfg.palm_facing_min_cos, cos


def thumb_opposition_err(
    tip_pos_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor | None,
    contact: torch.Tensor,
    cfg: CapUnscrewCurriculumConfig,
) -> torch.Tensor:
    """cos(theta/2) between the thumb and the mean direction of the rest.

    Two things have to hold at once, and each of the obvious forms breaks one.

    ``|sum of unit radials| / n`` over every contact -- the original -- correctly
    rejects a hand bunched on one side, but scores this hand's anatomy backwards:
    index through pinky sit within ~65 deg of each other, so every finger added
    to a wrap pushes the sum up. A two-finger pinch scored 1.000 against 0.464
    for a full five-finger wrap, and the policy duly delivered the pinch.

    Thumb against the *best-opposed single* finger fixes the count problem and
    loses the first property. Measured at epoch 1300: fingertips at -21, 69, 44,
    19, -13 deg -- the whole hand inside a 90 deg arc, thumb 8 deg from the pinky
    -- and it read 97% opposed, because the one thumb/index pair happened to span
    90 deg. Visibly not an opposed grasp.

    The mean direction of the non-thumb contacts keeps both. Adding a finger
    beside the others barely moves their centroid, so count still costs nothing;
    but the thumb now has to sit opposite where the hand actually is, not merely
    opposite some one finger. On that same epoch-1300 layout the others average
    30 deg against a thumb at -21, a 51 deg split, err 0.90 -- rejected.

    Scale is unchanged: for a thumb and one other finger theta apart this is
    still cos(theta/2), so 0.75 is still 83 deg.
    """
    rel = cap_local_probes(tip_pos_w, cap_pos_w, cap_quat_xyzw)
    radial = rel[..., :2]
    unit = (radial / torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(1e-6)).mean(dim=2)
    unit = unit / torch.linalg.norm(unit, dim=-1, keepdim=True).clamp_min(1e-6)

    others = contact[:, 1:].to(dtype=unit.dtype)[..., None]      # (N,4,1)
    mean_o = (unit[:, 1:, :] * others).sum(dim=1)                # (N,2)
    n_o = torch.linalg.norm(mean_o, dim=-1, keepdim=True)
    mean_o = mean_o / n_o.clamp_min(1e-6)

    cos = (unit[:, 0, :] * mean_o).sum(dim=-1)                   # (N,)
    err = ((cos + 1.0) * 0.5).clamp(0.0, 1.0).sqrt()             # = cos(theta/2)
    # Needs the thumb and at least one other, and the others must not cancel to
    # nothing -- a pair at 180 deg leaves no direction for the thumb to oppose.
    ok = contact[:, 0] & (contact[:, 1:].any(dim=-1)) & (n_o.squeeze(-1) > 1e-3)
    return torch.where(ok, err, torch.ones_like(err))


def cap_unscrew_curriculum_reward(
    unscrew_angle: torch.Tensor,
    prev_unscrew_angle: torch.Tensor,
    cap_rise: torch.Tensor,
    hold_timer: torch.Tensor,
    grasp_timer: torch.Tensor,
    dt: float,
    tip_pos_w: torch.Tensor,
    cap_pos_w: torch.Tensor,
    cap_quat_xyzw: torch.Tensor,
    wrist_pos_w: torch.Tensor,
    kc: float,
    tip_palm_dir_w: torch.Tensor | None = None,
    # 환경이 tools2 용으로 넘기는 해제 래치. 이 판(tools)은 사용하지 않지만
    # 시그니처에 없으면 TypeError 로 기존 체크포인트 재생이 깨진다.
    released_latch: torch.Tensor | None = None,
    wrist_quat_xyzw: torch.Tensor | None = None,
    tip_forces_w: torch.Tensor | None = None,
    tip_contact: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    cfg: CapUnscrewCurriculumConfig = DEFAULT_CFG,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Reward, success, updated hold timer, diagnostics.

    ``kc`` is the curriculum factor in (0, 1]; it multiplies every constraint
    term and none of the objective terms.
    """
    if tip_forces_w is None and tip_contact is None:
        raise ValueError("provide tip_contact or tip_forces_w")
    force_mag = (
        torch.zeros_like(tip_contact, dtype=cap_pos_w.dtype)
        if tip_forces_w is None
        else tip_force_magnitude(tip_forces_w)
    )
    if tip_contact is None:
        tip_contact = force_mag > cfg.contact_force_threshold
    raw_contact = tip_contact

    dist, on_band, depth = wall_distance_and_contact(tip_pos_w, cap_pos_w, cap_quat_xyzw, cfg)
    if tip_palm_dir_w is None:
        palm_ok = torch.ones_like(on_band)
        palm_cos = torch.zeros_like(dist)
        # Separate from palm_cos, which stays 0 here so the diagnostic reads
        # "unknown" rather than "square on". press multiplies by this, and with
        # no palm direction supplied it must not silently zero the term.
        press_dir = torch.ones_like(dist)
    else:
        palm_ok, palm_cos = palm_facing_mask(
            tip_pos_w, tip_palm_dir_w, cap_pos_w, cap_quat_xyzw, cfg
        )
        press_dir = palm_cos.clamp(0.0, 1.0)
    tip_contact = raw_contact & on_band & palm_ok
    contact_count = tip_contact.to(dtype=torch.int32).sum(dim=-1)
    thumb_contact = tip_contact[:, 0]
    closure_err = thumb_opposition_err(tip_pos_w, cap_pos_w, cap_quat_xyzw, tip_contact, cfg)

    # The precondition for every grasp-dependent payout: the thumb plus at
    # least one more finger. Without the thumb the other four are near-parallel
    # in azimuth, so closure_err is pinned around 1 and grasp_quality can never
    # leave ~0.5 -- which is exactly grasp_stable_min, so a one-sided press
    # cleared the gate and drew half the unscrew reward with nothing opposing
    # it. That is the pushing motion seen in the viewer. Opposition on this
    # hand can only come from the thumb, so require it rather than leaving the
    # closure term to discover it.
    grasp_ok = thumb_contact & (contact_count >= cfg.min_contact_tips)
    grasp_ok_f = grasp_ok.to(dtype=cap_pos_w.dtype)

    # Azimuth about the cap axis, in the cap frame -- the same geometry
    # closure_err uses, but available before contact, so it can steer the
    # placement rather than only score it.
    tips_local = cap_local_probes(tip_pos_w, cap_pos_w, cap_quat_xyzw)[:, :, -1, :]
    az = torch.atan2(tips_local[..., 1], tips_local[..., 0])          # (N,5)
    two_pi = 2.0 * math.pi
    def sep(a, b):
        d = a - b
        return (d - two_pi * torch.round(d / two_pi)).abs()
    sep_index = sep(az[:, 0], az[:, 1])
    sep_middle = sep(az[:, 0], az[:, 2])
    # Thumb against the MIDDLE finger, on tip position alone, and only while they
    # are closer together than opposition_min_sep_deg. Beyond it this is zero:
    # the term states a minimum, not a target.
    #
    # It replaces a cos(theta/2) that fell all the way to 180 deg, which pushed
    # too hard. That version did produce opposition -- 175 deg at epoch 362,
    # against a best of 51 in five earlier runs -- but the hand held that spread
    # above the cap and never came down: every fingertip sat at 30-59mm with the
    # band at 14.3-30. A spread that wide has nowhere to go on a 100mm cap except
    # over the top. Asking only for 90 deg leaves room to descend.
    #
    # Still on positions rather than contacts, which is what made the previous
    # version work at all: closure_err is pinned at 1.0 until a grasp exists, so
    # as a penalty it is a constant early on, and constants steer nothing.
    sep_ti = sep(az[:, 0], az[:, 2])
    thr_opp = math.radians(max(cfg.opposition_min_sep_deg, 1e-6))
    opposition_deficit = ((thr_opp - sep_ti) / thr_opp).clamp(0.0, 1.0)

    # --- objective --------------------------------------------------------
    # Continuous stand-in for the old binary gate. Both factors are needed:
    # closure alone would pay for two opposed grazes, count alone would pay for
    # the whole hand pressed on one side.
    contact_frac = (contact_count.to(cap_pos_w.dtype) / max(cfg.contact_ref, 1)).clamp(0.0, 1.0)
    # Graded by how many fingers are on, not gated on two of them.
    #
    # With the gate, the first finger to land changed the reward by exactly
    # nothing: measured at the epoch-406 pose, 0 contacts and 1 contact both
    # scored 0.84 and only the second contact moved it, by +1.07. There is no
    # one-finger-at-a-time route to a grasp, only a coordinated two-finger event
    # that pays nothing when it fails -- which is what the hand was doing,
    # trembling a millimetre off the wall.
    #
    # closure_err cannot fill the gap because it is undefined below two contacts
    # and returns its worst value there, so the first finger would still score
    # zero. Under two contacts the geometric opposition stands in: it is on tip
    # positions, so it is defined from the start and rewards putting the thumb
    # in the right place before anything touches.
    closure_term = torch.where(
        contact_count >= cfg.min_contact_tips,
        (1.0 - closure_err).clamp(0.0, 1.0),
        (1.0 - opposition_deficit).clamp(0.0, 1.0),
    )
    grasp_quality = closure_term * contact_frac * thumb_contact.to(dtype=cap_pos_w.dtype)

    stable = grasp_quality >= cfg.grasp_stable_min
    grasp_timer = torch.where(
        stable, grasp_timer + dt, (grasp_timer - cfg.grasp_hold_decay * dt).clamp_min(0.0)
    )
    grasp_hold = (grasp_timer / max(cfg.grasp_hold_time, 1e-6)).clamp(0.0, 1.0)

    frac = (unscrew_angle / max(cfg.unscrew_target_rad, 1e-6)).clamp(0.0, 1.0)
    # What the payout is scaled by. In turn-only mode this is normalised past
    # anything the joint can reach, so there is no point where turning further
    # stops paying and no shoulder for the policy to settle on; frac itself
    # stays on the target and is used only for success and diagnostics.
    if cfg.turn_progress:
        pay_frac = (
            ((unscrew_angle - prev_unscrew_angle) / max(dt, 1e-6))
            / max(cfg.unscrew_rate_ref, 1e-6)
        ).clamp(-1.0, 1.0)
    elif cfg.turn_only:
        pay_frac = (unscrew_angle / max(cfg.unscrew_norm_rad, 1e-6)).clamp(0.0, 1.0)
    else:
        pay_frac = frac
    # Still needs the full gate: grasp_quality now moves on one finger, and
    # turning the cap with one finger is exactly what min_contact_tips exists
    # to stop. The grading is for the penalty, not for the payout.
    reward_unscrew = pay_frac * grasp_quality * grasp_hold * grasp_ok_f

    lift_span = max(cfg.lift_success_height - cfg.lift_release_height, 1e-6)
    lift_frac = ((cap_rise - cfg.lift_release_height) / lift_span).clamp(0.0, 1.0)
    reward_lift = lift_frac * grasp_quality * grasp_hold * grasp_ok_f

    grip = (force_mag / cfg.grip_force_ref).clamp(0.0, 1.0) * tip_contact.to(dtype=force_mag.dtype)
    reward_grip = grip.mean(dim=-1) * grasp_ok_f
    # near is the exception, and deliberately so: it is the only term that pays
    # before any contact exists, so gating it on contact would leave nothing to
    # bring the hand to the cap in the first place. It is squeezed by grip
    # instead, which now needs the thumb, so it fades only once a real grasp
    # forms rather than on any touch.
    # Half-weighted by facing rather than fully, unlike press and pinch. This is
    # the only term that pays before any contact exists, and at a 60mm scale it
    # is mostly about finding the cap at all -- a hand still 5cm away has no
    # meaningful "which face" yet, so zeroing it on a bad cosine would remove the
    # one signal that gets the fingers there. Halving it still makes a correctly
    # turned approach worth twice a backhanded one, which is the gradient that
    # was missing when the thumb settled at cos -0.32 and stayed.
    near_face = 0.5 + 0.5 * press_dir
    reward_near = (torch.exp(-dist / cfg.tip_dist_scale) * near_face).mean(dim=-1) * (
        1.0 - cfg.near_gate_on_grip * reward_grip.clamp(0.0, 1.0)
    )
    # exp(-a/s) * exp(-b/s), written as one exp. Both distances have to fall for
    # this to move, and the thumb is one of them by construction.
    #
    # Scaled by palm facing for the same reason press is. Distance alone does not
    # say which face of the finger is presented, so pinch paid in full for a
    # thumb hovering 0.7mm off the wall with its back to it -- measured at epoch
    # 424 with the thumb at cos -0.32, accepted contacts down to 0.03 and the cap
    # turning 0.9 deg. near and pinch together were the whole reward, and neither
    # needed the hand to touch anything correctly.
    #
    # The nearest other finger is chosen after the coefficient, so a
    # well-oriented finger slightly further away beats a badly-oriented near one.
    d_thumb = dist[:, 0]
    face = torch.exp(-dist / max(cfg.pinch_dist_scale, 1e-6)) * press_dir
    reward_pinch = face[:, 0] * face[:, 1:].amax(dim=-1)
    d_other = dist[:, 1:].amin(dim=-1)
    # min(), so one finger leaning on the cap earns nothing and the thumb is
    # always one of the two. No band or palm-facing mask here: those belong to
    # the gate, and applying them would put this term behind the same step it
    # exists to smooth over.
    # Scaled by how squarely the pad faces the wall, not gated on it. Leaving
    # direction out was a mistake: press pays for force alone, so the cheapest
    # way to earn it is to shove with whichever part of the finger already
    # points at the cap. Measured over the first 374 epochs of the tip-only run,
    # press climbed 0.26 -> 0.95 while the palm-facing count fell 1.96 -> 0.41
    # and its cosine went negative -- the hand turning the cap 136 deg with the
    # backs of the fingers, the exact behaviour the palm gate exists to reject.
    #
    # A coefficient rather than a mask keeps the reason this term is ungated:
    # there is still no step to clear, the payout just grows as the pad squares
    # up, and dorsal contact earns nothing.
    press = (force_mag / max(cfg.grip_force_ref, 1e-6)).clamp(0.0, 1.0) * press_dir
    reward_press = torch.minimum(press[:, 0], press[:, 1:].amax(dim=-1))

    unscrewed = frac >= 1.0
    reward_release = unscrewed.to(cap_pos_w.dtype) * grasp_quality
    lifted = cap_rise >= (cfg.lift_success_height - cfg.lift_tol)
    opposed = closure_err <= 0.75
    grasped = thumb_contact & (contact_count >= cfg.min_contact_tips) & opposed
    # Held, not merely touched: the cap free-falls the moment the thread lets
    # go, so a bounce can spike past the height for a frame. hold_time keeps
    # that from counting.
    hold_frame = lifted
    updated_hold = torch.where(
        hold_frame, hold_timer + dt, (hold_timer - cfg.hold_decay_rate * dt).clamp_min(0.0)
    )
    success = updated_hold >= cfg.hold_time
    if cfg.turn_only:
        # The whole task: the thread let go. No height, no hold timer.
        success = unscrewed
    # The timer and the success test stay on the height alone; only the payout
    # is gated, so a cap balanced up there without a grasp earns nothing while
    # still being scored the same way it always was.
    reward_hold = (updated_hold / max(cfg.hold_time, 1e-6)).clamp(0.0, 1.0) * grasp_ok_f

    # Only what the task is actually scored on. Everything about the grasp moved
    # to the constraints below, as a deficit.
    #
    # The grasp terms were positive income, and that is what every run has
    # optimised instead of the task. Measured at epoch 387 of the fourth attempt:
    # 3.15 accepted contacts, thumb 96% at cos 0.95, cap turned 143 of 180 deg --
    # and a total of 3.5/step of which near, pinch and press were essentially all
    # of it, with unscrew, quality, ghold and grip together under 0.02. The hand
    # was turning the cap for free and being paid to hold still.
    #
    # A grasp that is already good pays nothing here, so there is no income to
    # protect by staying put; the only way to earn is to turn, lift and hold.
    # Letting the grasp go still costs, because the deficit reappears.
    # Turn-only drops the three terms that exist to carry the cap away after the
    # thread lets go. reward_release goes with them: it fires exactly at
    # frac >= 1.0, which is the 180-degree landmark this mode is meant to stop
    # encoding.
    _tail = 0.0 if cfg.turn_only else 1.0
    objective = (
        cfg.unscrew_weight * reward_unscrew
        + _tail * cfg.hold_weight * reward_hold
        + _tail * cfg.lift_weight * reward_lift
        + _tail * cfg.release_weight * reward_release
        # Approach stays a reward. Moving it to the constraints stalled the run
        # outright: contacts sat at 0.00 for 387 epochs and the deficit stopped
        # falling at epoch 200, because kc holds these at 40-50% through exactly
        # the window where the previous run built its grasp, while the objective
        # is gated behind a grasp that does not exist yet. Nothing bootstraps.
        + cfg.near_weight * reward_near
        + cfg.pinch_weight * reward_pinch
        + cfg.press_weight * reward_press
    )

    # --- constraints ------------------------------------------------------
    penetration = ((depth.amax(dim=-1) - cfg.penetration_tol).clamp_min(0.0) / cfg.penetration_ref).clamp(0.0, 1.0)
    reverse = (
        ((prev_unscrew_angle - unscrew_angle) / max(dt, 1e-6)).clamp_min(0.0) / max(cfg.reverse_ref, 1e-6)
    ).clamp(0.0, 1.0)
    rev_ref = max(cfg.reverse_angle_ref_frac * cfg.unscrew_target_rad, 1e-6)
    reverse_angle = ((-unscrew_angle).clamp_min(0.0) / rev_ref).clamp(0.0, 1.0)
    leash_r = torch.linalg.norm(wrist_pos_w[:, :2] - cap_pos_w[:, :2], dim=-1)
    leash = ((leash_r - cfg.wrist_leash_radius).clamp_min(0.0) / max(cfg.wrist_leash_radius, 1e-6)).clamp(0.0, 1.0)

    if wrist_quat_xyzw is None:
        palm_deg = torch.zeros_like(frac)
        palm = torch.zeros_like(frac)
    else:
        palm_local = torch.zeros_like(wrist_pos_w)
        palm_local[:, 0] = 1.0
        palm_w = _quat_rotate_xyzw(wrist_quat_xyzw, palm_local)
        palm_deg = torch.rad2deg((-palm_w[:, 2]).clamp(-1.0, 1.0).arccos())
        palm = ((palm_deg - cfg.palm_down_max_deg) / max(cfg.palm_down_ref_deg, 1e-6)).clamp(0.0, 1.0)

    tips = _as_probe_points(tip_pos_w)[:, :, -1, :]
    d_index = torch.linalg.norm(tips[:, 0] - tips[:, 1], dim=-1)
    d_middle = torch.linalg.norm(tips[:, 0] - tips[:, 2], dim=-1)
    thr = math.radians(max(cfg.thumb_pair_min_sep_deg, 1e-6))
    # min = soft AND: the cost clears as soon as *either* finger swings round
    # to oppose the thumb, which is the behaviour wanted.
    thumb_pair = torch.minimum(
        ((thr - sep_index) / thr).clamp(0.0, 1.0), ((thr - sep_middle) / thr).clamp(0.0, 1.0)
    )

    action_pen = torch.zeros_like(frac) if actions is None else actions.square().mean(dim=-1)

    # Each grasp term enters as its shortfall from perfect, so the sign flips but
    # the gradient does not: -(1 - near) moves exactly the way +near did. What
    # changes is the ceiling -- a perfect grasp scores 0 here instead of +8, so
    # holding one is no longer a way to earn.
    #
    # Under kc these start at 0.4 weight and grow, which is the intended order:
    # the objective is unscaled and available from the first step, the grasp
    # requirements tighten around it. The risk is the mirror of the old failure
    # -- early on the objective is gated behind a grasp the policy does not have
    # yet, and the pressure to build one is at 40% strength. near carries that,
    # and it is the largest of the six.
    constraints = (
        -cfg.penetration_penalty_weight * penetration
        - cfg.reverse_penalty_weight * reverse
        - cfg.reverse_angle_penalty_weight * reverse_angle
        - cfg.wrist_leash_weight * leash
        - cfg.palm_down_penalty_weight * palm
        - cfg.thumb_pair_penalty_weight * thumb_pair
        - cfg.action_penalty_weight * action_pen
        - cfg.opposition_penalty_weight * opposition_deficit
        - cfg.grasp_quality_weight * (1.0 - grasp_quality)
        - cfg.grasp_hold_weight * (1.0 - grasp_hold)
        - cfg.grip_weight * (1.0 - reward_grip)
    )

    reward = objective + kc * constraints

    terms = {
        "kc": torch.full_like(frac, kc),
        "unscrew_angle_deg": torch.rad2deg(unscrew_angle),
        "unscrew_frac": frac,
        "reward_unscrew": reward_unscrew,
        "reward_lift": reward_lift,
        "lift_frac": lift_frac,
        "grasp_quality": grasp_quality,
        "grasp_hold": grasp_hold,
        "grasp_timer": grasp_timer,
        "unscrewed": unscrewed.to(dtype=cap_pos_w.dtype),
        "cap_rise": cap_rise,
        "lifted": lifted.to(dtype=cap_pos_w.dtype),
        "separated": (cap_rise >= cfg.collar_overlap).to(dtype=cap_pos_w.dtype),
        "raw_contact_count": raw_contact.to(dtype=cap_pos_w.dtype).sum(dim=-1),
        "band_contact_count": on_band.to(dtype=cap_pos_w.dtype).sum(dim=-1),
        "palm_facing_count": palm_ok.to(dtype=cap_pos_w.dtype).sum(dim=-1),
        "palm_facing_cos": palm_cos.mean(dim=-1),
        "contact_count": contact_count.to(dtype=cap_pos_w.dtype),
        "thumb_contact": thumb_contact.to(dtype=cap_pos_w.dtype),
        "contact_per_finger": tip_contact.to(dtype=cap_pos_w.dtype),
        "palm_cos_per_finger": palm_cos,
        "dist_per_finger": dist,
        "z_per_finger": cap_local_probes(tip_pos_w, cap_pos_w, cap_quat_xyzw)[..., 2].mean(dim=-1),
        "grasp_ok": grasp_ok_f,
        "reward_near": reward_near,
        "reward_pinch": reward_pinch,
        "reward_press": reward_press,
        "reward_release": reward_release,
        "opposition_deficit": opposition_deficit,
        "thumb_surface_dist": d_thumb,
        "other_surface_dist": d_other,
        "grasp_contact": grasped.to(dtype=cap_pos_w.dtype),
        "closure_err": closure_err,
        "opposed": opposed.to(dtype=cap_pos_w.dtype),
        "reward_grip": reward_grip,
        "max_tip_force": force_mag.amax(dim=-1),
        "tip_surface_dist": dist.amin(dim=-1),
        "penetration_depth": depth.amax(dim=-1),
        "penetration_penalty": penetration,
        "reverse_penalty": reverse,
        "reverse_angle_penalty": reverse_angle,
        "wrist_cap_radius": leash_r,
        "leash_penalty": leash,
        "palm_down_deg": palm_deg,
        "palm_down_penalty": palm,
        "thumb_index_dist": d_index,
        "thumb_middle_dist": d_middle,
        "thumb_pair_penalty": thumb_pair,
        "thumb_index_sep_deg": torch.rad2deg(sep_index),
        "thumb_middle_sep_deg": torch.rad2deg(sep_middle),
        "hold_timer": updated_hold,
        "reward_hold": reward_hold,
        "success": success.to(dtype=cap_pos_w.dtype),
    }
    return reward, success, updated_hold, terms
