"""Isaac Gym replacement for the URDF ``<mimic>`` screw constraint.

Why this exists
---------------
``assets/tumbler/tumbler.urdf`` couples ``cap_lift`` (prismatic) to ``cap_spin``
(revolute) with a ``<mimic>`` tag, so one turn of the cap raises it by one
thread pitch. That tag maps to ``PhysxMimicJointAPI``, which only exists in
Isaac Sim / IsaacLab. **Isaac Gym ignores it**: loading the URDF there yields
two independent DOFs, and driving ``cap_spin`` one full turn moved ``cap_lift``
to 16.2 mm instead of the 10.0 mm the pitch calls for.

So the coupling is re-imposed here, on top of the same URDF, by driving
``cap_lift`` to ``multiplier * cap_spin + offset`` with a stiff PD every step,
applied as joint effort. Unlike teleporting the cap's root state (what
ManipTrans's ``cap_rot_only`` does), the force resolves through the solver, so
the constraint pushes back on whatever is holding the cap. That reaction is the
whole point: a policy can only learn to twist-and-hold if the thread resists.

Effort rather than a position drive because the thread has to *let go*: past
``cap_spin``'s upper limit the cap is unscrewed, and success means carrying it
away. Zeroing a force is a per-env tensor write; switching off a position drive
would mean rewriting actor DOF properties mid-episode.

The multiplier is read out of the URDF rather than hardcoded, so this stays in
sync with whatever ``make_screw_asset.py --pitch`` produced.

Usage
-----
    coupling = ScrewCoupling.from_urdf(URDF)
    coupling.bind(gym, env, actor)              # once per env, after create_actor
    ...
    coupling.apply(dof_pos, dof_targets)           # every step, before set_dof_position_target
    coupling.release_bound(gym, envs, actors)      # frees actors that finished unscrewing
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class MimicSpec:
    """One ``<mimic>`` relation: follower = multiplier * source + offset."""

    follower: str
    source: str
    multiplier: float
    offset: float
    # Source travel limits. The upper one is the end of the thread: past it the
    # cap is off and the coupling must let go. Both are needed to keep the
    # integrated angle inside what the joint can physically do.
    source_upper: float = float("inf")
    source_lower: float = float("-inf")
    # 소스 조인트의 <dynamics friction> (Nm). PhysX articulation 이 URDF DOF
    # friction 을 무시하므로 (0.6 Nm 를 넣어도 0.1 Nm 토크에 돌았다 -- 실측),
    # bind() 가 effort 제한 속도 드라이브로 대신 구현한다.
    source_friction: float = 0.0

    @property
    def pitch(self) -> float:
        """Thread pitch in metres per revolution (for a linear/angular pair)."""
        import math

        return self.multiplier * 2.0 * math.pi


def parse_mimic_joints(urdf_path: str) -> Dict[str, MimicSpec]:
    """Pull every ``<mimic>`` relation out of a URDF, keyed by follower joint."""
    root = ET.parse(urdf_path).getroot()
    out: Dict[str, MimicSpec] = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        name = joint.get("name")
        src = mimic.get("joint")
        upper, lower = float("inf"), float("-inf")
        fric = 0.0
        for j in root.findall("joint"):
            if j.get("name") == src:
                lim = j.find("limit")
                if lim is not None and lim.get("upper") is not None:
                    upper = float(lim.get("upper"))
                if lim is not None and lim.get("lower") is not None:
                    lower = float(lim.get("lower"))
                dyn = j.find("dynamics")
                if dyn is not None and dyn.get("friction") is not None:
                    fric = float(dyn.get("friction"))
        out[name] = MimicSpec(
            follower=name,
            source=src,
            multiplier=float(mimic.get("multiplier", "1.0")),
            offset=float(mimic.get("offset", "0.0")),
            source_upper=upper,
            source_lower=lower,
            source_friction=fric,
        )
    return out


class ScrewCoupling:
    """Drives mimic-follower DOFs from their source DOF, in Isaac Gym.

    ``stiffness`` has to be firm enough that the follower tracks under load but
    not so firm that the drive goes unstable at the sim timestep. PhysX's joint
    drive is implicit, so it tolerates far more than the explicit root-state
    servo elsewhere in this project did -- that one catapulted the cap at
    6.9 m/s when its gain went to 2000. Measured with check_screw_gym.py at
    dt=1/60, substeps=2, against a 20 N axial load:

        k=8000     free 0.110 mm   loaded 2.284 mm   slope err 9.00%
        k=40000    free 0.025 mm   loaded 0.384 mm   slope err 1.49%
        k=200000   free 0.004 mm   loaded 0.051 mm   slope err 0.20%   <- default
        k=1000000  free 0.000 mm   loaded 0.004 mm   slope err 0.01%

    Nothing went unstable even at 1e6, but 2e5 already tracks to 0.05 mm under
    load and leaves headroom. Re-run the check if the timestep or cap mass
    changes.
    """

    def __init__(
        self,
        specs: Dict[str, MimicSpec],
        stiffness: float = 200_000.0,
        damping: float = 1_000.0,
        max_effort: float = 200.0,
        ratchet: bool = False,
        engage_deg: float | None = None,
    ) -> None:
        self.specs = specs
        self.stiffness = stiffness
        self.damping = damping
        # 팔(RB5) 은 접촉을 통해 200N 을 넘는 축방향 힘을 낼 수 있어 나사가
        # "뜯긴다" (spin<180 인데 lift 가 한계까지 끌려 캡-몸체 분리, A4 실측).
        # 실물 나사는 기구학 구속이므로 유지력을 env var 로 올릴 수 있게 한다.
        # 기본 200 = 기존 hand-only 동작 그대로.
        import os as _os
        self.max_effort = float(_os.environ.get("CAP_SCREW_HOLD_FORCE", str(max_effort)))
        # Off by default. It stops the cumulative angle falling back, which also
        # stops the cap descending while still engaged -- but it removes any cost
        # for losing the grip, so a policy can bump the cap round in steps
        # instead of holding it. The descent that actually needs fixing happens
        # *after* release, which is release_bound's job, not this.
        self.ratchet = ratchet
        # Where the thread ends, in degrees, when that is not the joint's own
        # upper limit. Leaving the two equal is what stalled every episode at
        # 177-178: PhysX settles a joint against a hard stop a little short of
        # the limit, so an angle read that never quite reaches it can never trip
        # a threshold placed exactly there. Give the joint a couple of degrees
        # of headroom past the thread end and the reading passes cleanly.
        self.engage_deg = engage_deg
        self._peak: torch.Tensor | None = None
        self.follower_idx: list[int] = []
        self.source_idx: list[int] = []
        self.multiplier: torch.Tensor | None = None
        self.offset: torch.Tensor | None = None
        self.lower: torch.Tensor | None = None
        self.upper: torch.Tensor | None = None
        self._names: list[str] = []
        # free6 판(tumbler_free6.urdf)의 여분 자유도(cap_tx/ty/rx/ry). 잠긴 동안
        # 강성 드라이브로 0 에 고정하고 해제 때 풀어 캡을 완전 자유로 만든다.
        # 구 2자유도 URDF 에는 이 조인트가 없으므로 목록이 비고 동작이 같다.
        self.hold_idx: list[int] = []
        self._hold_local: list[int] = []
        self._source_local: list[int] = []
        self._source_limits: list[tuple[float, float]] = []
        # Isaac Gym reports revolute DOF positions wrapped into a 2*pi window
        # even when the URDF gives finite multi-turn limits: driving cap_spin at
        # 2 rad/s for 15 s read back as +4.87 rad, not 30 rad (30 - 8*pi). Using
        # that directly makes the cap screw back down every turn. So the
        # cumulative angle is integrated here from wrapped deltas.
        self._prev_src: torch.Tensor | None = None
        self._unwrapped: torch.Tensor | None = None

    @classmethod
    def from_urdf(cls, urdf_path: str, **kw) -> "ScrewCoupling":
        specs = parse_mimic_joints(urdf_path)
        if not specs:
            raise ValueError(f"no <mimic> joints found in {urdf_path}")
        return cls(specs, **kw)

    def bind(self, gym, env, actor, device: str = "cpu", num_envs: int = 1) -> None:
        """Resolve DOF indices and set drive modes on the actor.

        The source DOF is left undriven so the hand can turn it; only the
        follower is position-driven.
        """
        from isaacgym import gymapi

        names = gym.get_actor_dof_names(env, actor)
        self._names = names
        props = gym.get_actor_dof_properties(env, actor)

        # Two different index spaces. props[] is indexed actor-locally, but
        # apply() writes into the env-wide DOF tensor, where the hand's joints
        # come first. Using the actor-local index there drove finger joint 1
        # instead of cap_lift, and read finger joint 0 as the spin angle -- the
        # cap simply rested on its lower stop and looked perfectly stable.
        self.follower_idx, self.source_idx = [], []
        self._follower_local = []
        mults, offs = [], []
        for spec in self.specs.values():
            if spec.follower not in names or spec.source not in names:
                raise ValueError(
                    f"joint {spec.follower!r}/{spec.source!r} missing from actor DOFs {names}"
                )
            fi = names.index(spec.follower)
            si = names.index(spec.source)
            self._source_local.append(si)
            self._source_limits.append((float(props["lower"][si]), float(props["upper"][si])))
            self._follower_local.append(fi)
            self.follower_idx.append(gym.get_actor_dof_index(env, actor, fi, gymapi.DOMAIN_ENV))
            self.source_idx.append(gym.get_actor_dof_index(env, actor, si, gymapi.DOMAIN_ENV))
            mults.append(spec.multiplier)
            offs.append(spec.offset)

            # Position drive, deliberately. An explicit effort PD cannot be
            # this stiff: k=2e5 on the 0.05 kg cap gives omega_n*dt ~ 33 at
            # dt=1/120 and the cap was fired to the travel stop on the first
            # step. PhysX's position drive is implicit and stays clean at the
            # same gain (0.051 mm under a 20 N load). Release is handled by
            # zeroing this actor's gains instead -- see release_bound().
            props["driveMode"][fi] = gymapi.DOF_MODE_POS
            props["stiffness"][fi] = self.stiffness
            props["damping"][fi] = self.damping
            props["effort"][fi] = self.max_effort

        # 나사 저항 (URDF <dynamics friction> of the source joint). PhysX 가
        # articulation DOF friction 을 무시하므로, "목표 속도 0 + 큰 감쇠 +
        # effort = 마찰토크" 속도 드라이브로 대신 만든다: 저항 = min(D|w|, tau_f)
        # -- 사실상 쿨롱 마찰이고, 정지 상태 크리프는 tau/D (무시 가능) 뿐이다.
        # 현재 저항값을 기억해 둔다: release/rebind 가 props 를 get→set 으로
        # 통째로 되쓸 때 이 값을 명시 재스탬프해야 한다 (되쓰기가 낡은 값을
        # 실어와 저항이 조용히 되돌아가는 이력 결함 실측 -- 2026-08-08).
        if not hasattr(self, "_resistance_nm"):
            self._resistance_nm = max(
                (s.source_friction for s in self.specs.values()), default=0.0
            )
        for spec, si in zip(self.specs.values(), self._source_local):
            if spec.source_friction > 0.0:
                props["driveMode"][si] = gymapi.DOF_MODE_VEL
                props["stiffness"][si] = 0.0
                props["damping"][si] = 1000.0
                props["effort"][si] = self._resistance_nm
                print(f"[screw] {spec.source} 나사 저항 {self._resistance_nm:.2f} Nm "
                      "(effort 제한 속도 드라이브)", flush=True)

        self.hold_idx, self._hold_local = [], []
        for hn in ("cap_tx", "cap_ty", "cap_rx", "cap_ry"):
            if hn not in names:
                continue
            hi = names.index(hn)
            props["driveMode"][hi] = gymapi.DOF_MODE_POS
            props["stiffness"][hi] = self.stiffness
            props["damping"][hi] = self.damping
            props["effort"][hi] = self.max_effort
            self._hold_local.append(hi)
            self.hold_idx.append(gym.get_actor_dof_index(env, actor, hi, gymapi.DOMAIN_ENV))

        gym.set_actor_dof_properties(env, actor, props)
        self.multiplier = torch.tensor(mults, device=device, dtype=torch.float32)
        self.offset = torch.tensor(offs, device=device, dtype=torch.float32)
        # Follower travel limits, so the target is clamped the way the joint is
        # -- otherwise the drive fights the limit whenever the cap is turned
        # the wrong way and the reported error is meaningless.
        self.lower = torch.tensor(
            [float(props["lower"][i]) for i in self._follower_local], device=device, dtype=torch.float32
        )
        self.upper = torch.tensor(
            [float(props["upper"][i]) for i in self._follower_local], device=device, dtype=torch.float32
        )
        n_f = len(self.follower_idx)
        self._prev_src = torch.zeros(num_envs, n_f, device=device, dtype=torch.float32)
        self._unwrapped = torch.zeros(num_envs, n_f, device=device, dtype=torch.float32)
        self._peak = torch.zeros(num_envs, n_f, device=device, dtype=torch.float32)
        self.released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._release_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.engage_angle = torch.tensor(
            [self.specs[self._names[i]].source_upper for i in self._follower_local],
            device=device, dtype=torch.float32,
        )
        if self.engage_deg is not None:
            self.engage_angle = torch.full_like(self.engage_angle, math.radians(self.engage_deg))
        self.source_lower = torch.tensor(
            [self.specs[self._names[i]].source_lower for i in self._follower_local],
            device=device, dtype=torch.float32,
        )
        # The follower's upper limit now covers thread + free carry-away travel,
        # so the screw target must clamp at the thread end instead, or the drive
        # would keep hauling the cap up after it is already off.
        self.thread_upper = torch.minimum(self.multiplier * self.engage_angle + self.offset, self.upper)

    def zero_dofs(self, dof_state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Wind the screw back to its start. Call on episode reset, before reset().

        reset() only clears the accumulator; without this the joints keep
        whatever angle and lift the last episode left them at, so a cap that was
        unscrewed stays unscrewed into the next one. Seeding the accumulator from
        those stale positions then makes the new episode's zero the old episode's
        end.

        dof_state is the (E, D, 2) position/velocity view, so velocities go to
        zero too -- a cap still turning through the reset would otherwise carry
        its momentum across.
        """
        idx = slice(None) if env_ids is None else env_ids
        for k in self.follower_idx:
            dof_state[idx, k, 0] = 0.0
            dof_state[idx, k, 1] = 0.0
        for k in self.source_idx:
            dof_state[idx, k, 0] = 0.0
            dof_state[idx, k, 1] = 0.0
        for k in self.hold_idx:
            dof_state[idx, k, 0] = 0.0
            dof_state[idx, k, 1] = 0.0

    def reset(self, dof_pos: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Re-seed the cumulative angle. Call on episode reset, after zero_dofs."""
        src = dof_pos[:, self.source_idx]
        if env_ids is None:
            self._prev_src[:] = src
            self._unwrapped[:] = 0.0
            self._peak[:] = 0.0
            self.released[:] = False
        else:
            self._prev_src[env_ids] = src[env_ids]
            self._unwrapped[env_ids] = 0.0
            self._peak[env_ids] = 0.0
            self.released[env_ids] = False

    def cumulative_angle(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Advance and return the unwrapped source angle (N, n_follow), in rad."""
        import math

        src = dof_pos[:, self.source_idx]
        delta = src - self._prev_src
        # shortest-arc unwrap: a jump larger than pi is the 2*pi wrap, not motion
        delta = delta - (2.0 * math.pi) * torch.round(delta / (2.0 * math.pi))
        self._unwrapped += delta
        # Clamp to what the joint can actually reach. Integrating raw deltas let
        # the accumulator drift far outside the joint's own travel: with
        # cap_spin limited to [-0.02, 3.49] rad, the reported cumulative angle
        # reached -131 deg while the joint itself sat on its lower stop. That is
        # not rotation, it is the solver nudging a jammed joint, and it drove the
        # accumulated-reverse penalty to -1.59/step -- the single largest term in
        # the reward, charged for motion that never happened.
        torch.clamp_(self._unwrapped, self.source_lower, self.engage_angle)
        if self.ratchet:
            torch.maximum(self._unwrapped, self._peak, out=self._unwrapped)
            self._peak.copy_(self._unwrapped)
        self._prev_src.copy_(src)
        return self._unwrapped

    def targets(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """Follower targets (N, n_follow), clamped to the follower's travel."""
        theta = self.cumulative_angle(dof_pos)
        return torch.clamp(self.multiplier * theta + self.offset, self.lower, self.upper)

    def angle(self) -> torch.Tensor:
        """(N,) cumulative unscrew angle of the first follower, in radians."""
        return self._unwrapped[:, 0]

    def lift(self, dof_pos: torch.Tensor) -> torch.Tensor:
        """(N,) axial travel of the cap, straight off the joint.

        Preferred over differencing the cap's world z against a datum captured
        at reset: reset writes DOF state, not link state, so a rigid-body read
        there still holds the *previous* episode's cap height.
        """
        return dof_pos[:, self.follower_idx[0]]

    def apply(self, dof_pos: torch.Tensor, dof_targets: torch.Tensor) -> torch.Tensor:
        """Impose the thread. Call exactly once per step: it integrates the angle.

        While engaged the follower is driven to multiplier * cumulative angle,
        which is what makes the constraint push back on the hand. Past the
        thread's end the cap is unscrewed; the latch is set and release_bound()
        frees that actor so the hand can carry the cap away. The latch never
        clears within an episode -- a cap that came off does not screw itself
        back on by being turned backwards.
        """
        theta = self.cumulative_angle(dof_pos)
        self.released |= (theta >= self.engage_angle).any(dim=-1)
        # 해제 래치는 원시 조인트각으로도 건다. 누적각은 리셋 정착 딥·클램프·
        # 한 스텝 지연이 겹쳐 원시각보다 ~3도 뒤처질 수 있고 (allegro Al5 에서
        # 캡이 한계 182도에 닿아도 누적 179.x 로 래치가 영영 안 걸린 실측),
        # 이 조인트는 한계 182도 < 2pi 라 랩이 불가능하므로 원시각이 곧 참값이다.
        src_raw = dof_pos[:, self.source_idx]
        self.released |= (src_raw >= self.engage_angle).any(dim=-1)
        want = torch.clamp(self.multiplier * theta + self.offset, self.lower, self.thread_upper)
        # Released actors keep whatever target they had; their gains are zero, so
        # it has no effect and the joint is free over the carry-away stroke.
        dof_targets[:, self.follower_idx] = want
        if self.hold_idx:
            # 여분 자유도는 잠긴 동안 0. 해제된 env 는 게인이 0 이라 무시된다.
            dof_targets[:, self.hold_idx] = 0.0
        return dof_targets

    def release_bound(self, gym, envs, actors, dof_pos=None) -> int:
        """Free newly released actors, leaving the release height as a floor.

        The drive goes to zero so the cap can be lifted away, but the joint's
        lower limit is raised to wherever it was when the thread let go. An
        unscrewed cap therefore cannot drop back down its own travel: it stays
        where the last turn left it and the hand only has to carry it further.

        Cheap in practice: fires at most once per env per episode, and only for
        envs that actually finished unscrewing.
        """
        if self._release_done is None:
            self._release_done = torch.zeros_like(self.released)
        fresh = self.released & (~self._release_done)
        idx = torch.nonzero(fresh, as_tuple=False).flatten().tolist()
        # 진단용 누적 해제 횟수 (에피소드 리셋에도 지워지지 않는다).
        if not hasattr(self, "release_count"):
            self.release_count = torch.zeros_like(self.released, dtype=torch.long)
        self.release_count[fresh] += 1
        for i in idx:
            props = gym.get_actor_dof_properties(envs[i], actors[i])
            for k, fi in enumerate(self._follower_local):
                props["stiffness"][fi] = 0.0
                props["damping"][fi] = 0.0
                if self._hold_local:
                    # free6: 해제된 캡은 진짜 강체처럼 떨어져야 한다. 구판의
                    # "해제 높이 바닥"(되감김 방지)을 그대로 두면 z 하한이
                    # 막혀 캡이 제 높이에서 뒤집혀 대롱거리기만 한다. 낙하
                    # 종료(env 쪽)와 짝을 이뤄, 하한을 테이블 아래까지 연다.
                    props["lower"][fi] = -0.35
                elif dof_pos is not None:
                    # Never above the thread end: the drive sags a little under
                    # load, and a floor set above the cap would shove it up.
                    here = float(dof_pos[i, self.follower_idx[k]])
                    props["lower"][fi] = min(here, float(self.thread_upper[k]))
            for hi in self._hold_local:
                props["stiffness"][hi] = 0.0
                props["damping"][hi] = 0.0
            if self._hold_local:
                # free6 판에서만: 해제된 캡은 계속 돌 수 있어야 한다. 잠긴 동안의
                # spin 한계(나사끝+2도)는 나사산의 것이므로 함께 푼다.
                for si in self._source_local:
                    props["lower"][si] = -1.0e3
                    props["upper"][si] = 1.0e3
            self._stamp_resistance(props, env_i=i)
            gym.set_actor_dof_properties(envs[i], actors[i], props)
        self._release_done |= fresh
        return len(idx)

    def rebind_engaged(self, gym, envs, actors, env_ids) -> None:
        """Restore the thread on reset for actors that had been freed."""
        if self._release_done is None:
            return
        ids = [int(i) for i in env_ids if bool(self._release_done[int(i)])]
        for i in ids:
            props = gym.get_actor_dof_properties(envs[i], actors[i])
            for k, fi in enumerate(self._follower_local):
                props["stiffness"][fi] = self.stiffness
                props["damping"][fi] = self.damping
                props["effort"][fi] = self.max_effort
                props["lower"][fi] = float(self.lower[k])   # drop the floor again
            for hi in self._hold_local:
                props["stiffness"][hi] = self.stiffness
                props["damping"][hi] = self.damping
                props["effort"][hi] = self.max_effort
            for k, si in enumerate(self._source_local):
                lo, up = self._source_limits[k]
                props["lower"][si] = lo
                props["upper"][si] = up
            self._stamp_resistance(props, env_i=i)
            gym.set_actor_dof_properties(envs[i], actors[i], props)
            self._release_done[i] = False

    def _stamp_resistance(self, props, env_i: int | None = None) -> None:
        """props 구조체에 현재 나사 저항 드라이브를 써 넣는다.

        release_bound/rebind_engaged 처럼 get→수정→set 으로 props 를 통째로
        되쓰는 모든 경로가 set 직전에 호출해야 한다. 되쓰기가 낡은 소스 드라이브
        값을 실어와 저항이 조용히 바뀌는 것을 막는다. env_i 가 주어지고 per-env
        저항(resist_env)이 설정돼 있으면 그 env 의 값을 쓴다 (저항 랜덤화).
        """
        from isaacgym import gymapi

        r = getattr(self, "_resistance_nm", 0.0)
        re_ = getattr(self, "resist_env", None)
        if env_i is not None and re_ is not None:
            r = float(re_[env_i])
        for si in self._source_local:
            if r > 0.0:
                props["driveMode"][si] = gymapi.DOF_MODE_VEL
                props["stiffness"][si] = 0.0
                props["damping"][si] = 1000.0
                props["effort"][si] = r
            else:
                props["driveMode"][si] = gymapi.DOF_MODE_NONE
                props["stiffness"][si] = 0.0
                props["damping"][si] = 0.0

    def set_resistance(self, gym, envs, actors, torque_nm: float) -> None:
        """나사 저항(쿨롱 마찰 토크, Nm)을 런타임에 바꾼다 -- 저항 커리큘럼용.

        bind() 가 심는 것과 같은 "목표 속도 0 + 큰 감쇠 + effort=토크" 속도
        드라이브를 소스 조인트에 다시 쓴다. 다른 조인트의 드라이브 상태
        (해제된 캡 포함) 는 읽은 그대로 보존된다.
        """
        from isaacgym import gymapi

        import os as _os
        self._resistance_nm = float(torque_nm)
        if getattr(self, "resist_env", None) is not None:
            self.resist_env[:] = float(torque_nm)
        for env, actor in zip(envs, actors):
            props = gym.get_actor_dof_properties(env, actor)
            self._stamp_resistance(props)
            gym.set_actor_dof_properties(env, actor, props)

    def set_resistance_envs(self, gym, envs, actors, env_ids, torques) -> None:
        """env별 나사 저항 설정 (저항 랜덤화). torques 는 env_ids 와 같은 길이.

        per-env 값은 resist_env 에 기억돼 이후 release/rebind 재스탬프와
        _stamp_resistance(env_i) 가 이 값을 유지한다.
        """
        import torch as _torch

        if getattr(self, "resist_env", None) is None:
            self.resist_env = _torch.full(
                (len(envs),), float(getattr(self, "_resistance_nm", 0.0))
            )
        for i, t in zip(env_ids, torques):
            i = int(i)
            self.resist_env[i] = float(t)
            props = gym.get_actor_dof_properties(envs[i], actors[i])
            self._stamp_resistance(props, env_i=i)
            gym.set_actor_dof_properties(envs[i], actors[i], props)

    def describe(self) -> str:
        return "  ".join(
            f"{s.follower}<-{s.source} x{s.multiplier:.6e} (pitch {s.pitch * 1000:.1f}mm/rev)"
            for s in self.specs.values()
        )


