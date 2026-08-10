from .base import DexHand
from .decorators import register_dexhand
from abc import ABC, abstractmethod
import numpy as np
from main.dataset.transform import aa_to_rotmat


# ============================================================================
# Tesollo DG-5F (dg5fs) 5-finger anthropomorphic hand — 20 DOF (4 joints/finger)
#
# URDF finger index -> anatomical finger  (표준 관례 + 형상 검증으로 확정):
#   finger 1 -> thumb   (mount y=+0.018, 낮은 z, rest에서 옆으로 대향)
#   finger 2 -> index   (mount y=+0.028, rest 위)
#   finger 3 -> middle  (mount y=+0.005, rest 위)
#   finger 4 -> ring    (mount y=-0.018, rest 위)
#   finger 5 -> pinky   (mount y=-0.042, ring 너머, rest 위, 가장 짧음)
# y좌표가 index>middle>ring>pinky로 단조 감소, thumb만 반대편 대향 배치 → 해부학적으로 일치.
#
# 각 finger X 링크: link_X_1(knuckle/abduction) link_X_2(proximal,MCP) link_X_3(intermediate,PIP)
#   link_X_4(distal,DIP) link_X_tip(fixed tip).  조인트: joint_X_1(abduction) _2(MCP) _3(PIP) _4(DIP).
# ============================================================================


class DG5FS(DexHand, ABC):
    def __init__(self):
        super().__init__()
        self._urdf_path = None
        self.side = None
        self.name = "dg5fs"

        # body_names: index 0 = base, 그다음 thumb,index,middle,ring,pinky 순
        # (각 5링크: _1,_2,_3,_4,_tip). 이 순서가 bone_links / weight_idx 인덱스를 정의.
        self.body_names = [
            "link_base",  # 0
            # thumb (finger 1)
            "link_1_1", "link_1_2", "link_1_3", "link_1_4", "link_1_tip",   # 1-5
            # index (finger 2)
            "link_2_1", "link_2_2", "link_2_3", "link_2_4", "link_2_tip",   # 6-10
            # middle (finger 3)
            "link_3_1", "link_3_2", "link_3_3", "link_3_4", "link_3_tip",   # 11-15
            # ring (finger 4)
            "link_4_1", "link_4_2", "link_4_3", "link_4_4", "link_4_tip",   # 16-20
            # pinky (finger 5)
            "link_5_1", "link_5_2", "link_5_3", "link_5_4", "link_5_tip",   # 21-25
        ]

        # dof_names: 20 revolute joints, finger별 base->tip, body_names와 같은 finger 순서.
        self.dof_names = [
            "joint_1_1", "joint_1_2", "joint_1_3", "joint_1_4",  # thumb
            "joint_2_1", "joint_2_2", "joint_2_3", "joint_2_4",  # index
            "joint_3_1", "joint_3_2", "joint_3_3", "joint_3_4",  # middle
            "joint_4_1", "joint_4_2", "joint_4_3", "joint_4_4",  # ring
            "joint_5_1", "joint_5_2", "joint_5_3", "joint_5_4",  # pinky
        ]

        # MANO/SMPL-X keypoint -> dexhand body mapping.
        # 각 finger의 5링크를 MANO의 4키포인트에 매핑 (proximal이 2개 흡수).
        self.hand2dex_mapping = {
            "wrist": ["link_base"],
            "thumb_proximal": ["link_1_1", "link_1_2"],
            "thumb_intermediate": ["link_1_3"],
            "thumb_distal": ["link_1_4"],
            "thumb_tip": ["link_1_tip"],
            "index_proximal": ["link_2_1", "link_2_2"],
            "index_intermediate": ["link_2_3"],
            "index_distal": ["link_2_4"],
            "index_tip": ["link_2_tip"],
            "middle_proximal": ["link_3_1", "link_3_2"],
            "middle_intermediate": ["link_3_3"],
            "middle_distal": ["link_3_4"],
            "middle_tip": ["link_3_tip"],
            "ring_proximal": ["link_4_1", "link_4_2"],
            "ring_intermediate": ["link_4_3"],
            "ring_distal": ["link_4_4"],
            "ring_tip": ["link_4_tip"],
            "pinky_proximal": ["link_5_1", "link_5_2"],
            "pinky_intermediate": ["link_5_3"],
            "pinky_distal": ["link_5_4"],
            "pinky_tip": ["link_5_tip"],
        }
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        assert len(self.dex2hand_mapping.keys()) == len(self.body_names)

        # distal (마지막 구동) phalanx = contact body  [thumb, index, middle, ring, pinky]
        self.contact_body_names = [
            "link_1_4",  # thumb
            "link_2_4",  # index
            "link_3_4",  # middle
            "link_4_4",  # ring
            "link_5_4",  # pinky
        ]

        # skeleton edges (body_names 인덱스)
        self.bone_links = [
            [0, 1], [0, 6], [0, 11], [0, 16], [0, 21],  # base -> 각 finger root
            [1, 2], [2, 3], [3, 4], [4, 5],              # thumb
            [6, 7], [7, 8], [8, 9], [9, 10],             # index
            [11, 12], [12, 13], [13, 14], [14, 15],      # middle
            [16, 17], [17, 18], [18, 19], [19, 20],      # ring
            [21, 22], [22, 23], [23, 24], [24, 25],      # pinky
        ]

        self.weight_idx = {
            "thumb_tip": [5],
            "index_tip": [10],
            "middle_tip": [15],
            "ring_tip": [20],
            "pinky_tip": [25],
            # level_1: knuckle + proximal (link_X_1, link_X_2)
            "level_1_joints": [1, 2, 6, 7, 11, 12, 16, 17, 21, 22],
            # level_2: intermediate + distal (link_X_3, link_X_4)
            "level_2_joints": [3, 4, 8, 9, 13, 14, 18, 19, 23, 24],
        }

        # ? PID-controlled wrist pose mode (reference only, not our main method)
        self.Kp_rot = 0.8
        self.Ki_rot = 0.001
        self.Kd_rot = 0.01
        self.Kp_pos = 80
        self.Ki_pos = 0.005
        self.Kd_pos = 3

    def __str__(self):
        return self.name


@register_dexhand("dg5fs_rh")
class DG5FSRH(DG5FS):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/dg5fs_hand/dg5fs_right.urdf"
        self.side = "rh"
        # NOTE: 시작 추정값 (shadow/xhand RH 관례). GUI/리타겟팅에서 MANO 정렬로 조정 필요.
        self.relative_rotation = aa_to_rotmat(np.array([0, -np.pi / 2, 0]))
        # wrist를 손등(dorsal) 방향으로 1cm 이동. 손목-로컬 dorsal 단위벡터 × 거리(m).
        # 프레임마다 wrist 회전을 곱해 적용되므로 항상 손등 방향(월드 up이 아님).
        self.wrist_dorsal_offset = np.array([-0.274, 0.568, 0.776]) * 0.05

    def __str__(self):
        return super().__str__() + "_rh"


@register_dexhand("dg5fs_lh")
class DG5FSLH(DG5FS):
    def __init__(self):
        super().__init__()
        self._urdf_path = "assets/dg5fs_hand/dg5fs_left.urdf"
        self.side = "lh"
        self.relative_rotation = aa_to_rotmat(np.array([0, np.pi / 2, 0]))

    def __str__(self):
        return super().__str__() + "_lh"
