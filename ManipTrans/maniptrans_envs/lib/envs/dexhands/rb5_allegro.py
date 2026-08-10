"""RB5-850E 팔 + Allegro 오른손 결합 로봇 (캡 unscrew 팔 통합용).

AllegroRH 를 확장해 팔 6관절을 DOF 앞에 붙인다. base(link0) 고정이 전제라
retargeting 용 필드(relative_rotation 등)는 allegro 값을 그대로 물려받지만
실제로는 쓰이지 않는다 (cap 과제는 trajectory-free).
"""

import numpy as np

from .allegro import AllegroRH
from .decorators import register_dexhand


@register_dexhand("rb5_allegro_rh")
class Rb5AllegroRH(AllegroRH):
    ARM_BODIES = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "tcp"]
    ARM_DOFS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]

    def __init__(self):
        super().__init__()
        self.name = "rb5_allegro"
        # 결합 URDF (절대경로 -- task1 자산)
        import os as _os
        # RB5_ALLEGRO_URDF 로 마운트 변형 선택 (기본 = 원판 Rx90 마운트).
        # vmount = J6축 ⊥ 팜면 + 손목끝 축중심 + 갭 30mm (2026-08-07 사용자 설계).
        self._urdf_path = _os.environ.get(
            "RB5_ALLEGRO_URDF",
            "/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro.urdf")
        # 자기충돌 켬(필터 0). 결합 자산 rest/파지 유령 접촉 0 실측 (2026-08-07).
        self.self_collision = False
        n_arm = len(self.ARM_BODIES)
        # 팔 몸체가 body 목록 앞에 붙는다 -> allegro 쪽 인덱스 전부 +n_arm.
        self.body_names = self.ARM_BODIES + self.body_names
        self.dof_names = self.ARM_DOFS + self.dof_names
        self.hand2dex_mapping = {
            k: v for k, v in self.hand2dex_mapping.items()
        }
        self.hand2dex_mapping["wrist"] = ["base_link"]   # 손목 = 손 베이스 (팔 링크 아님)
        # 팔 몸체는 대응 인체 부위가 없다 -- dex2hand 재계산 시 wrist 로 몰아준다.
        for b in self.ARM_BODIES:
            self.hand2dex_mapping.setdefault("wrist", [])
            if b not in self.hand2dex_mapping["wrist"]:
                self.hand2dex_mapping["wrist"].append(b)
        self.dex2hand_mapping = self.reverse_mapping(self.hand2dex_mapping)
        self.weight_idx = {
            k: [i + n_arm for i in v] for k, v in self.weight_idx.items()
        }
        # contact_body_names 는 이름 기반이라 그대로 유효 (엄지/검지/중지/약지/약지중복)
        self.bone_links = [[a + n_arm, b + n_arm] for a, b in self.bone_links]

    def __str__(self):
        # 리타게팅 데이터 경로가 str(dexhand) 로 구성된다. rb5_allegro 전용
        # 데이터는 없고 cap 과제는 trajectory-free 라 내용도 안 쓰므로
        # allegro 데이터를 재사용한다 (손 부분 16 DOF 규격 동일).
        return "allegro_rh"
