"""allegro_hand_right.urdf 조인트 순서와 가동범위 (ManipTrans allegro 규약).

순서는 dexhands/allegro.py 의 dof_names 와 같아야 한다 -- URDF 순차(0..15)가
아니라 손가락별로 index -> thumb -> middle -> ring 이다. 저장되는 자세가 이
순서를 따라야 리셋에서 그대로 _q 에 들어간다.

allegro 는 4지(엄지/검지/중지/약지)라 학습 env 가 5번째 슬롯에 약지를 중복해
채운다. 여기(파지 최적화)에서는 실제 4손끝만 쓴다.
"""

# Isaac Gym 실측 DOF 순서 (get_asset_dof_names). index -> thumb -> middle -> ring.
# 주의: pytorch_kinematics 가 URDF 를 파싱한 순서(0..15 순차)와 다르다 -- Isaac Gym 은
# 엄지(12-15)를 검지 다음(슬롯 4-7)에 놓는다. GraspInitializer.apply 와 view_pose_allegro
# 모두 dof_pos 를 재정렬 없이 _q(=이 Isaac Gym 순서)에 직접 넣으므로, 저장 자세는 이 순서여야
# 한다. dexhands/allegro.py 의 dof_names 와도 일치한다. (FK 슬라이스는 DOF_NAMES.index 로
# pk 순서를 이 순서에 맞춰 풀므로 FK 는 그대로 정확하다.)
DOF_NAMES = [
    "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",       # index
    "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",   # thumb
    "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",       # middle
    "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",     # ring
]

# allegro_hand_right.urdf <limit> 에서 추출
DOF_LIMITS = {
    "joint_0.0": (-0.470000, 0.470000),
    "joint_1.0": (-0.196000, 1.610000),
    "joint_2.0": (-0.174000, 1.709000),
    "joint_3.0": (-0.227000, 1.618000),
    "joint_4.0": (-0.470000, 0.470000),
    "joint_5.0": (-0.196000, 1.610000),
    "joint_6.0": (-0.174000, 1.709000),
    "joint_7.0": (-0.227000, 1.618000),
    "joint_8.0": (-0.470000, 0.470000),
    "joint_9.0": (-0.196000, 1.610000),
    "joint_10.0": (-0.174000, 1.709000),
    "joint_11.0": (-0.227000, 1.618000),
    "joint_12.0": (0.263000, 1.396000),
    "joint_13.0": (-0.105000, 1.163000),
    "joint_14.0": (-0.189000, 1.644000),
    "joint_15.0": (-0.162000, 1.719000),
}
