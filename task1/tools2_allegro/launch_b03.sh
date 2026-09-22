#!/bin/bash
# B03 = 은닉 정본(B02) + grip 항 복구 3변수. 단일 목적: 캡 실접촉 2점 -> 4점.
#
# B02 실측 문제: 캡 압착률 엄지100% 중지100% / 검지 3% 약지 7%.
# 두 손가락은 PIP/DIP 를 신전 한계에 붙여 능동 회피(굴곡 여유 1.85~1.90 rad 남김).
# 원인 3층, 각각에 1변수:
#   (1) grip floor 4N + ramp 8N -> 만점에 12N 필요한데 실제 4~5.6N. reward_grip 0.09/step
#       으로 항이 사실상 꺼짐(설계 의도 최대 2.0). => floor 1,1,1,1 / ramp 4
#   (2) slip2 과금은 0.3N 부터인데 grip 보상은 4N 부터 -> 그 사이가 "보상 0 벌점만" 구간.
#       탐지 문턱은 유지하고(가벼운 역슬립 계속 잡음) 과금만 접촉력 비례. => CAP_SLIP2_FORCE_REF=4
#   (3) grip 평균이 5칸(5번째 = 약지 중복) -> 상한 2/5 로 희석. => CAP_GRIP_MEAN4=1
#   (4) grasp_quality 의 contact_frac = contact_count/contact_ref(기본 4) 인데 contact_count 는
#       5칸 합이라 엄지·중지·약지 3지만 붙어도 4 -> 1.0 포화. "4지 유도 항"이 3지에서 만점을
#       주고 검지 기여가 0 이 된다. => CONTACT_REF=5 GRASP_STABLE_MIN=0.10 (실슬롯 수와 일치)
# GRIP_W 는 정본값 2.0 유지 (스케일을 새로 키우는 게 아니라 설계값 복원).
# normalize_value/normalize_advantage 가 켜져 있어 절대 스케일보다 상대 비중이 중요.
#
# 판정: 캡 압착률(CAP_FSR_GEO_DEBUG=1, 현재 캡 기준) 4지 각 >50% 목표.
#       성능 회귀 감시 = 신선 env 결정론 재생 저항4 x mu{2,3,5} (정본 ep5400 = 44/45).
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.031 CAP_Z=0.246 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=1,1,1,1 CAP_GRIP_FLOOR_RAMP=4.0 GRIP_W=2.0 CAP_GRIP_MEAN4=1 CAP_SLIP2_FORCE_REF=4 CONTACT_REF=5 GRASP_STABLE_MIN=0.10 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B03_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_START=0.2 CAP_RESIST_END=4.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/rhinit_z193_up31.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount_siltip2.urdf \
  CAP_DEMO_FOLLOW_TUMBLER=1 CAP_TUMBLER_RESET_CREATION=1 CAP_RESET_FINGER_TARGETS=1 \
  CAP_ROT_GRIP=1 CAP_SCREW_HOLD_FORCE=2000 CAP_THUMB_ROT_W=2.0 CAP_UNSCREW_CONTACT_SCALE=1 \
  CAP_UNSCREW_THUMB_SCALE=1 CAP_THUMB_SCALE_REF=4.0 CAP_REWARD_ZONE_FORCE=1 \
  CAP_NO_RESET_ON_SUCCESS=1 CAP_OBS_HIDE_POS=1 CAP_OBS_HIDE_LINVEL=1 \
  CAP_OBS_TIP_FORCE6=1 CAP_TIP_FORCE_FILTER=0.3 \
  CAP_PROBE_BACKSPAN=0.024 PALM_FACING_COS=0.87 \
  CAP_ACCEL_LIMIT_HAND=50 CAP_ACCEL_LIMIT_ARM=50 \
  /home/leegyuwon/Documents/task1/tools2_allegro/train_cap_unscrew_arm.sh
