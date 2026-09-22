#!/bin/bash
# B06 = B05 구성(단단한 파지 2변수)을 **스크래치**로 + 저항 커리큘럼 시작 0.2 -> 1.0N·m.
#
# 왜 스크래치인가: B05(웜스타트, ep2500 이어받기)는 보상을 바꿔도 스타일이 안 바뀌었다.
#   저항 3.2N·m 에서 J6 기여 38%(정본 102%), 팁슬립 22.7mm — 오히려 악화.
#   원인 추정: J롤은 정본에서 저항 0.8N·m 시점에 이미 싹이 텄다(J6 기여 59%).
#   B04 가 그 구간을 "비비기"로 통과해 버렸고 B05 는 그 정책을 물려받았다.
#   RL 은 현재 정책 주변만 탐색하므로, 비비기로 sr 1.000 을 내는 국소최적에서
#   J롤이라는 전혀 다른 운동으로 건너뛰지 못한다. => 스타일이 정해지는 초반부터 다시.
#
# 왜 저항 시작 1.0 인가 (사용자 지정): 형성을 이끄는 항(near/pinch/grip/hold/closure)은
#   저항과 무관하므로 시작값을 올려도 파지 형성과 재추첨 판정은 영향이 없다.
#   저항이 붙는 항은 unscrew 하나뿐이다.
#
# 단서 — 이것만으로는 비비기가 안 막힐 수 있다:
#   비비기 최대 토크 τ = μ·r·ΣN (r=0.044, ΣN 실측 11~12N)
#     μ5.0 → 2.4~2.6N·m  |  μ3.0 → 1.5N·m  |  μ2.0 → 1.0N·m
#   커리큘럼 시작 마찰이 5.0 이라 1.0N·m 은 여전히 비벼서 열 수 있다(여유 2.6배).
#   실패하면 다음 카드는 "마찰 먼저 낮추기"(μ2 에서는 1.0N·m 이상 비비기 불가).
#
# B05 대비: CKPT 제거(스크래치), CAP_RESIST_START 2.1 -> 1.0.
# 판정: J6 기여율 60%+ 로 오르고 4지 압착률 유지. 슬립 정본 수준(~9~12mm).
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_TUMBLER_POS_NOISE_MM=2 CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.237 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=2,2,2,2 CAP_GRIP_FLOOR_RAMP=4.0 GRIP_W=2.0 CAP_GRIP_MEAN4=1 CONTACT_REF=5 GRASP_STABLE_MIN=0.10 CAP_SLIP2_W=7.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B06_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_START=1.0 CAP_RESIST_END=4.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/rhinit_z193_up22.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount_siltip2.urdf \
  CAP_DEMO_FOLLOW_TUMBLER=1 CAP_TUMBLER_RESET_CREATION=1 CAP_RESET_FINGER_TARGETS=1 \
  CAP_ROT_GRIP=1 CAP_SCREW_HOLD_FORCE=2000 CAP_THUMB_ROT_W=2.0 CAP_UNSCREW_CONTACT_SCALE=1 \
  CAP_UNSCREW_THUMB_SCALE=1 CAP_THUMB_SCALE_REF=4.0 CAP_REWARD_ZONE_FORCE=1 \
  CAP_NO_RESET_ON_SUCCESS=1 CAP_OBS_HIDE_POS=1 CAP_OBS_HIDE_LINVEL=1 \
  CAP_OBS_TIP_FORCE6=1 CAP_TIP_FORCE_FILTER=0.3 \
  CAP_PROBE_BACKSPAN=0.024 PALM_FACING_COS=0.87 \
  CAP_ACCEL_LIMIT_HAND=50 CAP_ACCEL_LIMIT_ARM=50 \
  /home/leegyuwon/Documents/task1/tools2_allegro/train_cap_unscrew_arm.sh
