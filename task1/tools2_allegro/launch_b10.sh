#!/bin/bash
# B10 = B09(z 슬립 불감대 3mm) + z 과금 sr 램프 0.5. B08 대비 실질 단일 변수(z 억제).
#
# B09 실패 이력: z_dead=3mm 을 램프 없이 걸었다가 **형성 39연속 실패**(14시간 소득 0).
#   ep400 판정 rew 평균 -678, 최고 -412, 합격 0회. 동일 구성에서 z_dead 만 뺀 B08 은
#   재추첨 0회로 형성했으므로 단일 변수 귀속이 명확하다.
#   원인: 파지를 만들려면 손끝이 캡 위를 더듬어야 하는데 수직 편차 3mm 는 곧바로 넘는다.
#   "닿는 순간 벌점"이 되어 정책이 손을 떼는 쪽으로 굳는다(A67 교훈).
#   kc(0.4->1)는 시간 기반이라 형성이 안 되고 있어도 계속 올라가 보호가 안 됐다.
#
# 대책: 역방향(CAP_SLIP2_BACK_SR_RAMP)과 같은 구조의 z 전용 sr 램프를 추가했다.
#   _pfz2 *= min(1, sr_ema / 0.5)  -> 성공률이 0.5 에 이르기 전에는 z 과금이 비례 축소된다.
#   형성이 끝난 뒤에야 z 억제가 실효되므로 회피가 생기지 않는다.
#
# 목적은 시뮬 성능이 아니라 **실물 팁 마모·캡 표면 보호**다. z 슬립 실측: 정본 6.5mm / B08 4.8mm.
#   성능이 떨어지면 되돌릴 것.
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_TUMBLER_POS_NOISE_MM=2 TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_r44_fric02_H228.urdf CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.250 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=4,2,4,4 CAP_GRIP_FLOOR_RAMP=8.0 GRIP_W=2.0 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_Z_DEAD_MM=3 CAP_SLIP2_Z_SR_RAMP=0.5 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B10_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_STEP=0.3 CAP_RESIST_START=0.2 CAP_RESIST_END=4.0 \
  INIT_POSE=/home/leegyuwon/Documents/task1/tools2_allegro/poses/rhinit_z193_H228.json \
  RB5_ALLEGRO_URDF=/home/leegyuwon/Documents/task1/assets/rb5_allegro/rb5_allegro_vmount_siltip2.urdf \
  CAP_DEMO_FOLLOW_TUMBLER=1 CAP_TUMBLER_RESET_CREATION=1 CAP_RESET_FINGER_TARGETS=1 \
  CAP_ROT_GRIP=1 CAP_SCREW_HOLD_FORCE=2000 CAP_THUMB_ROT_W=2.0 CAP_UNSCREW_CONTACT_SCALE=1 \
  CAP_UNSCREW_THUMB_SCALE=1 CAP_THUMB_SCALE_REF=4.0 CAP_REWARD_ZONE_FORCE=1 \
  CAP_NO_RESET_ON_SUCCESS=1 CAP_OBS_HIDE_POS=1 CAP_OBS_HIDE_LINVEL=1 \
  CAP_OBS_TIP_FORCE6=1 CAP_TIP_FORCE_FILTER=0.3 \
  CAP_PROBE_BACKSPAN=0.024 PALM_FACING_COS=0.87 \
  CAP_ACCEL_LIMIT_HAND=50 CAP_ACCEL_LIMIT_ARM=50 \
  /home/leegyuwon/Documents/task1/tools2_allegro/train_cap_unscrew_arm.sh
