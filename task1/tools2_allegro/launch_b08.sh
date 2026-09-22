#!/bin/bash
# B08 = 신 정본(launch_baseline_b02g.sh) + 2변수. 보상·관측은 정본 그대로.
#
# B03~B07 다섯 run 에서 정본 대비 얹었던 변경 중 근거가 남은 것만 유지한다:
#
#  [유지] CAP_TUMBLER_POS_NOISE_MM=2  — 실기 지그 오차 모사. 성능 실험이 아니라 실물 정합.
#         관측 기준은 공칭 고정이라 sim2real 격차를 안 만든다. B04 는 이 상태로 1차 추첨 형성 성공.
#         단서: 초반 게이트 전진이 약 35% 느려진다.
#  [유지] CAP_RESIST_STEP=0.3        — 저항 계단 0.1->0.3 (35계단 -> 13계단).
#         B07 이 저항 4.0N·m 에 ep2602 도달(정본 ep4032, 35% 단축). 12계단 전부 통과.
#         단서: B07 sr 이 0.65~0.85 로 낮았으나 grip·슬립 변경과 교란돼 원인 불명.
#               이번 run 에서 계단만 남기므로 sr>=0.95 면 계단은 무죄로 확정된다.
#
#  [폐기] grip 복구 4건(FLOOR 2,2,2,2 / RAMP 4 / MEAN4 / CONTACT_REF 5)
#         4지 접촉은 확실히 만들지만(검지 3%->100%) J롤을 죽인다(J6 106%->17~45%).
#         B04·B05·B06·B07 네 번 모두 재현. 4지와 J롤은 현 구조에서 배타적이다.
#  [폐기] CAP_SLIP2_W 7.0  — B06 슬립 9.0mm vs 정본 8.9mm, 차이 없음.
#  [폐기] CAP_SLIP2_DIR_W=2,8 — 역효과. J6 45%->17%, 역슬립 5.8->6.0mm, sr 하락.
#  [폐기] 저항 시작 0.5/1.0 — 형성만 어려워진다. 재추첨 정본 1회 / 1.0 은 5회 / 0.5 는 7회.
#
# 기구·배치는 신 정본과 동일(지그 22mm, 몸체 228mm, 캡 0.250, 마운트면-손등 32.0mm).
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_TUMBLER_POS_NOISE_MM=2 TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_r44_fric02_H228.urdf CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.250 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=4,2,4,4 CAP_GRIP_FLOOR_RAMP=8.0 GRIP_W=2.0 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B08_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
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
