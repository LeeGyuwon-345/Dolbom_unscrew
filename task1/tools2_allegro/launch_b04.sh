#!/bin/bash
# B04 = 은닉 정본(B02) + grip 항 복구 4변수. B03 에서 SLIP2_FORCE_REF 만 제거.
#
# B03 결과 (20000ep 완주, sr 0.993): 약지 접촉률은 정본 대비 2배(39% vs 21%)로 올랐으나
# **J롤이 사라졌다**(J6 기여 9~13% 고정, 정본은 저항 1.7N·m 에서 106% 로 전환).
# 원인 실측 — 동일 조건(저항 3.9N·m·μ5.0) 팁 슬립:
#     정본 11.9mm / 벌점 0.266/step   vs   B03 28.4mm / 벌점 0.391/step
# 즉 B03 은 "가볍게 4지를 얹고 문질러서" 연다. 접촉률 상승은 파지가 아니라 스침이다.
#
# 인과: 정본에서 J롤이 나온 이유는 슬립이 비쌌기 때문이다(SLIP2_W=5.0). 슬립 없이
# 캡을 돌리는 유일한 길이 손목째 돌리기(손과 캡이 함께 움직여 상대 미끄러짐 0)였다.
# CAP_SLIP2_FORCE_REF=4 는 접촉력 비례 과금이라 1~2N 의 가벼운 접촉을 25~50% 로 감액했고,
# GRIP_FLOOR=1 이 그 가벼운 접촉에 보상까지 줘서 "얹고 문지르기"가 싸고 이득인 전략이 됐다.
# => SLIP2_FORCE_REF 제거. 슬립은 다시 전액 과금(J롤 유도 복원), grip 쪽만 살린다.
#
# 남는 4변수 (모두 grip 신호 정상화):
#   (1) GRIP_FLOOR 4,2,4,4 -> 1,1,1,1  : floor 미만 무보상 사각지대 제거
#   (2) GRIP_FLOOR_RAMP 8.0 -> 4.0     : 만점 12N -> 5N (실제 힘 4~5.6N)
#   (3) CAP_GRIP_MEAN4=1               : grip 평균 5칸(약지 중복) -> 4칸
#   (4) CONTACT_REF 4 -> 5 (+ GRASP_STABLE_MIN 0.125 -> 0.10 연동)
#                                      : grasp_quality 가 3지에서 만점 포화하던 것 교정
#   주의: CAP_CONTACT_REF 는 런처가 "${CONTACT_REF:-4}" 로 받으므로 메타변수로 줘야 한다.
#
# 판정: 4지 압착률(CAP_FSR_GEO_DEBUG=1, 현재 캡 기준) + J6 기여율 + 팁 슬립(mm).
#       목표 = 검지·약지 >50% 이면서 J6 기여 100% 대(J롤 유지), 슬립 정본 수준(~12mm).
# 텀블러 상승 31mm -> 22mm (2026-09-02 사용자 지정, 실물 지그 변경):
#   TUMBLER_POS z 0.031->0.022, CAP_Z 0.246->0.237(캡 중심), 초기자세 up31 -> up22.
#   초기자세는 손목을 IK 로 정확히 9mm 내려 만들었다(팔6축만, 손16축 유지, 오차 0.002mm).
#   검증: 캡 대비 상대 기하가 up31 과 동일 — 캡벽 갭 검지-2.7 엄지-4.3 중지+2.9 약지-3.4mm,
#   팜(base_link) 캡상면 대비 +122mm, 엘보우업 OK.
# 텀블러 지그 오차 모사 (2026-09-02 사용자 지정, 실기 관점):
#   CAP_TUMBLER_POS_NOISE_MM=2 — 리셋마다 env 별 ±2mm 균일 노이즈를 xyz 에 준다.
#   **실제 텀블러만 움직이고 관측·데모 기준은 공칭 위치 고정.** demo_data(손목 기준자세·
#   cap_home·obj_to_joints)는 액터 생성 전에 CAP_GRASP_TUMBLER_POS 로 한 번 만들어지므로
#   노이즈를 모른다 = 로봇 PC 가 지그 좌표를 상수로 쓰는 실제 배포 상황과 동일(오차를 모른 채 잡는다).
#   보상·FSR 기하는 _cap_src(실제 강체)를 읽으므로 진짜 위치를 본다.
#   검증: 기본 off 에서 정본 ep5400 재생 100% 무회귀, 20mm 로 키워 리셋별 재추첨 확인.
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_TUMBLER_POS_NOISE_MM=2 CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.237 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=1,1,1,1 CAP_GRIP_FLOOR_RAMP=4.0 GRIP_W=2.0 CAP_GRIP_MEAN4=1 CONTACT_REF=5 GRASP_STABLE_MIN=0.10 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B04_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_START=0.2 CAP_RESIST_END=4.0 \
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
