#!/bin/bash
# ===== 은닉 정본 (2026-09-01 승격, 원본 run B02_20260831_164053) =====
# 실기 배포 계열 기준. 비전 불요, actor 는 캡 상태를 일절 안 본다.
#
# B02 = B01 + slip2 강화(W 3.0->5.0, 접촉문턱 1.0->0.3N) — 가벼운 접촉의 역슬립까지 과금
# B01 = A99 + 역슬립 sr 램프 0.5 (A67 교훈: 접촉 중에만 걸리는 과금은 형성기 손가락 떼기 유발)
# A99 = A98 + 비대칭 역슬립 불감대 2mm (순방향 8mm 유지)
# A98 = A96 + 팜다운 15도/W3.0
# A96 = A95 + MLP 추론 초기자세(rhinit_z193_up31)
# A95 = A94 + 텀블러 31mm 상승(실물 지그 정합) + 커리큘럼 저항 4Nm/마찰 2.0
# A94 = 각도·각속도 관측 제거 + 낙하벌 100 + 완주보너스 강화(RELEASE/LIFT 6) — 이전 은닉 정본
#
# 승격 근거 (2026-09-01 실측, 신선 env 결정론 재생):
#   - 커리큘럼 전 구간 완주: 저항 4Nm @ep4032, 마찰 2.0 @ep4569
#   - 20000ep 완주, 후반 퇴화 없음 (A94 는 ep15000 이후 붕괴)
#   - 배포 후보 ep5400: 저항4Nm x mu{2.0,3.0,5.0} 및 저항0.2/mu5.0 에서 44/45 (98%)
#     저항 8Nm 에서도 6/6. 하한은 mu1.5 (0/5)
#   - A95 대비 전 조건 상회 (A95 ep11200: 동일 조건 3~5/5)
#
# 알려진 한계 (2026-09-01 실측):
#   - 캡 실접촉은 엄지·중지 2점(각 100%), 검지 3% / 약지 7% — 두 손가락은 신전 한계에
#     붙여 능동 회피. 원인은 grip floor(4N) 아래 무보상 + slip2 과금(0.3N) 시작점 어긋남.
#     회전 토크는 J롤(팔 J4-6 28Nm)이 내므로 성능에는 안 나타난다.
#
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.031 CAP_Z=0.246 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=4,2,4,4 CAP_GRIP_FLOOR_RAMP=8.0 GRIP_W=2.0 CAP_SLIP2_W=5.0 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B02B_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
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
