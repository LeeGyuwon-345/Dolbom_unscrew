#!/bin/bash
# B07 = B06 + 저항 커리큘럼 시작 1.0 -> 0.5N·m (사용자 지정).
#
# B06 이력: 저항 시작 1.0 으로는 **형성에 5번 실패**했다(ep400~500 rew -437~-664).
#   6번째에 걸렸고 이후는 정상(ep1600 sr 0.967). 형성 자체는 저항과 무관하지만,
#   초기에 캡이 안 돌면 unscrew 수입이 없어 워처의 rew<0 판정선을 넘기 어렵다.
#   0.5 는 그 부담을 줄이면서 정본(0.2)보다는 높다.
#
# ⚠️ 기구 변경 동반: rb5_allegro_vmount_siltip2.urdf 의 hand_mount y 를
#   -0.060 -> -0.0615 로 고쳤다(2026-09-02). RB5 마운트면-손등 최소거리
#   30.5mm -> **32.0mm** (실측 정합). B06 까지는 30.5mm 기구로 학습됐다.
#   옛 기구 사본: rb5_allegro_vmount_siltip2_y060.urdf
#
# 미해결 과제 (B03~B06 공통): grip 항을 살리면 4지 접촉은 확실히 생기지만
#   J롤이 안 선다(J6 기여 12~45%, 정본 106%). 슬립을 정본 수준(9.0mm)까지
#   낮춘 B06 에서도 45% 였다. 4지 접촉과 J롤이 배타적일 가능성 —
#   손목을 크게 굴리면 4점 접촉이 깨지므로 grip 수입과 상충한다.
#   저항 2.6N·m 이상(μ5.0 에서 비비기 물리 한계)에서 어느 쪽으로 가는지가 관건.
# 저항 커리큘럼 계단 0.1 -> 0.3N·m (사용자 지정):
#   0.5->4.0 이 35계단에서 12계단으로 줄어 고저항 영역에 훨씬 빨리 닿는다.
#   의도는 속도만이 아니다 — 0.1 씩 오르면 정책이 매 단계 "비비기"를 조금씩 다듬어
#   적응해 버린다. 계단을 키우면 그 점진적 적응이 끊겨 다른 전략(J롤)을 찾을 여지가 생긴다.
#   위험: B02 의 마찰 2.0 전이처럼 큰 점프에서 sr 이 일시 붕괴할 수 있다(정상 경로).
# 텀블러 몸체 높이 215 -> 228mm (2026-09-02 사용자 지정):
#   자산 재생성: make_cyl_tumbler.py --r .044 --h .016 --R .044 --H 0.228 --capz 0.228
#                --spin-friction 0.2 -> tumbler_cyl_r44_fric02_H228.urdf
#   캡 원점(=cap_home) 월드 z = 텀블러루트 0.022 + 0.228 = **0.250** (캡 상면 0.266).
#   CAP_Z 는 월드 절대 z 다(관측 dump 로 확인: cap_home = 0.4, 0, CAP_Z).
#   초기자세: up22 에서 손목만 +13mm IK (팔6축, 오차 0.004mm, 손16축 불변)
#            -> poses/rhinit_z193_H228.json
#   검증: 캡 대비 상대 기하 동일 — 캡벽 갭 검지-2.6 엄지-4.3 중지+2.8 약지-3.6mm.
# 슬립 방향별 가중치 (2026-09-02 사용자 지정): CAP_SLIP2_DIR_W=2,8 (순방향 2 / 역방향 8).
#   벌점 = kc·( 2·mean(순방향+z 성분, 불감대 8mm) + 8·mean(역방향 성분, 불감대 2mm) )
#   켜면 공통 CAP_SLIP2_W(7.0)는 사용되지 않는다.
#   근거: 역슬립(팁이 캡보다 뒤처짐 = 캡이 손끝 밑으로 빠져나감)을 줄이는 길은
#   손목을 굴려 손 전체가 캡과 함께 도는 것뿐이다 — 손가락 가동범위로는 캡을 못 따라간다.
#   실측(저항 1.7~2.0N·m): 정본 순1.4/역7.3mm, B06 순1.3/역5.8mm — 양쪽 다 역방향이 4~5배.
#   순방향은 1.3~1.4mm 로 불감대(8mm) 안이라 사실상 무과금이고, 재배치(개깅) 탐색을 막지 않는다.
#   형성기 보호는 CAP_SLIP2_BACK_SR_RAMP=0.5 가 담당(성공률 EMA 도달 전 역방향 과금 비례 축소).
# TAG 자동 생성. setsid nohup 으로 기동할 것.
exec env TUMBLER_URDF=/home/leegyuwon/Documents/task1/assets/tumbler/tumbler_cyl_r44_fric02_H228.urdf CAP_TUMBLER_POS_NOISE_MM=2 CAP_SLIP2_BACK_DEAD_MM=2 CAP_SLIP2_BACK_SR_RAMP=0.5 CAP_FRIC_AUDIT=1 TUMBLER_POS=0.4,0,0.022 CAP_Z=0.250 CAP_RESIST_AUDIT=1 CAP_PRIV_RESIST=1 HIDE_RISE=0 CAP_REL_HOME_XY=1 CAP_OBS_COMPACT=1 CAP_ACTOR_PRIV_KEEP=22 PALM_DOWN_SR_RAMP=0.5 PALM_DOWN_W=3.0 PALM_DOWN_MAX=15 OPPOSE_W=2.0 OPPOSE_SEP=140 CAP_HIDE_CAP_QUAT=1 CAP_HIDE_CAP_VEL=1 CAP_REL_SPIN_FREE=1 CAP_BAND_LO=0.012 CAP_BAND_LO_SLACK_MM=2 CAP_RADIUS_LO=0.044 CAP_RADIUS_HI=0.044 CAP_CSCALE_FLOOR=0.25 CAP_GRIP_FLOOR=2,2,2,2 CAP_GRIP_FLOOR_RAMP=4.0 GRIP_W=2.0 CAP_GRIP_MEAN4=1 CONTACT_REF=5 GRASP_STABLE_MIN=0.10 CAP_SLIP2_W=7.0 CAP_SLIP2_DIR_W=2,8 CAP_SLIP2_DEAD_MM=8 CAP_SLIP2_REF_MM=45 CAP_SLIP2_CONTACT_N=0.3 EPISODE_STEPS=360 CAP_GRAVITY_RAMP_STEPS=1920 CAP_ENERGY_W_HAND=1.0 CAP_ENERGY_W_ARM=1.667 CAP_ENERGY_ALPHA=0.1 NUM_ENVS=512 MAX_ITER=20000 TAG=B07_$(date +%Y%m%d_%H%M%S) PYTHONUNBUFFERED=1 HEADLESS=true \
  EXTRA_HYDRA=task.task.randomize=False \
  ACTION_DELTA_SCALE_ARM=0.02 ARM_HAND_OBS=0 DROP_FAIL_PENALTY=100 RELEASE_W=6.0 LIFT_W=6.0 \
  HAND_FRICTION=5.0 OBJ_FRICTION=5.0 CAP_NONTIP_FRICTION=0.0 \
  CAP_FRICTION_CURRICULUM=1 CAP_FRICTION_CURR_START=5.0 CAP_FRICTION_CURR_END=2.0 CAP_FRICTION_CURR_STEP=0.5 \
  CAP_RESIST_CURRICULUM=1 CAP_RESIST_STEP=0.3 CAP_RESIST_START=0.5 CAP_RESIST_END=4.0 \
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
